"""
Error Analysis - qualitative analysis of model misclassifications.

This module collects false positive and false negative examples from the test set for each classical model, along with their linguistic features and text context.
It then generates plots comparing error rates across models, linguistic feature distributions for FP vs FN, and identifies posts that are systematically misclassified by multiple models.
Finally, it surfaces interpretable patterns in the errors that can be included in the technical report.
"""

import logging
from collections import Counter, defaultdict
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from datasets import DatasetDict

from .config import RESULTS_SUBDIRS
from .results import (
    cache_exists,
    json_exists,
    load_joblib,
    load_json,
    plot_path,
    record_step,
    sanitise_for_json,
    save_json,
)

logger = logging.getLogger(__name__)

# Linguistic features available in the preprocessed dataset
_LING_FEATURES = [
    "word_count",
    "char_count",
    "sentence_count",
    "avg_sentence_length",
    "type_token_ratio",
    "pronoun_ratio",
    "negation_ratio",
    "social_word_ratio",
    "emotion_word_ratio",
    "noun_ratio",
    "verb_ratio",
    "adj_ratio",
    "adv_ratio",
]

_CLR = {
    "TP": "#4C9BE8",
    "TN": "#4CE8A0",
    "FP": "#E8C84C",
    "FN": "#E8614C",
}


# Helpers
def _savefig(name: str) -> None:
    p = plot_path(name)
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Plot saved → {p}")


def _outcome(y_true: int, y_pred: int) -> str:
    """Return 'TP', 'TN', 'FP', or 'FN'."""
    if y_true == 1 and y_pred == 1:
        return "TP"
    if y_true == 0 and y_pred == 0:
        return "TN"
    if y_true == 0 and y_pred == 1:
        return "FP"  # model says lonely, actually not
    return "FN"                   # model says not lonely, actually is


def _row_features(example: dict) -> dict:
    """Extract serialisable feature dict from a preprocessed dataset row."""
    return {f: round(float(example.get(f, 0.0)), 4) for f in _LING_FEATURES}


def _describe_group(rows: list[dict], feat: str) -> dict:
    """Descriptive stats for a feature across a list of example dicts."""
    vals = [r.get(feat, 0.0) for r in rows]
    if not vals:
        return {}
    a = np.array(vals, dtype=float)
    return {
        "mean": round(float(np.mean(a)), 4),
        "median": round(float(np.median(a)), 4),
        "std": round(float(np.std(a)), 4),
    }


# Per-model error collection
def _collect_errors(
    model,
    X_test,
    y_test: np.ndarray,
    test_split,
    rep_name: str,
    model_name: str,
    max_examples: int = 50,
) -> dict:
    """
    Run model on X_test, collect FP and FN cases with full context.

    Parameters
    ----------
    model       : fitted sklearn model (or pipeline)
    X_test      : feature matrix for test split
    y_test      : ground-truth labels (numpy array)
    test_split  : HuggingFace Dataset — test split of the preprocessed dataset
    max_examples: max FPs and FNs to store individually (memory/storage limit)

    Returns
    -------
    dict with 'false_positives', 'false_negatives', 'summary'
    """
    preds = model.predict(X_test)

    fp_examples, fn_examples = [], []
    outcome_counts: Counter = Counter()

    for idx, (y_true, y_pred) in enumerate(zip(y_test, preds)):
        outcome = _outcome(int(y_true), int(y_pred))
        outcome_counts[outcome] += 1

        if outcome not in ("FP", "FN"):
            continue

        row = test_split[idx]
        example = {
            "dataset_idx": int(row.get("idx", idx)),
            "unique_id": str(row.get("unique_id", "")),
            "true_label": int(y_true),
            "predicted_label": int(y_pred),
            "outcome": outcome,
            # Text
            "original_text": str(row.get("text", ""))[:500],   # truncate for storage
            "cleaned_text": str(row.get("cleaned", ""))[:500],
            "word_count": int(row.get("word_count", 0)),
            # Linguistic features
            "features": _row_features(row),
        }

        if outcome == "FP" and len(fp_examples) < max_examples:
            fp_examples.append(example)
        elif outcome == "FN" and len(fn_examples) < max_examples:
            fn_examples.append(example)

    total = sum(outcome_counts.values())
    summary = {
        "representation": rep_name,
        "model": model_name,
        "total_test": total,
        **{k: outcome_counts[k] for k in ("TP", "TN", "FP", "FN")},
        "error_rate": round((outcome_counts["FP"] + outcome_counts["FN"]) / max(total, 1), 4),
        "fp_rate": round(outcome_counts["FP"] / max(outcome_counts["FP"] + outcome_counts["TN"], 1), 4),
        "fn_rate": round(outcome_counts["FN"] / max(outcome_counts["FN"] + outcome_counts["TP"], 1), 4),
    }

    # Linguistic feature averages per error type
    for bucket_name, bucket in [("false_positives", fp_examples), ("false_negatives", fn_examples)]:
        if bucket:
            summary[f"{bucket_name}_avg_features"] = {
                f: _describe_group(bucket, f)["mean"]
                for f in _LING_FEATURES
            }

    result = {
        "summary": summary,
        "false_positives": fp_examples,
        "false_negatives": fn_examples,
    }

    key = f"error_analysis_{rep_name}_{model_name}.json"
    save_json(key, sanitise_for_json(result))
    logger.info(
        f"  {rep_name}/{model_name} → "
        f"FP: {outcome_counts['FP']}, FN: {outcome_counts['FN']}, "
        f"error rate: {summary['error_rate']:.3f}"
    )
    return result


# Cross-model analysis
def _confusion_overlap(all_errors: dict[str, dict], test_split) -> dict:
    """
    Find posts misclassified by the most models — the 'hardest' examples.

    Returns a list of posts with their error count and which models got them wrong.
    """
    # Map unique_id → list of (rep, model) pairs that got it wrong
    post_errors: dict[str, list[str]] = defaultdict(list)

    for model_key, error_data in all_errors.items():
        for bucket in ("false_positives", "false_negatives"):
            for ex in error_data.get(bucket, []):
                uid = ex.get("unique_id", str(ex.get("dataset_idx", "")))
                post_errors[uid].append(model_key)

    # Sort by error count
    ranked = sorted(post_errors.items(), key=lambda x: len(x[1]), reverse=True)

    overlap = []
    for uid, model_keys in ranked[:20]:  # top-20 most confused
        overlap.append({
            "unique_id": uid,
            "n_models_wrong": len(model_keys),
            "models": model_keys,
        })

    return {"top_confused_posts": overlap}


def _plot_fp_fn_counts(summaries: list[dict]) -> None:
    """Bar chart of FP and FN counts per model."""
    if not summaries:
        return

    labels = [f"{s['representation']}\n{s['model']}" for s in summaries]
    fps = [s["FP"] for s in summaries]
    fns = [s["FN"] for s in summaries]

    x = np.arange(len(labels))
    width = 0.38

    fig, ax = plt.subplots(figsize=(max(12, len(labels) * 1.4), 5))
    bars1 = ax.bar(x - width / 2, fps, width, label="False Positives (FP)", color=_CLR["FP"], alpha=0.85)
    bars2 = ax.bar(x + width / 2, fns, width, label="False Negatives (FN)",  color=_CLR["FN"], alpha=0.85)
    ax.bar_label(bars1, fmt="%d", padding=2, fontsize=8)
    ax.bar_label(bars2, fmt="%d", padding=2, fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Count (test set)")
    ax.set_title("False Positives and False Negatives per Model (Test Set)", fontweight="bold")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    _savefig("error_analysis_fp_fn_counts.png")


def _plot_length_by_error_type(best_error_data: dict, rep_name: str, model_name: str) -> None:
    """
    Boxplot: word count distributions for TP / FP / FN / TN.
    Uses only the best model to keep the plot readable.
    Shows whether post length is a systematic predictor of errors.
    """
    # Rebuild from stored examples — we only have FP/FN stored individually;
    # for TP/TN we use the summary counts only (can't plot distribution without
    # iterating the full test set again, which requires the model).
    # So: plot what we have (FP, FN word counts) and note TP/TN unavailable.
    fp_lengths = [e["word_count"] for e in best_error_data.get("false_positives", [])]
    fn_lengths = [e["word_count"] for e in best_error_data.get("false_negatives", [])]

    if not fp_lengths and not fn_lengths:
        return

    data, tick_labels, colors = [], [], []
    if fp_lengths:
        data.append(fp_lengths)
        tick_labels.append(f"FP\n(n={len(fp_lengths)})")
        colors.append(_CLR["FP"])
    if fn_lengths:
        data.append(fn_lengths)
        tick_labels.append(f"FN\n(n={len(fn_lengths)})")
        colors.append(_CLR["FN"])

    fig, ax = plt.subplots(figsize=(7, 5))
    bp = ax.boxplot(data, labels=tick_labels, patch_artist=True, widths=0.5)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)

    ax.set_ylabel("Word Count")
    ax.set_title(
        f"Word Count Distribution by Error Type\n"
        f"({rep_name} / {model_name}, test set — up to 50 examples per bucket)",
        fontweight="bold",
        fontsize=11,
    )
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    _savefig("error_analysis_length_by_error_type.png")


def _plot_linguistic_fp_vs_fn(best_error_data: dict, rep_name: str, model_name: str) -> None:
    """
    Grouped bar: mean linguistic features for FP vs FN.
    Reveals which linguistic properties distinguish the two error types.
    """
    fp_examples = best_error_data.get("false_positives", [])
    fn_examples = best_error_data.get("false_negatives", [])

    if not fp_examples or not fn_examples:
        logger.warning("Not enough FP/FN examples to plot linguistic comparison.")
        return

    # Select ratio features (exclude raw counts for comparability)
    ratio_feats = [
        "pronoun_ratio", "negation_ratio", "social_word_ratio",
        "emotion_word_ratio", "type_token_ratio",
        "noun_ratio", "verb_ratio", "adj_ratio", "adv_ratio",
    ]

    fp_means = [
        np.mean([e["features"].get(f, 0.0) for e in fp_examples]) for f in ratio_feats
    ]
    fn_means = [
        np.mean([e["features"].get(f, 0.0) for e in fn_examples]) for f in ratio_feats
    ]

    x = np.arange(len(ratio_feats))
    width = 0.38

    fig, ax = plt.subplots(figsize=(14, 5))
    bars1 = ax.bar(x - width / 2, fp_means, width, label=f"False Positives (n={len(fp_examples)})",
                   color=_CLR["FP"], alpha=0.85)
    bars2 = ax.bar(x + width / 2, fn_means, width, label=f"False Negatives (n={len(fn_examples)})",
                   color=_CLR["FN"], alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels([f.replace("_", "\n") for f in ratio_feats], fontsize=8)
    ax.set_ylabel("Mean Feature Value")
    ax.set_title(
        f"Linguistic Features — False Positives vs False Negatives\n"
        f"({rep_name} / {model_name}, test set)",
        fontweight="bold",
        fontsize=12,
    )
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    _savefig("error_analysis_linguistic_fp_vs_fn.png")


def _plot_confusion_overlap(overlap_data: dict) -> None:
    """
    Bar chart: how many posts are confused by N models simultaneously.
    Answers: 'Are errors systematic or model-specific?'
    """
    posts = overlap_data.get("top_confused_posts", [])
    if not posts:
        return

    # Histogram of n_models_wrong
    counts = Counter(p["n_models_wrong"] for p in posts)
    ks = sorted(counts.keys())
    vs = [counts[k] for k in ks]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar([str(k) for k in ks], vs, color="#7c3aed", alpha=0.85)
    ax.set_xlabel("Number of models that misclassified the post")
    ax.set_ylabel("Number of posts")
    ax.set_title(
        "Error Overlap — Posts Misclassified by Multiple Models\n"
        "(higher = systematically hard examples)",
        fontweight="bold",
        fontsize=11,
    )
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    _savefig("error_analysis_confusion_overlap.png")


# Qualitative patterns
def _qualitative_patterns(best_error_data: dict) -> dict:
    """
    Surface interpretable patterns in FP/FN examples.

    Returns a dict of human-readable observations suitable for
    copying into the technical report.
    """
    fp = best_error_data.get("false_positives", [])
    fn = best_error_data.get("false_negatives", [])

    def _avg(examples, feat):
        vals = [e["features"].get(feat, 0.0) for e in examples]
        return round(float(np.mean(vals)), 4) if vals else 0.0

    observations = []

    # Pronoun ratio difference
    fp_pr = _avg(fp, "pronoun_ratio")
    fn_pr = _avg(fn, "pronoun_ratio")
    if abs(fp_pr - fn_pr) > 0.01:
        higher = "False Positives" if fp_pr > fn_pr else "False Negatives"
        observations.append(
            f"First-person pronoun ratio is higher in {higher} "
            f"(FP: {fp_pr:.3f}, FN: {fn_pr:.3f}), suggesting the model "
            f"may over-weight self-referential language as a loneliness signal."
        )

    # Negation ratio
    fp_neg = _avg(fp, "negation_ratio")
    fn_neg = _avg(fn, "negation_ratio")
    if abs(fp_neg - fn_neg) > 0.005:
        higher = "False Positives" if fp_neg > fn_neg else "False Negatives"
        observations.append(
            f"Negation ratio is higher in {higher} "
            f"(FP: {fp_neg:.3f}, FN: {fn_neg:.3f})."
        )

    # Word count
    fp_wc = np.mean([e["word_count"] for e in fp]) if fp else 0
    fn_wc = np.mean([e["word_count"] for e in fn]) if fn else 0
    if fp and fn:
        if fp_wc > fn_wc * 1.2:
            observations.append(
                f"False Positives tend to be longer posts (mean {fp_wc:.0f} words) "
                f"vs False Negatives ({fn_wc:.0f} words). Longer posts with emotional "
                f"vocabulary may be misclassified as lonely even without direct self-disclosure."
            )
        elif fn_wc > fp_wc * 1.2:
            observations.append(
                f"False Negatives tend to be longer posts (mean {fn_wc:.0f} words) "
                f"vs False Positives ({fp_wc:.0f} words). Longer lonely posts may "
                f"dilute signal density, causing the model to miss loneliness expressions."
            )

    # Emotion word ratio
    fp_em = _avg(fp, "emotion_word_ratio")
    fn_em = _avg(fn, "emotion_word_ratio")
    if abs(fp_em - fn_em) > 0.005:
        higher = "False Positives" if fp_em > fn_em else "False Negatives"
        observations.append(
            f"Emotion word ratio is higher in {higher} "
            f"(FP: {fp_em:.4f}, FN: {fn_em:.4f})."
        )

    if not observations:
        observations.append(
            "No strongly differentiated linguistic patterns observed between "
            "FP and FN examples for this model. Errors may be driven by "
            "contextual or pragmatic factors not captured by surface features."
        )

    return {
        "n_fp_analysed": len(fp),
        "n_fn_analysed": len(fn),
        "observations": observations,
    }


# Run Error Analysis
def run_error_analysis(
    all_results: list[dict],
    features_bundle: dict,
    dataset: DatasetDict,
) -> dict:
    """
    Run qualitative error analysis on all classical models and aggregate results.

    Parameters
    ----------
    all_results      : list of result dicts from train_and_compare
    features_bundle  : dict of representation → bundle
    dataset          : preprocessed DatasetDict (needs 'test' split)

    Returns
    -------
    dict — aggregated error analysis summary
    """
    if json_exists("error_analysis_summary.json"):
        logger.info("Error analysis results already exist — loading from cache.")
        return load_json("error_analysis_summary.json")

    logger.info("Running error analysis on test set...")

    test_split = dataset["test"]
    all_errors: dict[str, dict] = {}
    summaries: list[dict] = []

    # Evaluate each classical model
    for rep_name, rep_bundle in features_bundle.items():
        if not rep_bundle or "test" not in rep_bundle:
            continue

        X_test, y_test = rep_bundle["test"]

        for model_name in ("logistic_regression", "svm", "random_forest"):
            model_key = f"{rep_name}_{model_name}.joblib"
            if not cache_exists(model_key):
                logger.warning(f"Model not found: {model_key} — skipping.")
                continue

            model = load_joblib(model_key)
            error_data = _collect_errors(
                model, X_test, y_test, test_split,
                rep_name, model_name,
            )
            all_errors[f"{rep_name}_{model_name}"] = error_data
            summaries.append(error_data["summary"])

    if not summaries:
        logger.warning("No models found for error analysis.")
        return {}

    # Plots
    _plot_fp_fn_counts(summaries)

    # Find best model from all_results (by validation F1)
    best_key = None
    best_f1 = -1.0
    for r in all_results:
        rep, mdl = r.get("representation", ""), r.get("model", "")
        if rep == "distilbert":
            continue
        key = f"{rep}_{mdl}"
        if key in all_errors and (r.get("f1") or 0) > best_f1:
            best_f1 = r.get("f1", 0)
            best_key = key
            best_rep, best_mdl = rep, mdl

    if best_key and best_key in all_errors:
        best_data = all_errors[best_key]
        _plot_length_by_error_type(best_data, best_rep, best_mdl)
        _plot_linguistic_fp_vs_fn(best_data, best_rep, best_mdl)
        qualitative = _qualitative_patterns(best_data)
    else:
        qualitative = {"observations": ["Best model data unavailable."]}

    # Confusion overlap
    overlap = _confusion_overlap(all_errors, test_split)
    _plot_confusion_overlap(overlap)

    # Global summary
    summary = {
        "best_model_key": best_key,
        "n_models_analysed": len(summaries),
        "model_summaries": summaries,
        "confusion_overlap": overlap,
        "qualitative_patterns": qualitative,
        # Error rate ranking — useful for the report's model comparison table
        "error_rate_ranking": sorted(
            summaries,
            key=lambda s: s["error_rate"],
        ),
    }

    save_json("error_analysis_summary.json", sanitise_for_json(summary))
    record_step("error_analysis", meta={"n_models": len(summaries)})
    logger.info("Error analysis complete.")
    return summary