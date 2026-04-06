"""
Interpretability analysis for RQ4.

Answers: "What trade-offs exist between predictive performance and
interpretability when using different text representations for
loneliness self-disclosure detection?"

Outputs
───────
results/json/
  interpretability_tfidf_lr_coefficients.json
      Top-50 positive (lonely) and top-50 negative (non-lonely) LR
      coefficients from the tfidf + LogisticRegression model.

  interpretability_tfidf_ling_lr_coefficients.json
      Same for tfidf_ling + LR, split into tfidf tokens and
      linguistic feature names.

  interpretability_linguistic_only_lr_coefficients.json
      Full coefficient ranking for the 13-feature structural baseline.
      Direct answer to RQ2 — which structural features matter most.

  interpretability_distilbert_attention.json
      Per-token average attention weights from the last encoder layer,
      computed over the test set. Includes top-attended tokens per class.

  interpretability_summary.json
      Cross-representation interpretability comparison:
      - intrinsic interpretability score (human-defined)
      - top discriminative features / tokens per representation
      - computational cost (training time proxy)

results/plots/
  interp_tfidf_lr_coefficients.png
      Horizontal bar chart of top-25 positive + negative LR coefficients.

  interp_linguistic_only_coefficients.png
      Bar chart of all 13 structural feature coefficients.

  interp_distilbert_attention.png
      Heatmap of average attention on top-attended tokens.

  interp_representation_tradeoff.png
      2-axis scatter: F1 score vs interpretability score per representation.
"""

import logging

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from .config import RESULTS_SUBDIRS
from .features import _LING_FEATURE_NAMES
from .results import (
    cache_exists,
    load_joblib,
    load_json,
    json_exists,
    save_json,
    sanitise_for_json,
    plot_path,
    record_step,
)

logger = logging.getLogger(__name__)

# Interpretability scores (human-defined, 1–5 scale, higher = more interpretable)
# Based on: can a domain expert understand WHY a prediction was made?
INTERPRETABILITY_SCORES = {
    "linguistic_only": 5,  # 13 named features with direct clinical meaning
    "tfidf": 4,  # vocabulary weights — readable, but 15k features
    "tfidf_ling": 4,  # same + named features
    "word2vec": 2,  # 200-d dense vectors — no direct interpretation
    "sbert": 2,  # 384-d dense — even less interpretable
    "distilbert": 1,  # attention heads — partial, but opaque end-to-end
}

INTERPRETABILITY_RATIONALE = {
    "linguistic_only": "All 13 features have explicit clinical/psycholinguistic names and meaning (e.g. pronoun_ratio, negation_ratio). Coefficient = direct feature contribution.",
    "tfidf": "LR coefficients map directly to n-gram tokens. Top terms are human-readable. 15k features reduce parsimony but remain auditable.",
    "tfidf_ling": "Same as tfidf with 13 additional named features. Slight improvement over pure tfidf due to explicit linguistic dimensions.",
    "word2vec": "200-d dense embeddings. Nearest-neighbour analysis possible but coefficients have no direct linguistic meaning.",
    "sbert": "384-d contextual embeddings. No token-level attribution without additional probing. Black-box to end users.",
    "distilbert": "Attention weights provide partial token-level signal but are not faithful explanations (attention ≠ attribution). Requires SHAP/Integrated Gradients for reliable explanations.",
}

COMPUTATIONAL_COST = {
    "linguistic_only": "< 1s (fit)",
    "tfidf": "< 5s (fit + transform)",
    "tfidf_ling": "< 5s (fit + transform)",
    "word2vec": "~2 min (train from scratch)",
    "sbert": "~5 min (encode 5k docs on CPU)",
    "distilbert": "~30 min (fine-tune, T4 GPU required)",
}


# 1. TF-IDF LR coefficients
def _lr_coefficients_tfidf(
    model,
    feature_names: list[str],
    top_n: int = 50,
    name: str = "tfidf",
) -> dict:
    """
    Extract top-n positive (lonely) and top-n negative (non-lonely)
    LogisticRegression coefficients from a TF-IDF model.
    """
    # Walk pipeline to find the LR estimator
    lr = model
    while hasattr(lr, "steps"):  # sklearn Pipeline
        lr = lr.steps[-1][1]
    while hasattr(lr, "estimator"):  # meta-estimators
        lr = lr.estimator

    if not hasattr(lr, "coef_"):
        logger.warning(f"Model {name} has no coef_ — skipping coefficient extraction.")
        return {}

    coef = lr.coef_[0]  # shape (n_features,)
    n = len(feature_names)

    if len(coef) != n:
        logger.warning(
            f"Coef length {len(coef)} != feature_names length {n} for {name}."
        )
        n = min(len(coef), n)
        coef = coef[:n]
        feature_names = feature_names[:n]

    # Sort by coefficient value
    sorted_idx = np.argsort(coef)
    top_pos_idx = sorted_idx[-top_n:][::-1]  # highest → lonely
    top_neg_idx = sorted_idx[:top_n]  # lowest → non-lonely

    result = {
        "representation": name,
        "model": "logistic_regression",
        "n_features_total": len(coef),
        "top_n": top_n,
        "lonely_indicators": [
            {"feature": feature_names[i], "coefficient": round(float(coef[i]), 6)}
            for i in top_pos_idx
        ],
        "non_lonely_indicators": [
            {"feature": feature_names[i], "coefficient": round(float(coef[i]), 6)}
            for i in top_neg_idx
        ],
    }
    save_json(f"interpretability_{name}_lr_coefficients.json", result)
    logger.info(f"Saved {name} LR coefficients.")
    return result


def _plot_lr_coefficients(result: dict, name: str) -> None:
    if not result:
        return

    top_n = min(25, result.get("top_n", 25))
    lonely = result["lonely_indicators"][:top_n]
    non_lonely = result["non_lonely_indicators"][:top_n]

    # Two horizontal bar charts side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, max(8, top_n * 0.35)))

    def _hbar(ax, items, color, title):
        terms = [d["feature"] for d in reversed(items)]
        vals = [d["coefficient"] for d in reversed(items)]
        bars = ax.barh(terms, vals, color=color, alpha=0.85)
        ax.set_title(title, fontweight="bold", fontsize=11)
        ax.set_xlabel("LR Coefficient")
        ax.axvline(0, color="grey", linewidth=0.8, linestyle="--")
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="y", labelsize=8)

    _hbar(ax1, non_lonely, "#00d4ff", "Non-Lonely indicators (negative coef)")
    _hbar(ax2, lonely, "#f43f5e", "Lonely indicators (positive coef)")

    plt.suptitle(
        f"Top-{top_n} LR Coefficients — {name.replace('_', ' ').title()}",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    p = plot_path(f"interp_{name}_lr_coefficients.png")
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Plot saved → {p}")


# 2. Linguistic-only coefficients
def _lr_coefficients_linguistic(model, feature_names: list[str]) -> dict:
    """All 13 structural feature coefficients with full ranking."""
    lr = model
    while hasattr(lr, "steps"):
        lr = lr.steps[-1][1]
    while hasattr(lr, "estimator"):
        lr = lr.estimator

    if not hasattr(lr, "coef_"):
        logger.warning("linguistic_only model has no coef_ — skipping.")
        return {}

    coef = lr.coef_[0]
    sorted_idx = np.argsort(coef)[::-1]

    result = {
        "representation": "linguistic_only",
        "model": "logistic_regression",
        "n_features": len(coef),
        "features_ranked": [
            {
                "rank": int(i) + 1,
                "feature": feature_names[idx],
                "coefficient": round(float(coef[idx]), 6),
                "direction": "lonely" if coef[idx] > 0 else "non_lonely",
                "abs_rank": int(np.where(np.argsort(np.abs(coef))[::-1] == idx)[0][0])
                + 1,
            }
            for i, idx in enumerate(sorted_idx)
        ],
    }
    save_json("interpretability_linguistic_only_lr_coefficients.json", result)
    logger.info("Saved linguistic_only LR coefficients.")

    # Plot all 13 features
    fig, ax = plt.subplots(figsize=(10, 6))
    names = [d["feature"] for d in result["features_ranked"]]
    vals = [d["coefficient"] for d in result["features_ranked"]]
    colors = ["#f43f5e" if v > 0 else "#00d4ff" for v in vals]

    bars = ax.barh(names[::-1], vals[::-1], color=colors[::-1], alpha=0.85)
    ax.set_xlabel("LR Coefficient")
    ax.set_title(
        "Structural Feature Importances — Linguistic-Only Baseline (RQ2)",
        fontweight="bold",
        fontsize=12,
    )
    ax.axvline(0, color="grey", linewidth=0.8, linestyle="--")

    # Annotate coefficient values
    for bar, val in zip(bars, vals[::-1]):
        ax.text(
            val + (0.001 if val >= 0 else -0.001),
            bar.get_y() + bar.get_height() / 2,
            f"{val:+.4f}",
            va="center",
            ha="left" if val >= 0 else "right",
            fontsize=8,
            fontfamily="monospace",
        )

    ax.legend(
        handles=[
            mpatches.Patch(color="#f43f5e", label="→ Lonely"),
            mpatches.Patch(color="#00d4ff", label="→ Non-Lonely"),
        ],
        loc="lower right",
        fontsize=9,
    )
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    p = plot_path("interp_linguistic_only_coefficients.png")
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Plot saved → {p}")
    return result


# 3. DistilBERT attention weights
def _distilbert_attention(dataset, top_n: int = 30) -> dict:
    """
    Compute average last-layer attention weights over a sample of test posts,
    split by class. Returns top-attended tokens per class.

    Note: attention ≠ attribution. These are indicative signals, not
    faithful feature importances. This is documented in the summary.
    """
    bert_dir = RESULTS_SUBDIRS["bert"] / "best_model"
    if not bert_dir.exists():
        logger.warning("DistilBERT best_model not found — skipping attention analysis.")
        return {}

    logger.info("Computing DistilBERT attention weights...")
    tokenizer = AutoTokenizer.from_pretrained(str(bert_dir))
    model = AutoModelForSequenceClassification.from_pretrained(
        str(bert_dir), output_attentions=True
    )
    model.eval()

    # Sample up to 100 posts per class from test set
    test_data = dataset["test"]
    lonely_texts = [t for t, l in zip(test_data["text"], test_data["label"]) if l == 1][
        :100
    ]
    non_lonely_texts = [
        t for t, l in zip(test_data["text"], test_data["label"]) if l == 0
    ][:100]

    token_weights: dict[str, dict[str, list[float]]] = {"lonely": {}, "non_lonely": {}}

    def _process(texts: list[str], class_key: str) -> None:
        for text in texts:
            enc = tokenizer(
                text, return_tensors="pt", max_length=128, truncation=True, padding=True
            )
            with torch.no_grad():
                out = model(**enc)

            # out.attentions: tuple of (1, n_heads, seq_len, seq_len)
            # Take last layer, average over heads, then over query positions
            attn = out.attentions[-1][0]  # (n_heads, seq_len, seq_len)
            attn_mean = attn.mean(dim=0).mean(
                dim=0
            )  # (seq_len,) — avg attended-to weight
            attn_np = attn_mean.cpu().numpy()

            tokens = tokenizer.convert_ids_to_tokens(enc["input_ids"][0])
            for tok, weight in zip(tokens, attn_np):
                tok_clean = tok.replace("##", "").strip()
                if tok_clean in ("[CLS]", "[SEP]", "[PAD]", ""):
                    continue
                if tok_clean not in token_weights[class_key]:
                    token_weights[class_key][tok_clean] = []
                token_weights[class_key][tok_clean].append(float(weight))

    _process(lonely_texts, "lonely")
    _process(non_lonely_texts, "non_lonely")

    # Average weights per token, filter to tokens seen 3+ times
    def _top_tokens(class_key: str):
        avg = {
            tok: float(np.mean(weights))
            for tok, weights in token_weights[class_key].items()
            if len(weights) >= 3
        }
        sorted_toks = sorted(avg.items(), key=lambda x: x[1], reverse=True)
        return [
            {"token": t, "avg_attention": round(w, 6)} for t, w in sorted_toks[:top_n]
        ]

    lonely_top = _top_tokens("lonely")
    non_lonely_top = _top_tokens("non_lonely")

    result = {
        "representation": "distilbert",
        "method": "last_layer_mean_attention",
        "caveat": "Attention weights are indicative but not faithful attribution scores. They show which tokens the model attends to, not necessarily which drive the prediction.",
        "n_lonely_samples": len(lonely_texts),
        "n_non_lonely_samples": len(non_lonely_texts),
        "top_n": top_n,
        "lonely_top_tokens": lonely_top,
        "non_lonely_top_tokens": non_lonely_top,
    }
    save_json("interpretability_distilbert_attention.json", sanitise_for_json(result))

    # Plot top-15 tokens per class
    k = 15
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    def _plot_attn(ax, items, color, title):
        toks = [d["token"] for d in items[:k]][::-1]
        vals = [d["avg_attention"] for d in items[:k]][::-1]
        ax.barh(toks, vals, color=color, alpha=0.85)
        ax.set_title(title, fontweight="bold", fontsize=11)
        ax.set_xlabel("Avg attention weight")
        ax.spines[["top", "right"]].set_visible(False)

    _plot_attn(ax1, non_lonely_top, "#00d4ff", f"Non-Lonely — top-{k} attended tokens")
    _plot_attn(ax2, lonely_top, "#f43f5e", f"Lonely — top-{k} attended tokens")
    plt.suptitle(
        "DistilBERT — Average Last-Layer Attention (test set sample)",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    p = plot_path("interp_distilbert_attention.png")
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Plot saved → {p}")
    return result


# 4. Performance vs interpretability trade-off
def _plot_tradeoff(all_results: list[dict]) -> None:
    """
    2-axis scatter plot: F1 score vs interpretability score.
    Each point = best model for that representation.
    Directly visualises the RQ4 trade-off.
    """
    # Best F1 per representation from validation results
    best: dict[str, float] = {}
    for r in all_results:
        rep = r["representation"]
        f1 = r.get("f1", 0.0) or 0.0
        if rep not in best or f1 > best[rep]:
            best[rep] = f1

    reps = list(best.keys())
    f1s = [best[r] for r in reps]
    interp = [INTERPRETABILITY_SCORES.get(r, 1) for r in reps]
    colors = {
        "linguistic_only": "#10b981",
        "tfidf": "#00d4ff",
        "tfidf_ling": "#7c3aed",
        "word2vec": "#f59e0b",
        "sbert": "#3b82f6",
        "distilbert": "#f43f5e",
    }

    fig, ax = plt.subplots(figsize=(10, 7))

    for rep, f1, itp in zip(reps, f1s, interp):
        ax.scatter(
            itp,
            f1,
            s=180,
            color=colors.get(rep, "#888888"),
            zorder=5,
            edgecolors="white",
            linewidths=1.5,
        )
        ax.annotate(
            rep.replace("_", "\n"),
            (itp, f1),
            textcoords="offset points",
            xytext=(10, 4),
            fontsize=9,
            fontfamily="monospace",
            color=colors.get(rep, "#888888"),
        )

    # Annotation: ideal region
    ax.axhspan(
        0.94, 1.0, alpha=0.04, color="green", label="High performance zone (F1 > 0.94)"
    )
    ax.axvspan(
        3.5,
        5.5,
        alpha=0.04,
        color="blue",
        label="High interpretability zone (score ≥ 4)",
    )

    ax.set_xlabel(
        "Interpretability Score (1=opaque → 5=fully interpretable)", fontsize=11
    )
    ax.set_ylabel("Validation F1 Score", fontsize=11)
    ax.set_title(
        "RQ4: Performance vs Interpretability Trade-off\n(Best model per representation)",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_xlim(0.5, 5.8)
    ax.set_ylim(0.82, 1.0)
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.set_xticklabels(["1\n(opaque)", "2", "3", "4", "5\n(transparent)"])
    ax.grid(axis="both", alpha=0.3, linestyle="--")
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    p = plot_path("interp_representation_tradeoff.png")
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Plot saved → {p}")


# 5. Cross-representation summary
def _build_summary(
    all_results: list[dict],
    coef_results: dict[str, dict],
    attention_result: dict,
) -> dict:
    """Build interpretability_summary.json for the API and report."""
    best: dict[str, dict] = {}
    for r in all_results:
        rep = r["representation"]
        if rep not in best or (r.get("f1") or 0) > (best[rep].get("f1") or 0):
            best[rep] = r

    summary = {}
    for rep in list(INTERPRETABILITY_SCORES.keys()):
        entry: dict = {
            "representation": rep,
            "interpretability_score": INTERPRETABILITY_SCORES[rep],
            "interpretability_rationale": INTERPRETABILITY_RATIONALE.get(rep, ""),
            "computational_cost": COMPUTATIONAL_COST.get(rep, "unknown"),
            "best_f1": round(best.get(rep, {}).get("f1", 0.0) or 0.0, 4),
            "best_model": best.get(rep, {}).get("model", "n/a"),
        }

        # Top discriminative features per representation
        if rep == "linguistic_only" and "linguistic_only" in coef_results:
            ranked = coef_results["linguistic_only"].get("features_ranked", [])
            entry["top_lonely_features"] = [
                d["feature"] for d in ranked if d["direction"] == "lonely"
            ][:5]
            entry["top_non_lonely_features"] = [
                d["feature"] for d in ranked if d["direction"] == "non_lonely"
            ][:5]

        elif rep in ("tfidf", "tfidf_ling") and rep in coef_results:
            entry["top_lonely_tokens"] = [
                d["feature"]
                for d in coef_results[rep].get("lonely_indicators", [])[:10]
            ]
            entry["top_non_lonely_tokens"] = [
                d["feature"]
                for d in coef_results[rep].get("non_lonely_indicators", [])[:10]
            ]

        elif rep == "distilbert" and attention_result:
            entry["top_lonely_tokens"] = [
                d["token"] for d in attention_result.get("lonely_top_tokens", [])[:10]
            ]
            entry["top_non_lonely_tokens"] = [
                d["token"]
                for d in attention_result.get("non_lonely_top_tokens", [])[:10]
            ]
            entry["attention_caveat"] = attention_result.get("caveat", "")

        summary[rep] = entry

    save_json("interpretability_summary.json", summary)
    logger.info("Saved interpretability_summary.json")
    return summary


# Run Interpretability
def run_interpretability(
    all_results: list[dict],
    features_bundle: dict,
    dataset,
) -> dict:
    """
    Run all interpretability analyses and save results.

    Parameters
    ----------
    all_results       : list of validation result dicts (from train_and_compare)
    features_bundle   : dict of representation → bundle (from build_features)
    dataset           : preprocessed DatasetDict (for DistilBERT attention)

    Returns
    -------
    interpretability_summary dict
    """
    if json_exists("interpretability_summary.json"):
        logger.info("Interpretability results already exist — loading from cache.")
        return load_json("interpretability_summary.json")

    logger.info("Running interpretability analysis (RQ4)...")

    coef_results: dict[str, dict] = {}

    # Linguistic-only coefficients (RQ2 structural baseline)
    ling_model_key = "linguistic_only_logistic_regression.joblib"
    if cache_exists(ling_model_key):
        ling_model = load_joblib(ling_model_key)
        coef_results["linguistic_only"] = _lr_coefficients_linguistic(
            ling_model, list(_LING_FEATURE_NAMES)
        )
    else:
        logger.warning("linguistic_only LR model not found — run training first.")

    # TF-IDF LR coefficients
    tfidf_model_key = "tfidf_logistic_regression.joblib"
    if cache_exists(tfidf_model_key):
        tfidf_model = load_joblib(tfidf_model_key)
        tfidf_bundle = features_bundle.get("tfidf", {})
        feature_names = tfidf_bundle.get("feature_names", [])
        if feature_names:
            coef_results["tfidf"] = _lr_coefficients_tfidf(
                tfidf_model, feature_names, top_n=50, name="tfidf"
            )
            _plot_lr_coefficients(coef_results["tfidf"], "tfidf")
    else:
        logger.warning("tfidf LR model not found.")

    # TF-IDF + Linguistic LR coefficients
    tfidf_ling_model_key = "tfidf_ling_logistic_regression.joblib"
    if cache_exists(tfidf_ling_model_key):
        tfidf_ling_model = load_joblib(tfidf_ling_model_key)
        tfidf_bundle = features_bundle.get("tfidf", {})
        tfidf_names = tfidf_bundle.get("feature_names", [])
        all_names = list(tfidf_names) + list(_LING_FEATURE_NAMES)
        if all_names:
            coef_results["tfidf_ling"] = _lr_coefficients_tfidf(
                tfidf_ling_model, all_names, top_n=50, name="tfidf_ling"
            )
            _plot_lr_coefficients(coef_results["tfidf_ling"], "tfidf_ling")
    else:
        logger.warning("tfidf_ling LR model not found.")

    # DistilBERT attention
    attention_result = _distilbert_attention(dataset, top_n=30)

    # Trade-off scatter plot
    _plot_tradeoff(all_results)

    # Summary
    summary = _build_summary(all_results, coef_results, attention_result)

    record_step(
        "interpretability", meta={"representations_analysed": list(coef_results.keys())}
    )
    logger.info("Interpretability analysis complete.")
    return summary
