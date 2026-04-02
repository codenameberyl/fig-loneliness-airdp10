"""
Model evaluation — confusion matrices, ROC curves, classification reports,
and per-representation comparison charts.

All plots saved to results/plots/, all JSON to results/json/.
"""

import logging
import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    auc,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
import torch
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from .results import (
    plot_path,
    save_json,
    sanitise_for_json,
    record_step,
    load_joblib,
    cache_exists,
)
from .config import RESULTS_SUBDIRS

logger = logging.getLogger(__name__)

LABEL_NAMES = ["Non-Lonely", "Lonely"]


# Helpers
def _savefig(name: str) -> None:
    p = plot_path(name)
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Plot saved → {p}")


def _get_proba(model, X) -> np.ndarray | None:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        # Normalise to [0, 1] for plotting
        scores = (scores - scores.min()) / (scores.ptp() + 1e-9)
        return scores
    return None


# Per-model evaluation
def evaluate_classical_model(
    model,
    X_test,
    y_test,
    name: str,
    rep_name: str,
) -> dict:
    """Evaluate a trained sklearn model; save classification report, confusion matrix, and ROC as JSON."""
    model_key = f"{rep_name}_{name}"
    preds = model.predict(X_test)
    proba = _get_proba(model, X_test)

    report = classification_report(
        y_test, preds, target_names=LABEL_NAMES, output_dict=True
    )
    cm = confusion_matrix(y_test, preds)

    cm_data = _plot_confusion(cm, model_key)
    roc_data = _plot_roc(y_test, proba, model_key) if proba is not None else {}

    result = sanitise_for_json(
        {
            "representation": rep_name,
            "model": name,
            "classification_report": report,
            "confusion_matrix": cm_data,
            "roc_curve": roc_data,
        }
    )
    save_json(f"eval_full_{model_key}.json", result)
    return result


def evaluate_distilbert(
    trainer,
    dataset,
    tokenizer,
    rep_name: str = "distilbert",
    model_name: str = "distilbert",
) -> dict:
    """
    Evaluate fine-tuned DistilBERT on the test set.

    Parameters
    ----------
    rep_name / model_name : used to build the result file key so it matches
        the same pattern as classical models: {rep_name}_{model_name}.
    """
    model_key = f"{rep_name}_{model_name}"

    def _tokenize(batch):
        return tokenizer(
            batch["cleaned"],
            padding="max_length",
            truncation=True,
            max_length=256,
        )

    tokenized_test = dataset["test"].map(_tokenize, batched=True)
    tokenized_test = tokenized_test.rename_column("label", "labels")
    tokenized_test.set_format(
        "torch", columns=["input_ids", "attention_mask", "labels"]
    )

    output = trainer.predict(tokenized_test)
    logits = output.predictions
    labels = output.label_ids
    preds = np.argmax(logits, axis=1)

    # Stable softmax to avoid overflow (subtract row max before exp)
    logits_shifted = logits - logits.max(axis=1, keepdims=True)
    probs = np.exp(logits_shifted) / np.sum(
        np.exp(logits_shifted), axis=1, keepdims=True
    )
    positive_scores = probs[:, 1]

    # AUC
    try:
        roc_auc = round(float(roc_auc_score(labels, positive_scores)), 4)
    except Exception:
        roc_auc = None

    report = classification_report(
        labels, preds, target_names=LABEL_NAMES, output_dict=True
    )
    cm = confusion_matrix(labels, preds)

    cm_data = _plot_confusion(cm, model_key)
    roc_data = _plot_roc(labels, positive_scores, model_key)

    # Inject AUC into roc_data so it matches the classical model structure
    if roc_data and roc_auc is not None:
        roc_data["auc"] = roc_auc

    result = sanitise_for_json(
        {
            "representation": rep_name,
            "model": model_name,
            "roc_auc": roc_auc,
            "classification_report": report,
            "confusion_matrix": cm_data,
            "roc_curve": roc_data,
        }
    )
    save_json(f"eval_full_{model_key}.json", result)
    return result


# Plots
def _plot_confusion(cm: np.ndarray, name: str) -> dict:
    """Plot confusion matrix and persist raw cell values as JSON."""
    tn, fp, fn, tp = int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1])
    total = tn + fp + fn + tp
    cm_data = {
        "model_key": name,
        "labels": LABEL_NAMES,
        "matrix": cm.tolist(),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "total": total,
        "accuracy": round((tn + tp) / total, 4) if total else 0,
        "sensitivity": round(tp / (tp + fn), 4) if (tp + fn) else 0,
        "specificity": round(tn / (tn + fp), 4) if (tn + fp) else 0,
    }
    save_json(f"eval_confusion_{name}.json", cm_data)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=LABEL_NAMES)
    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(
        f"Confusion Matrix\n{name.replace('_', ' ').title()}", fontweight="bold"
    )
    _savefig(f"eval_confusion_{name}.png")
    return cm_data


def _plot_roc(y_true: np.ndarray, scores: np.ndarray, name: str) -> dict:
    """Plot ROC curve and persist fpr/tpr/thresholds + AUC as JSON."""
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    roc_auc = auc(fpr, tpr)

    # sklearn sets thresholds[0] = max_score + 1 (inf for probability outputs).
    # Clamp to the next real value so the array is finite throughout.
    thresholds = np.array([
        (
            t if (not math.isinf(t) and not math.isnan(t))
            else float(thresholds[1]) if len(thresholds) > 1 else 1.0
        )
        for t in thresholds
    ])

    # Optimal threshold via Youden J statistic (maximises sensitivity + specificity)
    j_scores = tpr - fpr
    best_idx = int(np.argmax(j_scores))

    roc_data = sanitise_for_json({
        "model_key": name,
        "auc": round(float(roc_auc), 4),
        "fpr": [round(float(v), 4) for v in fpr],
        "tpr": [round(float(v), 4) for v in tpr],
        "thresholds": [round(float(v), 4) for v in thresholds],
        "optimal_threshold": round(float(thresholds[best_idx]), 4),
        "optimal_fpr": round(float(fpr[best_idx]), 4),
        "optimal_tpr": round(float(tpr[best_idx]), 4),
    })
    save_json(f"eval_roc_{name}.json", roc_data)

    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, color="#E8614C", lw=2, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="grey", lw=1)
    plt.scatter(
        fpr[best_idx],
        tpr[best_idx],
        color="#2E2E2E",
        zorder=5,
        label=f"Best threshold = {thresholds[best_idx]:.3f}",
    )
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve\n{name.replace('_', ' ').title()}", fontweight="bold")
    plt.legend(loc="lower right")
    plt.tight_layout()
    _savefig(f"eval_roc_{name}.png")
    return roc_data


def plot_model_comparison(all_results: list[dict]) -> None:
    """Bar chart comparing F1 scores across all representation × model combinations."""
    if not all_results:
        return

    labels = [f"{r['representation']}\n{r['model']}" for r in all_results]
    f1_scores = [r.get("f1", 0) for r in all_results]
    auc_scores = [r.get("roc_auc", 0) for r in all_results]

    x = np.arange(len(labels))
    width = 0.4

    fig, ax = plt.subplots(figsize=(max(12, len(labels) * 1.5), 6))
    bars1 = ax.bar(
        x - width / 2, f1_scores, width, label="F1 Score", color="#4C9BE8", alpha=0.85
    )
    bars2 = ax.bar(
        x + width / 2, auc_scores, width, label="ROC-AUC", color="#E8614C", alpha=0.85
    )

    ax.bar_label(bars1, fmt="%.3f", padding=2, fontsize=8)
    ax.bar_label(bars2, fmt="%.3f", padding=2, fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.set_title(
        "Model Comparison — Validation F1 & AUC", fontweight="bold", fontsize=13
    )
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    _savefig("eval_model_comparison.png")


def plot_representation_comparison(best_per_rep: list[dict]) -> None:
    """Radar / grouped bar comparing best model per representation."""
    if not best_per_rep:
        return

    reps = [r["representation"] for r in best_per_rep]
    metrics = ["accuracy", "precision", "recall", "f1"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1"]

    x = np.arange(len(reps))
    width = 0.2
    colors = ["#4C9BE8", "#E8614C", "#4CE8A0", "#E8C84C"]

    fig, ax = plt.subplots(figsize=(max(10, len(reps) * 2.5), 6))
    for i, (metric, label, color) in enumerate(zip(metrics, metric_labels, colors)):
        vals = [r.get(metric, 0) for r in best_per_rep]
        offset = (i - 1.5) * width
        bars = ax.bar(x + offset, vals, width, label=label, color=color, alpha=0.85)
        ax.bar_label(bars, fmt="%.3f", padding=2, fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(reps, fontsize=9)
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.1)
    ax.set_title(
        "Best Model per Representation (Validation Set)", fontweight="bold", fontsize=13
    )
    ax.legend(loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    _savefig("eval_representation_comparison.png")


# Run Evaluation
def run_evaluation(
    all_results: list[dict],
    features_bundle: dict,
    dataset,
    best_rep: str,
    best_model_name: str,
) -> dict:
    """
    Evaluate the best model on the test set and generate all plots.

    Parameters
    ----------
    all_results      : list of validation result dicts from train_and_compare
    features_bundle  : dict of representation → (X, y) bundles
    dataset          : preprocessed DatasetDict (for DistilBERT)
    best_rep         : best representation name
    best_model_name  : best model name

    Returns
    -------
    dict with test metrics for the best model.
    """
    # Comparison charts
    plot_model_comparison(all_results)

    best_per_rep_cache = "best_per_representation.joblib"
    if cache_exists(best_per_rep_cache):
        best_per_rep = load_joblib(best_per_rep_cache)
    else:
        # Reconstruct from all_results
        seen: dict[str, dict] = {}
        for r in all_results:
            rep = r["representation"]
            if rep not in seen or r.get("f1", 0) > seen[rep].get("f1", 0):
                seen[rep] = r
        best_per_rep = list(seen.values())
    plot_representation_comparison(best_per_rep)

    # Best model test evaluation
    test_report = {}

    if best_rep == "distilbert":
        bert_dir = RESULTS_SUBDIRS["bert"] / "best_model"
        if bert_dir.exists():
            try:
                tokenizer = AutoTokenizer.from_pretrained(str(bert_dir))
                model = AutoModelForSequenceClassification.from_pretrained(
                    str(bert_dir)
                )

                training_args = TrainingArguments(
                    output_dir=str(RESULTS_SUBDIRS["bert"]),
                    per_device_eval_batch_size=16,
                    report_to="none",
                )
                trainer = Trainer(model=model, args=training_args)
                test_report = evaluate_distilbert(
                    trainer,
                    dataset,
                    tokenizer,
                    rep_name=best_rep,
                    model_name=best_model_name,
                )
            except Exception as e:
                logger.error(f"DistilBERT evaluation failed: {e}")
    else:
        model_cache = f"{best_rep}_{best_model_name}.joblib"
        if cache_exists(model_cache):
            model = load_joblib(model_cache)
            rep_bundle = features_bundle.get(best_rep, {})
            if rep_bundle and "test" in rep_bundle:
                X_test, y_test = rep_bundle["test"]
                test_report = evaluate_classical_model(
                    model, X_test, y_test, best_model_name, best_rep
                )

    if test_report:
        save_json("test_evaluation_report.json", test_report)
        logger.info(
            f"Test evaluation complete for {best_rep}/{best_model_name}. "
            f"F1: {test_report.get('classification_report', {}).get('1', {}).get('f1-score', 'n/a')}"
        )

    record_step("evaluation", meta={"best": f"{best_rep}/{best_model_name}"})
    return test_report
