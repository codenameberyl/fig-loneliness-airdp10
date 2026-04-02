"""
Model training for FIG-Loneliness.

For each text representation × each classical classifier, we:
  1. Train on the training split
  2. Evaluate on the validation split
  3. Save the trained model results

Additionally, DistilBERT is fine-tuned end-to-end.

Results are aggregated in a comparison table saved as JSON.
"""

import logging
from typing import Any

import numpy as np
from datasets import DatasetDict
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
import torch
from transformers import (
    AutoModelForSequenceClassification,
    EarlyStoppingCallback,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from .config import (
    BERT_BATCH_EVAL,
    BERT_BATCH_TRAIN,
    BERT_EPOCHS,
    BERT_LR,
    BERT_MAX_LENGTH,
    BERT_MODEL_NAME,
    BERT_WARMUP_RATIO,
    BERT_WEIGHT_DECAY,
    LR_PARAMS,
    RF_PARAMS,
    SVM_PARAMS,
    RESULTS_SUBDIRS,
)
from .results import cache_exists, load_joblib, save_joblib, save_json, record_step

logger = logging.getLogger(__name__)

# Metrics helpers
def _compute_metrics_from_preds(y_true, y_pred, y_proba=None) -> dict:
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary")
    acc = accuracy_score(y_true, y_pred)
    result = {
        "accuracy": round(float(acc), 4),
        "precision": round(float(p), 4),
        "recall": round(float(r), 4),
        "f1": round(float(f1), 4),
    }
    if y_proba is not None:
        try:
            result["roc_auc"] = round(float(roc_auc_score(y_true, y_proba)), 4)
        except Exception:
            pass
    return result


def _hf_compute_metrics(eval_pred):
    """HuggingFace Trainer compatible metrics function — includes ROC-AUC via softmax."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    # Stable softmax: subtract row max before exp to avoid overflow
    shifted = logits - logits.max(axis=1, keepdims=True)
    probs = np.exp(shifted) / np.sum(np.exp(shifted), axis=1, keepdims=True)
    positive_scores = probs[:, 1]
    return _compute_metrics_from_preds(labels, preds, y_proba=positive_scores)


# Classical classifier factory
def _make_classical_models() -> dict:
    return {
        "logistic_regression": LogisticRegression(**LR_PARAMS),
        "svm": make_pipeline(
            StandardScaler(with_mean=False),
            LinearSVC(**SVM_PARAMS),
        ),
        "random_forest": RandomForestClassifier(**RF_PARAMS),
    }


def _get_proba(model, X) -> np.ndarray | None:
    """Return probability estimates for the positive class if supported."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    return None


# Train one representation × all classifiers
def _train_representation(
    rep_name: str,
    rep_bundle: dict,
) -> list[dict]:
    """Train all classical classifiers on a single representation."""
    if not rep_bundle:
        logger.warning(
            f"Skipping '{rep_name}' — bundle is empty (dependency missing?)."
        )
        return []

    X_train, y_train = rep_bundle["train"]
    X_val, y_val = rep_bundle["val"]
    X_test, y_test = rep_bundle.get("test", (None, None))

    rows = []
    for model_name, model in _make_classical_models().items():
        cache_key = f"{rep_name}_{model_name}.joblib"

        if cache_exists(cache_key):
            logger.info(f"  Loading cached {rep_name}/{model_name}...")
            model = load_joblib(cache_key)
        else:
            logger.info(f"  Training {rep_name}/{model_name}...")
            model.fit(X_train, y_train)
            save_joblib(cache_key, model)

        preds = model.predict(X_val)
        proba = _get_proba(model, X_val)
        val_metrics = _compute_metrics_from_preds(y_val, preds, proba)

        row = {
            "representation": rep_name,
            "model": model_name,
            "split": "validation",
            **val_metrics,
        }

        if X_test is not None:
            test_preds = model.predict(X_test)
            test_proba = _get_proba(model, X_test)
            test_metrics = _compute_metrics_from_preds(y_test, test_preds, test_proba)
            row.update({f"test_{k}": v for k, v in test_metrics.items()})

        rows.append(row)
        logger.info(
            f"  {rep_name}/{model_name} → val F1: {val_metrics['f1']:.4f}"
            + (f" | test F1: {row.get('test_f1', 'n/a')}")
        )

    return rows


# DistilBERT fine-tuning
def train_distilbert(dataset: DatasetDict) -> dict:
    """Fine-tune DistilBERT for binary classification."""
    bert_output = RESULTS_SUBDIRS["bert"]
    bert_model_dir = bert_output / "best_model"
    bert_results_cache = "distilbert_results.joblib"

    if cache_exists(bert_results_cache):
        logger.info("Loading cached DistilBERT results...")
        return load_joblib(bert_results_cache)

    logger.info(f"Fine-tuning {BERT_MODEL_NAME}...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")

    tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)

    def _tokenize(batch):
        return tokenizer(
            batch["cleaned"],
            padding="max_length",
            truncation=True,
            max_length=BERT_MAX_LENGTH,
        )

    tokenized = dataset.map(_tokenize, batched=True, batch_size=128)
    tokenized = tokenized.rename_column("label", "labels")
    keep_cols = ["input_ids", "attention_mask", "labels"]
    tokenized.set_format("torch", columns=keep_cols)

    model = AutoModelForSequenceClassification.from_pretrained(
        BERT_MODEL_NAME, num_labels=2
    )

    # Calculate warmup steps from warmup ratio
    batch_size = BERT_BATCH_TRAIN if device == "cuda" else 8
    total_steps = len(tokenized["train"]) * BERT_EPOCHS // batch_size
    warmup_steps = int(total_steps * BERT_WARMUP_RATIO)

    training_args = TrainingArguments(
        output_dir=str(bert_output),
        eval_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=BERT_BATCH_EVAL if device == "cuda" else 16,
        num_train_epochs=BERT_EPOCHS,
        learning_rate=BERT_LR,
        weight_decay=BERT_WEIGHT_DECAY,
        warmup_steps=warmup_steps,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        fp16=(device == "cuda"),
        report_to="none",
        logging_steps=100,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        compute_metrics=_hf_compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    trainer.train()
    val_metrics_raw = trainer.evaluate(tokenized["validation"])

    val_metrics = {
        k.replace("eval_", ""): v
        for k, v in val_metrics_raw.items()
        if k.startswith("eval_") and k != "eval_loss"
    }

    row = {
        "representation": "distilbert",
        "model": "distilbert",
        "split": "validation",
        **{k: round(float(v), 4) for k, v in val_metrics.items()},
    }

    test_preds_output = trainer.predict(tokenized["test"])
    test_preds = np.argmax(test_preds_output.predictions, axis=1)
    labels = test_preds_output.label_ids
    probs = np.exp(test_preds_output.predictions) / np.sum(
        np.exp(test_preds_output.predictions), axis=1, keepdims=True
    )
    test_metrics = _compute_metrics_from_preds(labels, test_preds, probs[:, 1])
    row.update({f"test_{k}": v for k, v in test_metrics.items()})

    # Save model and tokenizer
    bert_model_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(bert_model_dir))
    tokenizer.save_pretrained(str(bert_model_dir))
    logger.info(f"DistilBERT saved → {bert_model_dir}")

    result = {"row": row, "trainer": trainer, "tokenizer": tokenizer}
    save_joblib(bert_results_cache, {"row": row})  # don't cache trainer

    logger.info(f"DistilBERT val F1: {row.get('f1', 'n/a')}")
    return result


# Aggregate comparison
def _best_per_representation(results: list[dict]) -> list[dict]:
    """Return the best-F1 model for each representation."""
    best: dict[str, dict] = {}
    for row in results:
        rep = row["representation"]
        if rep not in best or row["f1"] > best[rep]["f1"]:
            best[rep] = row
    return list(best.values())


# Train and Compare
def train_and_compare(
    features_bundle: dict,
    dataset: DatasetDict,
) -> tuple[str, str, list[dict]]:
    """
    Train all representation × model combinations and return results.

    Returns
    -------
    (best_representation, best_model_name, all_results)
    """
    if cache_exists("all_model_results.joblib"):
        logger.info("Loading cached model results...")
        all_results = load_joblib("all_model_results.joblib")
        record_step("train_models", meta={"source": "cache"})
    else:
        all_results: list[dict] = []

        for rep_name, rep_bundle in features_bundle.items():
            logger.info(f"── Representation: {rep_name} ──")
            rows = _train_representation(rep_name, rep_bundle)
            all_results.extend(rows)

        # DistilBERT
        bert_result = train_distilbert(dataset)
        if bert_result and "row" in bert_result:
            all_results.append(bert_result["row"])

        save_joblib("all_model_results.joblib", all_results)
        save_json("all_model_results.json", all_results)
        record_step("train_models", meta={"n_experiments": len(all_results)})

    # Find overall best by validation F1
    if not all_results:
        raise RuntimeError("No model results found — check dependencies.")

    best = max(all_results, key=lambda r: r.get("f1", 0))
    best_rep = best["representation"]
    best_model = best["model"]

    # Save summary: best per representation
    best_per_rep = _best_per_representation(all_results)
    save_json("best_per_representation.json", best_per_rep)

    logger.info(
        f"Best overall: {best_rep}/{best_model} — "
        f"val F1: {best.get('f1', 'n/a')}, "
        f"val AUC: {best.get('roc_auc', 'n/a')}"
    )
    return best_rep, best_model, all_results
