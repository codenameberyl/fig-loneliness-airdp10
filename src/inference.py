"""
Inference — load the best trained model and predict on new text.

Supports both classical (sklearn) models and DistilBERT.
The predictor is a singleton that is lazy-loaded on first call.
"""

import logging
import threading
from typing import Any

import numpy as np
from scipy.sparse import hstack, csr_matrix
from sentence_transformers import SentenceTransformer
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .config import RESULTS_SUBDIRS, BERT_MAX_LENGTH, W2V_VECTOR_SIZE, SBERT_MODEL
from .features import _ling_vector
from .results import cache_exists, load_joblib, load_json, json_exists
from .preprocessing import clean_text, nlp, _extract_tokens

logger = logging.getLogger(__name__)

# Constants
_MIN_CLEANED_WORDS = 3       # reject inputs shorter than this after cleaning
_MAX_CLEANED_CHARS = 10_000  # silently truncate inputs longer than this

# Singleton stat
_predictor: "Predictor | None" = None
_predictor_lock = threading.Lock()


# Input validation
class InvalidInputError(ValueError):
    """Raised when input text cannot be classified meaningfully."""
 
 
def _validate_and_clean(raw_text: str) -> tuple[str, list[str]]:
    """
    Validate, clean, and tokenise the input text.
 
    Returns
    -------
    (cleaned_text, tokens_no_stopwords)
 
    Raises
    ------
    InvalidInputError
        If the text is empty, non-string, or yields too few words after
        cleaning to make a reliable prediction.
    """
    if not isinstance(raw_text, str):
        raise InvalidInputError(
            f"Input must be a string, got {type(raw_text).__name__}."
        )
 
    raw_text = raw_text.strip()
    if not raw_text:
        raise InvalidInputError("Input text is empty.")
 
    # Truncate extreme inputs before cleaning (avoids spaCy memory issues)
    if len(raw_text) > _MAX_CLEANED_CHARS:
        logger.warning(
            f"Input truncated from {len(raw_text)} to {_MAX_CLEANED_CHARS} chars."
        )
        raw_text = raw_text[:_MAX_CLEANED_CHARS]
 
    cleaned = clean_text(raw_text)
 
    word_count = len(cleaned.split())
    if word_count < _MIN_CLEANED_WORDS:
        raise InvalidInputError(
            f"Input yields only {word_count} word(s) after cleaning "
            f"(minimum {_MIN_CLEANED_WORDS} required). "
            "Please provide more text for a meaningful prediction."
        )
 
    doc = list(nlp.pipe([cleaned]))[0]
    feats = _extract_tokens(doc)
    tokens = feats["tokens_no_stopwords"]
 
    return cleaned, tokens, feats


# Threshold loading
def _load_optimal_threshold(rep_name: str, model_name: str) -> float:
    """
    Load the ROC-derived optimal threshold for a given model, if available.
    Falls back to 0.5 if the ROC data was not saved.
 
    The threshold is derived from the Youden J statistic during evaluation
    (see evaluation.py → _plot_roc).
    """
    roc_key = f"eval_roc_{rep_name}_{model_name}.json"
    if json_exists(roc_key):
        try:
            roc_data = load_json(roc_key)
            threshold = float(roc_data.get("optimal_threshold", 0.5))
            logger.info(
                f"Using ROC-derived threshold {threshold:.4f} "
                f"for {rep_name}/{model_name}"
            )
            return threshold
        except Exception as e:
            logger.warning(f"Could not load optimal threshold: {e} — using 0.5")
    return 0.5


class Predictor:
    """Wraps the best trained model for single-document inference."""

    def __init__(self):
        self.best_rep: str = ""
        self.best_model_name: str = ""
        self.model: Any = None
        self.vectorizer: Any = None
        self.sbert_model: Any = None
        self.w2v_model: Any = None
        self.bert_tokenizer: Any = None
        self.bert_model: Any = None
        self.optimal_threshold: float = 0.5
        self._loaded = False

    def _resolve_best(self) -> tuple[str, str]:
        if json_exists("best_per_representation.json"):
            best_per_rep = load_json("best_per_representation.json")
            best = max(best_per_rep, key=lambda r: r.get("f1", 0))
            return best["representation"], best["model"]
        raise FileNotFoundError("No model results found. Run the pipeline first.")
 
    def load(self) -> None:
        if self._loaded:
            return
 
        self.best_rep, self.best_model_name = self._resolve_best()
        logger.info(f"Loading predictor: {self.best_rep}/{self.best_model_name}")
 
        if self.best_rep == "distilbert":
            self._load_bert()
        else:
            self._load_classical()
            self.optimal_threshold = _load_optimal_threshold(
                self.best_rep, self.best_model_name
            )
 
        self._loaded = True
        logger.info(
            f"Predictor ready — threshold: {self.optimal_threshold:.4f}"
        )
 
    def _load_classical(self) -> None:
        model_key = f"{self.best_rep}_{self.best_model_name}.joblib"
        if not cache_exists(model_key):
            raise FileNotFoundError(f"Model not found: {model_key}")
        self.model = load_joblib(model_key)
 
        if self.best_rep in ("tfidf", "tfidf_ling"):
            if cache_exists("tfidf_vectorizer.joblib"):
                self.vectorizer = load_joblib("tfidf_vectorizer.joblib")
        elif self.best_rep == "word2vec":
            if cache_exists("word2vec_model.joblib"):
                self.w2v_model = load_joblib("word2vec_model.joblib")
        elif self.best_rep == "sbert":
            sbert_model_dir = RESULTS_SUBDIRS["cache"] / "sbert_model"
            self.sbert_model = (
                SentenceTransformer(str(sbert_model_dir))
                if sbert_model_dir.exists()
                else SentenceTransformer(SBERT_MODEL)
            )

    def _load_bert(self) -> None:
        bert_dir = RESULTS_SUBDIRS["bert"] / "best_model"
        try:
            self.bert_tokenizer = AutoTokenizer.from_pretrained(str(bert_dir))
            self.bert_model = AutoModelForSequenceClassification.from_pretrained(
                str(bert_dir)
            )
            self.bert_model.eval()
        except Exception as e:
            raise RuntimeError(f"Failed to load DistilBERT: {e}") from e

    # Feature extraction
    def _featurise(self, cleaned: str, tokens: list[str], feats: dict) -> Any:
        if self.best_rep == "tfidf":
            return self.vectorizer.transform([cleaned])
 
        elif self.best_rep == "tfidf_ling":
            X_tfidf = self.vectorizer.transform([cleaned])
            feats["char_count"] = len(cleaned)
            X_ling = csr_matrix(np.array([_ling_vector(feats)]))
            return hstack([X_tfidf, X_ling])
 
        elif self.best_rep == "word2vec":
            if self.w2v_model is None:
                return np.zeros((1, W2V_VECTOR_SIZE))
            vecs = [self.w2v_model.wv[t] for t in tokens if t in self.w2v_model.wv]
            avg = np.mean(vecs, axis=0) if vecs else np.zeros(W2V_VECTOR_SIZE)
            return avg.reshape(1, -1)
 
        elif self.best_rep == "sbert":
            if self.sbert_model is None:
                raise RuntimeError("SBERT model not loaded.")
            return self.sbert_model.encode([cleaned])
 
        elif self.best_rep == "linguistic_only":
            feats["char_count"] = len(cleaned)
            return csr_matrix(np.array([_ling_vector(feats)]))
 
        raise ValueError(f"Unknown representation: {self.best_rep}")

    # Predict
    def predict(self, text: str) -> dict:
        """
        Predict loneliness self-disclosure for a raw text string.
 
        Parameters
        ----------
        text : str
            Raw input text (will be cleaned internally).
 
        Returns
        -------
        {
            "label": 0 | 1,
            "label_name": "non_lonely" | "lonely",
            "confidence": float,          # probability of predicted class
            "threshold_used": float,      # decision threshold applied
            "representation": str,
            "model": str,
        }
 
        Raises
        ------
        InvalidInputError
            If the text is empty or too short after cleaning.
        """
        self.load()

        # Validate & clean — raises InvalidInputError on bad input
        cleaned, tokens, feats = _validate_and_clean(text)
 
        if self.best_rep == "distilbert":
            return self._predict_bert(cleaned)
 
        X = self._featurise(cleaned, tokens, feats)
 
        # Score → threshold-based decision
        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(X)[0]
            positive_score = float(proba[1])
            pred = int(positive_score >= self.optimal_threshold)
            confidence = positive_score if pred == 1 else 1.0 - positive_score
        elif hasattr(self.model, "decision_function"):
            score = float(self.model.decision_function(X)[0])
            positive_score = float(1 / (1 + np.exp(-score)))  # sigmoid
            pred = int(positive_score >= self.optimal_threshold)
            confidence = positive_score if pred == 1 else 1.0 - positive_score
        else:
            pred = int(self.model.predict(X)[0])
            confidence = 1.0

        return {
            "label": pred,
            "label_name": "lonely" if pred == 1 else "non_lonely",
            "confidence": round(confidence, 4),
            "threshold_used": self.optimal_threshold,
            "representation": self.best_rep,
            "model": self.best_model_name,
        }
 
    def _predict_bert(self, cleaned: str) -> dict:
        enc = self.bert_tokenizer(
            cleaned,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=BERT_MAX_LENGTH,
        )
        with torch.no_grad():
            logits = self.bert_model(**enc).logits
        probs = torch.softmax(logits, dim=-1)[0].numpy()
        pred = int(np.argmax(probs))
        return {
            "label": pred,
            "label_name": "lonely" if pred == 1 else "non_lonely",
            "confidence": round(float(probs[pred]), 4),
            "threshold_used": 0.5,
            "representation": "distilbert",
            "model": "distilbert",
        }


# Get Predictor
def get_predictor() -> Predictor:
    """
    Return the singleton Predictor, loading on first call.
    Thread-safe: concurrent callers will wait for the first load to finish.
    """
    global _predictor
    if _predictor is None:
        with _predictor_lock:
            if _predictor is None:   # double-checked locking
                _predictor = Predictor()
    return _predictor
