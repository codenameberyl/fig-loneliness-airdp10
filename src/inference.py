"""
Inference — load the best trained model and predict on new text.

Supports both classical (sklearn) models and DistilBERT.
The predictor is a singleton that is lazy-loaded on first call.
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np

from .results import cache_exists, load_joblib, load_json, json_exists
from .config import RESULTS_SUBDIRS, BERT_MAX_LENGTH
from .preprocessing import clean_text, nlp, _extract_tokens

logger = logging.getLogger(__name__)


# Singleton stat
_predictor: "Predictor | None" = None


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
        self._loaded = False

    def _resolve_best(self) -> tuple[str, str]:
        """Determine best representation and model from saved results."""
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

        self._loaded = True
        logger.info("Predictor ready.")

    def _load_classical(self) -> None:
        model_key = f"{self.best_rep}_{self.best_model_name}.joblib"
        if not cache_exists(model_key):
            raise FileNotFoundError(f"Model result not found: {model_key}")
        self.model = load_joblib(model_key)

        if self.best_rep in ("tfidf", "tfidf_ling"):
            if cache_exists("tfidf_vectorizer.joblib"):
                self.vectorizer = load_joblib("tfidf_vectorizer.joblib")
        elif self.best_rep == "word2vec":
            if cache_exists("word2vec_model.joblib"):
                self.w2v_model = load_joblib("word2vec_model.joblib")
        elif self.best_rep == "sbert":
            try:
                from sentence_transformers import SentenceTransformer

                self.sbert_model = SentenceTransformer(
                    str(RESULTS_SUBDIRS["cache"] / "sbert_model")
                    if (RESULTS_SUBDIRS["cache"] / "sbert_model").exists()
                    else "all-MiniLM-L6-v2"
                )
            except ImportError:
                logger.warning("sentence-transformers not installed.")

    def _load_bert(self) -> None:
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            bert_dir = RESULTS_SUBDIRS["bert"] / "best_model"
            self.bert_tokenizer = AutoTokenizer.from_pretrained(str(bert_dir))
            self.bert_model = AutoModelForSequenceClassification.from_pretrained(
                str(bert_dir)
            )
            self.bert_model.eval()
        except Exception as e:
            raise RuntimeError(f"Failed to load DistilBERT: {e}") from e

    # Feature extraction for a single text
    def _featurise(self, cleaned: str, tokens: list[str]) -> np.ndarray | Any:
        """Convert cleaned text + tokens into the feature vector for the best rep."""
        from .features import _ling_vector
        from .config import W2V_VECTOR_SIZE

        if self.best_rep == "tfidf":
            return self.vectorizer.transform([cleaned])

        elif self.best_rep == "tfidf_ling":
            from scipy.sparse import hstack, csr_matrix

            X_tfidf = self.vectorizer.transform([cleaned])
            # Build a pseudo-example dict for _ling_vector
            doc = list(nlp.pipe([cleaned]))[0]
            feats = _extract_tokens(doc)
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

        raise ValueError(f"Unknown representation: {self.best_rep}")

    # Predict
    def predict(self, text: str) -> dict:
        """
        Predict loneliness self-disclosure for a raw text string.

        Returns
        -------
        {
            "label": 0 | 1,
            "label_name": "non_lonely" | "lonely",
            "confidence": float,
            "representation": str,
            "model": str,
        }
        """
        self.load()

        cleaned = clean_text(text)
        doc = list(nlp.pipe([cleaned]))[0]
        feats = _extract_tokens(doc)
        tokens = feats["tokens_no_stopwords"]

        if self.best_rep == "distilbert":
            return self._predict_bert(cleaned)

        X = self._featurise(cleaned, tokens)
        pred = int(self.model.predict(X)[0])
        confidence: float

        if hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(X)[0]
            confidence = float(proba[pred])
        elif hasattr(self.model, "decision_function"):
            score = float(self.model.decision_function(X)[0])
            # Sigmoid approximation
            confidence = float(1 / (1 + np.exp(-score)))
        else:
            confidence = 1.0

        return {
            "label": pred,
            "label_name": "lonely" if pred == 1 else "non_lonely",
            "confidence": round(confidence, 4),
            "representation": self.best_rep,
            "model": self.best_model_name,
        }

    def _predict_bert(self, cleaned: str) -> dict:
        import torch

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
            "representation": "distilbert",
            "model": "distilbert",
        }


def get_predictor() -> Predictor:
    """Return the singleton Predictor, loading on first call."""
    global _predictor
    if _predictor is None:
        _predictor = Predictor()
    return _predictor
