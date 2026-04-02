"""
Feature Engineering — five text representation strategies.

Representations
───────────────
  linguistic_only : 13 handcrafted linguistic features ONLY (sparse, 13-d)
                    → Answers RQ2: isolated structural/psycholinguistic baseline
  tfidf           : TF-IDF unigram+bigram vectors (sparse, 15k-d)
  tfidf_ling      : TF-IDF + 13 linguistic features (sparse, 15013-d)
  word2vec        : Averaged Word2Vec token embeddings (dense, 200-d)
  sbert           : Sentence-BERT sentence embeddings (dense, 384-d)

Each representation is built on the training set and transforms
all three splits (train / validation / test). Results are cached
as a joblib bundle so the API can serve feature matrices directly.

The DistilBERT representation is handled end-to-end in models.py
since it fine-tunes the encoder jointly with the classifier head.
"""

import logging
from typing import Any

import numpy as np
from datasets import DatasetDict
from gensim.models import Word2Vec
from sentence_transformers import SentenceTransformer
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer

from .results import cache_exists, load_joblib, save_joblib, record_step, save_json
from .config import (
    FIRST_PERSON_PRONOUNS,
    NEGATIONS,
    RESULTS_SUBDIRS,
    SBERT_BATCH_SIZE,
    SBERT_MODEL,
    TFIDF_MAX_FEATURES,
    TFIDF_NGRAM_RANGE,
    TFIDF_MIN_DF,
    TFIDF_MAX_DF,
    W2V_EPOCHS,
    W2V_MIN_COUNT,
    W2V_VECTOR_SIZE,
    W2V_WINDOW,
    W2V_WORKERS,
    SOCIAL_WORDS,
    EMOTION_WORDS,
)

logger = logging.getLogger(__name__)


# Linguistic feature vector
_LING_FEATURE_NAMES = [
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


def _ling_vector(example: dict[str, Any]) -> np.ndarray:
    """Extract a fixed-length linguistic feature vector from a preprocessed example."""
    return np.array(
        [example.get(f, 0.0) for f in _LING_FEATURE_NAMES], dtype=np.float32
    )


def _build_ling_matrix(split_data) -> csr_matrix:
    vectors = np.array([_ling_vector(ex) for ex in split_data])
    return csr_matrix(vectors)


# 1. Linguistic-only
def _build_linguistic_only(dataset: DatasetDict) -> dict:
    """
    Build a representation using ONLY the 13 handcrafted linguistic features —
    no vocabulary, no embeddings.

    Feature names (in order):
        word_count, char_count, sentence_count, avg_sentence_length,
        type_token_ratio, pronoun_ratio, negation_ratio,
        social_word_ratio, emotion_word_ratio,
        noun_ratio, verb_ratio, adj_ratio, adv_ratio
    """
    logger.info("Building Linguistic-Only features...")

    results = {}
    for split_name, ds_split in dataset.items():
        split_key = "val" if split_name == "validation" else split_name
        X = _build_ling_matrix(ds_split)
        y = np.array(ds_split["label"])
        results[split_key] = (X, y)

    results["n_features"] = len(_LING_FEATURE_NAMES)
    results["feature_names"] = _LING_FEATURE_NAMES

    logger.info(
        f"Linguistic-only: {len(_LING_FEATURE_NAMES)} features × "
        f"{results['train'][0].shape[0]:,} train samples"
    )
    return results


# 2. TF-IDF
def _build_tfidf(dataset: DatasetDict) -> dict:
    """Fit TF-IDF on train, transform all splits. Returns sparse matrices."""
    logger.info("Building TF-IDF features...")
    vectorizer = TfidfVectorizer(
        max_features=TFIDF_MAX_FEATURES,
        ngram_range=TFIDF_NGRAM_RANGE,
        min_df=TFIDF_MIN_DF,
        max_df=TFIDF_MAX_DF,
        sublinear_tf=True,
    )
    X_train = vectorizer.fit_transform(dataset["train"]["cleaned"])
    X_val = vectorizer.transform(dataset["validation"]["cleaned"])
    X_test = vectorizer.transform(dataset["test"]["cleaned"])

    y_train = np.array(dataset["train"]["label"])
    y_val = np.array(dataset["validation"]["label"])
    y_test = np.array(dataset["test"]["label"])

    save_joblib("tfidf_vectorizer.joblib", vectorizer)
    logger.info(f"TF-IDF vocab size: {len(vectorizer.vocabulary_):,}")

    return {
        "train": (X_train, y_train),
        "val": (X_val, y_val),
        "test": (X_test, y_test),
        "vectorizer": vectorizer,
        "feature_names": vectorizer.get_feature_names_out().tolist(),
        "n_features": X_train.shape[1],
    }


# 3. TF-IDF + Linguistic
def _build_tfidf_ling(dataset: DatasetDict, tfidf_bundle: dict) -> dict:
    """Horizontally stack TF-IDF with linguistic feature matrix."""
    logger.info("Building TF-IDF + Linguistic features...")
    vectorizer = tfidf_bundle["vectorizer"]

    results = {}
    for split_name, ds_split in dataset.items():
        split_key = "val" if split_name == "validation" else split_name
        X_tfidf = vectorizer.transform(ds_split["cleaned"])
        X_ling = _build_ling_matrix(ds_split)
        X_combined = hstack([X_tfidf, X_ling])
        y = np.array(ds_split["label"])
        results[split_key] = (X_combined, y)

    results["n_features"] = (
        tfidf_bundle["n_features"] + len(_LING_FEATURE_NAMES)
    )
    return results


# 4. Word2Vec (averaged token embeddings)
def _build_word2vec(dataset: DatasetDict) -> dict:
    """Train Word2Vec on train tokens, average embeddings per document."""
    logger.info("Training Word2Vec model...")

    train_tokens = dataset["train"]["tokens_no_stopwords"]
    model = Word2Vec(
        sentences=train_tokens,
        vector_size=W2V_VECTOR_SIZE,
        window=W2V_WINDOW,
        min_count=W2V_MIN_COUNT,
        workers=W2V_WORKERS,
        epochs=W2V_EPOCHS,
        seed=42,
    )
    save_joblib("word2vec_model.joblib", model)
    logger.info(f"Word2Vec vocab: {len(model.wv):,} words")

    def _avg_embedding(tokens: list[str]) -> np.ndarray:
        vecs = [model.wv[t] for t in tokens if t in model.wv]
        if not vecs:
            return np.zeros(W2V_VECTOR_SIZE, dtype=np.float32)
        return np.mean(vecs, axis=0).astype(np.float32)

    results = {}
    for split_name, ds_split in dataset.items():
        split_key = "val" if split_name == "validation" else split_name
        X = np.array([_avg_embedding(toks) for toks in ds_split["tokens_no_stopwords"]])
        y = np.array(ds_split["label"])
        results[split_key] = (X, y)

    results["n_features"] = W2V_VECTOR_SIZE
    results["model"] = model
    return results


# 5. Sentence-BERT
def _build_sbert(dataset: DatasetDict) -> dict:
    """Encode cleaned texts with Sentence-BERT."""
    logger.info(f"Encoding with Sentence-BERT ({SBERT_MODEL})...")
    model = SentenceTransformer(SBERT_MODEL)

    results = {}
    for split_name, ds_split in dataset.items():
        split_key = "val" if split_name == "validation" else split_name
        texts = ds_split["cleaned"]
        logger.info(f"  Encoding {split_key} ({len(texts):,} samples)...")
        X = model.encode(
            texts,
            batch_size=SBERT_BATCH_SIZE,
            show_progress_bar=True,
            convert_to_numpy=True,
        )
        y = np.array(ds_split["label"])
        results[split_key] = (X, y)
 
    results["n_features"] = model.get_sentence_embedding_dimension()

    sbert_model_dir = RESULTS_SUBDIRS["cache"] / "sbert_model"
    sbert_model_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(sbert_model_dir))
    logger.info(f"Saved SBERT model → {sbert_model_dir}")
    return results


# Build Features
def build_features(dataset: DatasetDict) -> dict:
    """
    Build all text representation bundles.

    Returns
    -------
    dict with keys: 'linguistic_only', 'tfidf', 'tfidf_ling', 'word2vec', 'sbert'
    Each value is a dict with keys 'train', 'val', 'test' → (X, y) tuples,
    plus 'n_features' and any fitted objects (vectorizer, model).
    """
    cache_name = "features_all_representations.joblib"

    if cache_exists(cache_name):
        logger.info("Loading cached feature bundle...")
        bundle = load_joblib(cache_name)
        record_step("build_features", meta={"source": "cache"})
        return bundle

    logger.info("Building all feature representations...")

    linguistic_only_bundle = _build_linguistic_only(dataset)
    tfidf_bundle = _build_tfidf(dataset)
    tfidf_ling_bundle = _build_tfidf_ling(dataset, tfidf_bundle)
    word2vec_bundle = _build_word2vec(dataset)
    sbert_bundle = _build_sbert(dataset)

    bundle = {
        "linguistic_only": linguistic_only_bundle,
        "tfidf": tfidf_bundle,
        "tfidf_ling": tfidf_ling_bundle,
        "word2vec": word2vec_bundle,
        "sbert": sbert_bundle,
    }

    save_joblib(cache_name, bundle)

    # Save feature metadata for API
    meta = {
        rep: {
            "n_features": b.get("n_features"),
            "available": bool(b),
        }
        for rep, b in bundle.items()
    }
    save_json("features_metadata.json", meta)

    record_step("build_features", meta={"representations": list(bundle.keys())})
    logger.info("Feature building complete.")
    return bundle
