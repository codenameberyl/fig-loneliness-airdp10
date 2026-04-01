

import logging
import re
import subprocess
import unicodedata
from pathlib import Path
from typing import Any

import bleach
import emoji
import ftfy
import spacy
from datasets import DatasetDict, load_from_disk

from .config import (
    RESULTS_SUBDIRS,
    SPACY_MODEL,
    SPACY_DISABLED,
    NEGATIONS,
    FIRST_PERSON_PRONOUNS,
    SOCIAL_WORDS,
    EMOTION_WORDS,
)
from .results import record_step, save_json, sanitise_for_json

logger = logging.getLogger(__name__)

PREPROCESS_CACHE = RESULTS_SUBDIRS["cache"] / "preprocessed_dataset"

# Compiled regex patterns
_URL = re.compile(r"https?://\S+|www\.\S+")
_MENTION = re.compile(r"@\w+")
_HASHTAG = re.compile(r"#(\w+)")
_SUBREDDIT = re.compile(r"r/\w+")
_MULTISPACE = re.compile(r"\s+")
_REPEATED = re.compile(r"(.)\1{3,}")  # 3+ repeated chars → 2


# spaCy model
def _load_spacy() -> spacy.language.Language:
    try:
        nlp = spacy.load(SPACY_MODEL, disable=SPACY_DISABLED)
    except OSError:
        logger.info(f"Downloading spaCy model: {SPACY_MODEL}")
        subprocess.run(["python", "-m", "spacy", "download", SPACY_MODEL], check=True)
        nlp = spacy.load(SPACY_MODEL, disable=SPACY_DISABLED)

    if "sentencizer" not in nlp.pipe_names:
        nlp.add_pipe("sentencizer")

    return nlp


nlp = _load_spacy()


# Text cleaning
def clean_text(text: str) -> str:
    #Return a normalised, lowercased, cleaned string
    if not isinstance(text, str):
        text = "" if text is None else str(text)

    text = ftfy.fix_text(text)
    text = unicodedata.normalize("NFKC", text)
    text = bleach.clean(text, tags=[], strip=True)

    text = _URL.sub(" ", text)
    text = _MENTION.sub(" ", text)
    text = _SUBREDDIT.sub(" ", text)
    text = emoji.replace_emoji(text, replace=" ")
    text = _HASHTAG.sub(r"\1", text)
    text = text.lower()
    text = _REPEATED.sub(r"\1\1", text)
    text = _MULTISPACE.sub(" ", text).strip()

    return text


# Token / linguistic feature extraction
def _extract_tokens(doc: spacy.tokens.Doc) -> dict:
    #Extract token lists, POS tags, and linguistic counts from a spaCy doc
    tokens_all: list[str] = []
    tokens_no_stop: list[str] = []
    lemmas: list[str] = []
    pos_tags: list[str] = []

    for t in doc:
        if not t.is_alpha or len(t.text) < 2:
            continue

        tok = t.text.lower()
        lem = t.lemma_.lower()
        tokens_all.append(tok)
        pos_tags.append(t.pos_)

        # Keep negations even though they're stop words
        if not t.is_stop or tok in NEGATIONS:
            tokens_no_stop.append(tok)
            lemmas.append(lem)

    n = len(tokens_all) or 1

    # POS counts (normalised)
    pos_counts = {}
    for p in pos_tags:
        pos_counts[p] = pos_counts.get(p, 0) + 1

    sentence_count = max(len(list(doc.sents)), 1)

    return {
        "tokens": tokens_all,
        "tokens_no_stopwords": tokens_no_stop,
        "lemmas": lemmas,
        "pos_tags": pos_tags,
        # Surface features
        "word_count": len(tokens_all),
        "sentence_count": sentence_count,
        "avg_sentence_length": len(tokens_all) / sentence_count,
        "type_token_ratio": len(set(tokens_all)) / n,
        # Linguistic / psycholinguistic features
        "pronoun_ratio": sum(1 for t in tokens_all if t in FIRST_PERSON_PRONOUNS) / n,
        "negation_ratio": sum(1 for t in tokens_all if t in NEGATIONS) / n,
        "social_word_ratio": sum(1 for t in tokens_all if t in SOCIAL_WORDS) / n,
        "emotion_word_ratio": sum(1 for t in tokens_all if t in EMOTION_WORDS) / n,
        # POS ratios
        "noun_ratio": pos_counts.get("NOUN", 0) / n,
        "verb_ratio": pos_counts.get("VERB", 0) / n,
        "adj_ratio": pos_counts.get("ADJ", 0) / n,
        "adv_ratio": pos_counts.get("ADV", 0) / n,
    }


# Batch mapping function
def _preprocess_batch(batch: dict) -> dict:
    #Process a batch of raw examples. Used with dataset.map(batched=True)
    cleaned_texts = [clean_text(t) for t in batch["text"]]
    docs = list(nlp.pipe(cleaned_texts, batch_size=64))

    out: dict[str, list] = {
        k: []
        for k in [
            "cleaned",
            "tokens",
            "tokens_no_stopwords",
            "lemmas",
            "pos_tags",
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
            "label",
        ]
    }

    for cleaned, doc, label_vec in zip(cleaned_texts, docs, batch["lonely"]):
        feats = _extract_tokens(doc)
        label = int(label_vec[1])  # [non_lonely, lonely] → scalar

        out["cleaned"].append(cleaned)
        out["char_count"].append(len(cleaned))
        out["label"].append(label)

        for k, v in feats.items():
            out[k].append(v)

    return out


# Preprocess Dataset
def preprocess_dataset(dataset: DatasetDict) -> DatasetDict:
   
    PREPROCESS_CACHE.parent.mkdir(parents=True, exist_ok=True)

    if PREPROCESS_CACHE.exists():
        logger.info(f"Loading cached preprocessed dataset from {PREPROCESS_CACHE}")
        processed = load_from_disk(str(PREPROCESS_CACHE))
        record_step("preprocess", meta={"source": "cache"})
        return processed

    logger.info("Running text preprocessing (this may take a few minutes)...")

    processed = dataset.map(
        _preprocess_batch,
        batched=True,
        batch_size=64,
        desc="Preprocessing",
        remove_columns=["lonely"],  # replaced by scalar 'label'
    )

    processed.save_to_disk(str(PREPROCESS_CACHE))
    logger.info(f"Preprocessed dataset saved → {PREPROCESS_CACHE}")

    # Save sample data for the API
    _save_samples(processed)

    # Persist feature column names for the API
    feature_cols = [
        c
        for c in processed["train"].column_names
        if c not in (
            "idx",
            "unique_id",
            "text",
            "cleaned",
            "tokens",
            "tokens_no_stopwords",
            "lemmas",
            "pos_tags",
            "label",
            "temporal",
            "interaction",
            "context_pri",
            "interpersonal_pri",
        )
    ]
    record_step(
        "preprocess",
        meta={
            "feature_columns": feature_cols,
            "train_size": len(processed["train"]),
            "val_size": len(processed["validation"]),
            "test_size": len(processed["test"]),
        },
    )

    return processed


def _save_samples(processed: DatasetDict, n_per_class: int = 50) -> None:
    
    SAMPLE_FIELDS = [
        "idx",
        "unique_id",
        "text",
        "cleaned",
        "word_count",
        "char_count",
        "sentence_count",
        "pronoun_ratio",
        "negation_ratio",
        "social_word_ratio",
        "emotion_word_ratio",
        "label",
    ]
    TOKEN_FIELDS = ["tokens", "lemmas"]

    result = {}
    for split_name, split_data in processed.items():
        lonely_samples, non_lonely_samples = [], []

        for i, row in enumerate(split_data):
            if (
                len(lonely_samples) >= n_per_class
                and len(non_lonely_samples) >= n_per_class
            ):
                break

            label = row.get("label", 0)
            bucket = lonely_samples if label == 1 else non_lonely_samples
            if len(bucket) >= n_per_class:
                continue

            sample = {f: row.get(f) for f in SAMPLE_FIELDS}
            # Truncate token lists for readability
            for tf in TOKEN_FIELDS:
                val = row.get(tf)
                sample[tf + "_preview"] = val[:20] if isinstance(val, list) else []

            # Round floats
            for k, v in sample.items():
                if isinstance(v, float):
                    sample[k] = round(v, 4)

            bucket.append(sample)

        result[split_name] = {
            "lonely": lonely_samples,
            "non_lonely": non_lonely_samples,
        }

    save_json("preprocessing_samples.json", sanitise_for_json(result))
    logger.info("Preprocessing samples saved → preprocessing_samples.json")
