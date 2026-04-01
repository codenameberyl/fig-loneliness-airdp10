

import os
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent

# Dataset
DATASET_DIR = BASE_DIR / "dataset"
DATASET_REPO = "https://huggingface.co/datasets/FIG-Loneliness/FIG-Loneliness"

DATASET_SPLITS = {
    "train": "train_set",
    "validation": "dev_set",
    "test": "test_set",
}

# Results Directory
RESULTS_DIR = BASE_DIR / "results"
RESULTS_SUBDIRS = {
    "cache": RESULTS_DIR / "cache",
    "plots": RESULTS_DIR / "plots",
    "json": RESULTS_DIR / "json",
    "bert": RESULTS_DIR / "bert",
    "logs": RESULTS_DIR / "logs",
}

# Preprocessing
SPACY_MODEL = "en_core_web_sm"
SPACY_DISABLED = ["ner", "parser"]



# Linguistic features
FIRST_PERSON_PRONOUNS = frozenset({"i", "me", "my", "mine", "myself"})
SOCIAL_WORDS = frozenset(
    {
        "friend",
        "friends",
        "people",
        "person",
        "someone",
        "anybody",
        "everyone",
        "family",
        "alone",
        "lonely",
        "loneliness",
        "isolated",
        "connection",
        "together",
        "relationship",
        "relationships",
    }
)
EMOTION_WORDS = frozenset(
    {
        "sad",
        "happy",
        "depressed",
        "anxious",
        "hurt",
        "pain",
        "empty",
        "hopeless",
        "worthless",
        "lost",
        "miss",
        "missing",
        "wish",
    }
)
NEGATIONS = frozenset(
    {"no", "not", "never", "none", "nobody", "nothing", "neither", "nor", "n't"}
)



# EDA
EDA_TOP_N_GRAMS = 20
EDA_WORDCLOUD_MAX_WORDS = 150
EDA_UMAP_N_COMPONENTS = 2
EDA_UMAP_N_NEIGHBORS = 15
EDA_TSNE_PERPLEXITY = 30

