

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

# TF-IDF
TFIDF_MAX_FEATURES = 15_000
TFIDF_NGRAM_RANGE = (1, 2)
TFIDF_MIN_DF = 3
TFIDF_MAX_DF = 0.95

# Word2Vec
W2V_VECTOR_SIZE = 200
W2V_WINDOW = 5
W2V_MIN_COUNT = 2
W2V_WORKERS = 4
W2V_EPOCHS = 10

# Sentence-BERT
SBERT_MODEL = "all-MiniLM-L6-v2"
SBERT_BATCH_SIZE = 64

# Text Representations (ordered for comparison)
REPRESENTATIONS = [
    "tfidf",
    "tfidf_ling",  # TF-IDF + linguistic features
    "word2vec",
    "sbert",
    "distilbert",
]

#Classical Models
CLASSICAL_MODELS = ["logistic_regression", "svm", "random_forest"]

LR_PARAMS = {"max_iter": 3000, "C": 1.0, "solver": "lbfgs"}
SVM_PARAMS = {"C": 1.0, "max_iter": 10_000, "dual": "auto"}
RF_PARAMS = {"n_estimators": 200, "n_jobs": -1, "random_state": 42}

# DistilBERT
BERT_MODEL_NAME = "distilbert-base-uncased"
BERT_MAX_LENGTH = 256
BERT_BATCH_TRAIN = 16
BERT_BATCH_EVAL = 32
BERT_EPOCHS = 3
BERT_WEIGHT_DECAY = 0.01
BERT_WARMUP_RATIO = 0.1
BERT_LR = 2e-5