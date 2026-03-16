"""
Model training and evaluation for loneliness self-disclosure classification.

Phases:
1. TF-IDF only
2. TF-IDF + Linguistic features (first-person ratio, social words, question marks, sentence length, sentiment)

Models:
- Logistic Regression
- Linear SVM
- Random Forest

Evaluation metrics:
- Accuracy
- Precision
- Recall
- F1-score
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier

from src.feature_extractors import extract_linguistic_features


# ─────────────────────────────────────────────
# 1. TF-IDF Feature Builder
# ─────────────────────────────────────────────
def build_tfidf_features(train_texts, val_texts, test_texts):
    """
    Transform text into TF-IDF vectors for train, validation, and test sets.
    """
    vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), stop_words="english")

    X_train = vectorizer.fit_transform(train_texts)
    X_val = vectorizer.transform(val_texts)
    X_test = vectorizer.transform(test_texts)

    return X_train, X_val, X_test, vectorizer


# ─────────────────────────────────────────────
# 2. Combine TF-IDF and Linguistic Features
# ─────────────────────────────────────────────
def combine_with_linguistic_features(X_tfidf, texts):
    """
    Extract linguistic features and concatenate to TF-IDF sparse matrix
    """
    ling_feats = extract_linguistic_features(texts)  # shape: (n_samples, 5)
    return np.hstack([X_tfidf.toarray(), ling_feats])


# ─────────────────────────────────────────────
# 3. Model Evaluation
# ─────────────────────────────────────────────
def evaluate_model(model, X, y, model_name, split_name="Validation"):
    """
    Evaluate a trained model on any split.
    Returns metrics dictionary.
    """
    predictions = model.predict(X)

    accuracy = accuracy_score(y, predictions)
    precision = precision_score(y, predictions)
    recall = recall_score(y, predictions)
    f1 = f1_score(y, predictions)

    print("\n" + "=" * 60)
    print(f"{split_name} Metrics for {model_name}")
    print("=" * 60)
    print(f"Accuracy  : {accuracy:.4f}")
    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1-score  : {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y, predictions, digits=4))

    return {
        "model": model_name,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


# ─────────────────────────────────────────────
# 4. Train and Evaluate All Models
# ─────────────────────────────────────────────
def train_and_evaluate_models(df, use_linguistic_features=False):
    """
    Train baseline models on TRAIN, evaluate on VALIDATION,
    then test best model on TEST split.

    Args:
        use_linguistic_features (bool): If True, concatenate linguistic features to TF-IDF
    """

    # Split dataset
    train_df = df[df["split"] == "train"]
    val_df   = df[df["split"] == "validation"]
    test_df  = df[df["split"] == "test"]

    # Extract texts and labels
    X_train_text, y_train = train_df["clean_text"], train_df["label"].to_numpy()
    X_val_text, y_val     = val_df["clean_text"], val_df["label"].to_numpy()
    X_test_text, y_test   = test_df["clean_text"], test_df["label"].to_numpy()

    # TF-IDF features
    X_train, X_val, X_test, vectorizer = build_tfidf_features(X_train_text, X_val_text, X_test_text)

    # Optionally add linguistic features
    if use_linguistic_features:
        print("\nAdding linguistic features to TF-IDF vectors...")
        X_train = combine_with_linguistic_features(X_train, X_train_text)
        X_val   = combine_with_linguistic_features(X_val, X_val_text)
        X_test  = combine_with_linguistic_features(X_test, X_test_text)

    # Define models
    models = [
        ("Logistic Regression", LogisticRegression(max_iter=200)),
        ("Linear SVM", LinearSVC()),
        ("Random Forest", RandomForestClassifier(n_estimators=200, random_state=42)),
    ]

    validation_results = []

    # ────────────────
    # Train on TRAIN, Evaluate on VALIDATION
    # ────────────────
    for name, model in models:
        print("\n" + "=" * 60)
        print(f"Training {name} on TRAIN set")
        print("=" * 60)

        model.fit(X_train, y_train)
        result = evaluate_model(model, X_val, y_val, name, split_name="Validation")
        validation_results.append(result)

    validation_df = pd.DataFrame(validation_results)

    # Select best model based on F1 on validation
    best_model_name = validation_df.sort_values("f1", ascending=False).iloc[0]["model"]
    best_model = dict(models)[best_model_name]

    print("\nBest model on validation set:", best_model_name)

    # ────────────────
    # Retrain best model on TRAIN, Evaluate on TEST
    # ────────────────
    best_model.fit(X_train, y_train)
    test_result = evaluate_model(best_model, X_test, y_test, best_model_name, split_name="Test")

    print("\nValidation Comparison:")
    print(validation_df.sort_values("f1", ascending=False))

    return validation_df, test_result, best_model, vectorizer