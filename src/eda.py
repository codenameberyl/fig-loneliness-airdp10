"""
Exploratory Data Analysis for FIG-Loneliness dataset.

Features:
- Clean structure
- Publication-style plots
- Automatic saving to results/eda/
- No redundant preprocessing
"""

import os
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2

from src.config import RESULTS_DIR
from src.preprocessing import preprocess_for_classical
from src.feature_extractors import extract_linguistic_features


# ─────────────────────────────────────────────
# Setup
# ─────────────────────────────────────────────

RESULTS_DIR = Path(RESULTS_DIR)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

sns.set_style("whitegrid")
sns.set_context("talk")
plt.rcParams["figure.figsize"] = (10, 6)


def save_plot(filename):
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / filename, dpi=300)
    plt.close()


# ─────────────────────────────────────────────
# 1. Dataset Overview
# ─────────────────────────────────────────────

def dataset_overview(df):

    print("\n===== DATASET OVERVIEW =====")
    print("Total samples:", len(df))
    print("\nClass distribution:")
    print(df["label"].value_counts())

    # Bar plot
    plt.figure()
    sns.countplot(data=df, x="label")
    plt.title("Class Distribution")
    plt.xlabel("Label (0=Non-Lonely, 1=Lonely)")
    plt.ylabel("Count")
    save_plot("class_distribution.png")


# ─────────────────────────────────────────────
# 2. Post Length Analysis
# ─────────────────────────────────────────────

def analyze_post_length(df):

    print("\n===== POST LENGTH ANALYSIS =====")

    df["text_length"] = df["text"].apply(lambda x: len(x.split()))

    print("\nAverage length by class:")
    print(df.groupby("label")["text_length"].mean())

    plt.figure()
    sns.boxplot(x="label", y="text_length", data=df)
    plt.title("Post Length by Class")
    plt.xlabel("Label")
    plt.ylabel("Number of Words")
    save_plot("post_length_boxplot.png")


# ─────────────────────────────────────────────
# 3. Most Frequent Words
# ─────────────────────────────────────────────

def most_frequent_words(df, top_n=15):

    print("\n===== MOST FREQUENT WORDS =====")

    df["clean_text"] = preprocess_for_classical(df["text"])

    lonely_words = Counter(" ".join(
        df[df["label"] == 1]["clean_text"]
    ).split()).most_common(top_n)

    words, counts = zip(*lonely_words)

    plt.figure()
    sns.barplot(x=list(counts), y=list(words))
    plt.title("Top Words in Lonely Posts")
    plt.xlabel("Frequency")
    save_plot("top_lonely_words.png")

    print("Top Lonely Words:", lonely_words)


# ─────────────────────────────────────────────
# 4. Discriminative TF-IDF Features
# ─────────────────────────────────────────────

def discriminative_tfidf_features(df, top_n=20):

    print("\n===== DISCRIMINATIVE TF-IDF FEATURES =====")

    clean_text = preprocess_for_classical(df["text"])

    vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
    X = vectorizer.fit_transform(clean_text)
    y = df["label"]

    chi2_scores, _ = chi2(X, y)

    feature_names = np.array(vectorizer.get_feature_names_out())
    top_indices = np.argsort(chi2_scores)[-top_n:]

    top_features = feature_names[top_indices]
    top_scores = chi2_scores[top_indices]

    plt.figure()
    sns.barplot(x=top_scores, y=top_features)
    plt.title("Top Discriminative TF-IDF Features")
    plt.xlabel("Chi-Square Score")
    save_plot("discriminative_tfidf.png")

    print("\nTop Features:")
    for f, s in zip(top_features[::-1], top_scores[::-1]):
        print(f"{f} | Score: {s:.3f}")


# ─────────────────────────────────────────────
# 5. Linguistic Feature Analysis
# ─────────────────────────────────────────────

def linguistic_feature_analysis(df):

    print("\n===== LINGUISTIC FEATURE ANALYSIS =====")

    clean_text = preprocess_for_classical(df["text"])
    features = extract_linguistic_features(clean_text)

    feature_names = [
        "First Person Ratio",
        "Social Word Ratio",
        "Question Marks",
        "Avg Sentence Length",
        "Sentiment Polarity"
    ]

    feature_df = pd.DataFrame(features, columns=feature_names)
    feature_df["label"] = df["label"].values

    print("\nMean Feature Values by Class:")
    print(feature_df.groupby("label").mean())

    for feature in feature_names:
        plt.figure()
        sns.boxplot(x="label", y=feature, data=feature_df)
        plt.title(f"{feature} by Class")
        save_plot(f"{feature.replace(' ', '_').lower()}.png")


# ─────────────────────────────────────────────
# 6. Sentiment Distribution
# ─────────────────────────────────────────────

def sentiment_distribution(df):

    print("\n===== SENTIMENT DISTRIBUTION =====")

    clean_text = preprocess_for_classical(df["text"])
    features = extract_linguistic_features(clean_text)

    sentiment_scores = features[:, -1]

    sentiment_df = pd.DataFrame({
        "sentiment": sentiment_scores,
        "label": df["label"]
    })

    plt.figure()
    sns.histplot(
        data=sentiment_df,
        x="sentiment",
        hue="label",
        bins=30,
        kde=True,
        element="step"
    )
    plt.title("Sentiment Distribution by Class")
    save_plot("sentiment_distribution.png")


# ─────────────────────────────────────────────
# Run Full Pipeline
# ─────────────────────────────────────────────

def run_eda(df):

    print("\nRunning full EDA pipeline...")
    dataset_overview(df)
    analyze_post_length(df)
    most_frequent_words(df)
    discriminative_tfidf_features(df)
    linguistic_feature_analysis(df)
    sentiment_distribution(df)

    print("\nAll plots saved to:", RESULTS_DIR)
    print("===== EDA COMPLETE =====")