"""
Preprocessing utilities for the FIG-Loneliness dataset.

Responsibilities:
- Text cleaning and normalisation
- Preparation for classical ML models
- Minimal preprocessing for transformer models
- Conversion of one-hot loneliness labels into scalar binary labels
- Dataset integrity validation

Expected dataset structure:
    - 'text' column (string)
    - 'lonely' column (one-hot vector: [non_lonely, lonely])
"""

import re
import numpy as np
import pandas as pd


# ─────────────────────────────────────────────
# 1. Text Cleaning Utilities
# ─────────────────────────────────────────────


def clean_text(text: str) -> str:
    """
    Perform light normalisation for classical ML models.

    Steps:
        - Lowercase text
        - Remove URLs
        - Collapse multiple spaces

    Parameters
    ----------
    text : str
        Raw input text.

    Returns
    -------
    str
        Cleaned text.
    """

    text = text.lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def preprocess_for_classical(text_series: pd.Series) -> pd.Series:
    """
    Apply cleaning pipeline for classical ML approaches
    (e.g., TF-IDF + Logistic Regression).

    Parameters
    ----------
    text_series : pd.Series

    Returns
    -------
    pd.Series
        Cleaned text series.
    """
    return text_series.apply(clean_text)


def preprocess_for_bert(text_series: pd.Series) -> pd.Series:
    """
    Minimal preprocessing for transformer-based models.

    BERT-like models require minimal cleaning since:
        - Tokenisation handles casing
        - Special tokens are important

    Parameters
    ----------
    text_series : pd.Series

    Returns
    -------
    pd.Series
        Lightly stripped text.
    """
    return text_series.str.strip()


# ─────────────────────────────────────────────
# 2. Dataset Preparation
# ─────────────────────────────────────────────


def prepare_binary_dataframe(dataset: dict) -> pd.DataFrame:
    """
    Convert DatasetDict into a single pandas DataFrame
    with binary labels.

    Output columns:
        - text  : original post
        - label : 0 (non-lonely) or 1 (lonely)
        - split : train / validation / test

    Assumptions
    -----------
    - dataset is a dictionary with keys:
        {"train", "validation", "test"}
    - Each split contains:
        - 'text'
        - 'lonely' as one-hot vector [non_lonely, lonely]

    Returns
    -------
    pd.DataFrame
        Combined dataset across all splits.
    """

    all_dfs = []

    for split_name, split_data in dataset.items():

        # Convert HuggingFace Dataset to pandas
        df = split_data.to_pandas()

        # ─────────────────────────────────────────
        # Integrity checks
        # ─────────────────────────────────────────
        if "text" not in df.columns:
            raise ValueError(f"'text' column missing in {split_name} split.")

        if "lonely" not in df.columns:
            raise ValueError(f"'lonely' column missing in {split_name} split.")

        # Validate one-hot structure
        sample_value = df["lonely"].iloc[0]

        if not isinstance(sample_value, (list, tuple, np.ndarray)):
            raise ValueError(
                f"'lonely' column in {split_name} is not a valid one-hot vector."
            )

        if len(sample_value) != 2:
            raise ValueError(f"'lonely' column in {split_name} does not have length 2.")

        # Convert one-hot → scalar label
        df["label"] = df["lonely"].apply(lambda x: int(list(x)[1]))

        # Keep only relevant columns
        df = df[["text", "label"]]
        df["split"] = split_name

        all_dfs.append(df)

    # Combine splits
    full_df = pd.concat(all_dfs, ignore_index=True)

    # ─────────────────────────────────────────
    # Summary statistics
    # ─────────────────────────────────────────
    print(f"\nCombined dataset size: {len(full_df)} samples")
    print("\nLabel distribution:")
    print(full_df["label"].value_counts())
    print("\nLabel proportions:")
    print(full_df["label"].value_counts(normalize=True))

    return full_df
