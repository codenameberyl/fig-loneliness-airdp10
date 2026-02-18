"""
Preprocessing module for FIG-Loneliness dataset.

This module:
- Loads predefined dataset splits
- Applies minimal text normalisation
- Converts one-hot loneliness labels into scalar binary labels
- Performs integrity validation checks
"""

import os
import re
from datasets import load_from_disk, DatasetDict


def load_fig_loneliness(root_path: str) -> DatasetDict:
    """
    Load the FIG-Loneliness dataset from disk.

    Parameters:
    -----------
    root_path : str
        Path to the cloned FIG-Loneliness dataset directory.

    Returns:
    --------
    DatasetDict
        A DatasetDict containing train, validation, and test splits.
    """

    train_set = load_from_disk(os.path.join(root_path, "train_set"))
    dev_set = load_from_disk(os.path.join(root_path, "dev_set"))
    test_set = load_from_disk(os.path.join(root_path, "test_set"))

    return DatasetDict({
        "train": train_set,
        "validation": dev_set,
        "test": test_set
    })


def basic_text_normalisation(text: str) -> str:
    """
    Apply minimal text preprocessing.

    IMPORTANT:
    We deliberately avoid heavy cleaning (e.g., stopword removal,
    stemming, lemmatisation) to preserve subtle loneliness signals.

    Steps:
    - Lowercase text
    - Strip leading/trailing whitespace
    - Replace URLs with <URL>
    - Collapse multiple spaces

    Parameters:
    -----------
    text : str

    Returns:
    --------
    str : cleaned text
    """

    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = text.strip()

    # Replace URLs with special placeholder token
    text = re.sub(r"http\S+|www\S+", "<URL>", text)

    # Collapse multiple spaces into single space
    text = re.sub(r"\s+", " ", text)

    return text


def apply_preprocessing(dataset: DatasetDict) -> DatasetDict:
    """
    Apply preprocessing to all dataset splits.

    For each example:
    - Create 'text_clean'
    - Convert one-hot lonely vector into scalar 'label'

    lonely format:
        [non-lonely, lonely]

    We convert:
        label = lonely[1]

    Returns:
    --------
    DatasetDict with added columns:
        - text_clean
        - label
    """

    def process(example):
        # Create cleaned text column (preserve original text)
        example["text_clean"] = basic_text_normalisation(example["text"])

        # Convert one-hot loneliness vector to scalar label
        # lonely = [non-lonely, lonely]
        example["label"] = example["lonely"][1]

        return example

    dataset = dataset.map(process)

    # Remove columns not needed for modelling
    # (We kept annotation vectors for EDA later)
    dataset = dataset.remove_columns(["idx", "unique_id"])

    return dataset


def validate_dataset(dataset: DatasetDict, num_samples_to_print: int = 3):
    """
    Perform integrity checks after preprocessing.

    Checks:
    - Empty cleaned text count
    - Label distribution
    - Print sample cleaned texts for verification

    Parameters:
    -----------
    dataset : DatasetDict
    num_samples_to_print : int
        Number of example cleaned texts to print per split.
    """

    for split in dataset.keys():
        print(f"\n==============================")
        print(f"Validating split: {split}")
        print(f"==============================")

        data = dataset[split]

        # Check for empty cleaned text
        empty_count = sum(
            1 for x in data if not x["text_clean"].strip()
        )
        print(f"Empty cleaned texts: {empty_count}")

        # Label distribution
        labels = data["label"]
        label_counts = {
            label: labels.count(label)
            for label in set(labels)
        }
        print(f"Label distribution: {label_counts}")

        # Print sample cleaned texts
        print(f"\nSample cleaned texts:")
        for i in range(min(num_samples_to_print, len(data))):
            print(f"\nExample {i+1}:")
            print("Original:", data[i]["text"][:200], "...")
            print("Cleaned :", data[i]["text_clean"][:200], "...")
