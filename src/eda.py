
import logging
from collections import Counter
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from datasets import DatasetDict
from wordcloud import WordCloud

from .results import load_json, json_exists, save_json, plot_path, record_step
from .config import EDA_TOP_N_GRAMS, EDA_WORDCLOUD_MAX_WORDS

logger = logging.getLogger(__name__)

# Colour palette
CLR_NON_LONELY = "#4C9BE8"
CLR_LONELY = "#E8614C"
CLR_BOTH = [CLR_NON_LONELY, CLR_LONELY]

LABEL_NAMES = {0: "Non-Lonely", 1: "Lonely"}

LINGUISTIC_FEATURES = [
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


# Helpers
def _split_by_label(split_data, col: str) -> tuple[list, list]:
    #Return values_non_lonely, values_lonely for a column in a spli
    non_lonely, lonely = [], []
    for label, val in zip(split_data["label"], split_data[col]):
        if label == 0:
            non_lonely.append(val)
        else:
            lonely.append(val)
    return non_lonely, lonely


def _describe(values: list[float]) -> dict:
    #Basic descriptive stats for a list of numbers
    a = np.array(values, dtype=float)
    return {
        "mean": float(np.mean(a)),
        "median": float(np.median(a)),
        "std": float(np.std(a)),
        "min": float(np.min(a)),
        "max": float(np.max(a)),
        "q25": float(np.percentile(a, 25)),
        "q75": float(np.percentile(a, 75)),
    }


def _savefig(name: str) -> None:
    p = plot_path(name)
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"Plot saved → {p}")


#  Class distribution
def eda_class_distribution(dataset: DatasetDict) -> dict:
    logger.info("EDA: class distribution")

    counts = {}
    for split_name, split_data in dataset.items():
        labels = split_data["label"]
        n_lonely = sum(labels)
        n_non_lonely = len(labels) - n_lonely
        counts[split_name] = {
            "total": len(labels),
            "lonely": n_lonely,
            "non_lonely": n_non_lonely,
            "lonely_pct": round(100 * n_lonely / len(labels), 2),
        }

    # Plot
    fig, axes = plt.subplots(1, len(counts), figsize=(5 * len(counts), 4))
    if len(counts) == 1:
        axes = [axes]

    for ax, (split_name, c) in zip(axes, counts.items()):
        bars = ax.bar(
            ["Non-Lonely", "Lonely"],
            [c["non_lonely"], c["lonely"]],
            color=CLR_BOTH,
            edgecolor="white",
            linewidth=1.2,
        )
        ax.bar_label(bars, fmt="%d", padding=3)
        ax.set_title(f"{split_name.capitalize()} Split", fontweight="bold")
        ax.set_ylabel("Count")
        ax.spines[["top", "right"]].set_visible(False)

    plt.suptitle("Class Distribution per Split", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    _savefig("eda_class_distribution.png")

    save_json("eda_class_distribution.json", counts)
    return counts


#  Text length distributions
def eda_length_stats(dataset: DatasetDict) -> dict:
    logger.info("EDA: text length statistics")

    train = dataset["train"]
    stats = {}
    for feat in ["word_count", "char_count", "sentence_count"]:
        non_lonely, lonely = _split_by_label(train, feat)
        stats[feat] = {
            "non_lonely": _describe(non_lonely),
            "lonely": _describe(lonely),
        }

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    titles = {
        "word_count": "Word Count",
        "char_count": "Character Count",
        "sentence_count": "Sentence Count",
    }

    for ax, feat in zip(axes, ["word_count", "char_count", "sentence_count"]):
        non_lonely, lonely = _split_by_label(train, feat)

        # Clip for readability
        cap = np.percentile(non_lonely + lonely, 97)
        non_lonely_c = [v for v in non_lonely if v <= cap]
        lonely_c = [v for v in lonely if v <= cap]

        ax.hist(
            non_lonely_c,
            bins=40,
            alpha=0.6,
            color=CLR_NON_LONELY,
            label="Non-Lonely",
            density=True,
        )
        ax.hist(
            lonely_c, bins=40, alpha=0.6, color=CLR_LONELY, label="Lonely", density=True
        )
        ax.set_title(titles[feat], fontweight="bold")
        ax.set_xlabel(titles[feat])
        ax.set_ylabel("Density")
        ax.legend()
        ax.spines[["top", "right"]].set_visible(False)

    plt.suptitle(
        "Text Length Distributions (Train Set, 97th percentile cap)",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    _savefig("eda_length_distribution.png")

    save_json("eda_length_stats.json", stats)
    return stats


#  Linguistic feature boxplots
def eda_linguistic_stats(dataset: DatasetDict) -> dict:
    logger.info("EDA: linguistic feature statistics")

    train = dataset["train"]
    ratio_features = [
        "pronoun_ratio",
        "negation_ratio",
        "social_word_ratio",
        "emotion_word_ratio",
        "noun_ratio",
        "verb_ratio",
        "adj_ratio",
        "adv_ratio",
        "type_token_ratio",
    ]

    stats = {}
    for feat in ratio_features:
        non_lonely, lonely = _split_by_label(train, feat)
        stats[feat] = {
            "non_lonely": _describe(non_lonely),
            "lonely": _describe(lonely),
        }

    # Boxplot
    n_feats = len(ratio_features)
    ncols = 3
    nrows = (n_feats + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 4 * nrows))
    axes_flat = axes.flatten()

    for ax, feat in zip(axes_flat, ratio_features):
        non_lonely, lonely = _split_by_label(train, feat)
        bp = ax.boxplot(
            [non_lonely, lonely],
            labels=["Non-Lonely", "Lonely"],
            patch_artist=True,
            notch=False,
            widths=0.5,
        )
        for patch, color in zip(bp["boxes"], CLR_BOTH):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax.set_title(feat.replace("_", " ").title(), fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)

    # Hide unused subplots
    for ax in axes_flat[n_feats:]:
        ax.set_visible(False)

    plt.suptitle(
        "Linguistic Feature Distributions by Class (Train Set)",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    _savefig("eda_linguistic_boxplots.png")

    save_json("eda_linguistic_stats.json", stats)
    return stats


#  N-gram analysis
def _count_ngrams(token_lists: list[list[str]], n: int) -> Counter:
    c: Counter = Counter()
    for tokens in token_lists:
        for i in range(len(tokens) - n + 1):
            ngram = " ".join(tokens[i : i + n])
            c[ngram] += 1
    return c


def eda_ngrams(dataset: DatasetDict, top_n: int = EDA_TOP_N_GRAMS) -> dict:
    logger.info("EDA: N-gram analysis")

    train = dataset["train"]
    non_lonely_tokens, lonely_tokens = _split_by_label(train, "tokens_no_stopwords")

    result = {}
    for n, label in [(1, "unigrams"), (2, "bigrams")]:
        nl_counts = _count_ngrams(non_lonely_tokens, n)
        l_counts = _count_ngrams(lonely_tokens, n)

        result[f"non_lonely_{label}"] = nl_counts.most_common(top_n)
        result[f"lonely_{label}"] = l_counts.most_common(top_n)

        # Plot side by side
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
        for ax, counts, color, title in [
            (ax1, nl_counts.most_common(top_n), CLR_NON_LONELY, "Non-Lonely"),
            (ax2, l_counts.most_common(top_n), CLR_LONELY, "Lonely"),
        ]:
            terms, freqs = zip(*counts) if counts else ([], [])
            ax.barh(
                list(reversed(terms)), list(reversed(freqs)), color=color, alpha=0.85
            )
            ax.set_title(
                f"{title} — Top {top_n} {label.capitalize()}", fontweight="bold"
            )
            ax.set_xlabel("Frequency")
            ax.spines[["top", "right"]].set_visible(False)

        plt.suptitle(
            f"Top {top_n} {label.capitalize()} by Class (Train Set)",
            fontsize=13,
            fontweight="bold",
        )
        plt.tight_layout()
        _savefig(f"eda_{label}.png")

    # Serialise Counter lists → JSON-safe
    for k in result:
        result[k] = [{"term": t, "count": c} for t, c in result[k]]

    save_json("eda_ngrams.json", result)
    return result


#  Word clouds
def eda_wordclouds(dataset: DatasetDict) -> None:
    #Attempt to generate word clouds; skip gracefully if wordcloud not installed
    logger.info("EDA: word clouds")

    train = dataset["train"]
    non_lonely_tokens, lonely_tokens = _split_by_label(train, "tokens_no_stopwords")

    for tokens_list, name, color in [
        (non_lonely_tokens, "non_lonely", CLR_NON_LONELY),
        (lonely_tokens, "lonely", CLR_LONELY),
    ]:
        text = " ".join(t for sublist in tokens_list for t in sublist)
        wc = WordCloud(
            width=900,
            height=500,
            background_color="white",
            max_words=EDA_WORDCLOUD_MAX_WORDS,
            colormap="Blues" if name == "non_lonely" else "Reds",
            collocations=False,
        ).generate(text)

        plt.figure(figsize=(12, 6))
        plt.imshow(wc, interpolation="bilinear")
        plt.axis("off")
        plt.title(
            f"Word Cloud — {LABEL_NAMES[0 if name == 'non_lonely' else 1]}",
            fontsize=14,
            fontweight="bold",
        )
        _savefig(f"eda_wordcloud_{name}.png")


#  POS tag distribution
def eda_pos_distribution(dataset: DatasetDict) -> dict:
    logger.info("EDA: POS distribution")

    train = dataset["train"]
    pos_stats: dict[str, dict[str, Counter]] = {
        "non_lonely": Counter(),
        "lonely": Counter(),
    }

    for label, pos_list in zip(train["label"], train["pos_tags"]):
        key = "lonely" if label == 1 else "non_lonely"
        for p in pos_list:
            pos_stats[key][p] += 1

    # Normalise
    result = {}
    for key, counter in pos_stats.items():
        total = sum(counter.values()) or 1
        result[key] = {p: round(c / total, 4) for p, c in counter.most_common(10)}

    # Plot
    all_pos = sorted(set(result["non_lonely"]) | set(result["lonely"]))
    x = np.arange(len(all_pos))
    width = 0.35

    fig, ax = plt.subplots(figsize=(13, 5))
    ax.bar(
        x - width / 2,
        [result["non_lonely"].get(p, 0) for p in all_pos],
        width,
        label="Non-Lonely",
        color=CLR_NON_LONELY,
        alpha=0.85,
    )
    ax.bar(
        x + width / 2,
        [result["lonely"].get(p, 0) for p in all_pos],
        width,
        label="Lonely",
        color=CLR_LONELY,
        alpha=0.85,
    )

    ax.set_xticks(x)
    ax.set_xticklabels(all_pos)
    ax.set_ylabel("Proportion of tokens")
    ax.set_title("POS Tag Distribution by Class (Train Set)", fontweight="bold")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    _savefig("eda_pos_distribution.png")

    save_json("eda_pos_distribution.json", result)
    return result


#  Feature correlation heatmap
def eda_correlation_heatmap(dataset: DatasetDict) -> None:
    logger.info("EDA: correlation heatmap")

    train = dataset["train"]
    numeric_cols = [c for c in LINGUISTIC_FEATURES if c in train.column_names] + [
        "label"
    ]
    df = pd.DataFrame({c: train[c] for c in numeric_cols})
    corr = df.corr()

    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(
        [c.replace("_", "\n") for c in corr.columns],
        fontsize=8,
        rotation=45,
        ha="right",
    )
    ax.set_yticklabels([c.replace("_", " ") for c in corr.columns], fontsize=8)

    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(
                j,
                i,
                f"{corr.values[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=7,
                color="black",
            )

    ax.set_title(
        "Feature Correlation Matrix (Train Set)", fontweight="bold", fontsize=13
    )
    plt.tight_layout()
    _savefig("eda_correlation_heatmap.png")


# Run EDA
def run_eda(dataset: DatasetDict) -> dict:

    if json_exists("eda_ngrams.json"):
        logger.info("EDA results already exist — loading from cache.")
        return {
            "class_distribution": load_json("eda_class_distribution.json"),
            "length_stats": load_json("eda_length_stats.json"),
            "linguistic_stats": load_json("eda_linguistic_stats.json"),
            "ngrams": load_json("eda_ngrams.json"),
            "pos_distribution": load_json("eda_pos_distribution.json"),
        }

    logger.info("Running full EDA...")

    class_dist = eda_class_distribution(dataset)
    length_stats = eda_length_stats(dataset)
    ling_stats = eda_linguistic_stats(dataset)
    ngrams = eda_ngrams(dataset)
    pos_dist = eda_pos_distribution(dataset)
    eda_wordclouds(dataset)
    eda_correlation_heatmap(dataset)

    summary = {
        "class_distribution": class_dist,
        "length_stats": length_stats,
        "linguistic_stats": ling_stats,
        "ngrams": ngrams,
        "pos_distribution": pos_dist,
    }

    save_json("eda_summary.json", summary)
    record_step("eda", meta={"plots_generated": True})

    logger.info("EDA complete.")
    return summary
