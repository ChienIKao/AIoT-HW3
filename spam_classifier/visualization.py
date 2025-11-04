"""Visualization utilities for the spam classifier dataset."""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib.pyplot as plt
import pandas as pd
from wordcloud import WordCloud

from .data import clean_text, prepare_dataset


def tokenize(texts: Iterable[str]) -> Counter[str]:
    """Tokenize cleaned texts into word frequency counts."""
    counter: Counter[str] = Counter()
    for text in texts:
        cleaned = clean_text(text)
        if not cleaned:
            continue
        counter.update(cleaned.split())
    return counter


def plot_top_tokens(
    frequencies: Counter[str],
    output_path: Path,
    top_n: int = 20,
) -> Path:
    """Render a horizontal bar chart of the most common tokens."""
    top_items = frequencies.most_common(top_n)
    if not top_items:
        raise ValueError("No tokens available to visualize.")

    words, counts = zip(*reversed(top_items))

    plt.figure(figsize=(8, 6))
    bars = plt.barh(words, counts, color="#1f77b4")
    plt.xlabel("Frequency")
    plt.ylabel("Token")
    plt.title(f"Top {top_n} Tokens by Frequency")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    return output_path


def generate_wordcloud(
    frequencies: Counter[str],
    output_path: Path,
) -> Path:
    """Generate and save a word cloud image from token frequencies."""
    if not frequencies:
        raise ValueError("No tokens available to visualize.")

    cloud = WordCloud(
        width=800,
        height=400,
        background_color="white",
        colormap="viridis",
    ).generate_from_frequencies(dict(frequencies))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cloud.to_file(output_path)
    return output_path


def visualize_dataset(
    data_path: Path,
    output_dir: Path,
    top_n: int = 20,
) -> dict[str, Path]:
    """Create both bar chart and word cloud visualizations for the dataset."""
    dataset = prepare_dataset(data_path)
    frequencies = tokenize(dataset["clean_text"])

    output_dir.mkdir(parents=True, exist_ok=True)
    bar_chart_path = output_dir / "top_tokens.png"
    wordcloud_path = output_dir / "wordcloud.png"

    paths = {
        "top_tokens": plot_top_tokens(frequencies, bar_chart_path, top_n=top_n),
        "wordcloud": generate_wordcloud(frequencies, wordcloud_path),
    }
    return paths


def class_distribution(
    dataset: pd.DataFrame,
    label_column: str = "label",
) -> pd.Series:
    """Return class distribution counts."""
    series = dataset[label_column].value_counts().sort_index()
    return series


def top_tokens_by_class(
    dataset: pd.DataFrame,
    label_column: str = "label",
    text_column: str = "clean_text",
    top_n: int = 10,
) -> Dict[str, List[tuple[str, int]]]:
    """Return top tokens per class."""
    results: Dict[str, List[tuple[str, int]]] = {}
    for label, group in dataset.groupby(label_column):
        counter: Counter[str] = Counter()
        for text in group[text_column]:
            if isinstance(text, str):
                counter.update(text.split())
        results[label] = counter.most_common(top_n)
    return results


def probability_bar(probability: float, threshold: float) -> plt.Figure:
    """Create a horizontal bar visualizing spam probability with threshold marker."""
    probability = max(0.0, min(1.0, probability))
    threshold = max(0.0, min(1.0, threshold))

    fig, ax = plt.subplots(figsize=(6, 1.2))
    ax.barh([0], [probability], color="#d62728" if probability >= threshold else "#1f77b4")
    ax.set_xlim(0, 1)
    ax.set_yticks([])
    ax.set_xlabel("Spam Probability")
    ax.axvline(threshold, color="black", linestyle="--", label=f"Threshold {threshold:.2f}")
    ax.legend(loc="upper right")
    ax.set_title("Spam Probability vs Threshold")
    plt.tight_layout()
    return fig


__all__ = [
    "visualize_dataset",
    "plot_top_tokens",
    "generate_wordcloud",
    "tokenize",
    "class_distribution",
    "top_tokens_by_class",
    "probability_bar",
]
