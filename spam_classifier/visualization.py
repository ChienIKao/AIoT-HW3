"""Visualization utilities for the spam classifier dataset."""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
from wordcloud import WordCloud

from .data import prepare_dataset, clean_text


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
    frequencies = tokenize(dataset["text"])

    output_dir.mkdir(parents=True, exist_ok=True)
    bar_chart_path = output_dir / "top_tokens.png"
    wordcloud_path = output_dir / "wordcloud.png"

    paths = {
        "top_tokens": plot_top_tokens(frequencies, bar_chart_path, top_n=top_n),
        "wordcloud": generate_wordcloud(frequencies, wordcloud_path),
    }
    return paths


__all__ = [
    "visualize_dataset",
    "plot_top_tokens",
    "generate_wordcloud",
    "tokenize",
]
