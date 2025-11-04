"""Data loading and preprocessing utilities for the spam classifier."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

import pandas as pd

DEFAULT_DATA_PATH = Path("datasets/sms_spam_no_header.csv")
_CLEAN_REGEX = re.compile(r"[^a-z0-9\s]+")


def clean_text(text: str) -> str:
    """Normalize text by lowercasing, removing punctuation, and collapsing whitespace."""
    normalized = text.lower()
    normalized = _CLEAN_REGEX.sub(" ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def load_raw_dataset(path: str | Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Load the SMS spam dataset, applying canonical column names."""
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Dataset not found at {source}")
    df = pd.read_csv(source, names=["label", "text"], encoding="utf-8")
    return prepare_dataframe(df)


def prepare_dataframe(
    df: pd.DataFrame,
    label_column: str = "label",
    text_column: str = "text",
) -> pd.DataFrame:
    """Clean a dataframe and return standardized columns."""
    if label_column not in df.columns or text_column not in df.columns:
        raise KeyError("Specified label/text columns not found in dataframe")

    data = df[[label_column, text_column]].copy()
    data = data.rename(columns={label_column: "label", text_column: "text"})
    data["label"] = data["label"].astype(str).str.strip().str.lower()
    data["text"] = data["text"].astype(str)
    data = data.dropna(subset=["label", "text"])
    data = data.drop_duplicates(subset=["text"])
    data["clean_text"] = data["text"].map(clean_text)
    data = data[data["clean_text"].str.len() > 0]
    return data.reset_index(drop=True)


def prepare_dataset(path: str | Path = DEFAULT_DATA_PATH) -> pd.DataFrame:
    """Load and clean the default dataset, returning de-duplicated records."""
    df = pd.read_csv(path, names=["label", "text"], encoding="utf-8")
    return prepare_dataframe(df)


def encode_labels(labels: Iterable[str]) -> list[int]:
    """Map ham/spam labels to binary targets."""
    mapping = {"ham": 0, "spam": 1}
    encoded = []
    for label in labels:
        key = label.strip().lower()
        if key not in mapping:
            raise ValueError(f"Unexpected label: {label}")
        encoded.append(mapping[key])
    return encoded


def decode_labels(targets: Iterable[int]) -> list[str]:
    """Map binary predictions back to ham/spam labels."""
    reverse = {0: "ham", 1: "spam"}
    decoded = []
    for target in targets:
        if target not in reverse:
            raise ValueError(f"Unexpected target value: {target}")
        decoded.append(reverse[target])
    return decoded
