"""Inference helpers for the spam classifier."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import joblib

from .data import clean_text
from .training import (
    DEFAULT_ARTIFACTS_DIR,
    METRICS_FILENAME,
    MODEL_FILENAME,
    VECTORIZER_FILENAME,
)


@lru_cache(maxsize=None)
def _load_cached(model_path: str, vectorizer_path: str):
    model_file = Path(model_path)
    vectorizer_file = Path(vectorizer_path)

    missing = [str(p) for p in (vectorizer_file, model_file) if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing model artifacts: " + ", ".join(missing))

    vectorizer = joblib.load(vectorizer_file)
    model = joblib.load(model_file)
    return vectorizer, model


def _resolve_paths(
    artifacts_dir: Optional[str | Path] = None,
    model_path: Optional[str | Path] = None,
    vectorizer_path: Optional[str | Path] = None,
) -> Tuple[Path, Path]:
    if artifacts_dir is None and model_path is None and vectorizer_path is None:
        artifacts_dir = DEFAULT_ARTIFACTS_DIR

    if artifacts_dir is not None:
        base = Path(artifacts_dir)
        model_file = Path(model_path) if model_path else base / MODEL_FILENAME
        vectorizer_file = Path(vectorizer_path) if vectorizer_path else base / VECTORIZER_FILENAME
    else:
        if model_path is None or vectorizer_path is None:
            raise ValueError(
                "Both model_path and vectorizer_path must be provided when artifacts_dir is None."
            )
        model_file = Path(model_path)
        vectorizer_file = Path(vectorizer_path)

    return model_file.resolve(), vectorizer_file.resolve()


def load_artifacts(
    artifacts_dir: str | Path | None = None,
    model_path: str | Path | None = None,
    vectorizer_path: str | Path | None = None,
):
    """Load vectorizer and model from disk, caching across calls."""
    model_file, vectorizer_file = _resolve_paths(artifacts_dir, model_path, vectorizer_path)
    return _load_cached(str(model_file), str(vectorizer_file))


def predict_messages(
    texts: Iterable[str],
    artifacts_dir: str | Path | None = None,
    model_path: str | Path | None = None,
    vectorizer_path: str | Path | None = None,
) -> List[Dict[str, float | str]]:
    """Predict spam probabilities for a list of messages."""
    vectorizer, model = load_artifacts(artifacts_dir, model_path, vectorizer_path)

    cleaned = [clean_text(str(text)) for text in texts]
    if not cleaned:
        return []

    features = vectorizer.transform(cleaned)
    probabilities = model.predict_proba(features)[:, 1]

    results: List[Dict[str, float | str]] = []
    for original, prob in zip(texts, probabilities):
        label = "spam" if prob >= 0.5 else "ham"
        results.append(
            {
                "text": original,
                "label": label,
                "spam_probability": float(prob),
            }
        )
    return results


def load_metrics(
    artifacts_dir: str | Path | None = None,
) -> Dict[str, float] | None:
    """Load training metrics JSON file if present."""
    base_dir = Path(artifacts_dir) if artifacts_dir is not None else DEFAULT_ARTIFACTS_DIR
    metrics_path = base_dir / METRICS_FILENAME
    if not metrics_path.exists():
        return None
    try:
        return json.loads(metrics_path.read_text())
    except json.JSONDecodeError:
        return None


def reset_cache() -> None:
    """Clear cached artifacts, primarily for testing."""
    _load_cached.cache_clear()


__all__ = ["load_artifacts", "predict_messages", "load_metrics", "reset_cache"]
