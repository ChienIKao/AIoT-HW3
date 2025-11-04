"""Model evaluation utilities."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)

from .data import encode_labels, prepare_dataset
from .reporting import save_confusion_matrix, save_roc_curve
from .service import load_artifacts
from .training import DEFAULT_ARTIFACTS_DIR

EVAL_METRICS_FILENAME = "eval_metrics.json"
EVAL_CONFUSION_FILENAME = "eval_confusion_matrix.png"
EVAL_ROC_FILENAME = "eval_roc_curve.png"


def evaluate_model(
    data_path: str | Path,
    artifacts_dir: str | Path = DEFAULT_ARTIFACTS_DIR,
) -> Dict[str, float]:
    """Evaluate persisted artifacts on a labeled dataset and save reports."""
    dataset = prepare_dataset(data_path)
    y_true = np.array(encode_labels(dataset["label"].tolist()))
    texts = dataset["clean_text"].tolist()

    vectorizer, model = load_artifacts(artifacts_dir=artifacts_dir)
    features = vectorizer.transform(texts)

    y_pred = model.predict(features)
    y_prob = model.predict_proba(features)[:, 1]

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    accuracy = accuracy_score(y_true, y_pred)
    try:
        roc_auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        roc_auc = 0.5

    metrics = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": float(roc_auc),
    }

    artifacts_path = Path(artifacts_dir)
    artifacts_path.mkdir(parents=True, exist_ok=True)

    (artifacts_path / EVAL_METRICS_FILENAME).write_text(json.dumps(metrics, indent=2))

    cm = confusion_matrix(y_true, y_pred)
    save_confusion_matrix(cm, artifacts_path / EVAL_CONFUSION_FILENAME, labels=("ham", "spam"))

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    save_roc_curve(fpr, tpr, artifacts_path / EVAL_ROC_FILENAME)

    return metrics


__all__ = [
    "evaluate_model",
    "EVAL_METRICS_FILENAME",
    "EVAL_CONFUSION_FILENAME",
    "EVAL_ROC_FILENAME",
]
