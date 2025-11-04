"""Training utilities for the spam classifier."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split

from .data import DEFAULT_DATA_PATH, decode_labels, encode_labels, prepare_dataset
from .reporting import save_confusion_matrix, save_pr_curve, save_roc_curve

DEFAULT_ARTIFACTS_DIR = Path("artifacts")
MODEL_FILENAME = "model.pkl"
VECTORIZER_FILENAME = "vectorizer.pkl"
METRICS_FILENAME = "metrics.json"
CONFUSION_MATRIX_FILENAME = "confusion_matrix.png"
ROC_CURVE_FILENAME = "roc_curve.png"
PR_CURVE_FILENAME = "pr_curve.png"
VALIDATION_PREDICTIONS_FILENAME = "validation_predictions.json"
F1_THRESHOLD = 0.95
RANDOM_STATE = 42


@dataclass
class TrainingResult:
    vectorizer: TfidfVectorizer
    model: LogisticRegression
    metrics: Dict[str, float]
    artifacts_dir: Path


def train_pipeline(
    data_path: str | Path = DEFAULT_DATA_PATH,
    artifacts_dir: str | Path = DEFAULT_ARTIFACTS_DIR,
) -> TrainingResult:
    """Run the full training workflow and persist artifacts."""
    dataset = prepare_dataset(data_path)
    texts = dataset["clean_text"].tolist()
    labels = np.array(encode_labels(dataset["label"].tolist()))

    X_train, X_val, y_train, y_val = train_test_split(
        texts,
        labels,
        test_size=0.2,
        stratify=labels,
        random_state=RANDOM_STATE,
    )

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        sublinear_tf=True,
    )
    X_train_vec = vectorizer.fit_transform(X_train)

    model = LogisticRegression(
        max_iter=1000,
        solver="liblinear",
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )
    model.fit(X_train_vec, y_train)

    X_val_vec = vectorizer.transform(X_val)
    y_pred = model.predict(X_val_vec)
    y_prob = model.predict_proba(X_val_vec)[:, 1]

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_val, y_pred, average="macro", zero_division=0
    )
    accuracy = accuracy_score(y_val, y_pred)
    try:
        roc_auc = roc_auc_score(y_val, y_prob)
    except ValueError:
        roc_auc = 0.5

    metrics = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": float(roc_auc),
    }

    if metrics["f1"] < F1_THRESHOLD:
        raise RuntimeError(
            f"F1-score {metrics['f1']:.3f} is below required threshold {F1_THRESHOLD}."
        )

    precision_curve, recall_curve, thresholds = precision_recall_curve(y_val, y_prob)

    artifacts_path = save_artifacts(
        vectorizer,
        model,
        metrics,
        y_val,
        y_pred,
        y_prob,
        precision_curve,
        recall_curve,
        thresholds,
        artifacts_dir,
    )
    return TrainingResult(
        vectorizer=vectorizer,
        model=model,
        metrics=metrics,
        artifacts_dir=artifacts_path,
    )


def save_artifacts(
    vectorizer: TfidfVectorizer,
    model: LogisticRegression,
    metrics: Dict[str, float],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    precision_curve: np.ndarray,
    recall_curve: np.ndarray,
    thresholds: np.ndarray,
    artifacts_dir: str | Path = DEFAULT_ARTIFACTS_DIR,
) -> Path:
    """Persist artifacts to disk."""
    target = Path(artifacts_dir)
    target.mkdir(parents=True, exist_ok=True)

    joblib.dump(vectorizer, target / VECTORIZER_FILENAME)
    joblib.dump(model, target / MODEL_FILENAME)

    metrics_path = target / METRICS_FILENAME
    metrics_path.write_text(json.dumps(metrics, indent=2))

    cm = confusion_matrix(y_true, y_pred)
    save_confusion_matrix(cm, target / CONFUSION_MATRIX_FILENAME, labels=("ham", "spam"))

    fpr, tpr, _ = roc_curve(y_true, y_prob)
    save_roc_curve(fpr, tpr, target / ROC_CURVE_FILENAME)

    save_pr_curve(precision_curve, recall_curve, thresholds, target / PR_CURVE_FILENAME)

    validation_payload = {
        "labels": decode_labels(y_true),
        "probabilities": y_prob.tolist(),
        "threshold": 0.5,
    }
    (target / VALIDATION_PREDICTIONS_FILENAME).write_text(json.dumps(validation_payload, indent=2))
    return target


__all__ = [
    "TrainingResult",
    "train_pipeline",
    "save_artifacts",
    "MODEL_FILENAME",
    "VECTORIZER_FILENAME",
    "METRICS_FILENAME",
    "CONFUSION_MATRIX_FILENAME",
    "ROC_CURVE_FILENAME",
    "PR_CURVE_FILENAME",
    "VALIDATION_PREDICTIONS_FILENAME",
]
