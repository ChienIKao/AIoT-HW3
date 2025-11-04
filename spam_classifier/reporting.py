"""Reporting helpers for model metrics and plots."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


def save_confusion_matrix(
    matrix: np.ndarray,
    output_path: Path,
    labels: Tuple[str, str] = ("ham", "spam"),
) -> Path:
    """Save a confusion matrix heatmap."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(matrix, interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(matrix.shape[1]),
        yticks=np.arange(matrix.shape[0]),
        xticklabels=labels,
        yticklabels=labels,
        ylabel="True label",
        xlabel="Predicted label",
        title="Confusion Matrix",
    )

    thresh = matrix.max() / 2.0 if matrix.size else 0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(
                j,
                i,
                f"{matrix[i, j]}",
                ha="center",
                va="center",
                color="white" if matrix[i, j] > thresh else "black",
            )

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def save_roc_curve(
    fpr: Iterable[float],
    tpr: Iterable[float],
    output_path: Path,
) -> Path:
    """Save an ROC curve plot."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(fpr, tpr, color="darkorange", lw=2, label="ROC curve")
    ax.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random chance")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Receiver Operating Characteristic")
    ax.legend(loc="lower right")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def save_pr_curve(
    precision: Sequence[float],
    recall: Sequence[float],
    thresholds: Sequence[float],
    output_path: Path,
) -> Path:
    """Save a precision-recall curve plot."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    thresholds = list(thresholds)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(recall, precision, color="green", lw=2, label="Precision-Recall curve")
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend(loc="lower left")

    if thresholds:
        for r, p, thresh in zip(recall[1:], precision[1:], thresholds):
            if abs(thresh - 0.5) < 1e-6:
                ax.scatter(r, p, color="red", label="Threshold 0.5")
                break

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


__all__ = ["save_confusion_matrix", "save_roc_curve", "save_pr_curve"]
