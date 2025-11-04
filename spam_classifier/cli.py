"""Command line interface for the spam classifier."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict

import click

from .data import DEFAULT_DATA_PATH
from .evaluation import evaluate_model
from .service import predict_messages
from .training import DEFAULT_ARTIFACTS_DIR, train_pipeline
from .visualization import visualize_dataset


def _print_metrics(metrics: Dict[str, float]) -> None:
    for key in ("accuracy", "precision", "recall", "f1", "roc_auc"):
        value = metrics.get(key)
        if value is None:
            continue
        if isinstance(value, float):
            click.echo(f"{key.title():<10}: {value:.4f}")
        else:
            click.echo(f"{key.title():<10}: {value}")


@click.group(help="Utilities for training, evaluating, and using the spam classifier.")
def cli() -> None:
    """CLI entry point."""


@cli.command()
@click.option(
    "--data",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=DEFAULT_DATA_PATH,
    show_default=True,
    help="Path to the raw SMS spam CSV dataset.",
)
@click.option(
    "--artifacts-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=DEFAULT_ARTIFACTS_DIR,
    show_default=True,
    help="Directory where model artifacts should be stored.",
)
def train(data: Path, artifacts_dir: Path) -> None:
    """Train the spam classifier and persist artifacts."""
    try:
        result = train_pipeline(data_path=data, artifacts_dir=artifacts_dir)
    except RuntimeError as err:
        click.echo(str(err), err=True)
        sys.exit(1)

    click.echo("Training completed.")
    _print_metrics(result.metrics)
    click.echo(f"Artifacts saved to {result.artifacts_dir}")


@cli.command(name="eval")
@click.option(
    "--data",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="Path to the labeled dataset for evaluation.",
)
@click.option(
    "--artifacts-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=DEFAULT_ARTIFACTS_DIR,
    show_default=True,
    help="Directory containing trained model artifacts.",
)
def evaluate(data: Path, artifacts_dir: Path) -> None:
    """Evaluate the persisted model on a labeled dataset."""
    try:
        metrics = evaluate_model(data_path=data, artifacts_dir=artifacts_dir)
    except FileNotFoundError as err:
        click.echo(str(err), err=True)
        sys.exit(2)

    click.echo("Evaluation completed.")
    _print_metrics(metrics)
    click.echo(f"Reports written to {artifacts_dir}")


@cli.command()
@click.option(
    "--text",
    type=str,
    required=False,
    help="Message to classify. If omitted, reads from stdin.",
)
@click.option(
    "--artifacts-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=DEFAULT_ARTIFACTS_DIR,
    show_default=True,
    help="Directory containing trained model artifacts.",
)
def infer(text: str | None, artifacts_dir: Path) -> None:
    """Predict whether a message is spam or ham."""
    message = text or sys.stdin.read().strip()
    if not message:
        raise click.BadParameter("Provide --text or pipe a message via stdin.")

    try:
        results = predict_messages([message], artifacts_dir=artifacts_dir)
    except FileNotFoundError as err:
        click.echo(str(err), err=True)
        sys.exit(2)

    if not results:
        click.echo("{}")
        return

    result = results[0]
    payload = {
        "label": result["label"],
        "spam_probability": round(result["spam_probability"], 4),
    }
    click.echo(json.dumps(payload))


@cli.command()
@click.option(
    "--data",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=DEFAULT_DATA_PATH,
    show_default=True,
    help="Path to the raw SMS spam CSV dataset.",
)
@click.option(
    "--output-dir",
    type=click.Path(file_okay=False, path_type=Path),
    default=DEFAULT_ARTIFACTS_DIR,
    show_default=True,
    help="Directory where visualization images should be written.",
)
@click.option(
    "--top-n",
    type=click.IntRange(5, 100),
    default=20,
    show_default=True,
    help="Number of top tokens to include in the frequency bar chart.",
)
def visualize(data: Path, output_dir: Path, top_n: int) -> None:
    """Generate token frequency bar chart and word cloud images."""
    try:
        outputs = visualize_dataset(data_path=data, output_dir=output_dir, top_n=top_n)
    except FileNotFoundError as err:
        click.echo(str(err), err=True)
        sys.exit(1)
    except ValueError as err:
        click.echo(str(err), err=True)
        sys.exit(1)

    for name, path in outputs.items():
        click.echo(f"{name.replace('_', ' ').title()} saved to {path}")


if __name__ == "__main__":
    cli()
