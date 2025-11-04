# Project Context

## Purpose
- Deliver a spec-driven spam email classifier covering data preparation, model training, CLI tooling, and a Streamlit demo.
- Provide reproducible commands (train, eval, infer, visualize) that persist artifacts for reuse across interfaces.
- Maintain OpenSpec documentation so future contributors can extend the system (e.g., batch inference, CSV export, additional visualizations).

## Tech Stack
- Python 3.10+
- Core libraries: pandas, numpy, scikit-learn, scipy, joblib
- CLI: Click
- Visualization: matplotlib, wordcloud
- Web demo: Streamlit
- Spec tooling: OpenSpec CLI

## Project Conventions

### Code Style
- Follow PEP 8 with type hints for public functions.
- Keep modules single-purpose (`data`, `training`, `service`, `evaluation`, `visualization`).
- Prefer pure functions; side effects (file writes) live in orchestration layers.

### Architecture Patterns
- `spam_classifier/` package split into:
  - `data.py` for loading/cleaning datasets
  - `training.py` for model training + artifact persistence
  - `service.py` for inference helpers and artifact loading
  - `evaluation.py` for offline evaluation using persisted artifacts
  - `visualization.py` and `reporting.py` for exploratory charts and metric plots
  - `cli.py` wiring Click commands; `cli.py` at repo root exposes the CLI module
- Artifacts stored under `artifacts/` (`model.pkl`, `vectorizer.pkl`, metrics JSON, confusion matrix, ROC curve, top token chart, word cloud).
- Streamlit app consumes the same artifacts to keep behaviour consistent.

### Testing Strategy
- Manual verification via CLI commands:
  - `python cli.py train ...`
  - `python cli.py eval ...`
  - `python cli.py infer ...`
  - `python cli.py visualize ...`
- Streamlit smoke test after training to confirm UI wiring.
- Future work: add automated unit tests for preprocessing and pipeline components.

### Git Workflow
- Each feature/change MUST be captured as an OpenSpec change (proposal + tasks + specs).
- Run `openspec validate <change-id> --strict` before requesting review.
- Commit artifacts and code together once the change is complete.

## Domain Context
- Dataset: SMS Spam Collection (`datasets/sms_spam_no_header.csv`).
- Labels: `ham` or `spam`; training enforces macro F1 ≥ 0.95.
- Artifacts reused across CLI and Streamlit to avoid retraining.

## Important Constraints
- Training splits data 80/20 with stratification and fixed `random_state=42` for reproducibility.
- TF-IDF features (1-2 grams, min_df=2, max_df=0.95, sublinear TF) + logistic regression (`liblinear`, class_weight balanced).
- ROC AUC fallback to 0.5 if the metric is undefined (all one class).

## External Dependencies
- No external services; all computation is local.
- Datasets bundled in repo; advanced features (batch .txt/.eml parsing) will require additional parsing utilities when implemented.

## Roadmap
- Implement optional/bonus features:
  - Batch inference for uploaded email files with CSV export of predictions.
  - Surface top tokens/word cloud directly inside Streamlit.
  - Visualize model confidence distribution in the demo.
- Add automated test coverage for preprocessing and service layers.
- Consider packaging the CLI as a console script for easier distribution.
