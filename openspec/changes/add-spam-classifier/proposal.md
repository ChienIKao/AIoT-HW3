## Why
- Provide a complete baseline for SMS spam detection required by the assignment brief.
- Ensure model, CLI, and demo behaviors are spec-driven and traceable via OpenSpec.

## What Changes
- Add a TF-IDF + logistic regression training pipeline that persists metrics and artifacts.
- Deliver a reusable inference service layer consumable by CLI commands and the Streamlit demo.
- Build a Click-based CLI for training and predicting spam vs ham labels.
- Ship a Streamlit demo app that classifies free-form text and displays model confidence.
- Author new spam-classifier, spam-classifier-cli, and spam-classifier-demo capability specs.

## Impact
- Introduces Python dependencies: pandas, scikit-learn, numpy, joblib, click, streamlit.
- Training requires ~1 minute on local CPU for the provided dataset.
- Produces serialized artifacts under `artifacts/` for reuse by CLI and demo.
- Requires developers to run `openspec validate add-spam-classifier --strict` before implementation approval.
