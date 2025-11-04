## ADDED Requirements
### Requirement: Baseline Spam Classifier Training
The project MUST ship a reproducible training entrypoint for the SMS spam dataset.

#### Scenario: Default training pipeline
- **GIVEN** the raw dataset at `datasets/sms_spam_no_header.csv`
- **WHEN** a developer runs `python -m spam_classifier.cli train --data datasets/sms_spam_no_header.csv`
- **THEN** the pipeline cleans the text, performs an 80/20 stratified split, vectorizes with TF-IDF, and trains a logistic regression model.
- **AND** the command persists `artifacts/vectorizer.pkl`, `artifacts/model.pkl`, and `artifacts/metrics.json`.
- **AND** the metrics file records at least `accuracy`, `precision`, `recall`, `f1`, and `roc_auc` for the validation split.
- **AND** the command saves `artifacts/confusion_matrix.png` and `artifacts/roc_curve.png` visualizing validation performance.
- **AND** the macro F1-score captured in `artifacts/metrics.json` is >= 0.95.

### Requirement: Reusable Model Artifacts
Training MUST persist artifacts that can be reloaded without re-training.

#### Scenario: Load artifacts from disk
- **GIVEN** the artifacts created by the training pipeline
- **WHEN** `spam_classifier.service.load_artifacts()` is invoked without arguments
- **THEN** it returns the TF-IDF vectorizer and logistic regression model ready for inference.
- **AND** the function raises a `FileNotFoundError` if any expected artifact is missing.

### Requirement: Batch Prediction API
The codebase MUST expose a reusable API to score arbitrary email content.

#### Scenario: Predict multiple messages
- **WHEN** `spam_classifier.service.predict_messages(["hello", "claim prize now"] )` is called
- **THEN** it returns a list of results where each item includes the original text, predicted label (`ham` or `spam`), and spam probability in `[0,1]`.
- **AND** the API loads artifacts lazily (only on first prediction) and reuses them for subsequent calls.

### Requirement: Offline Evaluation
The project MUST support evaluating an already-trained model on a labeled dataset.

#### Scenario: Evaluate on holdout data
- **GIVEN** a trained model and vectorizer in `artifacts/`
- **WHEN** `python -m spam_classifier.cli eval --data datasets/sms_spam_no_header.csv`
- **THEN** the command loads the persisted artifacts and reports accuracy, precision, recall, f1, and roc_auc to stdout.
- **AND** it refreshes `artifacts/eval_metrics.json`, `artifacts/eval_confusion_matrix.png`, and `artifacts/eval_roc_curve.png` with results for the provided dataset.
