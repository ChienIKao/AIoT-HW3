## MODIFIED Requirements
### Requirement: Baseline Spam Classifier Training
The project MUST ship a reproducible training entrypoint for the SMS spam dataset.

#### Scenario: Default training pipeline
- **AND** the command persists `artifacts/vectorizer.pkl`, `artifacts/model.pkl`, `artifacts/metrics.json`, `artifacts/confusion_matrix.png`, `artifacts/roc_curve.png`, and `artifacts/pr_curve.png`.
- **AND** the command saves `artifacts/validation_predictions.json` containing the validation labels and spam probabilities for threshold exploration.

### Requirement: Offline Evaluation
The project MUST support evaluating an already-trained model on a labeled dataset.

#### Scenario: Evaluate on holdout data
- **AND** the command saves `eval_pr_curve.png` alongside `eval_confusion_matrix.png` inside the artifacts directory.

## ADDED Requirements
### Requirement: Validation Predictions Artifact
The system MUST persist validation outcomes for interactive metrics.

#### Scenario: Store validation predictions
- **WHEN** the training pipeline completes successfully
- **THEN** it writes a JSON file containing arrays `labels` and `probabilities` aligned with the validation split order.
- **AND** the file includes the validation threshold (default 0.5) used during training metrics.
