## ADDED Requirements
### Requirement: Command Line Entry Point
The project MUST expose a Click-based CLI module for interacting with the spam classifier.

#### Scenario: Display CLI help
- **WHEN** `python -m spam_classifier.cli --help` is executed
- **THEN** the output lists the `train`, `eval`, `infer`, and `visualize` commands with short descriptions.

### Requirement: Train Command Wiring
Running training via the CLI MUST persist artifacts and report metrics for transparency.

#### Scenario: Train from CLI
- **WHEN** `python -m spam_classifier.cli train --data datasets/sms_spam_no_header.csv`
- **THEN** the command delegates to the training pipeline and prints accuracy, precision, recall, f1, and roc_auc with four decimal places.
- **AND** the command accepts optional `--artifacts-dir` to override the output directory (default `artifacts`).

### Requirement: Evaluate Command Behavior
Users MUST be able to score a labeled dataset with existing artifacts.

#### Scenario: Evaluate holdout dataset
- **WHEN** `python -m spam_classifier.cli eval --data datasets/sms_spam_no_header.csv`
- **THEN** the CLI prints accuracy, precision, recall, f1, and roc_auc for the dataset.
- **AND** the command saves `eval_metrics.json`, `eval_confusion_matrix.png`, and `eval_roc_curve.png` inside the artifacts directory.

### Requirement: Infer Command Behavior
Users MUST be able to classify an arbitrary message from the CLI without writing code.

#### Scenario: Infer single message
- **WHEN** `python -m spam_classifier.cli infer --text "Win cash now"`
- **THEN** the CLI prints a single-line JSON object with keys `label` and `spam_probability`.
- **AND** a spam probability >= 0.5 maps to the `spam` label.
- **AND** the command exits with code 0 on success and 2 when artifacts are missing.

### Requirement: Visualization Command
The CLI MUST provide an option to visualize token frequencies for exploratory analysis.

#### Scenario: Generate frequency plots
- **WHEN** `python -m spam_classifier.cli visualize --data datasets/sms_spam_no_header.csv --output-dir artifacts`
- **THEN** the command saves a bar chart image `artifacts/top_tokens.png` illustrating the top 20 tokens by frequency across the dataset.
- **AND** it saves a word cloud image `artifacts/wordcloud.png` derived from the same token frequency distribution.
- **AND** the command exits with code 0 after confirming the file paths, or 1 if dataset loading fails.
