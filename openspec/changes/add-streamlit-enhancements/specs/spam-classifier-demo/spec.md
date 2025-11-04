## MODIFIED Requirements
### Requirement: Streamlit Demo Launch
The repository MUST include a Streamlit app that demonstrates the spam classifier.

#### Scenario: Start demo locally
- **AND** the sidebar includes selectors for available dataset files under `datasets/` and dropdowns to choose label/text columns (defaulting to `label` and `text`).

### Requirement: Metrics Snapshot Panel
The demo MUST surface the latest saved evaluation metrics for transparency.

#### Scenario: Display metrics sidebar
- **AND** the panel shows accuracy, precision, recall, f1, and roc_auc rounded to three decimals.
- **AND** it updates live when the threshold slider changes (displaying recalculated precision/recall/f1).

## ADDED Requirements
### Requirement: Dataset Insights Visualization
The demo MUST visualize dataset distribution and per-class top tokens.

#### Scenario: Show dataset charts
- **WHEN** artifacts are available and a dataset is selected
- **THEN** the main panel renders a class distribution bar chart and two tables (or charts) listing the top tokens for ham and spam classes.

### Requirement: Diagnostics Gallery
The demo MUST display evaluation plots produced by the training pipeline.

#### Scenario: Show evaluation plots
- **WHEN** `confusion_matrix.png`, `roc_curve.png`, and `pr_curve.png` exist in the artifacts directory
- **THEN** the demo displays them in a dedicated "Model Diagnostics" section.
- **AND** it falls back to warning messages if any image is missing.

### Requirement: Threshold Explorer
The demo MUST provide an interactive threshold control for validation predictions.

#### Scenario: Adjust decision threshold
- **WHEN** the user moves the threshold slider between 0.1 and 0.9
- **THEN** the sidebar updates precision, recall, and F1 based on the stored validation probabilities.
- **AND** the live inference probability bar shows the current threshold marker.

### Requirement: Live Inference Enhancements
The demo MUST make live predictions more discoverable.

#### Scenario: Live probability bar
- **WHEN** the user classifies a message
- **THEN** the result section includes a horizontal bar indicating spam probability with a marker for the current threshold.

### Requirement: Quick Test Buttons
The demo MUST offer instant spam/ham examples for testing.

#### Scenario: Autofill examples
- **WHEN** the user clicks "Use spam example" or "Use ham example"
- **THEN** the text area autofills with the predefined sample and triggers prediction automatically.
