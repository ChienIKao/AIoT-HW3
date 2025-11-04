## ADDED Requirements
### Requirement: Streamlit Demo Launch
The repository MUST include a Streamlit app that demonstrates the spam classifier.

#### Scenario: Start demo locally
- **WHEN** a developer runs `streamlit run streamlit_app.py`
- **THEN** the page title is "Spam Email Classifier" and renders a multiline text area labeled "Email content".
- **AND** submitting text triggers prediction using the persisted artifacts and displays the label (`Ham` or `Spam`) and spam probability as a percentage.

### Requirement: Artifact Status Feedback
The demo MUST guide users when model artifacts are missing or stale.

#### Scenario: Missing artifacts message
- **GIVEN** `artifacts/model.pkl` does not exist
- **WHEN** the Streamlit app initializes
- **THEN** it renders an error callout prompting the user to run the training CLI before using the demo.

### Requirement: Metrics Snapshot Panel
The demo MUST surface the latest saved evaluation metrics for transparency.

#### Scenario: Display metrics sidebar
- **WHEN** artifacts include `metrics.json`
- **THEN** the demo shows a sidebar section titled "Validation Metrics" listing accuracy, precision, recall, f1, and roc_auc rounded to three decimals.
