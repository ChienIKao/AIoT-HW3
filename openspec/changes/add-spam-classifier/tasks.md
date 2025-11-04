## 1. Specification
- [x] 1.1 Draft spam-classifier, spam-classifier-cli, and spam-classifier-demo spec deltas.
- [x] 1.2 Run `openspec validate add-spam-classifier --strict` and resolve issues.

## 2. Implementation
- [x] 2.1 Build data ingestion, cleaning, vectorization, and training modules with persisted artifacts.
- [x] 2.2 Expose a prediction service with reusable load and predict helpers.
- [x] 2.3 Implement Click CLI commands for training and single-message prediction.
- [x] 2.4 Create Streamlit demo that loads the persisted model and classifies input text.
- [x] 2.5 Add visualization utilities and CLI command to export token frequency plots.
- [x] 2.6 Add offline evaluation workflow producing confusion matrix and ROC artifacts.

## 3. Validation & Docs
- [x] 3.1 Add README / usage docs covering CLI commands and Streamlit usage.
- [x] 3.2 Re-run `openspec validate add-spam-classifier --strict` and applicable tests.
- [x] 3.3 Document visualization workflow and refresh validation.
