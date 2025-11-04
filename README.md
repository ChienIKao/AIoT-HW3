# Spam Email Classifier

A complete SMS spam detection project that trains a TF-IDF + logistic regression model, exposes a reusable CLI, and ships a Streamlit demo.

## Getting Started
1. Create a virtual environment (recommended).
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Ensure the dataset `datasets/sms_spam_no_header.csv` is present (included in this repo).

## Training the Model
Run the training command to produce model artifacts, metrics, and diagnostic plots:
```bash
python cli.py train --data datasets/sms_spam_no_header.csv --artifacts-dir artifacts
```
This command prints accuracy, precision, recall, F1, and ROC AUC, and persists the following artifacts in the specified directory:
- `vectorizer.pkl` / `model.pkl`
- `metrics.json`
- `confusion_matrix.png`, `roc_curve.png`, `pr_curve.png`
- `validation_predictions.json`

## Evaluating the Model
Score a labeled dataset using the saved artifacts:
```bash
python cli.py eval --data datasets/sms_spam_no_header.csv --artifacts-dir artifacts
```
The evaluation exports `eval_metrics.json`, `eval_confusion_matrix.png`, `eval_roc_curve.png`, and `eval_pr_curve.png` alongside the printed metrics.

## Predicting from the CLI
Classify a single message via:
```bash
python cli.py infer --text "Win cash now" --artifacts-dir artifacts
```
The CLI outputs a JSON object containing the predicted label and spam probability. Omit `--text` to read message content from standard input instead.

## Streamlit Demo
Launch the interactive demo after training:
```bash
streamlit run streamlit_app.py
```
Key features of the dashboard:
- Dataset picker with configurable label/text columns (and header toggle).
- Sidebar threshold slider that recalculates precision, recall, and F1 using stored validation predictions.
- Live inference section with spam/ham quick-test buttons, probability bar, and threshold marker.
- Dataset insights highlighting class distribution and top tokens per class.
- Model diagnostics displaying confusion matrix, ROC curve, and PR curve images.

## Exploratory Visualizations
Export token frequency plots and a word cloud for quick dataset intuition:
```bash
python cli.py visualize --data datasets/sms_spam_no_header.csv --output-dir artifacts
```
This command produces `artifacts/top_tokens.png` (bar chart of the top tokens) and `artifacts/wordcloud.png`.

## Project Structure
- `spam_classifier/` - Python package with data prep, training, evaluation, visualization, and inference utilities.
- `datasets/` - Raw and processed SMS spam datasets.
- `artifacts/` - Default output location for trained model assets and reports.
- `streamlit_app.py` - Streamlit interface for interactive predictions.
- `openspec/` - Spec-driven project documentation and change tracking.
- `cli.py` - Convenience wrapper so commands can be run as `python cli.py ...`.

## Known Limitations & Backlog
- Batch upload / CSV export flows are not yet implemented (planned as a future OpenSpec change).
- No automated tests yet; run manual CLI commands to verify behaviour.
- Deployment scripts for Streamlit Cloud should include training or artifact upload before launch.

## OpenSpec
Active changes: `add-spam-classifier`, `add-streamlit-enhancements`
- Specs live under `openspec/changes/<change-id>/specs/`
- Validate with `openspec validate <change-id> --strict`

## Testing Notes
The dataset is imbalanced; the training pipeline uses stratified splitting, enforces a macro F1 >= 0.95, and reports ROC AUC to ensure baseline quality.
