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
This command prints accuracy, precision, recall, F1, and ROC AUC, and persists `vectorizer.pkl`, `model.pkl`, `metrics.json`, `confusion_matrix.png`, and `roc_curve.png` inside the specified artifacts directory.

## Evaluating the Model
Score a labeled dataset using the saved artifacts:
```bash
python cli.py eval --data datasets/sms_spam_no_header.csv --artifacts-dir artifacts
```
The evaluation exports `eval_metrics.json`, `eval_confusion_matrix.png`, and `eval_roc_curve.png` alongside the printed metrics.

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
The app loads persisted artifacts, lets you paste email text, visualizes the predicted label and probability, and shows the latest validation metrics in the sidebar.

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
- Streamlit demo currently links to generated plots via CLI; embedding token charts or confidence histograms remains TODO.
- No automated tests yet; run manual CLI commands to verify behaviour.
- Deployment scripts for Streamlit Cloud should include training or artifact upload before launch.

## OpenSpec
Active change: `add-spam-classifier`
- Specs live under `openspec/changes/add-spam-classifier/specs/`
- Validate with `openspec validate add-spam-classifier --strict`

## Testing Notes
The dataset is imbalanced; the training pipeline uses stratified splitting, enforces a macro F1 >= 0.95, and reports ROC AUC to ensure baseline quality.
