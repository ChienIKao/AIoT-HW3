from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

from spam_classifier.data import encode_labels, prepare_dataframe
from spam_classifier.service import (
    load_artifacts,
    load_metrics,
    load_validation_predictions,
    predict_messages,
)
from spam_classifier.visualization import (
    class_distribution,
    probability_bar,
    top_tokens_by_class,
)

ARTIFACTS_DIR = Path("artifacts")
DATASETS_DIR = Path("datasets")
DEFAULT_SPAM_EXAMPLE = "Congratulations! You have won a free cruise. Reply NOW to claim your prize!"
DEFAULT_HAM_EXAMPLE = "Hi there, just checking if we're still meeting for lunch tomorrow."


@st.cache_data(show_spinner=False)
def list_dataset_files(directory: Path) -> List[Path]:
    return sorted([p for p in directory.glob("*.csv") if p.is_file()], key=lambda p: p.name)


@st.cache_data(show_spinner=False)
def read_dataset(path_str: str, has_header: bool) -> pd.DataFrame:
    path = Path(path_str)
    if has_header:
        df = pd.read_csv(path)
    else:
        df = pd.read_csv(path, header=None)
        df.columns = [f"column_{i}" for i in range(df.shape[1])]
    return df


def compute_threshold_metrics(labels: List[str], probabilities: List[float], threshold: float) -> dict[str, float]:
    y_true = np.array(encode_labels(labels))
    probs = np.array(probabilities)
    preds = (probs >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, preds, average="binary", zero_division=0
    )
    accuracy = accuracy_score(y_true, preds)
    roc_auc = roc_auc_score(y_true, probs) if len(np.unique(y_true)) > 1 else 0.5
    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": float(roc_auc),
    }


st.set_page_config(page_title="Spam Email Classifier", page_icon="📧", layout="wide")
st.title("Spam Email Classifier")
st.write(
    "Provide an email or SMS message below to predict whether it is spam or ham using the"
    " trained TF-IDF + logistic regression model."
)

artifacts_ready = True
try:
    load_artifacts(ARTIFACTS_DIR)
except FileNotFoundError:
    artifacts_ready = False

metrics = load_metrics(ARTIFACTS_DIR) if artifacts_ready else None
validation_predictions = load_validation_predictions(ARTIFACTS_DIR) if artifacts_ready else None

st.sidebar.header("Configuration")
dataset_files = list_dataset_files(DATASETS_DIR)
selected_dataset = None
if dataset_files:
    dataset_display = [p.name for p in dataset_files]
    dataset_choice = st.sidebar.selectbox("Dataset file", options=dataset_display, index=0)
    selected_dataset = next((p for p in dataset_files if p.name == dataset_choice), None)
else:
    st.sidebar.warning("No CSV datasets found under the datasets/ directory.")

has_header = st.sidebar.checkbox("Dataset has header row", value=False)
label_column = ""
text_column = ""
prepared_dataset = None
if selected_dataset is not None:
    try:
        raw_df = read_dataset(str(selected_dataset), has_header)
        column_options = list(raw_df.columns)
        default_label_idx = 0
        default_text_idx = 1 if len(column_options) > 1 else 0
        label_column = st.sidebar.selectbox("Label column", column_options, index=default_label_idx)
        text_column = st.sidebar.selectbox("Text column", column_options, index=default_text_idx)
        prepared_dataset = prepare_dataframe(raw_df, label_column=label_column, text_column=text_column)
    except Exception as exc:  # pylint: disable=broad-except
        st.sidebar.error(f"Failed to load dataset: {exc}")
        prepared_dataset = None

threshold_default = 0.5
if validation_predictions:
    threshold_default = float(validation_predictions.get("threshold", 0.5))
threshold = st.sidebar.slider("Decision threshold", min_value=0.10, max_value=0.90, value=threshold_default, step=0.01)

if not artifacts_ready:
    st.error(
        "Model artifacts were not found. Run `python cli.py train` before launching the demo."
    )

st.sidebar.subheader("Validation Metrics")
if validation_predictions and artifacts_ready:
    dynamic_metrics = compute_threshold_metrics(
        validation_predictions["labels"],
        validation_predictions["probabilities"],
        threshold,
    )
else:
    dynamic_metrics = metrics or {}

for metric_name in ("accuracy", "precision", "recall", "f1", "roc_auc"):
    value = dynamic_metrics.get(metric_name)
    if value is not None:
        st.sidebar.metric(metric_name.upper(), f"{value:.3f}")
    elif metrics and metric_name in metrics:
        st.sidebar.metric(metric_name.upper(), f"{metrics[metric_name]:.3f}")
    else:
        st.sidebar.metric(metric_name.upper(), "-")

st.sidebar.markdown("---")
st.sidebar.caption("Threshold slider recalculates precision/recall/F1 using validation predictions.")

st.header("Live Inference")
if "message_input" not in st.session_state:
    st.session_state.message_input = ""
if "auto_classify" not in st.session_state:
    st.session_state.auto_classify = False

quick_cols = st.columns(2)
if quick_cols[0].button("Use spam example"):
    st.session_state.message_input = DEFAULT_SPAM_EXAMPLE
    st.session_state.auto_classify = True
if quick_cols[1].button("Use ham example"):
    st.session_state.message_input = DEFAULT_HAM_EXAMPLE
    st.session_state.auto_classify = True

text = st.text_area("Email content", height=180, placeholder="Paste an email or SMS message...", key="message_input")
classify_clicked = st.button("Classify") or st.session_state.auto_classify

if classify_clicked:
    st.session_state.auto_classify = False
    if not artifacts_ready:
        st.error("Artifacts missing. Train the model first.")
    elif not text.strip():
        st.warning("Please provide some text to classify.")
    else:
        result = predict_messages([text], artifacts_dir=ARTIFACTS_DIR)[0]
        label_display = "Spam" if result["label"] == "spam" else "Ham"
        probability = float(result["spam_probability"])
        st.subheader(f"Prediction: {label_display}")
        st.write(f"Spam probability: {probability * 100:.2f}% (threshold {threshold:.2f})")
        fig = probability_bar(probability, threshold)
        st.pyplot(fig, clear_figure=True)
        st.caption("Spam probability above the threshold is classified as spam.")

st.markdown("---")

if prepared_dataset is not None:
    st.header("Dataset Insights")
    class_counts = class_distribution(prepared_dataset)
    st.subheader("Class Distribution")
    st.bar_chart(class_counts)

    token_map = top_tokens_by_class(prepared_dataset, top_n=10)
    st.subheader("Top Tokens by Class")
    token_columns = st.columns(max(len(token_map), 1))
    for idx, (label, tokens) in enumerate(token_map.items()):
        df_tokens = pd.DataFrame(tokens, columns=["Token", "Count"])
        token_columns[idx].table(df_tokens)
else:
    st.info("Select a dataset and columns in the sidebar to view dataset insights.")

st.markdown("---")

st.header("Model Diagnostics")
plot_cols = st.columns(3)
plot_paths = [
    (ARTIFACTS_DIR / "confusion_matrix.png", "Confusion Matrix"),
    (ARTIFACTS_DIR / "roc_curve.png", "ROC Curve"),
    (ARTIFACTS_DIR / "pr_curve.png", "PR Curve"),
]
for col, (plot_path, title) in zip(plot_cols, plot_paths):
    if plot_path.exists():
        col.subheader(title)
        col.image(str(plot_path), width="stretch")
    else:
        col.warning(f"Missing {title.lower()}, retrain to regenerate.")

st.caption("Validation plots refresh when you retrain the model.")
