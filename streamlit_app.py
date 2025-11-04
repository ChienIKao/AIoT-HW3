from __future__ import annotations

from pathlib import Path

import streamlit as st

from spam_classifier.service import load_artifacts, load_metrics, predict_messages

ARTIFACTS_DIR = Path("artifacts")

st.set_page_config(page_title="Spam Email Classifier", page_icon="📧", layout="centered")
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

metrics = None
if artifacts_ready:
    metrics = load_metrics(ARTIFACTS_DIR)
    with st.sidebar:
        st.header("Validation Metrics")
        if metrics:
            for key in ("accuracy", "precision", "recall", "f1", "roc_auc"):
                value = metrics.get(key)
                if value is not None:
                    st.metric(label=key.replace("_", " ").title(), value=f"{value:.3f}")
        else:
            st.info("Metrics not found. Re-run training to refresh this section.")
else:
    st.error(
        "Model artifacts were not found. Run `python cli.py train` before"
        " launching the demo."
    )

text = st.text_area("Email content", height=200, placeholder="Paste an email or SMS message...")
if st.button("Classify"):
    if not artifacts_ready:
        st.error("Artifacts missing. Train the model first.")
    elif not text.strip():
        st.warning("Please provide some text to classify.")
    else:
        result = predict_messages([text], artifacts_dir=ARTIFACTS_DIR)[0]
        label_display = "Spam" if result["label"] == "spam" else "Ham"
        probability = result["spam_probability"] * 100
        if result["label"] == "spam":
            st.error(f"Prediction: {label_display} ({probability:.2f}% confidence)")
        else:
            st.success(f"Prediction: {label_display} ({probability:.2f}% confidence)")
        st.caption("Spam probability above 50% is considered spam.")
