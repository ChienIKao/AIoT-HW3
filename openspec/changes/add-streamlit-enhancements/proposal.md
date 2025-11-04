## Why
- Deliver the advanced Streamlit experience described in the assignment brief (dataset pickers, interactive metrics, visual diagnostics).
- Surface richer evaluation outputs (PR curve, threshold exploration) by extending the training pipeline and artifacts.
- Improve usability with quick test examples and visual probability feedback.

## What Changes
- Update training/evaluation workflows to persist precision-recall curves and validation predictions for dynamic thresholding.
- Enhance Streamlit app with dataset/column selectors, class distribution charts, top tokens by class, confusion/ROC/PR plots, threshold slider, probability bar, and quick test buttons.
- Add any supporting utilities required for per-class token frequencies and storing validation outputs.

## Impact
- Additional dependencies: none beyond existing stack (matplotlib/wordcloud already present).
- Streamlit session will load extra data (validation predictions) for interactive metrics.
- Artifacts directory gains `pr_curve.png` and `validation_predictions.json` (naming to decide in implementation).
