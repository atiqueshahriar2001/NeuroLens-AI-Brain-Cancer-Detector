# NeuroLens AI

**Explainable and Uncertainty-Aware Brain Tumor MRI Classification**

NeuroLens AI is a local Streamlit research and education dashboard. It runs a trained
PyTorch classifier on a brain MRI and presents an AI prediction, MC-Dropout uncertainty,
Grad-CAM++ and Integrated Gradients explanations, plus an Explanation Agreement Score (EAS).

> ⚠️ It does **not** provide a confirmed medical diagnosis.

## Features

- Automatic checkpoint discovery and architecture auto-selection for
  **CustomCNN, ResNet50, EfficientNet-B0, and MobileNetV3-Small**.
- Four-class probabilities (`glioma`, `meningioma`, `notumor`, `pituitary`).
- **Monte Carlo Dropout** inference → per-class mean & std, plus predictive entropy.
- **Grad-CAM++** heatmaps for the predicted class.
- **Integrated Gradients** (Captum) attribution maps.
- **Explanation Agreement Score (EAS)** = IoU of top-25% activated regions.
- CPU and CUDA support, cached model loading, graceful error handling.
- 100% local processing — images are never written to disk by the app.

## Supported classes

`glioma`, `meningioma`, `notumor`, `pituitary` — read from the checkpoint's
`class_names` field when present.

## Model checkpoint

The app never asks you to upload a model. Place the trained checkpoint
`neurolens_best.pth` in one of these locations (first match wins):
