# NeuroLens Streamlit App

This Streamlit application is built from the uploaded `Brain Tumor Detection(2).ipynb`.

## What it does

- Loads the notebook's saved `neurolens_best.pth` checkpoint.
- Automatically detects the saved `best_model_name`.
- Supports:
  - Custom CNN
  - ResNet50
  - EfficientNet-B0
  - MobileNetV3-Small
- Preprocesses MRI images at 224×224 using the notebook's ImageNet mean/std.
- Predicts one of:
  - glioma
  - meningioma
  - notumor
  - pituitary
- Runs MC Dropout uncertainty estimation.
- Shows class probabilities and standard deviation.
- Generates:
  - Grad-CAM++
  - Integrated Gradients
  - Explanation Agreement Score (EAS)
- Includes the notebook's medical/research disclaimer.

## 1. Create the model artifact

Run the notebook through the point where it saves:

`neurolens_best.pth`

The notebook already contains the save operation in Phase 3.

Optional: after the notebook calculates `low_thr` and `high_thr`, save them as:

```python
import json

with open("uncertainty_thresholds.json", "w") as f:
    json.dump({
        "low_thr": float(low_thr),
        "high_thr": float(high_thr)
    }, f)
```

## 2. Install dependencies

```bash
pip install -r requirements.txt
```

## 3. Run

```bash
streamlit run app.py
```

Then upload:

1. `neurolens_best.pth`
2. An MRI image

The uncertainty-threshold JSON is optional.

## 4. Deployment

For Streamlit Community Cloud:

- Push `app.py` and `requirements.txt` to GitHub.
- Deploy the repository.
- Upload the `.pth` checkpoint through the application UI.

For a private/production deployment, store the checkpoint as a protected server-side artifact instead of asking users to upload it.

## Important

This is a research/educational application. It is not a certified medical diagnostic device and should not be used as a substitute for professional medical assessment.
