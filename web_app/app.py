
import io
import json
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms

try:
    from captum.attr import IntegratedGradients
except Exception:
    IntegratedGradients = None


st.set_page_config(
    page_title="NeuroLens AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

IMG_SIZE = 224
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
DEFAULT_MC_SAMPLES = 30

CLASS_INFO = {
    "glioma": "Glioma",
    "meningioma": "Meningioma",
    "notumor": "No Tumor",
    "pituitary": "Pituitary",
}


# -----------------------------
# Model definitions
# -----------------------------
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, pool=True):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        ]
        if pool:
            layers.append(nn.MaxPool2d(2))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class CustomCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(3, 32, pool=True),
            ConvBlock(32, 64, pool=True),
            ConvBlock(64, 128, pool=True),
            ConvBlock(128, 256, pool=False),
            ConvBlock(256, 256, pool=True),
            nn.Dropout2d(0.3),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.35),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.pool(self.features(x)))


def build_model(name, num_classes):
    name = name.lower()

    if "resnet50" in name:
        model = models.resnet50(weights=None)
        model.fc = nn.Sequential(
            nn.Linear(model.fc.in_features, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.35),
            nn.Linear(256, num_classes),
        )
        return model

    if "efficientnet" in name:
        model = models.efficientnet_b0(weights=None)
        model.classifier = nn.Sequential(
            nn.Dropout(0.35),
            nn.Linear(model.classifier[1].in_features, num_classes),
        )
        return model

    if "mobilenet" in name:
        model = models.mobilenet_v3_small(weights=None)
        model.classifier[3] = nn.Linear(
            model.classifier[3].in_features, num_classes
        )
        return model

    return CustomCNN(num_classes)


# -----------------------------
# Loading
# -----------------------------
@st.cache_resource(show_spinner=False)
def load_checkpoint(file_bytes):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(io.BytesIO(file_bytes), map_location=device)

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state = checkpoint["model_state_dict"]
        class_names = checkpoint.get(
            "class_names",
            ["glioma", "meningioma", "notumor", "pituitary"],
        )
        model_name = checkpoint.get("best_model_name", "Custom CNN")
    else:
        state = checkpoint
        class_names = ["glioma", "meningioma", "notumor", "pituitary"]
        model_name = "Custom CNN"

    model = build_model(model_name, len(class_names))
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()

    return model, class_names, model_name, device


transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


def preprocess(image):
    rgb = image.convert("RGB")
    return transform(rgb)


# -----------------------------
# MC Dropout uncertainty
# -----------------------------
def enable_mc_dropout(model):
    for module in model.modules():
        if isinstance(module, (nn.Dropout, nn.Dropout2d)):
            module.train()
    return model


def mc_predict(model, tensor, device, n_samples=30):
    model.eval()
    enable_mc_dropout(model)

    x = tensor.unsqueeze(0).to(device)
    samples = []

    with torch.no_grad():
        for _ in range(n_samples):
            probs = torch.softmax(model(x), dim=1)[0].cpu().numpy()
            samples.append(probs)

    arr = np.asarray(samples)
    mean_probs = arr.mean(axis=0)
    std_probs = arr.std(axis=0)
    pred_idx = int(np.argmax(mean_probs))
    confidence = float(mean_probs[pred_idx])
    entropy = float(-np.sum(mean_probs * np.log(mean_probs + 1e-9)))

    model.eval()
    return mean_probs, std_probs, pred_idx, confidence, entropy


# -----------------------------
# Grad-CAM++
# -----------------------------
class GradCAMPlusPlus:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None

        self.fwd_handle = target_layer.register_forward_hook(
            self._save_activation
        )
        self.bwd_handle = target_layer.register_full_backward_hook(
            self._save_gradient
        )

    def _save_activation(self, module, inputs, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, tensor, class_idx):
        self.model.eval()
        x = tensor.unsqueeze(0).to(next(self.model.parameters()).device)
        x.requires_grad_(True)

        output = self.model(x)
        self.model.zero_grad(set_to_none=True)

        one_hot = torch.zeros_like(output)
        one_hot[0, class_idx] = 1.0
        output.backward(gradient=one_hot)

        grads = self.gradients.squeeze(0)
        acts = self.activations.squeeze(0)

        grads_2 = grads ** 2
        grads_3 = grads ** 3
        denom = 2.0 * grads_2 + (
            acts * grads_3
        ).sum(dim=(1, 2), keepdim=True)
        denom = torch.where(denom == 0, torch.ones_like(denom), denom)

        alpha = grads_2 / denom
        weights = (alpha * F.relu(grads)).sum(dim=(1, 2))

        heatmap = (weights[:, None, None] * acts).sum(0)
        heatmap = F.relu(heatmap).cpu().numpy()

        if heatmap.max() > 0:
            heatmap /= heatmap.max()

        return heatmap

    def close(self):
        self.fwd_handle.remove()
        self.bwd_handle.remove()


def get_last_conv(model):
    last = None
    for module in model.modules():
        if isinstance(module, nn.Conv2d):
            last = module
    if last is None:
        raise ValueError("No Conv2d layer found in model.")
    return last


def make_overlay(image_np, heatmap, alpha=0.45):
    heat = cv2.resize(
        heatmap, (image_np.shape[1], image_np.shape[0])
    )
    colored = cv2.applyColorMap(
        (heat * 255).astype(np.uint8),
        cv2.COLORMAP_JET,
    )
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    overlay = (
        (1 - alpha) * image_np.astype(np.float32)
        + alpha * colored.astype(np.float32)
    )
    return np.clip(overlay, 0, 255).astype(np.uint8)


# -----------------------------
# Integrated Gradients
# -----------------------------
def integrated_gradients_map(model, tensor, class_idx, device):
    if IntegratedGradients is None:
        raise RuntimeError(
            "Captum is not installed. Install it with: pip install captum"
        )

    model.eval()
    x = tensor.unsqueeze(0).to(device)
    x.requires_grad_(True)

    ig = IntegratedGradients(model)
    baseline = torch.zeros_like(x)

    attr = ig.attribute(
        x,
        baselines=baseline,
        target=class_idx,
        n_steps=32,
    )

    # Sum absolute attribution over RGB channels
    saliency = attr.detach().abs().sum(dim=1)[0].cpu().numpy()
    saliency -= saliency.min()

    if saliency.max() > 0:
        saliency /= saliency.max()

    return saliency


def explanation_agreement_score(map1, map2, resize_to=(14, 14)):
    h, w = resize_to
    m1 = cv2.resize(map1, (w, h))
    m2 = cv2.resize(map2, (w, h))

    t1 = np.percentile(m1, 75)
    t2 = np.percentile(m2, 75)

    b1 = (m1 >= t1).astype(np.uint8)
    b2 = (m2 >= t2).astype(np.uint8)

    intersection = (b1 & b2).sum()
    union = (b1 | b2).sum()

    return float(intersection / union) if union > 0 else 0.0


# -----------------------------
# UI
# -----------------------------
st.title("🧠 NeuroLens")
st.caption(
    "Lightweight, Explainable and Uncertainty-Aware Brain Tumor MRI "
    "Classification"
)

with st.sidebar:
    st.header("Model Setup")

    model_file = st.file_uploader(
        "Upload trained model (.pth)",
        type=["pth", "pt"],
        help="Use the neurolens_best.pth checkpoint generated by the notebook.",
    )

    st.divider()

    mc_samples = st.slider(
        "MC Dropout samples",
        min_value=5,
        max_value=50,
        value=DEFAULT_MC_SAMPLES,
        step=5,
    )

    run_xai = st.checkbox(
        "Generate XAI explanations",
        value=True,
    )

    st.divider()
    st.info(
        "Expected checkpoint: neurolens_best.pth\n\n"
        "The checkpoint should contain model_state_dict, class_names "
        "and best_model_name as produced by the notebook."
    )

if model_file is None:
    st.warning("Upload `neurolens_best.pth` from the notebook to start prediction.")
    st.markdown(
        """
### Supported classes
- Glioma
- Meningioma
- No Tumor
- Pituitary

### Pipeline
**MRI → Resize 224×224 → ImageNet normalization → trained model → "
**prediction + MC Dropout uncertainty + XAI**
        """
    )
    st.stop()

try:
    model, class_names, model_name, device = load_checkpoint(
        model_file.getvalue()
    )
except Exception as exc:
    st.error(f"Could not load the checkpoint: {exc}")
    st.stop()

st.success(f"Loaded model: **{model_name}**  |  Device: `{device}`")

uploaded = st.file_uploader(
    "Upload an MRI image",
    type=["jpg", "jpeg", "png", "bmp", "webp"],
)

if uploaded is None:
    st.info("Upload an MRI image above.")
    st.stop()

image = Image.open(uploaded).convert("RGB")
tensor = preprocess(image)

with st.spinner("Running prediction..."):
    mean_probs, std_probs, pred_idx, confidence, entropy = mc_predict(
        model, tensor, device, mc_samples
    )

pred_class = class_names[pred_idx]

# Top result
st.divider()
c1, c2, c3, c4 = st.columns(4)
c1.metric("Prediction", CLASS_INFO.get(pred_class, pred_class.title()))
c2.metric("Confidence", f"{confidence * 100:.2f}%")
c3.metric("Entropy", f"{entropy:.4f}")
c4.metric("MC Samples", mc_samples)

left, right = st.columns([1, 1])

with left:
    st.subheader("Input MRI")
    st.image(image, use_container_width=True)

with right:
    st.subheader("Class probabilities")
    probability_data = {
        CLASS_INFO.get(name, name.title()): float(mean_probs[i] * 100)
        for i, name in enumerate(class_names)
    }
    st.bar_chart(probability_data)

    rows = []
    for i, name in enumerate(class_names):
        rows.append({
            "Class": CLASS_INFO.get(name, name.title()),
            "Mean probability (%)": round(float(mean_probs[i] * 100), 2),
            "Std (%)": round(float(std_probs[i] * 100), 2),
        })
    st.dataframe(rows, use_container_width=True, hide_index=True)

# Uncertainty
st.subheader("Uncertainty assessment")
st.caption(
    "Predictive entropy is reported exactly from the notebook's MC Dropout "
    "approach. A calibrated Low/Medium/High label requires the validation "
    "entropy thresholds from the training run."
)

threshold_file = st.file_uploader(
    "Optional: upload uncertainty_thresholds.json",
    type=["json"],
    help="Optional file containing low_thr and high_thr from notebook calibration.",
)

if threshold_file is not None:
    try:
        thresholds = json.loads(threshold_file.getvalue().decode("utf-8"))
        low_thr = float(thresholds["low_thr"])
        high_thr = float(thresholds["high_thr"])

        if entropy <= low_thr:
            level, reliability = "Low", "High"
        elif entropy <= high_thr:
            level, reliability = "Medium", "Moderate"
        else:
            level, reliability = "High", "Low"

        u1, u2, u3 = st.columns(3)
        u1.metric("Uncertainty", level)
        u2.metric("Reliability", reliability)
        u3.metric("Thresholds", f"{low_thr:.3f} / {high_thr:.3f}")
    except Exception:
        st.error("Invalid uncertainty_thresholds.json format.")

# XAI
if run_xai:
    st.divider()
    st.subheader("Explainable AI")

    try:
        target_layer = get_last_conv(model)
        cam = GradCAMPlusPlus(model, target_layer)
        gcam = cam.generate(tensor, pred_idx)
        cam.close()

        ig = integrated_gradients_map(model, tensor, pred_idx, device)

        original_np = np.asarray(image)
        gcam_overlay = make_overlay(original_np, gcam)
        ig_overlay = make_overlay(original_np, ig)

        eas = explanation_agreement_score(gcam, ig)

        x1, x2, x3 = st.columns(3)
        with x1:
            st.image(
                original_np,
                caption="Original MRI",
                use_container_width=True,
            )
        with x2:
            st.image(
                gcam_overlay,
                caption="Grad-CAM++",
                use_container_width=True,
            )
        with x3:
            st.image(
                ig_overlay,
                caption="Integrated Gradients",
                use_container_width=True,
            )

        st.metric("Explanation Agreement Score (EAS)", f"{eas:.4f}")
        st.caption(
            "EAS is the IoU between the top-25% activated regions of the "
            "two explanation maps, following the notebook implementation."
        )

    except Exception as exc:
        st.error(f"XAI generation failed: {exc}")

st.divider()
st.warning(
    "Medical disclaimer: NeuroLens is an AI-assisted research/educational "
    "tool. It is not a certified medical diagnostic system and must not "
    "replace assessment by qualified healthcare professionals."
)
