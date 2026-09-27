"""NeuroLens AI: explainable, uncertainty-aware brain MRI classification."""

import io
import json
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, UnidentifiedImageError
from torchvision import models, transforms

try:
    from captum.attr import IntegratedGradients
except ImportError:
    IntegratedGradients = None


st.set_page_config(page_title="NeuroLens AI", page_icon="🧠", layout="wide")

APP_DIR = Path(__file__).resolve().parent
PROJECT_DIR = APP_DIR.parent
MODEL_PATHS = [
    PROJECT_DIR / "neurolens_best.pth",
    PROJECT_DIR / "models" / "neurolens_best.pth",
    PROJECT_DIR / "model" / "neurolens_best.pth",
    PROJECT_DIR / "checkpoints" / "neurolens_best.pth",
    APP_DIR / "neurolens_best.pth",
    APP_DIR / "models" / "neurolens_best.pth",
    APP_DIR / "model" / "neurolens_best.pth",
    APP_DIR / "checkpoints" / "neurolens_best.pth",
]
IMG_SIZE = 224
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
DEFAULT_MC_SAMPLES = 30
DEFAULT_CLASSES = ["glioma", "meningioma", "notumor", "pituitary"]
CLASS_LABELS = {
    "glioma": "Glioma", "meningioma": "Meningioma",
    "notumor": "No Tumor", "no_tumor": "No Tumor", "pituitary": "Pituitary",
}


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, pool=True):
        super().__init__()
        layers = [nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
                  nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True)]
        if pool:
            layers.append(nn.MaxPool2d(2))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class CustomCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(3, 32), ConvBlock(32, 64), ConvBlock(64, 128),
            ConvBlock(128, 256, pool=False), ConvBlock(256, 256), nn.Dropout2d(0.3),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(256, 256), nn.BatchNorm1d(256),
            nn.ReLU(inplace=True), nn.Dropout(0.35), nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.pool(self.features(x)))


def build_model(name, num_classes):
    normalized = str(name).lower().replace("-", "").replace("_", "")
    if "resnet50" in normalized:
        model = models.resnet50(weights=None)
        model.fc = nn.Sequential(nn.Linear(model.fc.in_features, 256),
                                 nn.BatchNorm1d(256), nn.ReLU(inplace=True),
                                 nn.Dropout(0.35), nn.Linear(256, num_classes))
        return model
    if "efficientnet" in normalized:
        model = models.efficientnet_b0(weights=None)
        model.classifier = nn.Sequential(nn.Dropout(0.35),
                                         nn.Linear(model.classifier[1].in_features, num_classes))
        return model
    if "mobilenet" in normalized:
        model = models.mobilenet_v3_small(weights=None)
        model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)
        return model
    return CustomCNN(num_classes)


def find_model_path():
    return next((path for path in MODEL_PATHS if path.is_file()), None)


@st.cache_resource(show_spinner="Loading trained NeuroLens model…")
def load_checkpoint(path_string, modified_time):
    del modified_time  # Included in Streamlit's cache key to reload replaced checkpoints.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        checkpoint = torch.load(path_string, map_location=device, weights_only=False)
    except TypeError:  # Compatibility with older PyTorch versions.
        checkpoint = torch.load(path_string, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state = checkpoint["model_state_dict"]
        class_names = checkpoint.get("class_names", DEFAULT_CLASSES)
        model_name = checkpoint.get("best_model_name", "Custom CNN")
    elif isinstance(checkpoint, dict) and checkpoint and all(torch.is_tensor(v) for v in checkpoint.values()):
        state, class_names, model_name = checkpoint, DEFAULT_CLASSES, "Custom CNN"
    else:
        raise ValueError("Unsupported checkpoint format")
    model = build_model(model_name, len(class_names))
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model, list(class_names), str(model_name), device


TRANSFORM = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)), transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])


def enable_mc_dropout(model):
    for module in model.modules():
        if isinstance(module, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            module.train()


def mc_predict(model, tensor, device, n_samples):
    model.eval()
    enable_mc_dropout(model)
    x = tensor.unsqueeze(0).to(device)
    samples = []
    try:
        with torch.no_grad():
            for _ in range(n_samples):
                samples.append(torch.softmax(model(x), dim=1)[0].cpu().numpy())
    finally:
        model.eval()
    probabilities = np.asarray(samples)
    mean_probs, std_probs = probabilities.mean(axis=0), probabilities.std(axis=0)
    prediction = int(np.argmax(mean_probs))
    entropy = float(-np.sum(mean_probs * np.log(mean_probs + 1e-9)))
    return mean_probs, std_probs, prediction, float(mean_probs[prediction]), entropy


class GradCAMPlusPlus:
    def __init__(self, model, target_layer):
        self.model, self.activations, self.gradients = model, None, None
        self.fwd = target_layer.register_forward_hook(self._save_activation)
        self.bwd = target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module, inputs, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, tensor, class_idx, device):
        self.model.eval()
        x = tensor.unsqueeze(0).to(device)
        output = self.model(x)
        self.model.zero_grad(set_to_none=True)
        output[0, class_idx].backward()
        grads, acts = self.gradients[0], self.activations[0]
        grads2, grads3 = grads.square(), grads.pow(3)
        denominator = 2 * grads2 + (acts * grads3).sum((1, 2), keepdim=True)
        denominator = torch.where(denominator == 0, torch.ones_like(denominator), denominator)
        alpha = grads2 / denominator
        weights = (alpha * F.relu(grads)).sum((1, 2))
        heatmap = F.relu((weights[:, None, None] * acts).sum(0)).cpu().numpy()
        maximum = float(heatmap.max())
        if maximum > 0:
            heatmap /= maximum
        return heatmap

    def close(self):
        self.fwd.remove()
        self.bwd.remove()


def get_last_conv(model):
    layer = next((m for m in reversed(list(model.modules())) if isinstance(m, nn.Conv2d)), None)
    if layer is None:
        raise ValueError("No convolutional layer found")
    return layer


def integrated_gradients_map(model, tensor, class_idx, device):
    if IntegratedGradients is None:
        raise RuntimeError("Integrated Gradients requires Captum. Install it with `pip install captum`.")
    model.eval()
    x = tensor.unsqueeze(0).to(device)
    attribution = IntegratedGradients(model).attribute(
        x, baselines=torch.zeros_like(x), target=class_idx, n_steps=32)
    saliency = attribution.detach().abs().sum(dim=1)[0].cpu().numpy()
    saliency -= saliency.min()
    if saliency.max() > 0:
        saliency /= saliency.max()
    return saliency


def make_overlay(image_np, heatmap, alpha=0.45):
    heat = cv2.resize(heatmap, (image_np.shape[1], image_np.shape[0]))
    colored = cv2.cvtColor(cv2.applyColorMap(np.uint8(np.clip(heat, 0, 1) * 255),
                                            cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    return np.clip((1 - alpha) * image_np.astype(np.float32) + alpha * colored, 0, 255).astype(np.uint8)


def agreement_score(map1, map2, size=(14, 14)):
    h, w = size
    a, b = cv2.resize(map1, (w, h)), cv2.resize(map2, (w, h))
    mask_a, mask_b = a >= np.percentile(a, 75), b >= np.percentile(b, 75)
    union = np.logical_or(mask_a, mask_b).sum()
    return float(np.logical_and(mask_a, mask_b).sum() / union) if union else 0.0


def display_class(name):
    return CLASS_LABELS.get(str(name).lower(), str(name).replace("_", " ").title())


def load_thresholds(uploaded_file):
    if uploaded_file is None:
        return None
    try:
        data = json.loads(uploaded_file.getvalue().decode("utf-8"))
        low, high = float(data["low_thr"]), float(data["high_thr"])
        if not np.isfinite(low) or not np.isfinite(high) or low < 0 or high < low:
            return None
        return low, high
    except (ValueError, TypeError, KeyError, UnicodeDecodeError):
        return None


def uncertainty_label(entropy, thresholds):
    if thresholds is None:
        return "Unavailable", "Unavailable"
    low, high = thresholds
    if entropy <= low:
        return "Low", "High"
    if entropy <= high:
        return "Medium", "Moderate"
    return "High", "Low"


st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
.stApp { background: #f4f7fb; color: #14243a; }
h1,h2,h3 { font-family: 'Manrope', sans-serif; letter-spacing: -0.03em; }
.hero { padding: 1.5rem 1.8rem; border-radius: 22px; color: white;
 background: linear-gradient(120deg,#102b48,#146b7a); margin-bottom: 1.2rem; }
.hero h1 { color:white; margin:0; font-size:2.2rem; }
.hero p { color:#d8edf0; margin:.5rem 0 0; font-size:1.03rem; }
.section-label { color:#257887; font-size:.78rem; font-weight:700; letter-spacing:.12em; text-transform:uppercase; }
[data-testid="stMetric"] { background:white; border:1px solid #e4ebf2; border-radius:16px; padding:1rem 1.1rem; box-shadow:0 3px 12px #1934510a; }
div[data-testid="stFileUploader"] { background:white; border:1px dashed #79aab4; border-radius:18px; padding:1rem; }
.disclaimer { background:#fff8e7; border-left:5px solid #e6a523; border-radius:12px; padding:1rem 1.2rem; color:#583f12; }
</style>""", unsafe_allow_html=True)

st.markdown("""<div class="hero"><h1>🧠 NeuroLens AI</h1>
<p>Explainable &amp; Uncertainty-Aware Brain Tumor MRI Classification</p>
<p>NeuroLens AI analyzes brain MRI images using deep learning and provides AI-assisted predictions together with confidence, uncertainty and visual explanations.</p></div>""", unsafe_allow_html=True)

model_path = find_model_path()
model = class_names = model_name = device = None
load_error = False
if model_path is not None:
    try:
        model, class_names, model_name, device = load_checkpoint(
            str(model_path), model_path.stat().st_mtime_ns)
    except Exception:
        load_error = True

with st.sidebar:
    st.header("🧠 AI Model")
    if model is not None:
        st.success("✅ AI Model Loaded")
        st.markdown(f"**Status:** ✅ Loaded  \n**Model:** {model_name}  \n**Device:** {str(device).upper()}  \n**Classes:** {len(class_names)}")
    elif load_error:
        st.error("❌ Unable to load the trained model. Verify that neurolens_best.pth matches the expected model architecture.")
    else:
        st.warning("⚠️ Trained model not found.\n\nPlease place neurolens_best.pth inside the project directory.")
    st.divider()
    mc_samples = st.slider("MC Dropout Samples", 5, 50, DEFAULT_MC_SAMPLES, 5)
    run_xai = st.checkbox("Generate XAI explanations", value=True)
    st.divider()
    threshold_file = st.file_uploader("Optional: uncertainty_thresholds.json", type=["json"],
                                      help="Optional validation calibration thresholds (low_thr and high_thr).")
    st.caption("MRI images are analyzed locally for this session and are not permanently stored.")

if model is None:
    if load_error:
        st.error("❌ Unable to load the trained model.\n\nPlease verify that neurolens_best.pth matches the expected model architecture.")
    else:
        st.warning("⚠️ AI model not found.\n\nExpected: `neurolens_best.pth`\n\nPlease place the trained model in the project directory, `models/`, `model/`, or `checkpoints/` folder.")
    st.stop()

st.markdown('<div class="section-label">MRI analysis</div>', unsafe_allow_html=True)
st.subheader("📤 Upload Brain MRI")
uploaded = st.file_uploader("Choose a brain MRI image", type=["jpg", "jpeg", "png", "bmp", "webp"],
                            label_visibility="collapsed")
if uploaded is None:
    st.info("Upload an MRI image to start the analysis. The prediction runs automatically after upload.")
    st.stop()

try:
    image = Image.open(io.BytesIO(uploaded.getvalue())).convert("RGB")
    tensor = TRANSFORM(image)
except (UnidentifiedImageError, OSError, ValueError):
    st.error("❌ Unable to process this image. Please upload a valid MRI image.")
    st.stop()

st.caption(f"Image dimensions: {image.width} × {image.height} pixels")
with st.spinner("Analyzing MRI with the trained model…"):
    try:
        mean_probs, std_probs, pred_idx, confidence, entropy = mc_predict(
            model, tensor, device, mc_samples)
    except Exception:
        st.error("❌ Unable to analyze this image with the loaded model. Please check that the checkpoint and class configuration are valid.")
        st.stop()
predicted_class = display_class(class_names[pred_idx])
thresholds = load_thresholds(threshold_file)
level, reliability = uncertainty_label(entropy, thresholds)

st.divider()
st.markdown('<div class="section-label">Model output</div>', unsafe_allow_html=True)
st.header("🧠 AI Prediction")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Prediction", predicted_class)
m2.metric("Confidence", f"{confidence * 100:.2f}%")
m3.metric("Predictive Entropy", f"{entropy:.4f}")
m4.metric("MC Samples", mc_samples)

st.header("📊 Class Probability Analysis")
prob_col, image_col = st.columns([1.15, 0.85])
with prob_col:
    probability_rows = [{"Class": display_class(name),
                         "Mean Probability (%)": round(float(mean_probs[i] * 100), 2),
                         "Standard Deviation (%)": round(float(std_probs[i] * 100), 2)}
                        for i, name in enumerate(class_names)]
    st.bar_chart({row["Class"]: row["Mean Probability (%)"] for row in probability_rows},
                 y_label="Mean probability (%)")
    st.dataframe(probability_rows, use_container_width=True, hide_index=True)
with image_col:
    st.image(image, caption="Original MRI", use_container_width=True)

st.header("🔍 Uncertainty Analysis")
u1, u2, u3, u4 = st.columns(4)
u1.metric("Predictive Entropy", f"{entropy:.4f}")
u2.metric("Uncertainty Level", level)
u3.metric("Reliability", reliability)
u4.metric("MC Dropout Samples", mc_samples)
if thresholds is None:
    st.info("Calibrated uncertainty level is unavailable because validation thresholds have not been provided.")
    if threshold_file is not None:
        st.caption("Threshold file is invalid. It must contain numeric low_thr and high_thr values, with high_thr ≥ low_thr.")
else:
    st.caption(f"Validation thresholds: low = {thresholds[0]:.4f}, high = {thresholds[1]:.4f}")

eas = None
xai_error = None
if run_xai:
    st.divider()
    st.header("🔬 Explainable AI")
    progress = st.progress(0, text="Generating Grad-CAM++ explanation…")
    cam = None
    try:
        cam = GradCAMPlusPlus(model, get_last_conv(model))
        gcam = cam.generate(tensor, pred_idx, device)
        cam.close()
        cam = None
        progress.progress(50, text="Generating Integrated Gradients explanation…")
        ig = integrated_gradients_map(model, tensor, pred_idx, device)
        original_np = np.asarray(image)
        gcam_overlay = make_overlay(original_np, gcam)
        ig_overlay = make_overlay(original_np, ig)
        eas = agreement_score(gcam, ig)
        progress.progress(100, text="Explanations ready")
        x1, x2, x3 = st.columns(3)
        x1.image(original_np, caption="Original MRI", use_container_width=True)
        x2.image(gcam_overlay, caption="Grad-CAM++", use_container_width=True)
        x3.image(ig_overlay, caption="Integrated Gradients", use_container_width=True)
        d1, d2, d3 = st.columns(3)
        d1.caption("Reference image used for both explanations.")
        d2.caption("Highlights image regions that contributed to the predicted class.")
        d3.caption("Shows input features attributed to the predicted class using a zero baseline.")
        st.metric("Explanation Agreement Score (EAS)", f"{eas:.4f}")
        st.caption("EAS measures the spatial agreement between the important regions identified by Grad-CAM++ and Integrated Gradients, using IoU of their top 25% activated regions.")
    except Exception as exc:
        xai_error = str(exc)
        if cam is not None:
            cam.close()
        progress.empty()
        if IntegratedGradients is None:
            st.info("Integrated Gradients is unavailable because Captum is not installed. Install the `captum` package to enable it.")
        else:
            st.warning("XAI explanations could not be generated for this image. The AI prediction and uncertainty results are still available.")
        with st.expander("XAI details"):
            st.write(xai_error)

st.divider()
st.header("📋 Analysis Summary")
summary = [
    {"Measure": "MRI Image", "Result": "Analyzed"},
    {"Measure": "AI Prediction", "Result": predicted_class},
    {"Measure": "Confidence", "Result": f"{confidence * 100:.2f}%"},
    {"Measure": "Predictive Entropy", "Result": f"{entropy:.4f}"},
    {"Measure": "Uncertainty", "Result": level},
    {"Measure": "Reliability", "Result": reliability},
    {"Measure": "Explanation Agreement Score", "Result": f"{eas:.4f}" if eas is not None else ("Unavailable" if run_xai else "Not generated")},
]
st.dataframe(summary, use_container_width=True, hide_index=True)
st.markdown("<div class='disclaimer'><strong>⚠️ Medical Disclaimer</strong><br>NeuroLens AI is an AI-assisted research and educational tool. It is not a certified medical diagnostic system and must not replace assessment, diagnosis, or treatment by qualified healthcare professionals.</div>", unsafe_allow_html=True)
