"""
NeuroLens AI
Explainable, Uncertainty-Aware Brain Tumor MRI Classification (Streamlit app)
"""

import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image, UnidentifiedImageError

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import (
    resnet50,
    efficientnet_b0,
    mobilenet_v3_small,
)

warnings.filterwarnings("ignore")

try:
    from captum.attr import IntegratedGradients
    CAPTUM_AVAILABLE = True
except ImportError:
    CAPTUM_AVAILABLE = False


# =====================================================================
# CONFIG
# =====================================================================
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]
IMG_SIZE = 224
MC_SAMPLES = 30
MAX_HISTORY = 100
CHECKPOINT_NAME = "neurolens_best.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

st.set_page_config(
    page_title="NeuroLens AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =====================================================================
# INLINE SVG ICONS (Lucide-style, stroke-based)
# =====================================================================
_SVG_PATHS = {
    "brain": (
        '<path d="M12 5a3 3 0 1 0-5.997.125 4 4 0 0 0-2.526 5.77 '
        '4 4 0 0 0 .556 6.588A4 4 0 1 0 12 18Z"/>'
        '<path d="M12 5a3 3 0 1 1 5.997.125 4 4 0 0 1 2.526 5.77 '
        '4 4 0 0 1-.556 6.588A4 4 0 1 1 12 18Z"/>'
    ),
    "microscope": (
        '<path d="M6 18h8"/><path d="M3 22h18"/>'
        '<path d="M14 22a7 7 0 1 0 0-14h-1"/>'
        '<path d="M9 14h2"/>'
        '<path d="M9 12a2 2 0 0 1-2-2V6h6v4a2 2 0 0 1-2 2Z"/>'
        '<path d="M12 6V3a1 1 0 0 0-1-1H9a1 1 0 0 0-1 1v3"/>'
    ),
    "history": (
        '<path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8"/>'
        '<path d="M3 3v5h5"/><path d="M12 7v5l4 2"/>'
    ),
    "settings": (
        '<path d="M12.22 2h-.44a2 2 0 0 0-2 2v.18a2 2 0 0 1-1 1.73l-.43.25'
        'a2 2 0 0 1-2 0l-.15-.08a2 2 0 0 0-2.73.73l-.22.38a2 2 0 0 0 .73 2.73'
        'l.15.1a2 2 0 0 1 1 1.72v.51a2 2 0 0 1-1 1.74l-.15.09a2 2 0 0 0-.73 2.73'
        'l.22.38a2 2 0 0 0 2.73.73l.15-.08a2 2 0 0 1 2 0l.43.25a2 2 0 0 1 1 1.73'
        'V20a2 2 0 0 0 2 2h.44a2 2 0 0 0 2-2v-.18a2 2 0 0 1 1-1.73l.43-.25'
        'a2 2 0 0 1 2 0l.15.08a2 2 0 0 0 2.73-.73l.22-.39a2 2 0 0 0-.73-2.73'
        'l-.15-.08a2 2 0 0 1-1-1.74v-.5a2 2 0 0 1 1-1.74l.15-.09a2 2 0 0 0 .73-2.73'
        'l-.22-.38a2 2 0 0 0-2.73-.73l-.15.08a2 2 0 0 1-2 0l-.43-.25a2 2 0 0 1-1-1.73'
        'V4a2 2 0 0 0-2-2z"/><circle cx="12" cy="12" r="3"/>'
    ),
    "search": '<circle cx="11" cy="11" r="8"/><path d="m21 21-4.3-4.3"/>',
    "trash": (
        '<path d="M3 6h18"/>'
        '<path d="M19 6v14c0 1-1 2-2 2H7c-1 0-2-1-2-2V6"/>'
        '<path d="M8 6V4c0-1 1-2 2-2h4c1 0 2 1 2 2v2"/>'
    ),
    "chart": (
        '<path d="M3 3v16a2 2 0 0 0 2 2h16"/>'
        '<path d="M18 17V9"/><path d="M13 17V5"/><path d="M8 17v-3"/>'
    ),
    "flame": (
        '<path d="M8.5 14.5A2.5 2.5 0 0 0 11 12c0-1.38-.5-2-1-3-1.072-2.143'
        '-.224-4.054 2-6 .5 2.5 2 4.9 4 6.5 2 1.6 3 3.5 3 5.5a7 7 0 1 1-14 0'
        'c0-1.153.433-2.294 1-3a2.5 2.5 0 0 0 2.5 2.5z"/>'
    ),
    "alert": (
        '<path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16'
        'a2 2 0 0 0 1.73-3Z"/><path d="M12 9v4"/><path d="M12 17h.01"/>'
    ),
    "check": (
        '<path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/>'
        '<polyline points="22 4 12 14.01 9 11.01"/>'
    ),
    "info": (
        '<circle cx="12" cy="12" r="10"/>'
        '<path d="M12 16v-4"/><path d="M12 8h.01"/>'
    ),
    "activity": '<polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>',
}


def svg(name, size="1em", color="currentColor", stroke="2"):
    """Return an inline SVG string for the given icon name."""
    paths = _SVG_PATHS[name]
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="{stroke}" '
        f'stroke-linecap="round" stroke-linejoin="round" '
        f'style="vertical-align:-0.15em;display:inline-block;">{paths}</svg>'
    )


# =====================================================================
# STYLES
# =====================================================================
st.markdown("""
<style>
:root {
  --bg: #0f172a;
  --bg2: #1e293b;
  --card: rgba(30,41,59,0.6);
  --border: rgba(255,255,255,0.08);
  --accent: #0ea5e9;
  --accent2: #38bdf8;
  --text: #f8fafc;
  --muted: #94a3b8;
  --success: #10b981;
  --warn: #f59e0b;
  --danger: #ef4444;
}
.stApp { background: var(--bg); color: var(--text); }
#MainMenu, footer, header[data-testid="stHeader"] { visibility: hidden; }

[data-testid="stSidebar"] {
  background: var(--bg2);
  border-right: 1px solid var(--border);
}

/* --- brand --- */
.sidebar-brand {
  display: flex; align-items: center; gap: 0.65rem;
  padding: 0.2rem 0 0.9rem 0.2rem;
  border-bottom: 1px solid var(--border);
  margin-bottom: 0.9rem;
}
.sidebar-brand-icon {
  width: 38px; height: 38px; border-radius: 10px;
  display: flex; align-items: center; justify-content: center;
  background: linear-gradient(135deg, #0ea5e9, #06b6d4);
  color: #fff;
  box-shadow: 0 0 14px rgba(14,165,233,0.4);
}
.sidebar-title {
  font-size: 1.15rem; font-weight: 800; color: var(--text);
  letter-spacing: -0.01em; line-height: 1.1;
}
.sidebar-sub {
  font-size: 0.66rem; color: var(--muted);
  text-transform: uppercase; letter-spacing: 0.1em; font-weight: 600;
  margin-top: 2px;
}

[data-testid="stSidebar"] .stButton > button {
  width: 100%; text-align: left;
  padding: 0.55rem 0.8rem; margin: 0.15rem 0;
  border-radius: 10px;
  background: rgba(255,255,255,0.03);
  border: 1px solid var(--border);
  color: var(--muted); font-weight: 600;
  transition: all 0.2s ease;
}
[data-testid="stSidebar"] .stButton > button:hover {
  background: rgba(14,165,233,0.12);
  border-color: rgba(56,189,248,0.4);
  color: var(--text);
  transform: translateX(3px);
}

/* --- hero --- */
.hero {
  padding: 2.2rem; border-radius: 20px;
  background: var(--card);
  border: 1px solid var(--border);
  margin-bottom: 1.5rem;
}
.hero h1 {
  margin: 0; font-size: 2.3rem; font-weight: 800;
  letter-spacing: -0.02em;
  display: flex; align-items: center; gap: 0.6rem;
}
.hero p { color: var(--muted); font-size: 1rem; max-width: 640px; }

.badge {
  display: inline-flex; align-items: center; gap: 0.4rem;
  padding: 0.32rem 0.8rem;
  border-radius: 6px; font-size: 0.72rem; font-weight: 700;
  background: rgba(14,165,233,0.15);
  border: 1px solid rgba(56,189,248,0.35);
  color: var(--accent2);
  margin-bottom: 0.8rem;
}

/* --- cards --- */
.card {
  padding: 1.25rem; border-radius: 16px;
  background: var(--card); border: 1px solid var(--border);
}
.card h3 {
  margin-top: 0; font-size: 1.05rem;
  display: flex; align-items: center; gap: 0.5rem;
}

.metric-card {
  padding: 1.1rem; border-radius: 16px;
  background: var(--card); border: 1px solid var(--border);
  text-align: center;
}
.metric-value { font-size: 1.8rem; font-weight: 800; margin: 0.3rem 0; }
.metric-label {
  font-size: 0.7rem; color: var(--muted); font-weight: 600;
  text-transform: uppercase; letter-spacing: 0.1em;
}
.metric-icon { display: flex; justify-content: center; color: var(--accent2); }

/* --- diagnostic panel --- */
.diag-panel {
  padding: 1.4rem; border-radius: 18px; margin-top: 1rem;
  background: var(--card);
  border: 1px solid rgba(56,189,248,0.35);
}
.diag-title {
  font-size: 0.72rem; color: var(--muted);
  text-transform: uppercase; letter-spacing: 0.1em; font-weight: 700;
  display: flex; align-items: center; gap: 0.4rem;
}
.diag-pred { font-size: 1.6rem; font-weight: 800; margin: 0.35rem 0; }

.badge-benign {
  background: rgba(16,185,129,0.15); color: var(--success);
  border: 1px solid rgba(16,185,129,0.3);
}
.badge-malignant {
  background: rgba(239,68,68,0.15); color: var(--danger);
  border: 1px solid rgba(239,68,68,0.3);
}
.medical-badge {
  display: inline-flex; align-items: center; gap: 0.35rem;
  padding: 0.32rem 0.75rem; border-radius: 6px;
  font-size: 0.72rem; font-weight: 700;
}

/* --- probability rows --- */
.prob-row { margin: 0.4rem 0; }
.prob-name {
  font-size: 0.85rem; color: var(--text); font-weight: 600;
  display: flex; justify-content: space-between;
}
.prob-bar-bg {
  height: 6px; border-radius: 3px; overflow: hidden;
  background: rgba(255,255,255,0.06); margin-top: 5px;
}
.prob-bar-fill {
  height: 100%; border-radius: 3px;
  background: linear-gradient(90deg, #0ea5e9, #06b6d4);
}

/* --- empty state --- */
.empty {
  padding: 2.5rem; border-radius: 18px;
  background: var(--card); border: 1px solid var(--border);
  text-align: center; color: var(--muted);
}
.empty-icon { display: flex; justify-content: center; color: var(--muted); margin-bottom: 0.6rem; }

/* --- footer --- */
.footer {
  margin-top: 2.5rem; padding: 1.2rem;
  border-top: 1px solid var(--border);
  color: var(--muted); font-size: 0.85rem; text-align: center;
}

/* section heading with icon */
.section-title {
  display: flex; align-items: center; gap: 0.5rem;
  font-size: 1.1rem; font-weight: 700;
  margin: 1.2rem 0 0.6rem 0;
}

/* status pills */
.status-ok { color: var(--success); display: inline-flex; align-items: center; gap: 0.35rem; }
.status-bad { color: var(--danger);  display: inline-flex; align-items: center; gap: 0.35rem; }
</style>
""", unsafe_allow_html=True)


# =====================================================================
# TRANSFORMS
# =====================================================================
test_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(NORM_MEAN, NORM_STD),
])


# =====================================================================
# MODELS
# =====================================================================
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, pool=True):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
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
            ConvBlock(3, 32, pool=True),      # 0
            ConvBlock(32, 64, pool=True),     # 1
            ConvBlock(64, 128, pool=True),    # 2
            ConvBlock(128, 256, pool=False),  # 3
            ConvBlock(256, 256, pool=True),   # 4
            nn.Dropout2d(0.3),                # 5
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
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)


def build_resnet50(num_classes):
    model = resnet50(weights=None)
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(inplace=True),
        nn.Dropout(0.35),
        nn.Linear(256, num_classes),
    )
    return model


def build_efficientnet_b0(num_classes):
    model = efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.35),
        nn.Linear(in_features, num_classes),
    )
    return model


def build_mobilenet_v3_small(num_classes):
    model = mobilenet_v3_small(weights=None)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)
    return model


# =====================================================================
# CHECKPOINT LOADING
# =====================================================================
def find_checkpoint():
    here = Path(__file__).resolve().parent
    candidates = [
        here / CHECKPOINT_NAME,
        here.parent / CHECKPOINT_NAME,
        here / "models" / CHECKPOINT_NAME,
        here.parent / "models" / CHECKPOINT_NAME,
        here / "model" / CHECKPOINT_NAME,
        here.parent / "model" / CHECKPOINT_NAME,
        here / "checkpoints" / CHECKPOINT_NAME,
        here.parent / "checkpoints" / CHECKPOINT_NAME,
        here / "web_app" / CHECKPOINT_NAME,
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


@st.cache_resource(show_spinner=False)
def load_neurolens_model(path_str):
    path = Path(path_str)
    ckpt = torch.load(path, map_location=DEVICE)

    class_names = ckpt.get("class_names",
                           ["glioma", "meningioma", "notumor", "pituitary"])
    num_classes = ckpt.get("num_classes", len(class_names))
    model_name = ckpt.get("best_model_name", "CustomCNN")

    state = ckpt.get("model_state_dict", ckpt)
    state = {k.replace("module.", "", 1) if k.startswith("module.") else k: v
             for k, v in state.items()}

    if model_name == "ResNet50":
        model = build_resnet50(num_classes)
    elif model_name == "EfficientNet-B0":
        model = build_efficientnet_b0(num_classes)
    elif model_name == "MobileNetV3-Small":
        model = build_mobilenet_v3_small(num_classes)
    else:
        model = CustomCNN(num_classes)

    model.load_state_dict(state, strict=True)
    model.to(DEVICE)
    model.eval()
    return model, class_names, model_name


# =====================================================================
# MC DROPOUT PREDICTION
# =====================================================================
def enable_mc_dropout(model):
    """Keep BatchNorm in eval, but switch Dropout layers on."""
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            m.train()


def predict_with_uncertainty(model, image, class_names, n_samples=MC_SAMPLES):
    model.eval()
    enable_mc_dropout(model)

    tensor = test_transforms(image).unsqueeze(0).to(DEVICE)

    all_probs = []
    with torch.no_grad():
        for _ in range(n_samples):
            logits = model(tensor)
            probs = F.softmax(logits, dim=1)[0].cpu().numpy()
            all_probs.append(probs)

    all_probs = np.array(all_probs)
    mean_probs = all_probs.mean(axis=0)
    std_probs = all_probs.std(axis=0)

    idx = int(np.argmax(mean_probs))
    confidence = float(mean_probs[idx]) * 100.0

    eps = 1e-9
    entropy = float(-np.sum(mean_probs * np.log(mean_probs + eps)))

    mean_dict = {class_names[i]: float(mean_probs[i] * 100) for i in range(len(class_names))}
    std_dict = {class_names[i]: float(std_probs[i] * 100) for i in range(len(class_names))}

    return {
        "prediction": class_names[idx],
        "class_index": idx,
        "confidence": confidence,
        "mean_probs": mean_dict,
        "std_probs": std_dict,
        "entropy": entropy,
    }


# =====================================================================
# GRAD-CAM++
# =====================================================================
def get_target_layer(model, model_name):
    if model_name == "ResNet50":
        return model.layer4[-1].conv3
    if model_name == "EfficientNet-B0":
        return model.features[-1]
    if model_name == "MobileNetV3-Small":
        return model.features[-1]
    return model.features[4].block[0]


def generate_gradcam_pp(model, image, model_name, target_class):
    model.eval()
    target_layer = get_target_layer(model, model_name)

    store = {}

    def fwd_hook(m, i, o):
        store["A"] = o.detach()

    def bwd_hook(m, gi, go):
        store["dY"] = go[0].detach()

    h1 = target_layer.register_forward_hook(fwd_hook)
    h2 = target_layer.register_full_backward_hook(bwd_hook)

    try:
        tensor = test_transforms(image).unsqueeze(0).to(DEVICE).requires_grad_(True)
        model.zero_grad()
        output = model(tensor)
        output[0, target_class].backward()

        A = store["A"][0]
        dY = store["dY"][0]

        dY2 = dY ** 2
        dY3 = dY ** 3
        eps = 1e-8
        denom = 2.0 * dY2 + (A * dY3).sum(dim=(1, 2), keepdim=True) + eps
        alpha = dY2 / denom
        weights = (alpha * F.relu(dY)).sum(dim=(1, 2))

        cam = F.relu((weights.view(-1, 1, 1) * A).sum(dim=0))
        cam = cam.cpu().numpy()
        cam = cam - cam.min()
        if cam.max() > 0:
            cam = cam / cam.max()
        return cam
    finally:
        h1.remove()
        h2.remove()


# =====================================================================
# INTEGRATED GRADIENTS (via Captum)
# =====================================================================
def generate_integrated_gradients(model, image, target_class):
    if not CAPTUM_AVAILABLE:
        return None
    model.eval()
    tensor = test_transforms(image).unsqueeze(0).to(DEVICE)
    baseline = torch.zeros_like(tensor)

    ig = IntegratedGradients(model)
    attrs = ig.attribute(tensor, baselines=baseline, target=target_class, n_steps=32)

    attr = attrs[0].cpu().numpy()
    attr = np.abs(attr).sum(axis=0)
    attr = attr - attr.min()
    if attr.max() > 0:
        attr = attr / attr.max()
    return attr


# =====================================================================
# EXPLANATION AGREEMENT SCORE (IoU of top 25%)
# =====================================================================
def compute_eas(cam_a, cam_b, top_frac=0.25):
    if cam_a is None or cam_b is None:
        return None

    img_b = Image.fromarray((cam_b * 255).astype(np.uint8))
    img_b = img_b.resize((cam_a.shape[1], cam_a.shape[0]))
    cam_b_resized = np.array(img_b).astype(np.float32) / 255.0

    thr_a = np.quantile(cam_a, 1.0 - top_frac)
    thr_b = np.quantile(cam_b_resized, 1.0 - top_frac)

    mask_a = cam_a >= thr_a
    mask_b = cam_b_resized >= thr_b

    union = np.logical_or(mask_a, mask_b).sum()
    if union == 0:
        return 0.0
    inter = np.logical_and(mask_a, mask_b).sum()
    return float(inter / union)


# =====================================================================
# OVERLAY HELPER
# =====================================================================
def overlay_heatmap(image, cam, alpha=0.4):
    import matplotlib.cm as cm
    img = np.array(image.resize((IMG_SIZE, IMG_SIZE))).astype(np.float32) / 255.0
    heat = cm.jet(cam)[..., :3]
    overlay = (1 - alpha) * img + alpha * heat
    overlay = np.clip(overlay, 0, 1)
    return overlay


# =====================================================================
# SESSION STATE
# =====================================================================
defaults = {
    "nav": "Analyze",
    "last_image": None,
    "last_result": None,
    "last_gradcam": None,
    "last_ig": None,
    "last_eas": None,
    "history": [],
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


# =====================================================================
# LOAD MODEL
# =====================================================================
checkpoint_path = find_checkpoint()
model, class_names, model_name = None, None, None
model_error = None

if checkpoint_path is None:
    model_error = f"Checkpoint '{CHECKPOINT_NAME}' not found."
else:
    try:
        model, class_names, model_name = load_neurolens_model(str(checkpoint_path))
    except Exception as e:
        model_error = f"Failed to load checkpoint: {e}"


# =====================================================================
# SIDEBAR
# =====================================================================
with st.sidebar:
    st.markdown(
        f'<div class="sidebar-brand">'
        f'  <div class="sidebar-brand-icon">{svg("brain", "22")}</div>'
        f'  <div>'
        f'    <div class="sidebar-title">NeuroLens AI</div>'
        f'    <div class="sidebar-sub">Explainable · Uncertainty-Aware</div>'
        f'  </div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    nav_items = [
        ("Analyze",  ":material/biotech:"),
        ("History",  ":material/history:"),
        ("About",    ":material/settings:"),
    ]
    for label, mat_icon in nav_items:
        prefix = "▶  " if st.session_state.nav == label else ""
        if st.button(
            f"{prefix}{label}",
            key=f"nav_{label}",
            icon=mat_icon,
            use_container_width=True,
        ):
            st.session_state.nav = label
            st.rerun()

    st.markdown("---")

    total = len(st.session_state.history)
    avg_conf = float(np.mean([h["confidence"] for h in st.session_state.history])) \
        if st.session_state.history else 0.0
    st.metric("Total scans", total)
    st.metric("Avg. confidence", f"{avg_conf:.1f}%")

    st.markdown("---")
    if model_error:
        st.markdown(
            f'<div class="status-bad">{svg("alert", "16")} Engine unavailable</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="status-ok">{svg("check", "16")} {model_name}</div>',
            unsafe_allow_html=True,
        )
        st.caption(f"Device: {DEVICE}")
        if not CAPTUM_AVAILABLE:
            st.warning("Captum not installed — Integrated Gradients disabled.")

    if st.button(
        "Clear history",
        icon=":material/delete:",
        use_container_width=True,
    ):
        st.session_state.history = []
        st.session_state.last_result = None
        st.session_state.last_image = None
        st.session_state.last_gradcam = None
        st.session_state.last_ig = None
        st.session_state.last_eas = None
        st.rerun()


nav = st.session_state.nav


# =====================================================================
# PAGE: ANALYZE
# =====================================================================
if nav == "Analyze":
    st.markdown(
        f'<div class="hero">'
        f'  <div class="badge">{svg("activity", "14")} AI-Assisted Research Tool</div>'
        f'  <h1>{svg("brain", "1em", color="#38bdf8")} Brain MRI Analysis</h1>'
        f'  <p>Upload a brain MRI scan for classification, uncertainty estimation, '
        f'     and explanation via Grad-CAM++ and Integrated Gradients. '
        f'     All processing runs locally.</p>'
        f'</div>',
        unsafe_allow_html=True,
    )

    if model_error:
        st.error(model_error)
    else:
        uploaded = st.file_uploader(
            "Upload Brain MRI",
            type=["jpg", "jpeg", "png", "bmp", "webp"],
            key="mri_upload",
        )

        if uploaded is not None:
            try:
                image = Image.open(uploaded).convert("RGB")
            except UnidentifiedImageError:
                st.error("Invalid image file.")
                st.stop()
            except Exception as e:
                st.error(f"Could not read image: {e}")
                st.stop()

            col1, col2 = st.columns([1, 2])

            with col1:
                st.markdown("**Input MRI**")
                st.image(image, use_container_width=True)

            with col2:
                st.markdown("**Diagnostic Report**")
                run = st.button(
                    "Run Analysis",
                    icon=":material/search:",
                    type="primary",
                    use_container_width=True,
                )

                if run:
                    with st.spinner("Running MC Dropout inference..."):
                        result = predict_with_uncertainty(model, image, class_names)

                    with st.spinner("Computing Grad-CAM++..."):
                        try:
                            cam = generate_gradcam_pp(
                                model, image, model_name, result["class_index"]
                            )
                        except Exception as e:
                            cam = None
                            st.warning(f"Grad-CAM++ failed: {e}")

                    ig = None
                    if CAPTUM_AVAILABLE:
                        with st.spinner("Computing Integrated Gradients..."):
                            try:
                                ig = generate_integrated_gradients(
                                    model, image, result["class_index"]
                                )
                            except Exception as e:
                                st.warning(f"Integrated Gradients failed: {e}")

                    eas = compute_eas(cam, ig)

                    result["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    result["model"] = model_name

                    st.session_state.last_image = image
                    st.session_state.last_result = result
                    st.session_state.last_gradcam = cam
                    st.session_state.last_ig = ig
                    st.session_state.last_eas = eas

                    st.session_state.history.append(result)
                    if len(st.session_state.history) > MAX_HISTORY:
                        st.session_state.history = st.session_state.history[-MAX_HISTORY:]

            # ---------------- RESULTS ----------------
            if st.session_state.last_result is not None:
                result = st.session_state.last_result
                cam = st.session_state.last_gradcam
                ig = st.session_state.last_ig
                eas = st.session_state.last_eas

                is_tumor = result["prediction"].lower() != "notumor"
                badge_cls = "badge-malignant" if is_tumor else "badge-benign"
                if is_tumor:
                    badge_txt = f'{svg("alert", "14")} Tumor detected'
                else:
                    badge_txt = f'{svg("check", "14")} No tumor detected'

                st.markdown(
                    f'<div class="diag-panel">'
                    f'  <div class="diag-title">{svg("activity", "14")} Diagnosis</div>'
                    f'  <div class="diag-pred">{result["prediction"]}</div>'
                    f'  <span class="medical-badge {badge_cls}">{badge_txt}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

                m1, m2, m3 = st.columns(3)
                m1.metric("Confidence", f"{result['confidence']:.2f}%")
                m2.metric("Entropy", f"{result['entropy']:.3f}")
                m3.metric("Model", result["model"])

                # -------- Probabilities --------
                st.markdown(
                    f'<div class="section-title">{svg("chart", "20", color="#38bdf8")} '
                    f'Class Probabilities (MC Dropout)</div>',
                    unsafe_allow_html=True,
                )
                for label in class_names:
                    mean_p = result["mean_probs"][label]
                    std_p = result["std_probs"][label]
                    st.markdown(
                        f'<div class="prob-row">'
                        f'  <div class="prob-name">'
                        f'    <span>{label}</span>'
                        f'    <span>{mean_p:.2f}% ± {std_p:.2f}%</span>'
                        f'  </div>'
                        f'  <div class="prob-bar-bg">'
                        f'    <div class="prob-bar-fill" style="width:{mean_p:.2f}%;"></div>'
                        f'  </div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

                # -------- Uncertainty level --------
                std_max = max(result["std_probs"].values())
                if std_max < 5:
                    level = "Low uncertainty"
                elif std_max < 15:
                    level = "Medium uncertainty"
                else:
                    level = "High uncertainty"
                st.info(f"Uncertainty level (max std): **{std_max:.2f}%** → {level}")

                # -------- Explanations --------
                if cam is not None or ig is not None:
                    st.markdown(
                        f'<div class="section-title">{svg("flame", "20", color="#38bdf8")} '
                        f'Explainability</div>',
                        unsafe_allow_html=True,
                    )
                    tabs = st.tabs(["Grad-CAM++", "Integrated Gradients", "Overlay"])

                    with tabs[0]:
                        if cam is not None:
                            import matplotlib.pyplot as plt
                            fig, ax = plt.subplots(figsize=(4, 4))
                            ax.imshow(cam, cmap="jet")
                            ax.axis("off")
                            st.pyplot(fig)
                            plt.close(fig)
                        else:
                            st.write("Grad-CAM++ unavailable.")

                    with tabs[1]:
                        if ig is not None:
                            import matplotlib.pyplot as plt
                            fig, ax = plt.subplots(figsize=(4, 4))
                            ax.imshow(ig, cmap="hot")
                            ax.axis("off")
                            st.pyplot(fig)
                            plt.close(fig)
                        else:
                            st.write("Integrated Gradients unavailable (Captum missing or failed).")

                    with tabs[2]:
                        if cam is not None:
                            overlay = overlay_heatmap(image, cam)
                            st.image(overlay, caption="Grad-CAM++ overlay",
                                     use_container_width=True)

                    if eas is not None:
                        st.metric("Explanation Agreement Score (IoU, top 25%)", f"{eas:.3f}")
                        st.caption(
                            "EAS measures overlap between Grad-CAM++ and Integrated "
                            "Gradients. It is descriptive only — no clinical threshold "
                            "is implied."
                        )


# =====================================================================
# PAGE: HISTORY
# =====================================================================
elif nav == "History":
    st.markdown(
        f'<div class="section-title">{svg("history", "22", color="#38bdf8")} '
        f'Prediction History</div>',
        unsafe_allow_html=True,
    )

    if not st.session_state.history:
        st.markdown(
            f'<div class="empty">'
            f'  <div class="empty-icon">{svg("history", "40")}</div>'
            f'  <div>No predictions yet.</div>'
            f'</div>',
            unsafe_allow_html=True,
        )
    else:
        for item in reversed(st.session_state.history):
            conf = item["confidence"]
            color = "#10b981" if conf >= 80 else "#f59e0b" if conf >= 60 else "#ef4444"
            st.markdown(
                f'<div class="card" style="margin-bottom:0.6rem;">'
                f'  <div style="display:flex;justify-content:space-between;align-items:center;">'
                f'    <div>'
                f'      <b>{item["prediction"]}</b>'
                f'      <div style="color:var(--muted);font-size:0.8rem;">'
                f'        {item["timestamp"]} · {item["model"]}'
                f'      </div>'
                f'    </div>'
                f'    <div style="color:{color};font-weight:800;font-size:1.1rem;">'
                f'      {conf:.1f}%'
                f'    </div>'
                f'  </div>'
                f'</div>',
                unsafe_allow_html=True,
            )


# =====================================================================
# PAGE: ABOUT
# =====================================================================
elif nav == "About":
    st.markdown(
        f'<div class="section-title">{svg("settings", "22", color="#38bdf8")} '
        f'About NeuroLens AI</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        f'<div class="card">'
        f'  <h3>{svg("brain", "20", color="#38bdf8")} Overview</h3>'
        f'  <p>NeuroLens AI is a research &amp; education Streamlit app for brain MRI '
        f'  classification with four classes: '
        f'  <b>glioma, meningioma, notumor, pituitary</b>.</p>'
        f'  <p>It provides:</p>'
        f'  <ul>'
        f'    <li>Multi-architecture support (Custom CNN, ResNet50, '
        f'        EfficientNet-B0, MobileNetV3-Small)</li>'
        f'    <li>Monte Carlo Dropout uncertainty estimation</li>'
        f'    <li>Grad-CAM++ explanations</li>'
        f'    <li>Integrated Gradients (via Captum)</li>'
        f'    <li>Explanation Agreement Score (IoU of top-quartile regions)</li>'
        f'  </ul>'
        f'</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        f'<div class="card" style="margin-top:1rem;">'
        f'  <h3>{svg("settings", "20", color="#38bdf8")} System</h3>'
        f'  <ul>'
        f'    <li><b>Device:</b> {DEVICE}</li>'
        f'    <li><b>Model:</b> {model_name if model_name else "—"}</li>'
        f'    <li><b>Classes:</b> {", ".join(class_names) if class_names else "—"}</li>'
        f'    <li><b>Input size:</b> {IMG_SIZE} × {IMG_SIZE}</li>'
        f'    <li><b>Captum available:</b> {"yes" if CAPTUM_AVAILABLE else "no"}</li>'
        f'  </ul>'
        f'</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        f'<div class="card" style="margin-top:1rem;border-color:rgba(245,158,11,0.35);">'
        f'  <h3 style="color:#f59e0b;">{svg("alert", "20", color="#f59e0b")} '
        f'    Medical Disclaimer</h3>'
        f'  <p>NeuroLens AI is an <b>AI-assisted research and education tool</b>. '
        f'  It is <b>not</b> a certified medical diagnostic device and must <b>not</b> '
        f'  replace assessment by a qualified healthcare professional.</p>'
        f'</div>',
        unsafe_allow_html=True,
    )


# =====================================================================
# FOOTER
# =====================================================================
st.markdown(
    f'<div class="footer">'
    f'{svg("brain", "14", color="#38bdf8")} NeuroLens AI · v2.1 · '
    f'© {datetime.now().year} · Device: {DEVICE} · '
    f'Model: {model_name if model_name else "—"}'
    f'</div>',
    unsafe_allow_html=True,
)