# =============================================================================
# NeuroLens AI — Neurodiagnostic Intelligence Platform
# Reconstructed Edition
# =============================================================================

import warnings
import time
import io
import copy
import json
import hashlib
import platform
from importlib import metadata
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image, UnidentifiedImageError

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import resnet50, efficientnet_b0

warnings.filterwarnings("ignore")


# =============================================================================
# PAGE CONFIG  (must be first Streamlit call)
# =============================================================================

st.set_page_config(
    page_title="NeuroLens AI",
    page_icon=":material/neurology:",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =============================================================================
# CONSTANTS
# =============================================================================

NORM_MEAN        = [0.485, 0.456, 0.406]
NORM_STD         = [0.229, 0.224, 0.225]
IMG_SIZE         = 224
MC_SAMPLES_DEF   = 20
MAX_HISTORY      = 200
CLASS_NAMES      = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]
MAX_UPLOAD_BYTES = 200 * 1024 * 1024
MAX_IMAGE_PIXELS = 50_000_000
PAGE_LABELS      = ["Home", "MRI Analysis", "XAI Lab", "Dashboard", "History", "Settings"]

try:
    Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS
except Exception:
    pass

BASE_DIR   = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "neurolens_best.pth"
if not MODEL_PATH.exists():
    for _candidate in [BASE_DIR.parent / "neurolens_best.pth"]:
        if _candidate.exists():
            MODEL_PATH = _candidate
            break

DEFAULT_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

test_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(NORM_MEAN, NORM_STD),
])


# =============================================================================
# GLOBAL CSS  — clean, minimal, purposeful
# =============================================================================

st.markdown(
    '<link rel="preconnect" href="https://fonts.googleapis.com">'
    '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
    '<link href="https://fonts.googleapis.com/css2?'
    'family=Inter:wght@400;500;600;700&'
    'family=JetBrains+Mono:wght@400;600&display=swap" '
    'rel="stylesheet">',
    unsafe_allow_html=True,
)

st.markdown("""
<style>
/* ── Tokens ─────────────────────────────────────────────── */
:root {
  --bg:       #0b1628;
  --surface:  #111f38;
  --border:   rgba(96,165,250,.18);
  --cyan:     #22d3ee;
  --emerald:  #34d399;
  --amber:    #fbbf24;
  --danger:   #f87171;
  --text:     #e2ecf9;
  --muted:    #7f9bbd;
  --mono:     'JetBrains Mono', monospace;
  --sans:     'Inter', system-ui, sans-serif;
  --radius:   10px;
}

/* ── Base ────────────────────────────────────────────────── */
html, body,
[data-testid="stAppViewContainer"],
[data-testid="stApp"] {
  background: var(--bg) !important;
  font-family: var(--sans);
  -webkit-font-smoothing: antialiased;
}
[data-testid="stAppViewContainer"] > .main {
  background:
    radial-gradient(ellipse 90% 50% at 50% -10%, rgba(34,211,238,.12), transparent 60%),
    linear-gradient(180deg, #0d2347 0%, #080f1e 60%) !important;
  background-attachment: fixed;
}
[data-testid="stMainBlockContainer"] {
  max-width: 1320px !important;
  padding: 1.5rem 1.4rem 3rem !important;
}

/* ── Sidebar ─────────────────────────────────────────────── */
[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #0e2455 0%, #060d1c 100%) !important;
  border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] > div:first-child {
  padding: 0 .9rem 1rem !important;
}

/* ── Headings ────────────────────────────────────────────── */
h1 { font-size: 1.7rem !important; font-weight: 700; color: var(--text); letter-spacing: -.02em; }
h2 { font-size: 1.25rem !important; font-weight: 700; color: var(--text); }
h3 { font-size: 1rem !important; font-weight: 600; color: var(--text); }
h4 { font-size: .875rem !important; font-weight: 600; color: var(--muted); text-transform: uppercase; letter-spacing: .06em; }

/* ── Card ────────────────────────────────────────────────── */
.nl-card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 1.1rem 1.2rem;
  height: 100%;
}
.nl-card h3 { margin: 0 0 .5rem; }
.nl-card p  { color: var(--muted); font-size: .85rem; line-height: 1.6; margin: .25rem 0; }

/* ── Metric card ─────────────────────────────────────────── */
.nl-metric {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: 1rem 1.1rem;
}
.nl-metric-value {
  font-size: 1.55rem;
  font-weight: 700;
  color: var(--text);
  font-variant-numeric: tabular-nums;
  letter-spacing: -.02em;
}
.nl-metric-label {
  font-size: .72rem;
  font-weight: 600;
  color: var(--muted);
  text-transform: uppercase;
  letter-spacing: .08em;
  margin-top: .3rem;
}

/* ── Diagnostic result ───────────────────────────────────── */
.nl-result {
  background: linear-gradient(135deg, rgba(37,99,235,.14), rgba(16,185,129,.07));
  border: 1px solid rgba(34,211,238,.40);
  border-radius: var(--radius);
  padding: 1.4rem 1.5rem;
  margin: .75rem 0;
}
.nl-result-label {
  font-size: .68rem;
  font-weight: 700;
  color: var(--cyan);
  text-transform: uppercase;
  letter-spacing: .1em;
  margin-bottom: .5rem;
}
.nl-result-prediction {
  font-size: 2rem;
  font-weight: 700;
  color: var(--text);
  letter-spacing: -.025em;
  line-height: 1;
}
.nl-result-conf {
  font-size: .9rem;
  color: var(--muted);
  margin-top: .35rem;
}

/* ── Prob grid ───────────────────────────────────────────── */
.nl-prob-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
  gap: .6rem;
  margin: .5rem 0;
}
.nl-prob-card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  padding: .9rem 1rem;
  position: relative;
}
.nl-prob-card.top {
  border-color: rgba(34,211,238,.55);
  background: linear-gradient(135deg, rgba(37,99,235,.18), rgba(16,185,129,.10));
}
.nl-prob-value {
  font-size: 1.35rem;
  font-weight: 700;
  color: var(--text);
  font-family: var(--mono);
}
.nl-prob-card.top .nl-prob-value { color: var(--cyan); }
.nl-prob-label { font-size: .78rem; color: var(--muted); margin-top: .3rem; }
.nl-prob-top-tag {
  position: absolute;
  top: .5rem; right: .6rem;
  font-size: .58rem;
  font-weight: 700;
  color: var(--cyan);
  text-transform: uppercase;
  letter-spacing: .07em;
  background: rgba(34,211,238,.14);
  border: 1px solid rgba(34,211,238,.40);
  border-radius: 999px;
  padding: .1rem .45rem;
}

/* ── Uncertainty card ────────────────────────────────────── */
.nl-unc {
  background: rgba(139,92,246,.09);
  border: 1px solid rgba(139,92,246,.32);
  border-radius: var(--radius);
  padding: .9rem 1.1rem;
  margin: .6rem 0;
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 1rem;
  flex-wrap: wrap;
}
.nl-unc-title { font-size: .68rem; font-weight: 700; color: #a78bfa; text-transform: uppercase; letter-spacing: .08em; }
.nl-unc-value { font-size: 1.25rem; font-weight: 700; font-family: var(--mono); }
.nl-unc-band  { font-size: .76rem; margin-top: .15rem; font-weight: 600; }

/* ── XAI agree ───────────────────────────────────────────── */
.nl-agree {
  background: rgba(34,211,238,.07);
  border: 1px solid rgba(34,211,238,.28);
  border-radius: var(--radius);
  padding: .85rem 1.1rem;
  margin: .5rem 0;
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 1rem;
  flex-wrap: wrap;
}
.nl-agree-score { font-size: 1.5rem; font-weight: 700; font-family: var(--mono); letter-spacing: -.02em; }

/* ── History row ─────────────────────────────────────────── */
.nl-hist-row {
  display: flex;
  align-items: center;
  gap: 1rem;
  padding: .8rem 1rem;
  border: 1px solid var(--border);
  border-radius: var(--radius);
  background: var(--surface);
  margin: .4rem 0;
}
.nl-hist-title { font-weight: 600; color: var(--text); font-size: .9rem; }
.nl-hist-meta  { font-size: .72rem; color: var(--muted); margin-top: .15rem; font-family: var(--mono); }

/* ── Confidence badge ────────────────────────────────────── */
.nl-badge {
  display: inline-block;
  padding: .2rem .65rem;
  border-radius: 999px;
  font-size: .72rem;
  font-weight: 700;
  font-family: var(--mono);
}
.nl-badge-hi  { background: rgba(52,211,153,.14); border: 1px solid rgba(52,211,153,.42); color: #34d399; }
.nl-badge-mid { background: rgba(251,191,36,.14);  border: 1px solid rgba(251,191,36,.42);  color: #fbbf24; }
.nl-badge-lo  { background: rgba(248,113,113,.14); border: 1px solid rgba(248,113,113,.42); color: #f87171; }

/* ── Distribution bar ────────────────────────────────────── */
.nl-dist {
  padding: .7rem .9rem;
  border: 1px solid var(--border);
  border-radius: var(--radius);
  background: var(--surface);
  margin: .35rem 0;
}
.nl-dist-row { display: flex; justify-content: space-between; align-items: center; margin-bottom: .45rem; }
.nl-dist-name  { font-size: .84rem; font-weight: 600; color: var(--text); }
.nl-dist-count { font-size: .74rem; color: var(--muted); font-family: var(--mono); }
.nl-dist-track { height: 6px; border-radius: 999px; background: rgba(96,165,250,.12); overflow: hidden; }
.nl-dist-fill  { height: 100%; border-radius: 999px; background: linear-gradient(90deg, var(--cyan), var(--emerald)); }

/* ── Engine status block ─────────────────────────────────── */
.nl-engine {
  padding: 1rem 1.1rem;
  border-radius: var(--radius);
  margin: .5rem 0;
}
.nl-engine.ok  { background: rgba(16,185,129,.07); border: 1px solid rgba(16,185,129,.38); }
.nl-engine.err { background: rgba(239,68,68,.07);  border: 1px solid rgba(239,68,68,.35); }
.nl-engine-title { font-weight: 700; margin-bottom: .6rem; }
.nl-engine.ok  .nl-engine-title { color: var(--emerald); }
.nl-engine.err .nl-engine-title { color: var(--danger); }
.nl-engine p { font-size: .82rem; color: var(--muted); margin: .2rem 0; }

/* ── Spec grid (engine ready sub-info) ───────────────────── */
.nl-spec-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: .5rem;
  margin-top: .75rem;
}
.nl-spec-item {
  background: rgba(37,99,235,.12);
  border: 1px solid rgba(34,211,238,.22);
  border-radius: 8px;
  padding: .55rem .75rem;
}
.nl-spec-key   { font-size: .6rem; font-weight: 700; color: var(--muted); text-transform: uppercase; letter-spacing: .08em; }
.nl-spec-value { font-size: .85rem; font-weight: 700; color: var(--text); font-family: var(--mono); margin-top: .15rem; }

/* ── Empty state ─────────────────────────────────────────── */
.nl-empty {
  padding: 3rem 1rem;
  text-align: center;
  border: 1.5px dashed rgba(96,165,250,.28);
  border-radius: var(--radius);
  color: var(--muted);
}
.nl-empty p { font-size: .9rem; margin: .4rem 0 0; }

/* ── Activity feed ───────────────────────────────────────── */
.nl-feed { max-height: 370px; overflow-y: auto; }
.nl-feed-row {
  display: flex;
  gap: .6rem;
  padding: .4rem .2rem;
  border-bottom: 1px solid rgba(96,165,250,.10);
  font-size: .74rem;
}
.nl-feed-row:last-child { border-bottom: 0; }
.nl-feed-time { color: var(--muted); font-family: var(--mono); flex: 0 0 auto; }
.nl-feed-msg  { color: var(--text); }
.nl-feed-row.err  .nl-feed-msg { color: var(--danger); }
.nl-feed-row.warn .nl-feed-msg { color: var(--amber); }
.nl-feed-row.ok   .nl-feed-msg { color: var(--emerald); }

/* ── Sidebar brand block ─────────────────────────────────── */
.nl-brand {
  padding: .9rem .4rem .75rem;
  border-bottom: 1px solid var(--border);
  margin-bottom: .6rem;
}
.nl-brand-name { font-size: 1.05rem; font-weight: 700; color: var(--text); letter-spacing: -.01em; }
.nl-brand-name span { color: var(--cyan); }
.nl-brand-sub  { font-size: .72rem; color: var(--muted); margin-top: .15rem; }

/* ── Scrollbar ───────────────────────────────────────────── */
::-webkit-scrollbar { width: 8px; }
::-webkit-scrollbar-track { background: rgba(5,14,34,.6); }
::-webkit-scrollbar-thumb { background: rgba(34,211,238,.35); border-radius: 8px; }

/* ── Disclaimer ──────────────────────────────────────────── */
.nl-disclaimer {
  border-left: 3px solid var(--amber);
  background: rgba(245,158,11,.07);
  border-radius: 0 var(--radius) var(--radius) 0;
  padding: .75rem 1rem;
  font-size: .82rem;
  color: var(--muted);
  margin: .8rem 0;
  line-height: 1.6;
}
.nl-disclaimer b { color: var(--amber); }

/* ── Latest summary ──────────────────────────────────────── */
.nl-summary-row {
  display: flex;
  justify-content: space-between;
  padding: .45rem 0;
  border-bottom: 1px solid rgba(96,165,250,.10);
  font-size: .8rem;
}
.nl-summary-row:last-child { border-bottom: 0; }
.nl-summary-key { color: var(--muted); font-weight: 500; }
.nl-summary-val { color: var(--text); font-family: var(--mono); font-weight: 600; }

/* ── Steps row ───────────────────────────────────────────── */
.nl-step {
  text-align: center;
  padding: .9rem .5rem;
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  height: 100%;
}
.nl-step-num {
  width: 30px; height: 30px;
  margin: 0 auto .5rem;
  display: grid; place-items: center;
  border-radius: 50%;
  background: linear-gradient(135deg, var(--cyan), var(--emerald));
  color: #fff;
  font-size: .78rem;
  font-weight: 700;
}
.nl-step-title { font-size: .8rem; font-weight: 600; color: var(--text); }

/* ── Footer ──────────────────────────────────────────────── */
.nl-footer {
  margin-top: 2.5rem;
  padding-top: 1.2rem;
  border-top: 1px solid var(--border);
  display: flex;
  justify-content: space-between;
  align-items: center;
  flex-wrap: wrap;
  gap: .75rem;
  font-size: .76rem;
  color: var(--muted);
}
.nl-footer a { color: var(--cyan); text-decoration: none; }

/* ── LP bars ─────────────────────────────────────────────── */
.lp-wrap { border: 1px solid var(--border); border-radius: var(--radius); padding: 1rem 1.1rem; background: var(--surface); }
.lp-head { font-size: .65rem; font-weight: 700; color: var(--cyan); text-transform: uppercase; letter-spacing: .1em; margin-bottom: .7rem; }
.lp-row  { display: flex; align-items: center; gap: .6rem; margin: .32rem 0; }
.lp-label { min-width: 120px; font-size: .76rem; color: var(--muted); }
.lp-track { flex: 1; height: 5px; border-radius: 999px; background: rgba(59,130,246,.14); overflow: hidden; }
.lp-fill  { height: 100%; width: 0%; border-radius: 999px; background: linear-gradient(90deg, var(--cyan), var(--emerald)); transition: width 1.1s cubic-bezier(.2,.8,.2,1); }
.lp-mc    { background: linear-gradient(90deg, #a78bfa, var(--cyan)) !important; }
.lp-val   { min-width: 50px; text-align: right; font-family: var(--mono); font-size: .72rem; color: var(--text); font-weight: 600; }
.lp-sec   { font-size: .6rem; font-weight: 700; color: var(--muted); text-transform: uppercase; letter-spacing: .1em; margin: .6rem 0 .25rem; }

/* ── Ticker ──────────────────────────────────────────────── */
.nl-ticker {
  display: flex;
  align-items: center;
  gap: 1.4rem;
  overflow-x: auto;
  padding: .55rem .85rem;
  background: rgba(14,40,88,.55);
  border: 1px solid var(--border);
  border-radius: var(--radius);
  margin-bottom: 1rem;
  font-size: .72rem;
  font-family: var(--mono);
  scrollbar-width: none;
  white-space: nowrap;
}
.nl-ticker::-webkit-scrollbar { display: none; }
.nl-ticker-item { color: var(--muted); }
.nl-ticker-item span { color: var(--text); font-weight: 600; margin-left: .3rem; }

/* ── Info rows (Settings) ────────────────────────────────── */
.nl-info-row {
  display: flex;
  justify-content: space-between;
  padding: .5rem 0;
  border-bottom: 1px solid var(--border);
  font-size: .82rem;
}
.nl-info-row:last-child { border-bottom: 0; }
.nl-info-key { color: var(--muted); }
.nl-info-val { color: var(--text); font-family: var(--mono); font-weight: 600; }

@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after {
    animation-duration: .01ms !important;
    transition-duration: .01ms !important;
  }
}
</style>
""", unsafe_allow_html=True)


# =============================================================================
# HELPERS
# =============================================================================

def _e(text) -> str:
    """HTML-escape a value."""
    if text is None:
        return ""
    return (
        str(text)
        .replace("&", "&amp;").replace("<", "&lt;")
        .replace(">", "&gt;").replace('"', "&quot;").replace("'", "&#39;")
    )


def _st_width(container: bool) -> dict:
    try:
        major, minor = (int(p) for p in st.__version__.split(".")[:2])
        if (major, minor) >= (1, 42):
            return {"width": "stretch" if container else "content"}
    except Exception:
        pass
    return {"use_container_width": bool(container)}


def _W() -> dict:   return _st_width(True)
def _Wc() -> dict:  return _st_width(False)


def uncertainty_band(u: float) -> Tuple[str, str]:
    if u < 0.05:  return "Very High Reliability", "#34d399"
    if u < 0.12:  return "High Reliability",      "#22d3ee"
    if u < 0.22:  return "Moderate Reliability",  "#fbbf24"
    return              "Low Reliability",         "#f87171"


def conf_badge_class(conf: float) -> str:
    if conf >= 80: return "hi"
    if conf >= 60: return "mid"
    return "lo"


def _dark_fig(w=6, h=2.8):
    fig, ax = plt.subplots(figsize=(w, h))
    bg = "#0c1a30"
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)
    for s in ("top", "right"):    ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):  ax.spines[s].set_color("#1e3a5f")
    ax.tick_params(colors="#7f9bbd", labelsize=8)
    return fig, ax


# =============================================================================
# MODEL ARCHITECTURE
# =============================================================================

class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, kernel=3, stride=1, padding=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel, stride, padding, bias=False),
            nn.BatchNorm2d(out_c),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel, 1, padding, bias=False),
            nn.BatchNorm2d(out_c),
            nn.SiLU(inplace=True),
        )
        self.skip = (
            nn.Sequential(nn.Conv2d(in_c, out_c, 1, stride, bias=False), nn.BatchNorm2d(out_c))
            if in_c != out_c or stride != 1 else nn.Identity()
        )

    def forward(self, x):
        return self.block(x) + self.skip(x)


class CustomCNN(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(3,   32, stride=2),
            nn.MaxPool2d(2),
            ConvBlock(32,  64, stride=2),
            nn.MaxPool2d(2),
            ConvBlock(64, 128, stride=2),
            ConvBlock(128, 128),
        )
        self.pool       = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.45),
            nn.Linear(128, 128),
            nn.SiLU(inplace=True),
            nn.Dropout(0.30),
            nn.Linear(128, num_classes),
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.classifier(self.pool(self.features(x)))


def build_resnet50(n):
    m = resnet50(weights=None)
    m.fc = nn.Sequential(nn.Linear(m.fc.in_features, 256), nn.BatchNorm1d(256),
                         nn.ReLU(inplace=True), nn.Dropout(0.35), nn.Linear(256, n))
    return m


def build_efficientnet_b0(n):
    m = efficientnet_b0(weights=None)
    m.classifier = nn.Sequential(nn.Dropout(0.35), nn.Linear(m.classifier[1].in_features, n))
    return m


# =============================================================================
# MODEL LOADING
# =============================================================================

def _read_checkpoint(path: Path) -> dict:
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        ckpt = torch.load(path, map_location="cpu")
    except Exception as e:
        try:
            ckpt = torch.load(path, map_location="cpu", weights_only=False)
        except Exception as e2:
            raise RuntimeError(f"Could not load checkpoint: {e}; {e2}") from e2
    return ckpt if isinstance(ckpt, dict) else {"model_state_dict": ckpt}


@st.cache_resource(show_spinner=False)
def load_model(path_str: str, force_cpu: bool):
    path   = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")

    device = torch.device("cpu" if (force_cpu or not torch.cuda.is_available()) else "cuda")
    if device.type == "cuda":
        try: torch.backends.cudnn.benchmark = True
        except Exception: pass

    ckpt  = _read_checkpoint(path)
    ckpt_classes = ckpt.get("class_names", CLASS_NAMES)
    if ([n.casefold().replace(" ", "") for n in ckpt_classes]
            != [n.casefold().replace(" ", "") for n in CLASS_NAMES]):
        raise ValueError(f"Class mismatch. Expected {CLASS_NAMES}; found {ckpt_classes}.")

    n_cls = ckpt.get("num_classes", len(CLASS_NAMES))
    arch  = ckpt.get("best_model_name", "CustomCNN")
    if arch == "Custom CNN": arch = "CustomCNN"
    if arch not in {"CustomCNN", "ResNet50", "EfficientNet-B0"}:
        raise ValueError(f"Unsupported architecture: {arch}")

    model = {"ResNet50": build_resnet50, "EfficientNet-B0": build_efficientnet_b0}.get(
        arch, lambda n: CustomCNN(n))(n_cls)

    sd = {(k[7:] if k.startswith("module.") else k): v for k, v in ckpt.get("model_state_dict", ckpt).items()}
    try:
        model.load_state_dict(sd, strict=True)
    except RuntimeError:
        msd = model.state_dict()
        model.load_state_dict({k: v for k, v in sd.items()
                               if k in msd and tuple(v.shape) == tuple(msd[k].shape)}, strict=False)

    model.to(device).eval()
    return model, list(CLASS_NAMES), arch, device


# =============================================================================
# INFERENCE
# =============================================================================

def predict_image(image, model, class_names, device):
    model.eval()
    t0 = time.perf_counter()
    tensor = test_transforms(image).unsqueeze(0).to(device)
    pre_ms = (time.perf_counter() - t0) * 1000

    if device.type == "cuda": torch.cuda.synchronize(device)
    t1 = time.perf_counter()
    with torch.inference_mode():
        probs = F.softmax(model(tensor), dim=1)[0].detach().cpu().numpy()
    if device.type == "cuda": torch.cuda.synchronize(device)
    inf_ms = (time.perf_counter() - t1) * 1000

    idx = int(np.argmax(probs))
    return (class_names[idx], float(probs[idx]*100),
            {class_names[i]: float(probs[i]*100) for i in range(len(class_names))},
            pre_ms, inf_ms)


def mc_dropout_predict(image, model, class_names, device, n=MC_SAMPLES_DEF):
    if model is None: return None
    n = max(1, min(int(n), 500))

    def _enable(m):
        if isinstance(m, (nn.Dropout, nn.Dropout2d)): m.train()

    model.eval()
    preds = []
    try:
        model.apply(_enable)
        tensor = test_transforms(image).unsqueeze(0).to(device)
        with torch.no_grad():
            for _ in range(n):
                preds.append(F.softmax(model(tensor), dim=1)[0].detach().cpu().numpy())
    finally:
        model.eval()

    if not preds: return None
    arr   = np.stack(preds)
    mean  = arr.mean(0); std = arr.std(0)
    idx   = int(np.argmax(mean))
    unc   = float(std[idx])
    band, color = uncertainty_band(unc)
    return {
        "mean_probs":  {class_names[i]: float(mean[i]*100) for i in range(len(class_names))},
        "std_probs":   {class_names[i]: float(std[i]*100)  for i in range(len(class_names))},
        "uncertainty": unc, "band": band, "color": color,
        "prediction":  class_names[idx], "confidence": float(mean[idx]*100),
    }


# =============================================================================
# GRAD-CAM / GRAD-CAM++
# =============================================================================

def _fig_to_png(fig, dpi=160) -> bytes:
    buf = io.BytesIO()
    try:
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=dpi,
                    facecolor=fig.get_facecolor(), edgecolor="none")
    finally:
        plt.close(fig)
    return buf.getvalue()


def _target_layer(model, arch):
    candidates = {
        "ResNet50":       [lambda m: m.layer4[-1].conv3, lambda m: m.layer4[-1].conv2, lambda m: m.layer4[-1]],
        "EfficientNet-B0":[lambda m: m.features[-1], lambda m: m.features[-2]],
    }.get(arch, [lambda m: m.features[4].block[0], lambda m: m.features[4], lambda m: m.features[-1]])
    for fn in candidates:
        try:
            layer = fn(model)
            if isinstance(layer, nn.Module): return layer
        except Exception: continue
    for m in model.modules():
        if isinstance(m, nn.Conv2d): last = m
    return last


def _hook_out(out):
    return (out[0] if isinstance(out, tuple) else out).detach()


def _cam_norm(raw):
    c = np.maximum(raw, 0)
    c -= c.min()
    if c.max() > 0: c /= c.max()
    return c


def _overlay(image, cam, cmap, alpha) -> bytes:
    orig = np.asarray(image).astype(np.float32) / 255.0
    fig, ax = plt.subplots(figsize=(5, 5), dpi=120)
    fig.patch.set_alpha(0)
    ax.imshow(orig)
    ax.imshow(cam, cmap=cmap, alpha=alpha, extent=(0, orig.shape[1], orig.shape[0], 0))
    ax.axis("off"); fig.tight_layout(pad=0)
    return _fig_to_png(fig)


def generate_gradcam(image, model, arch, device) -> bytes:
    model.eval()
    acts, grds = [], []
    tl  = _target_layer(model, arch)
    fwd = tl.register_forward_hook(lambda m, i, o: acts.append(_hook_out(o)))
    bwd = tl.register_full_backward_hook(lambda m, gi, go: grds.append(go[0].detach()))
    try:
        tensor = test_transforms(image).unsqueeze(0).to(device)
        model.zero_grad()
        out = model(tensor)
        out[0, int(out.argmax(1).item())].backward()
        act = acts[0][0]; grd = grds[0][0]
        w   = grd.mean(dim=(1, 2), keepdim=True)
        cam = _cam_norm((w * act).sum(0).detach().cpu().numpy())
        return _overlay(image, cam, "jet", 0.44)
    finally:
        try: fwd.remove()
        except: pass
        try: bwd.remove()
        except: pass


def generate_gradcam_pp(image, model, arch, device) -> bytes:
    model.eval()
    acts, grds = [], []
    tl  = _target_layer(model, arch)
    fwd = tl.register_forward_hook(lambda m, i, o: acts.append(_hook_out(o)))
    bwd = tl.register_full_backward_hook(lambda m, gi, go: grds.append(go[0].detach()))
    try:
        tensor = test_transforms(image).unsqueeze(0).to(device)
        model.zero_grad()
        out = model(tensor)
        out[0, int(out.argmax(1).item())].backward()
        a  = acts[0][0].cpu().numpy(); g = grds[0][0].cpu().numpy()
        an = g ** 2
        ad = 2 * g**2 + (a * g**3).sum(axis=(1,2), keepdims=True) + 1e-8
        w  = (an / ad * np.maximum(g, 0)).sum(axis=(1, 2))
        raw = sum(ww * aa for ww, aa in zip(w, a))
        return _overlay(image, _cam_norm(raw), "inferno", 0.46)
    finally:
        try: fwd.remove()
        except: pass
        try: bwd.remove()
        except: pass


def explanation_agreement(image, model, arch, device) -> Optional[float]:
    if model is None: return None
    model.eval()
    acts, grds = [], []
    tl  = _target_layer(model, arch)
    fwd = tl.register_forward_hook(lambda m, i, o: acts.append(_hook_out(o)))
    bwd = tl.register_full_backward_hook(lambda m, gi, go: grds.append(go[0].detach()))
    try:
        tensor = test_transforms(image).unsqueeze(0).to(device)
        model.zero_grad()
        out = model(tensor)
        out[0, int(out.argmax(1).item())].backward()
    finally:
        try: fwd.remove()
        except: pass
        try: bwd.remove()
        except: pass

    if not acts or not grds: return None
    try:
        act = acts[0][0]; grd = grds[0][0]
        w   = grd.mean(dim=(1,2), keepdim=True)
        c1  = F.relu((w * act).sum(0)).detach().cpu().numpy()
        c1  = c1 / (c1.max() + 1e-8)
        a   = act.detach().cpu().numpy(); g = grd.cpu().numpy()
        an  = g**2; ad = 2*g**2 + (a*g**3).sum(axis=(1,2), keepdims=True) + 1e-8
        w2  = (an/ad * np.maximum(g, 0)).sum(axis=(1,2))
        c2  = sum(ww*aa for ww, aa in zip(w2, a))
        c2  = np.maximum(c2, 0); c2 = c2 / (c2.max() + 1e-8)
        f1, f2 = c1.flatten(), c2.flatten()
        if f1.std() < 1e-8 or f2.std() < 1e-8: return None
        r = float(np.corrcoef(f1, f2)[0, 1])
        return float(np.clip(r, 0, 1)) if np.isfinite(r) else None
    except Exception:
        return None


# =============================================================================
# CHARTING
# =============================================================================

def _trend_plot(history, key, ylabel, color):
    if len(history) < 2: return None
    fig, ax = _dark_fig()
    vals = [h[key] for h in history if h.get(key) is not None]
    xs   = list(range(1, len(vals) + 1))
    ax.plot(xs, vals, marker="o", lw=2, ms=3.5, color=color)
    ax.fill_between(xs, vals, alpha=0.09, color=color)
    ax.set_xlabel("Analysis #", color="#7f9bbd", fontsize=8)
    ax.set_ylabel(ylabel,       color="#7f9bbd", fontsize=8)
    ax.grid(True, alpha=0.08, ls="--", color="#7f9bbd")
    fig.tight_layout(pad=0.5)
    return fig


def plot_confidence_trend(history): return _trend_plot(history, "confidence",  "Confidence %",    "#22d3ee")
def plot_latency_trend(history):    return _trend_plot(history, "latency_ms",  "Latency (ms)",    "#a78bfa")
def plot_uncertainty_history(history):
    data = [h for h in history if h.get("uncertainty") is not None]
    if len(data) < 2: return None
    fig, ax = _dark_fig()
    vals = [h["uncertainty"] for h in data]
    xs   = list(range(1, len(vals) + 1))
    ax.plot(xs, vals, marker="o", lw=2, ms=3.5, color="#fbbf24")
    ax.fill_between(xs, vals, alpha=0.09, color="#fbbf24")
    ax.set_xlabel("Analysis #", color="#7f9bbd", fontsize=8)
    ax.set_ylabel("Uncertainty σ", color="#7f9bbd", fontsize=8)
    ax.grid(True, alpha=0.08, ls="--", color="#7f9bbd")
    fig.tight_layout(pad=0.5)
    return fig


# =============================================================================
# SESSION STATE
# =============================================================================

DEFAULTS: Dict[str, Any] = {
    "nav":                      "Home",
    "prediction_history":       [],
    "last_result":              None,
    "mc_result":                None,
    "gradcam_png":              None,
    "gradcam_pp_png":           None,
    "activity_log":             [],
    "live_predictions_count":   0,
    "live_avg_confidence":      0.0,
    "live_throughput":          0.0,
    "live_class_counts":        {c: 0 for c in CLASS_NAMES},
    "live_session_start":       None,
    "force_cpu":                False,
    "mc_samples":               MC_SAMPLES_DEF,
    "settings_confirm_reset":   False,
    "_page_changed":            False,
    "_prev_nav":                "Home",
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = copy.deepcopy(v)


# =============================================================================
# SESSION HELPERS
# =============================================================================

def navigate_to(page: str) -> None:
    st.session_state._prev_nav     = st.session_state.nav
    st.session_state.nav           = page
    st.session_state._page_changed = True
    st.rerun()


def log_activity(msg: str, level: str = "info") -> None:
    st.session_state.activity_log.append({
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        "message":   msg,
        "level":     level,
    })
    if len(st.session_state.activity_log) > 100:
        st.session_state.activity_log = st.session_state.activity_log[-100:]


def update_live_stats(result: dict, latency_ms: float) -> None:
    n  = st.session_state.live_predictions_count + 1
    st.session_state.live_predictions_count = n
    prev_avg = st.session_state.live_avg_confidence
    st.session_state.live_avg_confidence = prev_avg + (result["confidence"] - prev_avg) / n
    st.session_state.live_class_counts[result["prediction"]] = (
        st.session_state.live_class_counts.get(result["prediction"], 0) + 1
    )
    if st.session_state.live_session_start:
        elapsed = max((datetime.now() - st.session_state.live_session_start).total_seconds(), 1)
        st.session_state.live_throughput = n / (elapsed / 60)


def clear_prediction_history() -> None:
    st.session_state.prediction_history = []
    st.session_state.last_result        = None
    st.session_state.mc_result          = None
    st.session_state.gradcam_png        = None
    st.session_state.gradcam_pp_png     = None
    st.session_state.activity_log       = []
    st.session_state.live_predictions_count = 0
    st.session_state.live_avg_confidence    = 0.0
    st.session_state.live_throughput        = 0.0
    st.session_state.live_class_counts      = {c: 0 for c in CLASS_NAMES}
    st.session_state.live_session_start     = None


# =============================================================================
# MODEL BOOT
# =============================================================================

model_error: Optional[str] = None
model: Optional[nn.Module] = None
class_names: List[str]     = CLASS_NAMES
model_name: Optional[str]  = None
DEVICE: torch.device       = DEFAULT_DEVICE

try:
    model, class_names, model_name, DEVICE = load_model(
        str(MODEL_PATH), bool(st.session_state.force_cpu)
    )
except Exception as exc:
    model_error = str(exc)

if st.session_state.live_session_start is None:
    st.session_state.live_session_start = datetime.now()


# =============================================================================
# RENDER HELPERS
# =============================================================================

def render_ticker() -> None:
    hist = st.session_state.prediction_history
    total = st.session_state.live_predictions_count
    avg_c = st.session_state.live_avg_confidence
    thr   = st.session_state.live_throughput
    latest_pred = hist[-1]["prediction"] if hist else "—"

    st.markdown(
        f'<div class="nl-ticker">'
        f'<span class="nl-ticker-item">Scans<span>{total}</span></span>'
        f'<span class="nl-ticker-item">Avg Confidence<span>{avg_c:.1f}%</span></span>'
        f'<span class="nl-ticker-item">Throughput<span>{thr:.2f}/min</span></span>'
        f'<span class="nl-ticker-item">Last Result<span>{_e(latest_pred)}</span></span>'
        f'<span class="nl-ticker-item">Model<span>{_e(model_name or "—")}</span></span>'
        f'<span class="nl-ticker-item">Device<span>{_e(str(DEVICE))}</span></span>'
        f'</div>',
        unsafe_allow_html=True,
    )


def render_prob_bars(result: dict, mc_result: Optional[dict] = None) -> None:
    items = list(result["probabilities"].items())
    top   = result["prediction"]

    bars = "".join(
        f'<div class="lp-row">'
        f'<div class="lp-label">{_e(n)}</div>'
        f'<div class="lp-track"><div class="lp-fill" data-target="{p:.2f}" style="width:0%"></div></div>'
        f'<div class="lp-val">{p:.1f}%</div></div>'
        for n, p in items
    )

    mc_rows = ""
    if mc_result:
        mc_std = mc_result["std_probs"]
        mc_rows = '<div class="lp-sec">MC Dropout · Bayesian</div>' + "".join(
            f'<div class="lp-row">'
            f'<div class="lp-label">{_e(n)}'
            f'<span style="color:#a78bfa;font-family:var(--mono);font-size:.68rem"> ±{mc_std.get(n,0):.1f}%</span></div>'
            f'<div class="lp-track"><div class="lp-fill lp-mc" data-target="{p:.2f}" style="width:0%"></div></div>'
            f'<div class="lp-val">{p:.1f}%</div></div>'
            for n, p in mc_result["mean_probs"].items()
        )

    h = 110 + 34 * len(items) + (130 + 34 * len(items) if mc_result else 0)
    components.html(f"""
    <style>
    .lp-wrap{{padding:1rem 1.1rem;border-radius:10px;border:1px solid rgba(96,165,250,.18);
    background:#111f38;font-family:Inter,sans-serif;color:#e2ecf9}}
    .lp-head{{font-size:.65rem;font-weight:700;color:#22d3ee;text-transform:uppercase;letter-spacing:.1em;margin-bottom:.7rem}}
    .lp-row{{display:flex;align-items:center;gap:.6rem;margin:.32rem 0}}
    .lp-label{{min-width:120px;font-size:.76rem;color:#7f9bbd}}
    .lp-track{{flex:1;height:5px;border-radius:999px;background:rgba(59,130,246,.14);overflow:hidden}}
    .lp-fill{{height:100%;width:0%;border-radius:999px;background:linear-gradient(90deg,#22d3ee,#34d399);transition:width 1.1s cubic-bezier(.2,.8,.2,1)}}
    .lp-mc{{background:linear-gradient(90deg,#a78bfa,#22d3ee)!important}}
    .lp-val{{min-width:50px;text-align:right;font-family:'JetBrains Mono',monospace;font-size:.72rem;color:#e2ecf9;font-weight:600}}
    .lp-sec{{font-size:.6rem;font-weight:700;color:#7f9bbd;text-transform:uppercase;letter-spacing:.1em;margin:.6rem 0 .25rem}}
    </style>
    <div class="lp-wrap">
      <div class="lp-head">Live Probability · {_e(top)} {result['confidence']:.1f}%</div>
      <div class="lp-sec">Standard Inference</div>
      {bars}
      {mc_rows}
    </div>
    <script>
    requestAnimationFrame(()=>{{
      document.querySelectorAll('.lp-fill').forEach(el=>{{
        const t=parseFloat(el.getAttribute('data-target'));
        requestAnimationFrame(()=>{{el.style.width=t+'%';}});
      }});
    }});
    </script>""", height=h)


def render_activity_feed() -> None:
    log  = st.session_state.activity_log[-14:][::-1]
    if not log:
        st.markdown('<p style="color:var(--muted);font-size:.8rem">No activity yet.</p>',
                    unsafe_allow_html=True)
        return
    rows = "".join(
        f'<div class="nl-feed-row {e["level"] if e["level"] != "info" else ""}">'
        f'<span class="nl-feed-time">{_e(e["timestamp"])}</span>'
        f'<span class="nl-feed-msg">{_e(e["message"])}</span></div>'
        for e in log
    )
    st.markdown(f'<div class="nl-feed">{rows}</div>', unsafe_allow_html=True)


def render_footer() -> None:
    year = datetime.now().year
    st.markdown(
        f'<div class="nl-footer">'
        f'<span>© {year} <strong>NeuroLens AI</strong> · Developed by MD. Atique Shahriar &amp; Aronna Das</span>'
        f'<span>PyTorch {torch.__version__.split("+")[0]} · Streamlit {st.__version__} · '
        f'{"CUDA" if DEVICE.type == "cuda" else "CPU"}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )


# =============================================================================
# SIDEBAR
# =============================================================================

with st.sidebar:
    hist_sb     = st.session_state.prediction_history
    total_scans = len(hist_sb)
    engine_ok   = MODEL_PATH.exists() and model_error is None

    dot   = "🟢" if engine_ok else "🔴"
    label = "Engine ready" if engine_ok else "Engine offline"

    st.markdown(
        f'<div class="nl-brand">'
        f'<div class="nl-brand-name">NeuroLens <span>AI</span></div>'
        f'<div class="nl-brand-sub">{dot} {label}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    nav_map = {
        "Home":         "🏠 Home",
        "MRI Analysis": "🔬 MRI Analysis",
        "XAI Lab":      "👁 XAI Lab",
        "Dashboard":    "📊 Dashboard",
        "History":      "🕘 History",
        "Settings":     "⚙️ Settings",
    }
    for page, label_txt in nav_map.items():
        active = st.session_state.nav == page
        if st.button(
            label_txt + (f"  ({total_scans})" if page in ("MRI Analysis", "History") and total_scans else ""),
            key=f"nav_{page.lower().replace(' ','_')}",
            type="primary" if active else "secondary",
            **_W(),
        ):
            navigate_to(page)

    st.divider()

    if hist_sb:
        avg_c = float(np.mean([h["confidence"] for h in hist_sb]))
        st.caption(f"Session · {total_scans} scans · avg {avg_c:.1f}%")
    else:
        st.caption("No scans this session")

    if st.button("Clear History", key="sb_clear", **_W()):
        st.session_state.sb_clear_confirm = True
    if st.session_state.get("sb_clear_confirm"):
        st.warning("Clear all session data?")
        c1, c2 = st.columns(2)
        if c1.button("Yes", key="sb_clear_yes", **_W()):
            clear_prediction_history()
            st.session_state.sb_clear_confirm = False
            st.rerun()
        if c2.button("No", key="sb_clear_no", **_W()):
            st.session_state.sb_clear_confirm = False


# =============================================================================
# ROUTER
# =============================================================================

nav = st.session_state.nav
if nav not in PAGE_LABELS:
    nav = "Home"
    st.session_state.nav = "Home"


# ═══════════════════════════════════════════════════════════════
# HOME
# ═══════════════════════════════════════════════════════════════

if nav == "Home":
    render_ticker()

    st.markdown("## NeuroLens AI")
    st.markdown(
        "<p style='color:var(--muted);max-width:60ch;margin-bottom:1.2rem'>"
        "Upload a brain MRI scan to receive an AI classification across four tumor types, "
        "with Grad-CAM explainability and MC Dropout uncertainty quantification.</p>",
        unsafe_allow_html=True,
    )

    _, c, _ = st.columns([2, 1, 2])
    with c:
        if st.button("Run MRI Analysis", type="primary", key="home_cta", **_W()):
            navigate_to("MRI Analysis")

    st.write("")

    # Metrics
    hist = st.session_state.prediction_history
    m1, m2, m3, m4 = st.columns(4)
    avg_lat = f"{np.mean([x.get('latency_ms',0) for x in hist]):.0f} ms" if hist else "—"
    for col, (val, lbl) in zip([m1, m2, m3, m4], [
        (st.session_state.live_predictions_count,          "Scans"),
        (f"{st.session_state.live_avg_confidence:.1f}%",  "Avg Confidence"),
        (f"{st.session_state.live_throughput:.2f}/min",   "Throughput"),
        (avg_lat,                                          "Avg Inference"),
    ]):
        with col:
            st.markdown(
                f'<div class="nl-metric">'
                f'<div class="nl-metric-value">{_e(val)}</div>'
                f'<div class="nl-metric-label">{lbl}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.write("")

    # Feature cards
    c1, c2, c3, c4 = st.columns(4)
    for col, (title, body) in zip([c1, c2, c3, c4], [
        ("Brain MRI Classification",  "Classifies into Glioma, Meningioma, No Tumor, or Pituitary tumor."),
        ("MC Dropout Uncertainty",     "Bayesian uncertainty via repeated stochastic forward passes."),
        ("Dual XAI (Grad-CAM / ++)",   "Grad-CAM and Grad-CAM++ heatmaps with Pearson agreement score."),
        ("Real-Time Inference",        "Sub-100 ms on GPU; CPU fallback always available."),
    ]):
        with col:
            st.markdown(
                f'<div class="nl-card"><h3>{title}</h3><p>{body}</p></div>',
                unsafe_allow_html=True,
            )

    st.write("")
    st.markdown("#### Analysis Pipeline")
    pcols = st.columns(7)
    for col, (num, title) in zip(pcols, [
        ("1", "Upload"), ("2", "Preprocess"), ("3", "Inference"),
        ("4", "MC Dropout"), ("5", "Grad-CAM"), ("6", "Grad-CAM++"), ("7", "Report"),
    ]):
        with col:
            st.markdown(
                f'<div class="nl-step"><div class="nl-step-num">{num}</div>'
                f'<div class="nl-step-title">{title}</div></div>',
                unsafe_allow_html=True,
            )

    st.write("")

    # Engine status
    if model_error:
        st.markdown(
            f'<div class="nl-engine err"><div class="nl-engine-title">Neural Engine Unavailable</div>'
            f'<p>Expected checkpoint: <code>{_e(MODEL_PATH)}</code></p></div>',
            unsafe_allow_html=True,
        )
        with st.expander("Technical Details"):
            st.code(model_error)
    else:
        specs = [
            ("Architecture", model_name), ("Device", str(DEVICE)),
            ("Classes", ", ".join(class_names)),
            ("MC Passes", str(st.session_state.get("mc_samples", MC_SAMPLES_DEF))),
        ]
        spec_html = "".join(
            f'<div class="nl-spec-item"><div class="nl-spec-key">{_e(k)}</div>'
            f'<div class="nl-spec-value">{_e(v)}</div></div>'
            for k, v in specs
        )
        st.markdown(
            f'<div class="nl-engine ok">'
            f'<div class="nl-engine-title">Neural Engine Ready</div>'
            f'<div class="nl-spec-grid">{spec_html}</div></div>',
            unsafe_allow_html=True,
        )

    if st.session_state.last_result:
        st.write("")
        st.markdown("#### Last Analysis")
        render_prob_bars(st.session_state.last_result, mc_result=st.session_state.mc_result)

    render_footer()


# ═══════════════════════════════════════════════════════════════
# MRI ANALYSIS
# ═══════════════════════════════════════════════════════════════

elif nav == "MRI Analysis":
    st.markdown("## MRI Analysis")
    render_ticker()

    st.markdown(
        '<div class="nl-disclaimer">'
        '<b>Research use only.</b> This tool is not a medical device and must not be used '
        'for clinical diagnosis or treatment decisions.</div>',
        unsafe_allow_html=True,
    )

    if model_error:
        st.error(f"Neural engine unavailable: {model_error}")
        st.stop()

    col_up, col_res = st.columns([1, 1.3])

    with col_up:
        st.markdown("#### Upload Scan")
        uploaded = st.file_uploader(
            "Brain MRI image (JPEG / PNG / TIFF)",
            type=["jpg","jpeg","png","tif","tiff","bmp","webp"],
            key="mri_uploader",
        )

        run_mc    = st.checkbox("MC Dropout Uncertainty", value=True)
        run_xcam  = st.checkbox("Dual Grad-CAM XAI",     value=True)

    image_id = None
    image    = None

    if uploaded:
        raw = uploaded.read()
        if len(raw) > MAX_UPLOAD_BYTES:
            st.error("File exceeds 200 MB limit.")
        else:
            try:
                image    = Image.open(io.BytesIO(raw)).convert("RGB")
                image_id = hashlib.md5(raw).hexdigest()[:12]
                with col_up:
                    st.image(image, caption=f"{uploaded.name} ({image.size[0]}×{image.size[1]})", **_W())
                    if st.button("Analyze", type="primary", key="run_analysis", **_W()):
                        with col_res:
                            with st.status("Running analysis…", expanded=True) as status:
                                try:
                                    t_start = time.perf_counter()
                                    st.write("Running inference…")
                                    pred, conf, probs, pre_ms, inf_ms = predict_image(
                                        image, model, class_names, DEVICE
                                    )

                                    mc_res = None
                                    if run_mc:
                                        st.write("MC Dropout…")
                                        mc_res = mc_dropout_predict(
                                            image, model, class_names, DEVICE,
                                            n=int(st.session_state.get("mc_samples", MC_SAMPLES_DEF))
                                        )

                                    agree = None
                                    gc_png = pp_png = None
                                    gc_ms = pp_ms = 0.0
                                    if run_xcam:
                                        st.write("Grad-CAM…")
                                        t_gc = time.perf_counter()
                                        try: gc_png = generate_gradcam(image, model, model_name, DEVICE)
                                        except Exception as ex: st.warning(f"Grad-CAM: {ex}")
                                        gc_ms = (time.perf_counter() - t_gc) * 1000

                                        st.write("Grad-CAM++…")
                                        t_pp = time.perf_counter()
                                        try: pp_png = generate_gradcam_pp(image, model, model_name, DEVICE)
                                        except Exception as ex: st.warning(f"Grad-CAM++: {ex}")
                                        pp_ms = (time.perf_counter() - t_pp) * 1000

                                        try: agree = explanation_agreement(image, model, model_name, DEVICE)
                                        except Exception: pass

                                    total_ms = (time.perf_counter() - t_start) * 1000

                                    result = {
                                        "image_id":             image_id,
                                        "prediction":           pred,
                                        "confidence":           conf,
                                        "probabilities":        probs,
                                        "latency_ms":           inf_ms,
                                        "preprocessing_ms":     pre_ms,
                                        "gradcam_ms":           gc_ms,
                                        "gradcam_pp_ms":        pp_ms,
                                        "total_ms":             total_ms,
                                        "model":                model_name,
                                        "uncertainty":          mc_res["uncertainty"] if mc_res else None,
                                        "mc_band":              mc_res["band"]        if mc_res else None,
                                        "gradcam_available":    gc_png is not None,
                                        "gradcam_pp_available": pp_png is not None,
                                        "agreement_score":      agree,
                                        "timestamp":            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                    }

                                    st.session_state.last_result    = result
                                    st.session_state.mc_result      = mc_res
                                    st.session_state.gradcam_png    = gc_png
                                    st.session_state.gradcam_pp_png = pp_png

                                    hist = st.session_state.prediction_history
                                    hist.append(result)
                                    if len(hist) > MAX_HISTORY: hist.pop(0)

                                    update_live_stats(result, inf_ms)
                                    log_activity(f"Analyzed: {pred} ({conf:.1f}%)", "ok")

                                    status.update(label="Analysis complete", state="complete", expanded=False)

                                except Exception as exc:
                                    err = str(exc).lower()
                                    msg = ("CUDA OOM — enable Force CPU in Settings." if "out of memory" in err
                                           else f"Analysis failed: {exc}")
                                    status.error(msg)
                                    log_activity(f"Failed: {exc}", "err")

            except (UnidentifiedImageError, Exception) as exc:
                st.error(f"Could not open image: {exc}")

    # ── Results ──────────────────────────────────────────────────
    if (image_id and st.session_state.last_result
            and st.session_state.last_result.get("image_id") == image_id):

        result = st.session_state.last_result
        mc_res = st.session_state.mc_result
        gc_png = st.session_state.get("gradcam_png")
        pp_png = st.session_state.get("gradcam_pp_png")
        conf_v = result["confidence"]
        bc     = conf_badge_class(conf_v)

        with col_res:
            # Primary result
            st.markdown(
                f'<div class="nl-result">'
                f'<div class="nl-result-label">AI Classification</div>'
                f'<div class="nl-result-prediction">{_e(result["prediction"])}</div>'
                f'<div class="nl-result-conf">{conf_v:.2f}% confidence &nbsp;'
                f'<span class="nl-badge nl-badge-{bc}">'
                f'{"High" if bc=="hi" else "Moderate" if bc=="mid" else "Low"}</span></div>'
                f'</div>',
                unsafe_allow_html=True,
            )

            # Timing metrics
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Confidence",    f"{conf_v:.2f}%")
            m2.metric("Inference",     f"{result['latency_ms']:.1f} ms")
            m3.metric("Preprocess",    f"{result.get('preprocessing_ms',0):.1f} ms")
            m4.metric("Total",         f"{result.get('total_ms',0):.1f} ms")

            # MC Uncertainty
            if mc_res:
                unc, band, color = mc_res["uncertainty"], mc_res["band"], mc_res["color"]
                st.markdown(
                    f'<div class="nl-unc">'
                    f'<div><div class="nl-unc-title">MC Dropout · {st.session_state.get("mc_samples", MC_SAMPLES_DEF)} passes</div>'
                    f'<div class="nl-unc-value" style="color:{color}">σ = {unc:.4f}</div>'
                    f'<div class="nl-unc-band" style="color:{color}">{_e(band)}</div></div>'
                    f'<div style="text-align:right;font-size:.78rem;color:var(--muted)">'
                    f'MC conf: <strong style="color:#22d3ee">{mc_res["confidence"]:.2f}%</strong><br>'
                    f'Prediction: <strong>{_e(mc_res["prediction"])}</strong></div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            # Agreement
            agree = result.get("agreement_score")
            if agree is not None:
                ac = "#34d399" if agree >= 0.7 else "#fbbf24" if agree >= 0.5 else "#f87171"
                al = "High Agreement" if agree >= 0.7 else "Moderate Agreement" if agree >= 0.5 else "Low Agreement"
                st.markdown(
                    f'<div class="nl-agree">'
                    f'<span style="font-size:.78rem;color:var(--muted)">Grad-CAM ↔ Grad-CAM++ Pearson correlation</span>'
                    f'<div style="text-align:right">'
                    f'<div class="nl-agree-score" style="color:{ac}">{agree:.3f}</div>'
                    f'<div style="font-size:.7rem;color:{ac};font-weight:600">{_e(al)}</div></div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            # Probability grid
            st.markdown("#### Probability Distribution")
            grid_html = '<div class="nl-prob-grid">' + "".join(
                f'<div class="nl-prob-card {"top" if lbl == result["prediction"] else ""}">'
                f'<div class="nl-prob-value">{p:.1f}%</div>'
                f'<div class="nl-prob-label">{_e(lbl)}</div>'
                f'{"<div class=\"nl-prob-top-tag\">Top</div>" if lbl == result["prediction"] else ""}'
                f'</div>'
                for lbl, p in result["probabilities"].items()
            ) + '</div>'
            st.markdown(grid_html, unsafe_allow_html=True)

            # Probability bars
            st.markdown("#### Live Probability Stream")
            render_prob_bars(result, mc_result=mc_res)

            # XAI heatmaps
            if gc_png or pp_png:
                st.markdown("#### Dual XAI Visualization")
                gc_col, pp_col = st.columns(2)
                with gc_col:
                    st.markdown("**Grad-CAM** (jet · α=0.44)")
                    if gc_png: st.image(gc_png, **_W())
                with pp_col:
                    st.markdown("**Grad-CAM++** (inferno · α=0.46)")
                    if pp_png: st.image(pp_png, **_W())

            # Downloads
            st.markdown("#### Export")
            dl1, dl2, dl3 = st.columns(3)
            if gc_png:
                with dl1:
                    st.download_button("Download Grad-CAM", gc_png, "gradcam.png", "image/png",
                                       key="dl_gc", **_W())
            if pp_png:
                with dl2:
                    st.download_button("Download Grad-CAM++", pp_png, "gradcam_pp.png", "image/png",
                                       key="dl_pp", **_W())
            report_lines = [
                "NeuroLens AI — MRI Analysis Report",
                "=" * 52,
                f"Timestamp         : {result['timestamp']}",
                f"Architecture      : {result['model']}",
                f"Device            : {DEVICE}",
                "-" * 52,
                f"Prediction        : {result['prediction']}",
                f"Confidence        : {result['confidence']:.4f}%",
                "Probabilities:",
                *[f"  {k:<14}: {v:.4f}%" for k, v in result["probabilities"].items()],
                "-" * 52,
                f"MC Uncertainty σ  : {result.get('uncertainty', 'N/A')}",
                f"Reliability Band  : {result.get('mc_band', 'N/A')}",
                f"Agreement Score   : {f'{agree:.4f}' if agree is not None else 'N/A'}",
                "-" * 52,
                f"Preprocessing     : {result.get('preprocessing_ms', 0):.2f} ms",
                f"Inference         : {result['latency_ms']:.2f} ms",
                f"Total             : {result.get('total_ms', 0):.2f} ms",
            ]
            with dl3:
                st.download_button("Download Report (.txt)", "\n".join(report_lines),
                                   "neurolens_report.txt", "text/plain",
                                   key="dl_report", type="primary", **_W())

    render_footer()


# ═══════════════════════════════════════════════════════════════
# XAI LAB
# ═══════════════════════════════════════════════════════════════

elif nav == "XAI Lab":
    st.markdown("## XAI Lab")
    render_ticker()

    hist   = st.session_state.prediction_history
    gc_png = st.session_state.get("gradcam_png")
    pp_png = st.session_state.get("gradcam_pp_png")

    if not hist:
        st.markdown('<div class="nl-empty"><h3>No analyses yet</h3>'
                    '<p>Run an MRI analysis first to view explanations here.</p></div>',
                    unsafe_allow_html=True)
    else:
        result = st.session_state.last_result
        agree  = result.get("agreement_score") if result else None

        st.markdown("#### Heatmaps — Most Recent Analysis")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Grad-CAM** (jet colormap)")
            if gc_png:
                st.image(gc_png, **_W())
                st.download_button("Download Grad-CAM", gc_png, "gradcam.png", "image/png", key="xai_gc")
            else:
                st.info("Grad-CAM not available for this analysis.")
        with c2:
            st.markdown("**Grad-CAM++** (inferno colormap)")
            if pp_png:
                st.image(pp_png, **_W())
                st.download_button("Download Grad-CAM++", pp_png, "gradcam_pp.png", "image/png", key="xai_pp")
            else:
                st.info("Grad-CAM++ not available for this analysis.")

        if agree is not None:
            ac = "#34d399" if agree >= 0.7 else "#fbbf24" if agree >= 0.5 else "#f87171"
            al = "High" if agree >= 0.7 else "Moderate" if agree >= 0.5 else "Low"
            st.markdown(
                f'<div class="nl-agree" style="margin-top:.8rem">'
                f'<div><strong style="color:var(--text)">Explanation Agreement Score</strong><br>'
                f'<span style="font-size:.8rem;color:var(--muted)">Pearson correlation between the two heatmaps</span></div>'
                f'<div style="text-align:right">'
                f'<div class="nl-agree-score" style="color:{ac}">{agree:.3f}</div>'
                f'<div style="font-size:.72rem;color:{ac};font-weight:600">{_e(al)} Agreement</div>'
                f'</div></div>',
                unsafe_allow_html=True,
            )

        # Historical agreement trend
        agree_vals = [h["agreement_score"] for h in hist if h.get("agreement_score") is not None]
        if len(agree_vals) >= 2:
            st.markdown("#### Agreement Score Trend")
            fig, ax = _dark_fig()
            xs = list(range(1, len(agree_vals)+1))
            ax.plot(xs, agree_vals, marker="o", lw=2, ms=3.5, color="#22d3ee")
            ax.fill_between(xs, agree_vals, alpha=0.09, color="#22d3ee")
            ax.axhline(0.7, ls="--", color="#34d399", alpha=0.5, lw=1.2, label="High threshold")
            ax.axhline(0.5, ls="--", color="#fbbf24", alpha=0.5, lw=1.2, label="Moderate threshold")
            ax.legend(fontsize=7, facecolor="#0c1a30", labelcolor="#7f9bbd", framealpha=0.7)
            ax.set_xlabel("Analysis #", color="#7f9bbd", fontsize=8)
            ax.set_ylabel("Agreement", color="#7f9bbd", fontsize=8)
            ax.grid(True, alpha=0.08, ls="--", color="#7f9bbd")
            ax.set_ylim(-0.05, 1.05)
            fig.tight_layout(pad=0.5)
            st.pyplot(fig, **_W())
            plt.close(fig)

    render_footer()


# ═══════════════════════════════════════════════════════════════
# DASHBOARD
# ═══════════════════════════════════════════════════════════════

elif nav == "Dashboard":
    st.markdown("## Dashboard")
    render_ticker()

    dash_hist = st.session_state.prediction_history
    if not dash_hist:
        st.markdown('<div class="nl-empty"><h3>No data yet</h3>'
                    '<p>Run at least one MRI analysis to populate the dashboard.</p></div>',
                    unsafe_allow_html=True)
    else:
        total    = st.session_state.live_predictions_count
        avg_c    = st.session_state.live_avg_confidence
        unique   = len([k for k, v in st.session_state.live_class_counts.items() if v > 0])
        avg_lat  = float(np.mean([h.get("latency_ms", 0) for h in dash_hist]))
        unc_data = [h["uncertainty"] for h in dash_hist if h.get("uncertainty") is not None]
        avg_unc  = float(np.mean(unc_data)) if unc_data else None

        m1, m2, m3, m4, m5 = st.columns(5)
        for col, (val, lbl) in zip([m1, m2, m3, m4, m5], [
            (total,                                            "Total Scans"),
            (f"{avg_c:.1f}%",                                 "Avg Confidence"),
            (unique,                                           "Classes Seen"),
            (f"{avg_lat:.0f} ms",                             "Avg Latency"),
            (f"σ={avg_unc:.4f}" if avg_unc is not None else "—", "Avg Uncertainty"),
        ]):
            with col:
                st.markdown(
                    f'<div class="nl-metric">'
                    f'<div class="nl-metric-value">{_e(val)}</div>'
                    f'<div class="nl-metric-label">{lbl}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        st.write("")
        col_l, col_r = st.columns([2, 1])

        with col_l:
            st.markdown("#### Prediction Distribution")
            counts = st.session_state.live_class_counts
            mx = max(counts.values()) if max(counts.values()) > 0 else 1
            for label, count in counts.items():
                pct = (count / mx) * 100
                live_pct = (count / total) * 100 if total else 0
                st.markdown(
                    f'<div class="nl-dist">'
                    f'<div class="nl-dist-row">'
                    f'<div class="nl-dist-name">{_e(label)}</div>'
                    f'<div class="nl-dist-count">{count} · {live_pct:.1f}%</div></div>'
                    f'<div class="nl-dist-track"><div class="nl-dist-fill" style="width:{pct:.0f}%"></div></div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            st.markdown("#### Confidence Trend")
            fig = plot_confidence_trend(dash_hist)
            if fig: st.pyplot(fig, **_W()); plt.close(fig)
            else:   st.caption("Need ≥ 2 analyses.")

            st.markdown("#### Inference Latency")
            fig = plot_latency_trend(dash_hist)
            if fig: st.pyplot(fig, **_W()); plt.close(fig)
            else:   st.caption("Need ≥ 2 analyses.")

            if unc_data:
                st.markdown("#### Uncertainty Trend")
                fig = plot_uncertainty_history(dash_hist)
                if fig: st.pyplot(fig, **_W()); plt.close(fig)

        with col_r:
            st.markdown("#### Activity Feed")
            render_activity_feed()

            st.write("")
            st.markdown("#### Latest Result")
            latest = dash_hist[-1]
            rows = [
                ("Prediction",   latest["prediction"]),
                ("Confidence",   f"{latest['confidence']:.2f}%"),
                ("Architecture", latest["model"]),
                ("Inference",    f"{latest.get('latency_ms',0):.0f} ms"),
                ("Uncertainty σ",f"{latest['uncertainty']:.4f}" if latest.get("uncertainty") is not None else "—"),
                ("Agreement",    f"{latest['agreement_score']:.3f}" if latest.get("agreement_score") is not None else "—"),
                ("Timestamp",    latest["timestamp"]),
            ]
            st.markdown(
                '<div class="nl-card">' + "".join(
                    f'<div class="nl-summary-row">'
                    f'<span class="nl-summary-key">{_e(k)}</span>'
                    f'<span class="nl-summary-val">{_e(v)}</span></div>'
                    for k, v in rows
                ) + '</div>',
                unsafe_allow_html=True,
            )

    render_footer()


# ═══════════════════════════════════════════════════════════════
# HISTORY
# ═══════════════════════════════════════════════════════════════

elif nav == "History":
    st.markdown("## History")
    render_ticker()

    hist_list = st.session_state.prediction_history
    if not hist_list:
        st.markdown('<div class="nl-empty"><h3>No history</h3>'
                    '<p>Analyses you run will appear here.</p></div>',
                    unsafe_allow_html=True)
    else:
        # Filters
        fc1, fc2, fc3, fc4 = st.columns([1, 1, 1, 2])
        pred_f   = fc1.selectbox("Prediction", ["All"] + CLASS_NAMES, key="hist_pred")
        conf_f   = fc2.selectbox("Confidence", ["All","High (≥80%)","Moderate (60–79%)","Low (<60%)"], key="hist_conf")
        sort_ord = fc3.selectbox("Sort", ["Newest first","Oldest first"], key="hist_sort")
        search_q = fc4.text_input("Search (model / class)", key="hist_q")

        filtered = []
        for rec in hist_list:
            c = rec["confidence"]
            if pred_f != "All" and rec["prediction"] != pred_f: continue
            if conf_f == "High (≥80%)"       and c < 80:         continue
            if conf_f == "Moderate (60–79%)" and not (60<=c<80): continue
            if conf_f == "Low (<60%)"        and c >= 60:         continue
            q = search_q.strip().lower()
            if q and q not in rec["model"].lower() and q not in rec["prediction"].lower(): continue
            filtered.append(rec)

        if sort_ord == "Newest first":
            filtered = list(reversed(filtered))

        if not filtered:
            st.info("No records match the current filters.")
        else:
            st.caption(f"Showing {len(filtered)} of {len(hist_list)} records")
            for item in filtered:
                conf = item["confidence"]
                bc   = conf_badge_class(conf)
                unc  = f" · σ={item['uncertainty']:.4f}" if item.get("uncertainty") is not None else ""
                ag   = f" · Agr={item['agreement_score']:.3f}" if item.get("agreement_score") is not None else ""

                st.markdown(
                    f'<div class="nl-hist-row">'
                    f'<div style="flex:1">'
                    f'<div class="nl-hist-title">{_e(item["prediction"])}</div>'
                    f'<div class="nl-hist-meta">{_e(item["timestamp"])} · {_e(item["model"])}{unc}{ag}</div>'
                    f'</div>'
                    f'<div><span class="nl-badge nl-badge-{bc}">{conf:.1f}%</span></div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                with st.expander(f"Details — {item['timestamp']}"):
                    d1, d2 = st.columns(2)
                    with d1:
                        st.write(f"**Prediction:** {item['prediction']}")
                        st.write(f"**Confidence:** {item['confidence']:.4f}%")
                        st.write(f"**Architecture:** {item['model']}")
                        st.write(f"**Inference:** {item.get('latency_ms',0):.1f} ms")
                    with d2:
                        if item.get("uncertainty") is not None:
                            st.write(f"**MC Uncertainty σ:** {item['uncertainty']:.6f}")
                            st.write(f"**Reliability:** {item.get('mc_band','—')}")
                        if item.get("agreement_score") is not None:
                            st.write(f"**Agreement Score:** {item['agreement_score']:.4f}")
                        st.write("**Probabilities:**")
                        for cls, prob in item.get("probabilities", {}).items():
                            st.write(f"  · {cls}: {prob:.4f}%")

        # CSV export
        st.write("")
        df_rows = []
        for rec in hist_list:
            row = {k: rec.get(k) for k in
                   ["timestamp","prediction","confidence","model","latency_ms",
                    "preprocessing_ms","total_ms","uncertainty","mc_band","agreement_score"]}
            row.update({f"prob_{k}": v for k, v in rec.get("probabilities", {}).items()})
            df_rows.append(row)
        df = pd.DataFrame(df_rows)
        st.download_button("Export as CSV", df.to_csv(index=False),
                           "neurolens_history.csv", "text/csv",
                           key="hist_csv", **_W())

    render_footer()


# ═══════════════════════════════════════════════════════════════
# SETTINGS
# ═══════════════════════════════════════════════════════════════

elif nav == "Settings":
    st.markdown("## Settings")

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("#### Model")
        info_rows = [
            ("Status",        "Ready" if not model_error else "Failed"),
            ("Architecture",  model_name or "—"),
            ("Device",        str(DEVICE)),
            ("Input",         f"{IMG_SIZE}×{IMG_SIZE}"),
            ("Classes",       ", ".join(CLASS_NAMES)),
            ("Checkpoint",    MODEL_PATH.name if MODEL_PATH.exists() else "Not found"),
            ("MC Passes",     str(st.session_state.get("mc_samples", MC_SAMPLES_DEF))),
        ]
        st.markdown(
            '<div class="nl-card">' + "".join(
                f'<div class="nl-info-row">'
                f'<span class="nl-info-key">{_e(k)}</span>'
                f'<span class="nl-info-val">{_e(v)}</span></div>'
                for k, v in info_rows
            ) + '</div>',
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown("#### System")
        try:
            tv = metadata.version("torchvision")
        except Exception:
            tv = "—"
        sys_rows = [
            ("PyTorch",      torch.__version__),
            ("Torchvision",  tv),
            ("Streamlit",    st.__version__),
            ("Python",       platform.python_version()),
            ("CUDA",         str(torch.cuda.is_available())),
            ("CUDA Device",  torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"),
            ("XAI",          "Grad-CAM · Grad-CAM++"),
        ]
        st.markdown(
            '<div class="nl-card">' + "".join(
                f'<div class="nl-info-row">'
                f'<span class="nl-info-key">{_e(k)}</span>'
                f'<span class="nl-info-val">{_e(v)}</span></div>'
                for k, v in sys_rows
            ) + '</div>',
            unsafe_allow_html=True,
        )

    st.write("")
    st.markdown("#### Runtime Controls")

    rc1, rc2 = st.columns(2)
    with rc1:
        prev_cpu = bool(st.session_state.get("force_cpu", False))
        def _on_cpu():
            try: load_model.clear()
            except Exception: pass
        st.toggle("Force CPU Inference", key="force_cpu", on_change=_on_cpu,
                  help="Reloads the engine on CPU.")
        if st.session_state.force_cpu != prev_cpu:
            st.info("Device changed — reloading engine…")
            st.rerun()
        if st.session_state.force_cpu and DEVICE.type == "cpu":
            st.success("Running on CPU (forced).")
        elif not st.session_state.force_cpu and DEVICE.type == "cuda":
            st.success("Running on CUDA.")
        else:
            st.warning("CUDA unavailable; running on CPU.")

    with rc2:
        mc_val = st.number_input(
            "MC Dropout passes",
            min_value=5, max_value=200, step=5,
            value=int(st.session_state.get("mc_samples", MC_SAMPLES_DEF)),
            key="mc_samples_widget",
            help="More passes → better uncertainty estimate, slower runtime.",
        )
        if mc_val != st.session_state.get("mc_samples"):
            st.session_state.mc_samples = int(mc_val)

    st.write("")
    st.markdown("#### Technical Details")
    with st.expander("Checkpoint & Runtime"):
        st.code(
            f"Checkpoint : {MODEL_PATH}\n"
            f"Device     : {DEVICE}  ·  Input: {IMG_SIZE}×{IMG_SIZE}\n"
            f"Norm mean  : {NORM_MEAN}\n"
            f"Norm std   : {NORM_STD}\n"
            f"Force CPU  : {st.session_state.force_cpu}  ·  MC passes: {st.session_state.mc_samples}"
            + (f"\n\nError:\n{model_error}" if model_error else "")
        )

    st.write("")
    st.markdown("#### Reset Session")
    if st.button("Reset All Session Data", type="primary", key="settings_reset"):
        st.session_state.settings_confirm_reset = True
    if st.session_state.get("settings_confirm_reset", False):
        st.warning("This will clear all analyses, history, and live stats.")
        r1, r2, _ = st.columns([1, 1, 4])
        if r1.button("Confirm", key="settings_confirm_yes", **_W()):
            for k, v in DEFAULTS.items():
                st.session_state[k] = copy.deepcopy(v)
            st.session_state.settings_confirm_reset = False
            st.success("Session cleared.")
            navigate_to("Home")
        if r2.button("Cancel", key="settings_confirm_no", **_W()):
            st.session_state.settings_confirm_reset = False

    render_footer()
