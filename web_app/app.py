import warnings
import time
import io
import copy
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
from torchvision.models import resnet50, efficientnet_b0, mobilenet_v3_small

warnings.filterwarnings("ignore")


st.set_page_config(
    page_title="NeuroLens AI · Brain MRI Intelligence",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


NORM_MEAN      = [0.485, 0.456, 0.406]
NORM_STD       = [0.229, 0.224, 0.225]
IMG_SIZE       = 224
MC_SAMPLES_DEF = 20
MAX_HISTORY    = 200
CLASS_NAMES    = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]
MAX_UPLOAD_BYTES = 200 * 1024 * 1024

try:
    Image.MAX_IMAGE_PIXELS = 50_000_000
except Exception:
    pass

BASE_DIR   = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "neurolens_best.pth"
if not MODEL_PATH.exists():
    for _c in [BASE_DIR.parent / "neurolens_best.pth"]:
        if _c.exists():
            MODEL_PATH = _c
            break

DEFAULT_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

test_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(NORM_MEAN, NORM_STD),
])

CLASS_INFO = {
    "Glioma": {
        "color": "#f87171",
        "desc": "Malignant tumor arising from glial cells. Most common primary brain tumor.",
        "severity": "High",
    },
    "Meningioma": {
        "color": "#fbbf24",
        "desc": "Tumor arising from meninges. Usually benign but can cause pressure symptoms.",
        "severity": "Moderate",
    },
    "No Tumor": {
        "color": "#34d399",
        "desc": "No detectable tumor mass in the MRI scan.",
        "severity": "None",
    },
    "Pituitary": {
        "color": "#a78bfa",
        "desc": "Tumor in the pituitary gland. Usually benign; may affect hormonal function.",
        "severity": "Moderate",
    },
}


st.markdown(
    '<link rel="preconnect" href="https://fonts.googleapis.com">'
    '<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800'
    '&family=JetBrains+Mono:wght@400;500;600&display=swap" rel="stylesheet">',
    unsafe_allow_html=True,
)

st.markdown("""
<style>
:root {
  --bg:       #08121f;
  --bg2:      #0a1828;
  --surf:     #0f1e33;
  --surf2:    #142543;
  --border:   rgba(125,165,220,.14);
  --border2:  rgba(125,165,220,.24);
  --cyan:     #22d3ee;
  --emerald:  #34d399;
  --amber:    #fbbf24;
  --danger:   #f87171;
  --violet:   #a78bfa;
  --text:     #e8eefb;
  --text2:    #b9cbe2;
  --muted:    #7f9bbd;
  --mono:     'JetBrains Mono', monospace;
  --sans:     'Inter', system-ui, sans-serif;
  --r:        12px;
  --r-lg:     16px;
  --ease:     cubic-bezier(.22,.61,.36,1);
}

#MainMenu, footer, header,
[data-testid="stToolbar"],
[data-testid="stDecoration"],
[data-testid="stStatusWidget"],
[data-testid="stHeader"],
.stDeployButton, .stAppDeployButton {
  display: none !important;
  height: 0 !important;
}

html, body,
[data-testid="stAppViewContainer"],
[data-testid="stApp"] {
  background: var(--bg) !important;
  font-family: var(--sans);
  color: var(--text);
  -webkit-font-smoothing: antialiased;
}
[data-testid="stAppViewContainer"] > .main {
  background:
    radial-gradient(1100px 600px at 50% -15%, rgba(34,211,238,.09), transparent 55%),
    radial-gradient(900px 500px at 95% 10%, rgba(167,139,250,.07), transparent 55%),
    linear-gradient(180deg, #0a1a30 0%, #050c18 60%, #040911 100%) !important;
}
[data-testid="stMainBlockContainer"] {
  max-width: 1340px !important;
  padding: 1rem clamp(.8rem,2vw,1.6rem) 0 !important;
}
[data-testid="stAppViewContainer"] > .main > div:first-child {
  padding-top: 0 !important;
}

[data-testid="stSidebar"] {
  background: linear-gradient(180deg,#0b1e3d 0%,#060f1f 100%) !important;
  border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] > div:first-child { padding:.85rem .8rem 1.2rem !important; }
[data-testid="stSidebar"] hr { border-color:var(--border) !important; margin:1rem 0 !important; }
[data-testid="stSidebar"] .stButton > button {
  justify-content:flex-start !important; text-align:left !important;
  border-radius:10px !important; padding:.6rem .85rem !important;
  font-size:.875rem !important; font-weight:500 !important;
  min-height:42px !important;
  transition: background .18s var(--ease), border-color .18s var(--ease), color .18s var(--ease) !important;
}
[data-testid="stSidebar"] .stButton > button[kind="secondary"] {
  background:transparent !important; border:1px solid transparent !important; color:var(--text2) !important;
}
[data-testid="stSidebar"] .stButton > button[kind="secondary"]:hover {
  background:rgba(34,211,238,.07) !important; border-color:rgba(34,211,238,.22) !important; color:var(--text) !important;
}
[data-testid="stSidebar"] .stButton > button[kind="primary"] {
  background:linear-gradient(135deg,rgba(34,211,238,.17),rgba(52,211,153,.10)) !important;
  border:1px solid rgba(34,211,238,.42) !important; color:var(--text) !important;
  position:relative;
}
[data-testid="stSidebar"] .stButton > button[kind="primary"]::before {
  content:""; position:absolute; left:-1px; top:20%; bottom:20%;
  width:3px; border-radius:3px;
  background:linear-gradient(180deg,var(--cyan),var(--emerald));
}

.stButton > button, .stDownloadButton > button {
  border-radius:10px !important; font-weight:600 !important; font-size:.875rem !important;
  transition: transform .12s var(--ease), box-shadow .18s var(--ease) !important;
}
.stButton > button[kind="primary"], .stDownloadButton > button[kind="primary"] {
  background:linear-gradient(135deg,#0ea5e9,#06b6d4) !important;
  border:1px solid rgba(255,255,255,.10) !important; color:#031422 !important;
  box-shadow:0 6px 18px -6px rgba(34,211,238,.55) !important;
}
.stButton > button[kind="primary"]:hover { transform:translateY(-1px); }

h1 { font-size:clamp(1.35rem,2.6vw,1.75rem) !important; font-weight:800 !important; letter-spacing:-.022em !important; color:var(--text) !important; }
h2 { font-size:clamp(1.15rem,2.2vw,1.4rem) !important; font-weight:700 !important; color:var(--text) !important; }
h3 { font-size:clamp(.95rem,1.6vw,1.05rem) !important; font-weight:700 !important; color:var(--text) !important; }

.nl-page { animation: fadeUp .35s ease-out both; }
@keyframes fadeUp { from{opacity:0;transform:translateY(8px)} to{opacity:1;transform:translateY(0)} }

.nl-hero {
  position:relative; padding:2.4rem 2rem 2rem; margin:0 0 1.5rem;
  border-radius:var(--r-lg); border:1px solid var(--border2);
  background:
    radial-gradient(900px 300px at 15% 0%,rgba(34,211,238,.18),transparent 60%),
    radial-gradient(700px 250px at 95% 100%,rgba(167,139,250,.13),transparent 60%),
    linear-gradient(180deg,rgba(20,37,67,.85),rgba(12,22,40,.85));
  overflow:hidden;
}
.nl-hero::before {
  content:""; position:absolute; inset:0;
  background-image: linear-gradient(rgba(125,165,220,.04) 1px,transparent 1px),
                    linear-gradient(90deg,rgba(125,165,220,.04) 1px,transparent 1px);
  background-size:44px 44px;
  -webkit-mask-image:radial-gradient(ellipse at center,#000 20%,transparent 75%);
  pointer-events:none;
}
.nl-hero-inner { position:relative; z-index:1; }
.nl-badge {
  display:inline-flex; align-items:center; gap:.45rem;
  padding:.3rem .75rem; border-radius:999px;
  background:rgba(34,211,238,.10); border:1px solid rgba(34,211,238,.35);
  color:var(--cyan); font-size:.72rem; font-weight:700;
  font-family:var(--mono); letter-spacing:.04em; margin-bottom:1rem;
}
.nl-badge .dot {
  width:6px;height:6px;border-radius:50%;background:var(--emerald);
  box-shadow:0 0 0 3px rgba(52,211,153,.2);
  animation:pulse 2.4s ease infinite;
}
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.5} }
.nl-hero-title {
  font-size:clamp(2rem,4.4vw,3rem) !important; font-weight:800 !important;
  line-height:1.06 !important; letter-spacing:-.035em !important;
  color:var(--text) !important; margin:0 0 .8rem !important; max-width:20ch;
}
.nl-hero-title .g {
  background:linear-gradient(135deg,var(--cyan),var(--emerald));
  -webkit-background-clip:text; background-clip:text; -webkit-text-fill-color:transparent;
}
.nl-hero-sub { font-size:.98rem; color:var(--text2); line-height:1.65; max-width:58ch; margin:0 0 1.3rem; }
.nl-stats { display:flex; gap:2rem; flex-wrap:wrap; padding-top:1.2rem; border-top:1px solid var(--border); }
.nl-stat-v { font-size:1.3rem; font-weight:800; color:var(--text); font-family:var(--mono); }
.nl-stat-l { font-size:.67rem; font-weight:700; color:var(--muted); text-transform:uppercase; letter-spacing:.10em; }

.nl-card {
  background:linear-gradient(180deg,rgba(20,37,67,.85),rgba(15,30,51,.85));
  border:1px solid var(--border); border-radius:var(--r); padding:1.05rem 1.15rem;
  box-shadow:0 1px 2px rgba(0,0,0,.35), inset 0 1px 0 rgba(255,255,255,.04);
  transition:border-color .2s,transform .2s,box-shadow .2s;
}
.nl-card:hover { border-color:rgba(34,211,238,.28); transform:translateY(-2px); }

.nl-metric {
  background:linear-gradient(180deg,rgba(20,37,67,.9),rgba(15,30,51,.9));
  border:1px solid var(--border); border-radius:var(--r); padding:.9rem 1rem;
  position:relative; overflow:hidden;
}
.nl-metric::after {
  content:""; position:absolute; inset:0;
  background:radial-gradient(300px 100px at 100% 0%,rgba(34,211,238,.09),transparent 70%);
  pointer-events:none;
}
.nl-metric-label { font-size:.67rem; font-weight:700; color:var(--muted); text-transform:uppercase; letter-spacing:.09em; }
.nl-metric-value { font-size:clamp(1.2rem,2vw,1.5rem); font-weight:800; color:var(--text); letter-spacing:-.022em; }

.nl-result {
  position:relative;
  background:radial-gradient(600px 120px at 0% 0%,rgba(34,211,238,.15),transparent 60%),
             linear-gradient(135deg,rgba(37,99,235,.15),rgba(16,185,129,.08));
  border:1px solid rgba(34,211,238,.4); border-radius:var(--r-lg);
  padding:1.3rem 1.4rem; margin:.5rem 0 .8rem;
  overflow:hidden;
}
.nl-result::before {
  content:""; position:absolute; left:0; top:0; bottom:0; width:4px;
  background:linear-gradient(180deg,var(--cyan),var(--emerald));
}
.nl-result-label { font-size:.66rem; font-weight:800; color:var(--cyan); text-transform:uppercase; letter-spacing:.12em; margin-bottom:.5rem; }
.nl-result-pred { font-size:clamp(1.5rem,3vw,2rem); font-weight:800; color:var(--text); letter-spacing:-.028em; line-height:1.05; }
.nl-result-conf { font-size:.88rem; color:var(--text2); margin-top:.5rem; }

.nl-prob-bar-wrap { margin:.3rem 0; }
.nl-prob-label { font-size:.8rem; color:var(--text2); font-weight:500; margin-bottom:.22rem; display:flex; justify-content:space-between; }
.nl-prob-track { height:8px; background:rgba(255,255,255,.06); border-radius:4px; overflow:hidden; }
.nl-prob-fill { height:100%; border-radius:4px; transition:width .6s var(--ease); }

.nl-section-head {
  display:flex; align-items:center; gap:.55rem;
  font-size:.95rem; font-weight:700; color:var(--text);
  margin:.9rem 0 .55rem;
}
.nl-section-ico {
  width:26px; height:26px; display:grid; place-items:center;
  border-radius:8px; background:rgba(34,211,238,.10);
  border:1px solid rgba(34,211,238,.28); color:var(--cyan);
}

.nl-unc-band {
  display:inline-flex; align-items:center; gap:.4rem;
  padding:.25rem .7rem; border-radius:6px; font-size:.78rem; font-weight:700;
}
.nl-unc-low    { background:rgba(52,211,153,.12); border:1px solid rgba(52,211,153,.35); color:#34d399; }
.nl-unc-medium { background:rgba(251,191,36,.12);  border:1px solid rgba(251,191,36,.35);  color:#fbbf24; }
.nl-unc-high   { background:rgba(248,113,113,.12); border:1px solid rgba(248,113,113,.35); color:#f87171; }

.nl-hist-item {
  background:linear-gradient(180deg,rgba(20,37,67,.8),rgba(15,30,51,.8));
  border:1px solid var(--border); border-radius:var(--r); padding:.8rem 1rem;
  margin-bottom:.5rem; display:flex; align-items:center; gap:1rem;
  transition:border-color .18s;
}
.nl-hist-item:hover { border-color:rgba(34,211,238,.25); }
.nl-hist-pred { font-weight:700; color:var(--text); font-size:.9rem; }
.nl-hist-meta { font-size:.75rem; color:var(--muted); }

.nl-footer {
  margin-top:2.5rem; padding:1.4rem 0 1rem;
  border-top:1px solid var(--border);
  text-align:center; color:var(--muted); font-size:.78rem;
}

.nl-disc {
  background:rgba(251,191,36,.08); border:1px solid rgba(251,191,36,.3);
  border-radius:var(--r); padding:.75rem 1rem; font-size:.82rem; color:var(--text2);
  margin-bottom:1rem;
}

[data-testid="stFileUploadDropzone"] {
  background:linear-gradient(135deg,rgba(34,211,238,.05),rgba(52,211,153,.03)) !important;
  border:2px dashed rgba(34,211,238,.4) !important;
  border-radius:var(--r-lg) !important; transition:border-color .2s !important;
}
[data-testid="stFileUploadDropzone"]:hover {
  border-color:rgba(34,211,238,.7) !important;
  background:rgba(34,211,238,.08) !important;
}
[data-testid="stExpander"] {
  background:var(--surf) !important; border:1px solid var(--border) !important;
  border-radius:var(--r) !important;
}
[data-testid="stSlider"] .stSlider { color:var(--cyan) !important; }
.stTabs [data-baseweb="tab-list"] { background:var(--surf) !important; border-radius:var(--r) !important; gap:.25rem; padding:.25rem; }
.stTabs [data-baseweb="tab"] { border-radius:8px !important; color:var(--text2) !important; font-weight:500 !important; }
.stTabs [aria-selected="true"] { background:rgba(34,211,238,.15) !important; color:var(--text) !important; }
[data-testid="stProgress"] > div > div { background:linear-gradient(90deg,var(--cyan),var(--emerald)) !important; }
</style>
""", unsafe_allow_html=True)


def _ico(name: str, size: int = 16) -> str:
    icons = {
        "brain":    '<path d="M9.5 2A2.5 2.5 0 0 1 12 4.5v15a2.5 2.5 0 0 1-4.96.44 2.5 2.5 0 0 1-2.96-3.08 3 3 0 0 1-.34-5.58 2.5 2.5 0 0 1 1.32-4.24 2.5 2.5 0 0 1 1.98-3A2.5 2.5 0 0 1 9.5 2Z"/><path d="M14.5 2A2.5 2.5 0 0 0 12 4.5v15a2.5 2.5 0 0 0 4.96.44 2.5 2.5 0 0 0 2.96-3.08 3 3 0 0 0 .34-5.58 2.5 2.5 0 0 0-1.32-4.24 2.5 2.5 0 0 0-1.98-3A2.5 2.5 0 0 0 14.5 2Z"/>',
        "scan":     '<path d="M3 7V5a2 2 0 0 1 2-2h2"/><path d="M17 3h2a2 2 0 0 1 2 2v2"/><path d="M21 17v2a2 2 0 0 1-2 2h-2"/><path d="M7 21H5a2 2 0 0 1-2-2v-2"/><path d="M7 12h10"/><circle cx="12" cy="12" r="2.4"/>',
        "eye":      '<path d="M2 12s3.5-7 10-7 10 7 10 7-3.5 7-10 7-10-7-10-7Z"/><circle cx="12" cy="12" r="3"/>',
        "chart":    '<path d="M3 3v18h18"/><path d="M18 17V9"/><path d="M13 17V5"/><path d="M8 17v-3"/>',
        "history":  '<path d="M3 12a9 9 0 1 0 3-6.7L3 8"/><path d="M3 3v5h5"/><path d="M12 7v5l3 2"/>',
        "settings": '<circle cx="12" cy="12" r="3"/><path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 1 1-4 0v-.09a1.65 1.65 0 0 0-1-1.51 1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 1 1 0-4h.09a1.65 1.65 0 0 0 1.51-1 1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06a1.65 1.65 0 0 0 1.82.33h.01a1.65 1.65 0 0 0 1-1.51V3a2 2 0 1 1 4 0v.09a1.65 1.65 0 0 0 1 1.51h.01a1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82v.01a1.65 1.65 0 0 0 1.51 1H21a2 2 0 1 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1Z"/>',
        "home":     '<path d="m3 9 9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/><polyline points="9 22 9 12 15 12 15 22"/>',
        "shield":   '<path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10Z"/>',
        "trash":    '<polyline points="3 6 5 6 21 6"/><path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6"/><path d="M10 11v6M14 11v6"/>',
        "download": '<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/>',
        "zap":      '<polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/>',
        "cpu":      '<rect x="4" y="4" width="16" height="16" rx="2"/><rect x="9" y="9" width="6" height="6"/><path d="M9 2v2M15 2v2M9 20v2M15 20v2M2 9h2M2 15h2M20 9h2M20 15h2"/>',
        "info":     '<circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/>',
        "refresh":  '<polyline points="23 4 23 10 17 10"/><polyline points="1 20 1 14 7 14"/><path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15"/>',
        "alert":    '<circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/>',
    }
    paths = icons.get(name, "")
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 24 24" fill="none" stroke="currentColor" '
        f'stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" '
        f'style="display:inline-block;vertical-align:-.18em;flex-shrink:0">'
        f'{paths}</svg>'
    )


def section_head(icon: str, label: str) -> None:
    st.markdown(
        f'<div class="nl-section-head">'
        f'<span class="nl-section-ico">{_ico(icon, 14)}</span>{label}</div>',
        unsafe_allow_html=True,
    )


def metric_card(label: str, value: str, accent: str = "#22d3ee") -> str:
    return (
        f'<div class="nl-metric">'
        f'<div class="nl-metric-label">{label}</div>'
        f'<div class="nl-metric-value" style="color:{accent}">{value}</div>'
        f'</div>'
    )


def prob_bar(cls: str, prob: float, color: str) -> str:
    pct = prob * 100
    return (
        f'<div class="nl-prob-bar-wrap">'
        f'<div class="nl-prob-label"><span>{cls}</span>'
        f'<span style="font-family:var(--mono);font-size:.78rem">{pct:.2f}%</span></div>'
        f'<div class="nl-prob-track">'
        f'<div class="nl-prob-fill" style="width:{pct:.1f}%;background:{color}"></div>'
        f'</div></div>'
    )


def footer() -> None:
    st.markdown(
        '<div class="nl-footer">'
        '🧠 <strong>NeuroLens AI</strong> &nbsp;·&nbsp; '
        'Research tool — not a certified medical diagnostic system &nbsp;·&nbsp; '
        'Always consult a licensed medical professional'
        '</div>',
        unsafe_allow_html=True,
    )


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
    def __init__(self, num_classes: int = 4):
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
            nn.Linear(256, 256), nn.BatchNorm1d(256),
            nn.ReLU(inplace=True), nn.Dropout(0.35),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.pool(self.features(x)))


def _build_resnet(nc):
    from torchvision.models import ResNet50_Weights
    m = resnet50(weights=None)
    m.fc = nn.Sequential(
        nn.Linear(m.fc.in_features, 256), nn.BatchNorm1d(256),
        nn.ReLU(inplace=True), nn.Dropout(0.35), nn.Linear(256, nc))
    return m


def _build_efficientnet(nc):
    from torchvision.models import EfficientNet_B0_Weights
    m = efficientnet_b0(weights=None)
    m.classifier = nn.Sequential(
        nn.Dropout(0.35), nn.Linear(m.classifier[1].in_features, nc))
    return m


def _build_mobilenet(nc):
    from torchvision.models import MobileNet_V3_Small_Weights
    m = mobilenet_v3_small(weights=None)
    in_f = m.classifier[3].in_features
    m.classifier[3] = nn.Linear(in_f, nc)
    return m


ARCH_BUILDERS = {
    "Custom CNN":        lambda nc: CustomCNN(nc),
    "ResNet50":          _build_resnet,
    "EfficientNet-B0":   _build_efficientnet,
    "MobileNetV3-Small": _build_mobilenet,
}


@st.cache_resource(show_spinner=False)
def load_model(force_cpu: bool = False):
    device = torch.device("cpu") if force_cpu else DEFAULT_DEVICE

    if not MODEL_PATH.exists():
        return None, None, None, "Model checkpoint not found. Place neurolens_best.pth next to app.py.", device

    try:
        ckpt = torch.load(MODEL_PATH, map_location="cpu")
        nc = ckpt.get("num_classes", 4)
        arch_name = ckpt.get("best_model_name", "Custom CNN")

        builder = ARCH_BUILDERS.get(arch_name)
        if builder is None:
            arch_name = "Custom CNN"
            builder = ARCH_BUILDERS["Custom CNN"]

        model = builder(nc)
        state = ckpt.get("model_state_dict", ckpt)
        model.load_state_dict(state, strict=False)
        model.to(device).eval()
        return model, arch_name, nc, None, device
    except Exception as e:
        try:
            model = CustomCNN(4)
            model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
            model.to(device).eval()
            return model, "Custom CNN", 4, None, device
        except Exception as e2:
            return None, None, None, str(e2), device


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    return total, total * 4 / 1024 ** 2


class GradCAMPlusPlus:
    def __init__(self, model, target_layer):
        self.model, self.target_layer = model, target_layer
        self.activations = self.gradients = None
        self._hooks = [
            target_layer.register_forward_hook(
                lambda m, i, o: setattr(self, 'activations', o.detach())),
            target_layer.register_backward_hook(
                lambda m, gi, go: setattr(self, 'gradients', go[0].detach())),
        ]

    def generate(self, tensor, class_idx, device):
        self.model.eval()
        tensor = tensor.unsqueeze(0).to(device).requires_grad_(True)
        out = self.model(tensor)
        self.model.zero_grad()
        one_hot = torch.zeros_like(out)
        one_hot[0, class_idx] = 1.0
        out.backward(gradient=one_hot, retain_graph=True)

        grads = self.gradients.squeeze(0)
        acts  = self.activations.squeeze(0)
        g2 = grads ** 2; g3 = grads ** 3
        denom = 2.0 * g2 + (acts * g3).sum(dim=(1, 2), keepdim=True)
        denom = torch.where(denom == 0, torch.ones_like(denom), denom)
        alpha = g2 / denom
        weights = (alpha * F.relu(grads)).sum(dim=(1, 2))
        hm = F.relu((weights[:, None, None] * acts).sum(0)).cpu().numpy()
        if hm.max() > 0:
            hm /= hm.max()
        return hm

    def remove(self):
        for h in self._hooks:
            h.remove()


def get_last_conv(model):
    last = None
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            last = m
    return last


def enable_mc_dropout(model):
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d)):
            m.train()
    return model


def mc_predict(model, tensor, device, n=20):
    model.eval()
    enable_mc_dropout(model)
    inp = tensor.unsqueeze(0).to(device)
    probs = []
    with torch.no_grad():
        for _ in range(n):
            p = torch.softmax(model(inp), 1).squeeze(0).cpu().numpy()
            probs.append(p)
    arr  = np.array(probs)
    mean = arr.mean(0)
    std  = arr.std(0)
    pred = int(np.argmax(mean))
    conf = float(mean[pred])
    ent  = float(-np.sum(mean * np.log(mean + 1e-9)))
    return mean, std, pred, conf, ent


def single_predict(model, tensor, device):
    model.eval()
    with torch.no_grad():
        inp   = tensor.unsqueeze(0).to(device)
        probs = torch.softmax(model(inp), 1).squeeze(0).cpu().numpy()
    pred = int(np.argmax(probs))
    conf = float(probs[pred])
    return probs, pred, conf


def unc_band(ent, lo, hi):
    if ent <= lo:
        return "Low", "nl-unc-low", "High"
    elif ent <= hi:
        return "Medium", "nl-unc-medium", "Moderate"
    else:
        return "High", "nl-unc-high", "Low"


def compute_gradcam(model, tensor, pred_idx, device):
    layer = get_last_conv(model)
    if layer is None:
        return None
    gc = GradCAMPlusPlus(model, layer)
    try:
        hm = gc.generate(tensor, pred_idx, device)
        gc.remove()
        return hm
    except Exception:
        gc.remove()
        return None


def overlay_heatmap(orig_np, hm):
    import cv2
    H, W = orig_np.shape[:2]
    hm_r = cv2.resize(hm, (W, H))
    colored = cv2.applyColorMap((hm_r * 255).astype(np.uint8), cv2.COLORMAP_JET)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    return (0.55 * orig_np + 0.45 * colored).astype(np.uint8)


def init_session():
    defaults = {
        "nav":              "Home",
        "history":          [],
        "force_cpu":        False,
        "mc_samples":       MC_SAMPLES_DEF,
        "unc_lo":           0.3,
        "unc_hi":           0.8,
        "total_analyses":   0,
        "tumor_detected":   0,
        "mean_conf":        0.0,
        "conf_sum":         0.0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_session()

if st.session_state.get("_do_reset"):
    keys_to_keep = {"force_cpu", "mc_samples", "unc_lo", "unc_hi"}
    for k in list(st.session_state.keys()):
        if k not in keys_to_keep and k != "_do_reset":
            del st.session_state[k]
    init_session()
    del st.session_state["_do_reset"]
    st.rerun()

model, arch_name, num_classes, model_error, DEVICE = load_model(
    st.session_state.get("force_cpu", False)
)
if model is None and not model_error:
    model_error = "Model could not be loaded."


def render_sidebar():
    with st.sidebar:
        st.markdown(
            '<div style="display:flex;align-items:center;gap:.65rem;padding:.3rem 0 .8rem">'
            '<div style="width:36px;height:36px;border-radius:10px;display:grid;'
            'place-items:center;background:linear-gradient(135deg,rgba(34,211,238,.22),'
            'rgba(52,211,153,.14));border:1px solid rgba(34,211,238,.42);color:#22d3ee">'
            '🧠</div>'
            '<div><div style="font-weight:800;font-size:.95rem;color:#e8eefb">NeuroLens AI</div>'
            '<div style="font-size:.7rem;color:#7f9bbd">Brain MRI Intelligence</div></div>'
            '</div>',
            unsafe_allow_html=True,
        )
        st.markdown("---")

        pages = [
            ("Home",         "home"),
            ("MRI Analysis", "scan"),
            ("XAI Lab",      "eye"),
            ("Dashboard",    "chart"),
            ("History",      "history"),
            ("Settings",     "settings"),
        ]
        for label, icon in pages:
            active = st.session_state.nav == label
            kind = "primary" if active else "secondary"
            if st.button(f"{_ico(icon, 15)}  {label}", key=f"nav_{label}",
                         type=kind, use_container_width=True):
                st.session_state.nav = label
                st.rerun()

        st.markdown("---")
        status = "🔴 No checkpoint" if model_error else f"🟢 {arch_name}"
        st.markdown(
            f'<div style="font-size:.75rem;color:#7f9bbd;padding:.4rem .6rem">'
            f'Model: <span style="color:#e8eefb;font-weight:600">{status}</span><br>'
            f'Device: <span style="color:#e8eefb;font-weight:600">{str(DEVICE).upper()}</span><br>'
            f'Analyses: <span style="color:#22d3ee;font-weight:700">'
            f'{st.session_state.total_analyses}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )


def render_home():
    st.markdown('<div class="nl-page">', unsafe_allow_html=True)
    n = st.session_state.total_analyses
    td = st.session_state.tumor_detected
    mc = (st.session_state.conf_sum / n * 100) if n > 0 else 0.0

    st.markdown(f"""
<div class="nl-hero">
<div class="nl-hero-inner">
  <div class="nl-badge"><span class="dot"></span> AI Research Platform v2.0</div>
  <h1 class="nl-hero-title">
    Intelligent<br><span class="g">Brain MRI</span><br>Analysis
  </h1>
  <p class="nl-hero-sub">
    NeuroLens classifies brain tumors from MRI scans using lightweight deep learning,
    explainable AI (Grad-CAM++), and Monte Carlo Dropout uncertainty estimation.
    Built for research and educational use.
  </p>
  <div class="nl-stats">
    <div><div class="nl-stat-v">{n}</div><div class="nl-stat-l">Analyses Run</div></div>
    <div><div class="nl-stat-v">{td}</div><div class="nl-stat-l">Tumors Detected</div></div>
    <div><div class="nl-stat-v">{mc:.1f}%</div><div class="nl-stat-l">Mean Confidence</div></div>
    <div><div class="nl-stat-v">4</div><div class="nl-stat-l">Tumor Classes</div></div>
  </div>
</div>
</div>
""", unsafe_allow_html=True)

    st.markdown(
        '<div class="nl-disc">⚠️ <strong>Medical Disclaimer:</strong> '
        'This is an AI research tool for educational purposes only. '
        'It is <strong>not</strong> a certified diagnostic system. '
        'All predictions must be reviewed by a licensed medical professional.</div>',
        unsafe_allow_html=True,
    )

    section_head("zap", "Capabilities")
    cards = [
        ("scan",     "MRI Classification",    "4-class tumor detection with confidence scores and probability distribution."),
        ("eye",      "Explainable AI",         "Grad-CAM++ heatmaps visualize which brain regions drive each prediction."),
        ("shield",   "Uncertainty Estimation", "Monte Carlo Dropout quantifies prediction reliability across stochastic passes."),
        ("chart",    "Analytics Dashboard",    "Session-wide statistics, confidence trends, and class distribution charts."),
    ]
    cols = st.columns(4)
    for col, (icon, title, desc) in zip(cols, cards):
        with col:
            st.markdown(
                f'<div class="nl-card">'
                f'<h3>{_ico(icon, 16)} {title}</h3>'
                f'<p>{desc}</p></div>',
                unsafe_allow_html=True,
            )

    st.markdown("")
    section_head("brain", "Tumor Classes")
    ccols = st.columns(4)
    for col, (cls, info) in zip(ccols, CLASS_INFO.items()):
        with col:
            sev_color = {"High": "#f87171", "Moderate": "#fbbf24", "None": "#34d399"}[info["severity"]]
            st.markdown(
                f'<div class="nl-card" style="border-left:3px solid {info["color"]}">'
                f'<div style="font-weight:700;font-size:.9rem;color:{info["color"]};margin-bottom:.4rem">{cls}</div>'
                f'<div style="font-size:.78rem;color:#b9cbe2;line-height:1.5">{info["desc"]}</div>'
                f'<div style="margin-top:.5rem;font-size:.7rem;font-weight:700;color:{sev_color}">'
                f'Severity: {info["severity"]}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    footer()
    st.markdown('</div>', unsafe_allow_html=True)


def render_analysis():
    st.markdown('<div class="nl-page">', unsafe_allow_html=True)
    section_head("scan", "MRI Analysis")

    if model_error:
        st.error(f"⚠️ Model not loaded: {model_error}")
        st.info("Place `neurolens_best.pth` in the same folder as `app.py` and restart.")
        footer()
        st.markdown('</div>', unsafe_allow_html=True)
        return

    use_mc = st.sidebar.toggle("Monte Carlo Dropout", value=True,
                                help="Enable uncertainty estimation (slower)")
    mc_n = st.sidebar.slider("MC Passes", 5, 100, st.session_state.mc_samples, 5) if use_mc else 1
    if use_mc:
        st.session_state.mc_samples = mc_n

    uploaded = st.file_uploader(
        "Upload a Brain MRI scan (JPG / PNG / WEBP / BMP)",
        type=["jpg", "jpeg", "png", "webp", "bmp"],
        help=f"Max {MAX_UPLOAD_BYTES // 1024 // 1024} MB",
    )

    if uploaded is None:
        st.markdown(
            '<div style="text-align:center;padding:3rem 1rem;color:#7f9bbd;font-size:.9rem">'
            '📁 Upload an MRI image above to begin analysis</div>',
            unsafe_allow_html=True,
        )
        footer()
        st.markdown('</div>', unsafe_allow_html=True)
        return

    if uploaded.size > MAX_UPLOAD_BYTES:
        st.error("File too large.")
        footer()
        st.markdown('</div>', unsafe_allow_html=True)
        return

    try:
        img = Image.open(io.BytesIO(uploaded.read())).convert("RGB")
    except (UnidentifiedImageError, Exception) as e:
        st.error(f"Cannot read image: {e}")
        footer()
        st.markdown('</div>', unsafe_allow_html=True)
        return

    tensor = test_transforms(img)
    img_np = np.array(img)

    left_col, right_col = st.columns([1, 1.6], gap="large")

    with left_col:
        section_head("scan", "Input MRI")
        st.image(img, use_container_width=True)
        st.caption(f"Size: {img.width}×{img.height} px · {uploaded.size // 1024} KB")

    with right_col:
        section_head("zap", "Running Analysis…")
        t0 = time.time()
        with st.spinner("Classifying…"):
            if use_mc:
                mean_probs, std_probs, pred_idx, conf, entropy = mc_predict(
                    model, tensor, DEVICE, n=mc_n)
                probs = mean_probs
            else:
                probs, pred_idx, conf = single_predict(model, tensor, DEVICE)
                std_probs = np.zeros_like(probs)
                entropy = float(-np.sum(probs * np.log(probs + 1e-9)))
        latency = (time.time() - t0) * 1000

        pred_class = CLASS_NAMES[pred_idx]
        cls_color  = CLASS_INFO[pred_class]["color"]

        lo = st.session_state.unc_lo
        hi = st.session_state.unc_hi
        unc_level, unc_cls, reliability = unc_band(entropy, lo, hi)

        st.markdown(
            f'<div class="nl-result">'
            f'<div class="nl-result-label">{_ico("brain",12)} Diagnostic Result</div>'
            f'<div class="nl-result-pred" style="color:{cls_color}">{pred_class.upper()}</div>'
            f'<div class="nl-result-conf">'
            f'Confidence: <strong>{conf*100:.2f}%</strong>'
            f'&nbsp;·&nbsp; Latency: <strong>{latency:.0f} ms</strong>'
            f'</div></div>',
            unsafe_allow_html=True,
        )

        if use_mc:
            st.markdown(
                f'<div style="margin:.4rem 0 .7rem">'
                f'Uncertainty: <span class="nl-unc-band {unc_cls}">{unc_level}</span>'
                f'&nbsp; Reliability: <strong>{reliability}</strong>'
                f'&nbsp; Entropy: <code style="font-size:.78rem">{entropy:.4f}</code>'
                f'</div>',
                unsafe_allow_html=True,
            )

        section_head("chart", "Class Probabilities")
        bar_colors = [CLASS_INFO[c]["color"] for c in CLASS_NAMES]
        for i, (cls, p, c) in enumerate(zip(CLASS_NAMES, probs, bar_colors)):
            err = f" ±{std_probs[i]*100:.1f}%" if use_mc else ""
            st.markdown(
                f'<div class="nl-prob-bar-wrap">'
                f'<div class="nl-prob-label">'
                f'<span>{cls}{err}</span>'
                f'<span style="font-family:var(--mono);font-size:.78rem">{p*100:.2f}%</span>'
                f'</div>'
                f'<div class="nl-prob-track">'
                f'<div class="nl-prob-fill" style="width:{p*100:.1f}%;background:{c}"></div>'
                f'</div></div>',
                unsafe_allow_html=True,
            )

        if use_mc:
            with st.expander("MC Dropout Details"):
                mc_data = {"Class": CLASS_NAMES,
                           "Mean %": [f"{p*100:.3f}" for p in mean_probs],
                           "Std %":  [f"{s*100:.3f}" for s in std_probs]}
                st.dataframe(pd.DataFrame(mc_data), hide_index=True, use_container_width=True)

    st.markdown("---")
    section_head("eye", "Grad-CAM++ Explanation")

    with st.spinner("Generating heatmap…"):
        hm = compute_gradcam(model, tensor, pred_idx, DEVICE)

    if hm is not None:
        import cv2
        H, W = img_np.shape[:2]
        hm_r = cv2.resize(hm, (W, H))
        colored = cv2.applyColorMap((hm_r * 255).astype(np.uint8), cv2.COLORMAP_JET)
        colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
        overlay = (0.55 * img_np + 0.45 * colored).astype(np.uint8)

        c1, c2, c3 = st.columns(3)
        with c1:
            st.image(img_np, caption="Original MRI", use_container_width=True)
        with c2:
            st.image((hm_r * 255).astype(np.uint8), caption="Grad-CAM++ Heatmap",
                     use_container_width=True, clamp=True)
        with c3:
            st.image(overlay, caption="Overlay", use_container_width=True)

        st.caption(
            "🔴 Red = high activation (model focuses here) · "
            "🔵 Blue = low activation · "
            "Grad-CAM++ highlights discriminative regions for the predicted class."
        )
    else:
        st.info("Grad-CAM++ not available for this architecture.")

    st.markdown("---")
    section_head("brain", "Clinical Notes")
    info = CLASS_INFO[pred_class]
    sev_color = {"High": "#f87171", "Moderate": "#fbbf24", "None": "#34d399"}[info["severity"]]
    st.markdown(
        f'<div class="nl-card" style="border-left:3px solid {cls_color}">'
        f'<div style="font-weight:700;font-size:.95rem;color:{cls_color};margin-bottom:.4rem">'
        f'{pred_class}</div>'
        f'<div style="font-size:.84rem;color:#b9cbe2;line-height:1.6">{info["desc"]}</div>'
        f'<div style="margin-top:.5rem;font-size:.75rem;font-weight:700;color:{sev_color}">'
        f'Severity: {info["severity"]} &nbsp;·&nbsp; '
        f'<span style="color:#7f9bbd;font-weight:400">This is an AI estimate — consult a physician.</span>'
        f'</div></div>',
        unsafe_allow_html=True,
    )

    record = {
        "timestamp":   datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "prediction":  pred_class,
        "confidence":  conf * 100,
        "model":       arch_name or "Unknown",
        "latency_ms":  round(latency, 1),
        "entropy":     round(entropy, 6),
        "uncertainty": unc_level if use_mc else "N/A",
        "mc_used":     use_mc,
        "probabilities": {c: round(float(p) * 100, 4)
                          for c, p in zip(CLASS_NAMES, probs)},
        "img_hash": hashlib.md5(img.tobytes()).hexdigest()[:10],
    }
    hist = st.session_state.history
    hist.insert(0, record)
    if len(hist) > MAX_HISTORY:
        hist.pop()
    st.session_state.history = hist
    st.session_state.total_analyses += 1
    if pred_class != "No Tumor":
        st.session_state.tumor_detected += 1
    st.session_state.conf_sum += conf

    footer()
    st.markdown('</div>', unsafe_allow_html=True)


def render_xai():
    st.markdown('<div class="nl-page">', unsafe_allow_html=True)
    section_head("eye", "XAI Lab — Explainability Explorer")

    st.markdown(
        '<div class="nl-disc">📖 The XAI Lab lets you inspect how the model reaches its decisions. '
        'Upload an MRI scan to generate Grad-CAM++ heatmaps and explore activation patterns.'
        '</div>',
        unsafe_allow_html=True,
    )

    if model_error:
        st.error(f"Model not loaded: {model_error}")
        footer()
        st.markdown('</div>', unsafe_allow_html=True)
        return

    uploaded = st.file_uploader("Upload MRI for XAI analysis", type=["jpg", "jpeg", "png", "webp"])
    if uploaded is None:
        st.markdown(
            '<div style="text-align:center;padding:2.5rem;color:#7f9bbd">Upload an MRI above.</div>',
            unsafe_allow_html=True,
        )
        footer()
        st.markdown('</div>', unsafe_allow_html=True)
        return

    img  = Image.open(io.BytesIO(uploaded.read())).convert("RGB")
    img_np = np.array(img)
    tensor = test_transforms(img)

    probs, pred_idx, conf = single_predict(model, tensor, DEVICE)
    pred_class = CLASS_NAMES[pred_idx]

    st.markdown(
        f'<div style="margin:.5rem 0;font-size:.9rem">'
        f'Prediction: <strong style="color:{CLASS_INFO[pred_class]["color"]}">{pred_class}</strong>'
        f' &nbsp;·&nbsp; Confidence: <strong>{conf*100:.2f}%</strong></div>',
        unsafe_allow_html=True,
    )

    target_label = st.selectbox("Target class for heatmap", CLASS_NAMES, index=pred_idx)
    target_idx   = CLASS_NAMES.index(target_label)

    if st.button("Generate Grad-CAM++ Heatmap", type="primary"):
        with st.spinner("Computing activations…"):
            hm = compute_gradcam(model, tensor, target_idx, DEVICE)

        if hm is not None:
            import cv2
            H, W = img_np.shape[:2]
            hm_r = cv2.resize(hm, (W, H))
            colored = cv2.applyColorMap((hm_r * 255).astype(np.uint8), cv2.COLORMAP_JET)
            colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
            overlay = (0.55 * img_np + 0.45 * colored).astype(np.uint8)

            tab1, tab2, tab3 = st.tabs(["Original", "Heatmap", "Overlay"])
            with tab1:
                st.image(img_np, use_container_width=True)
            with tab2:
                fig, ax = plt.subplots(figsize=(6, 5),
                                       facecolor="#08121f")
                im = ax.imshow(hm_r, cmap="jet", vmin=0, vmax=1)
                ax.set_title(f"Grad-CAM++ · {target_label}", color="#e8eefb", fontsize=12)
                ax.axis("off")
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
            with tab3:
                st.image(overlay, use_container_width=True)

            with st.expander("About Grad-CAM++"):
                st.markdown("""
**Grad-CAM++** (Gradient-weighted Class Activation Mapping++) computes a weighted
combination of feature maps from the last convolutional layer, using second-order
gradient weights to produce more accurate and complete class-discriminative localization.

- **Red / warm** regions = areas the model strongly associated with the target class  
- **Blue / cool** regions = areas with low activation for the target class  
- The heatmap is upsampled and overlaid on the original MRI for interpretation
""")
        else:
            st.error("Grad-CAM++ could not be computed for this model architecture.")

    footer()
    st.markdown('</div>', unsafe_allow_html=True)


def render_dashboard():
    st.markdown('<div class="nl-page">', unsafe_allow_html=True)
    section_head("chart", "Analytics Dashboard")

    hist = st.session_state.history
    n    = st.session_state.total_analyses

    m1, m2, m3, m4 = st.columns(4)
    mc = (st.session_state.conf_sum / n * 100) if n > 0 else 0.0
    td = st.session_state.tumor_detected
    with m1:
        st.markdown(metric_card("Total Analyses", str(n), "#22d3ee"), unsafe_allow_html=True)
    with m2:
        st.markdown(metric_card("Tumors Detected", str(td), "#f87171"), unsafe_allow_html=True)
    with m3:
        st.markdown(metric_card("Mean Confidence", f"{mc:.1f}%", "#34d399"), unsafe_allow_html=True)
    with m4:
        rate = f"{td/n*100:.0f}%" if n > 0 else "—"
        st.markdown(metric_card("Detection Rate", rate, "#a78bfa"), unsafe_allow_html=True)

    if not hist:
        st.markdown(
            '<div style="text-align:center;padding:3rem;color:#7f9bbd">'
            'Run some analyses first to see statistics here.</div>',
            unsafe_allow_html=True,
        )
        footer()
        st.markdown('</div>', unsafe_allow_html=True)
        return

    st.markdown("")
    col_l, col_r = st.columns(2)

    with col_l:
        section_head("brain", "Class Distribution")
        cls_counts = {c: 0 for c in CLASS_NAMES}
        for r in hist:
            cls_counts[r["prediction"]] = cls_counts.get(r["prediction"], 0) + 1

        fig, ax = plt.subplots(figsize=(5, 3.5), facecolor="#0f1e33")
        colors = [CLASS_INFO[c]["color"] for c in CLASS_NAMES]
        vals   = [cls_counts[c] for c in CLASS_NAMES]
        bars   = ax.bar(CLASS_NAMES, vals, color=colors, edgecolor="none", width=0.55)
        ax.set_facecolor("#0f1e33")
        ax.tick_params(colors="#b9cbe2", labelsize=9)
        ax.spines[:].set_visible(False)
        ax.set_ylabel("Count", color="#7f9bbd", fontsize=9)
        for bar, v in zip(bars, vals):
            if v:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + .1,
                        str(v), ha="center", va="bottom", color="#e8eefb", fontsize=9)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with col_r:
        section_head("zap", "Confidence Over Time")
        confs = [r["confidence"] for r in reversed(hist[-30:])]
        if confs:
            fig, ax = plt.subplots(figsize=(5, 3.5), facecolor="#0f1e33")
            ax.plot(confs, color="#22d3ee", linewidth=2, marker="o", markersize=4)
            ax.fill_between(range(len(confs)), confs, alpha=0.15, color="#22d3ee")
            ax.set_facecolor("#0f1e33")
            ax.tick_params(colors="#b9cbe2", labelsize=9)
            ax.spines[:].set_visible(False)
            ax.set_ylabel("Confidence %", color="#7f9bbd", fontsize=9)
            ax.set_ylim(0, 105)
            ax.axhline(np.mean(confs), color="#34d399", linewidth=1.2,
                       linestyle="--", alpha=0.7)
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)

    st.markdown("")
    section_head("info", "Uncertainty Distribution")
    if any(r.get("mc_used") for r in hist):
        ents = [r["entropy"] for r in hist if r.get("mc_used") and r.get("entropy") is not None]
        if ents:
            fig, ax = plt.subplots(figsize=(10, 3), facecolor="#0f1e33")
            ax.hist(ents, bins=20, color="#22d3ee", edgecolor="none", alpha=0.75)
            ax.axvline(st.session_state.unc_lo, color="#34d399", linestyle="--",
                       linewidth=1.5, label=f"Low thr ({st.session_state.unc_lo:.2f})")
            ax.axvline(st.session_state.unc_hi, color="#fbbf24", linestyle="--",
                       linewidth=1.5, label=f"High thr ({st.session_state.unc_hi:.2f})")
            ax.set_facecolor("#0f1e33")
            ax.tick_params(colors="#b9cbe2", labelsize=9)
            ax.spines[:].set_visible(False)
            ax.set_xlabel("Predictive Entropy", color="#7f9bbd", fontsize=9)
            ax.legend(fontsize=8, facecolor="#142543", labelcolor="#e8eefb")
            fig.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
    else:
        st.info("Enable Monte Carlo Dropout in MRI Analysis to see uncertainty data.")

    footer()
    st.markdown('</div>', unsafe_allow_html=True)


def render_history():
    st.markdown('<div class="nl-page">', unsafe_allow_html=True)
    section_head("history", "Analysis History")

    hist = st.session_state.history
    if not hist:
        st.markdown(
            '<div style="text-align:center;padding:3rem;color:#7f9bbd">'
            'No analyses recorded yet. Run an MRI analysis first.</div>',
            unsafe_allow_html=True,
        )
        footer()
        st.markdown('</div>', unsafe_allow_html=True)
        return

    c1, c2 = st.columns([3, 1])
    with c1:
        filter_cls = st.multiselect("Filter by class", CLASS_NAMES)
    with c2:
        if st.button(f"{_ico('trash',14)}  Clear History", use_container_width=True):
            st.session_state.history = []
            st.rerun()

    filtered = [r for r in hist if (not filter_cls or r["prediction"] in filter_cls)]
    st.caption(f"Showing {len(filtered)} of {len(hist)} records")

    for item in filtered[:50]:
        cls   = item["prediction"]
        color = CLASS_INFO.get(cls, {}).get("color", "#22d3ee")
        unc   = item.get("uncertainty", "—")
        st.markdown(
            f'<div class="nl-hist-item">'
            f'<div style="width:10px;height:10px;border-radius:50%;'
            f'background:{color};flex-shrink:0"></div>'
            f'<div style="flex:1">'
            f'<div class="nl-hist-pred">{cls}</div>'
            f'<div class="nl-hist-meta">'
            f'{item["timestamp"]} &nbsp;·&nbsp; '
            f'Conf: {item["confidence"]:.2f}% &nbsp;·&nbsp; '
            f'Unc: {unc} &nbsp;·&nbsp; '
            f'{item["model"]} &nbsp;·&nbsp; '
            f'{item["latency_ms"]} ms</div>'
            f'</div></div>',
            unsafe_allow_html=True,
        )

    st.markdown("")
    df_rows = []
    for r in filtered:
        row = {k: r.get(k) for k in ["timestamp", "prediction", "confidence",
                                       "model", "latency_ms", "entropy", "uncertainty"]}
        row.update({f"prob_{k}": v for k, v in r.get("probabilities", {}).items()})
        df_rows.append(row)
    df = pd.DataFrame(df_rows)
    st.download_button(
        f"{_ico('download',14)}  Export CSV",
        df.to_csv(index=False), "neurolens_history.csv", "text/csv",
        use_container_width=True,
    )

    footer()
    st.markdown('</div>', unsafe_allow_html=True)


def render_settings():
    st.markdown('<div class="nl-page">', unsafe_allow_html=True)
    section_head("settings", "Settings")

    c1, c2 = st.columns(2)
    with c1:
        section_head("brain", "Model Info")
        rows = [
            ("Status",       "Ready" if not model_error else "Failed"),
            ("Architecture", arch_name or "—"),
            ("Device",       str(DEVICE)),
            ("Input Size",   f"{IMG_SIZE}×{IMG_SIZE}"),
            ("Classes",      ", ".join(CLASS_NAMES)),
            ("Checkpoint",   MODEL_PATH.name if MODEL_PATH.exists() else "Not found"),
        ]
        if model:
            tot, sz = count_params(model)
            rows += [("Parameters", f"{tot:,}"), ("Size", f"{sz:.2f} MB")]
        st.table(pd.DataFrame(rows, columns=["Property", "Value"]))

    with c2:
        section_head("cpu", "System Info")
        try:
            tv = metadata.version("torchvision")
        except Exception:
            tv = "—"
        sys_rows = [
            ("PyTorch",     torch.__version__),
            ("Torchvision", tv),
            ("Streamlit",   st.__version__),
            ("Python",      platform.python_version()),
            ("CUDA",        str(torch.cuda.is_available())),
            ("CUDA Device", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"),
        ]
        st.table(pd.DataFrame(sys_rows, columns=["Property", "Value"]))

    st.markdown("")
    section_head("cpu", "Runtime Controls")
    rc1, rc2 = st.columns(2)
    with rc1:
        prev = bool(st.session_state.force_cpu)
        st.toggle("Force CPU Inference", key="force_cpu",
                  on_change=lambda: load_model.clear(),
                  help="Reload model on CPU")
        if st.session_state.force_cpu != prev:
            st.info("Device changed — reloading…")
            st.rerun()
    with rc2:
        mc_v = st.number_input("MC Dropout Passes", 5, 200, st.session_state.mc_samples, 5)
        if mc_v != st.session_state.mc_samples:
            st.session_state.mc_samples = int(mc_v)

    st.markdown("")
    section_head("alert", "Uncertainty Thresholds")
    u1, u2 = st.columns(2)
    with u1:
        lo = st.slider("Low→Medium threshold (entropy)", 0.0, 2.0,
                       float(st.session_state.unc_lo), 0.05)
        st.session_state.unc_lo = lo
    with u2:
        hi = st.slider("Medium→High threshold (entropy)", 0.0, 2.0,
                       float(st.session_state.unc_hi), 0.05)
        st.session_state.unc_hi = hi

    st.markdown("")
    section_head("refresh", "Reset Session")
    if st.button("Reset All Session Data", type="primary"):
        st.session_state.settings_confirm = True
    if st.session_state.get("settings_confirm"):
        st.warning("This clears all analyses, history, and statistics.")
        r1, r2, _ = st.columns([1, 1, 4])
        if r1.button("Confirm"):
            st.session_state["_do_reset"] = True
            st.rerun()
        if r2.button("Cancel"):
            st.session_state.settings_confirm = False

    footer()
    st.markdown('</div>', unsafe_allow_html=True)


PAGES = {
    "Home":         render_home,
    "MRI Analysis": render_analysis,
    "XAI Lab":      render_xai,
    "Dashboard":    render_dashboard,
    "History":      render_history,
    "Settings":     render_settings,
}


def main():
    render_sidebar()
    nav = st.session_state.nav
    if nav not in PAGES:
        nav = "Home"
        st.session_state.nav = "Home"
    PAGES[nav]()


if __name__ == "__main__" or True:
    main()
