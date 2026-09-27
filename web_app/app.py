# =============================================================================
# NeuroLens AI — Neurodiagnostic Intelligence Platform
# Reconstructed Edition · v2 (SVG iconography · responsive redesign)
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
# PAGE CONFIG
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
# SVG ICON LIBRARY
# =============================================================================

SVG_ICONS: Dict[str, str] = {
    "brain": (
        '<path d="M9.5 2A2.5 2.5 0 0 1 12 4.5v15a2.5 2.5 0 0 1-4.96.44 2.5 2.5 0 0 1-2.96-3.08 3 3 0 0 1-.34-5.58 2.5 2.5 0 0 1 1.32-4.24 2.5 2.5 0 0 1 1.98-3A2.5 2.5 0 0 1 9.5 2Z"/>'
        '<path d="M14.5 2A2.5 2.5 0 0 0 12 4.5v15a2.5 2.5 0 0 0 4.96.44 2.5 2.5 0 0 0 2.96-3.08 3 3 0 0 0 .34-5.58 2.5 2.5 0 0 0-1.32-4.24 2.5 2.5 0 0 0-1.98-3A2.5 2.5 0 0 0 14.5 2Z"/>'
    ),
    "home": '<path d="m3 9 9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/><polyline points="9 22 9 12 15 12 15 22"/>',
    "scan": (
        '<path d="M3 7V5a2 2 0 0 1 2-2h2"/><path d="M17 3h2a2 2 0 0 1 2 2v2"/>'
        '<path d="M21 17v2a2 2 0 0 1-2 2h-2"/><path d="M7 21H5a2 2 0 0 1-2-2v-2"/>'
        '<path d="M7 12h10"/><circle cx="12" cy="12" r="2.4"/>'
    ),
    "eye": '<path d="M2 12s3.5-7 10-7 10 7 10 7-3.5 7-10 7-10-7-10-7Z"/><circle cx="12" cy="12" r="3"/>',
    "chart": '<path d="M3 3v18h18"/><path d="M18 17V9"/><path d="M13 17V5"/><path d="M8 17v-3"/>',
    "history": '<path d="M3 12a9 9 0 1 0 3-6.7L3 8"/><path d="M3 3v5h5"/><path d="M12 7v5l3 2"/>',
    "settings": (
        '<circle cx="12" cy="12" r="3"/>'
        '<path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 1 1-4 0v-.09a1.65 1.65 0 0 0-1-1.51 1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 1 1 0-4h.09a1.65 1.65 0 0 0 1.51-1 1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06a1.65 1.65 0 0 0 1.82.33h.01a1.65 1.65 0 0 0 1-1.51V3a2 2 0 1 1 4 0v.09a1.65 1.65 0 0 0 1 1.51h.01a1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82v.01a1.65 1.65 0 0 0 1.51 1H21a2 2 0 1 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1Z"/>'
    ),
    "activity": '<path d="M22 12h-4l-3 9L9 3l-3 9H2"/>',
    "upload": '<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="17 8 12 3 7 8"/><line x1="12" y1="3" x2="12" y2="15"/>',
    "download": '<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/>',
    "zap": '<polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/>',
    "shield": '<path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10Z"/>',
    "check": '<path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/>',
    "alert": '<circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="12"/><line x1="12" y1="16" x2="12.01" y2="16"/>',
    "info": '<circle cx="12" cy="12" r="10"/><line x1="12" y1="16" x2="12" y2="12"/><line x1="12" y1="8" x2="12.01" y2="8"/>',
    "cpu": '<rect x="4" y="4" width="16" height="16" rx="2"/><rect x="9" y="9" width="6" height="6"/><path d="M9 2v2M15 2v2M9 20v2M15 20v2M2 9h2M2 15h2M20 9h2M20 15h2"/>',
    "layers": '<polygon points="12 2 2 7 12 12 22 7 12 2"/><polyline points="2 17 12 22 22 17"/><polyline points="2 12 12 17 22 12"/>',
    "target": '<circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/>',
    "sparkles": '<path d="M9.9 2.5 12 7l4.5 2.1L12 11l-2.1 4.5L7.9 11 3.4 8.9 7.9 7z"/><path d="M19 14l1 2 2 1-2 1-1 2-1-2-2-1 2-1z"/>',
    "pulse": '<polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>',
    "file": '<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/><line x1="16" y1="13" x2="8" y2="13"/><line x1="16" y1="17" x2="8" y2="17"/>',
    "trash": '<polyline points="3 6 5 6 21 6"/><path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6"/><path d="M10 11v6M14 11v6"/><path d="M9 6V4a1 1 0 0 1 1-1h4a1 1 0 0 1 1 1v2"/>',
    "refresh": '<polyline points="23 4 23 10 17 10"/><polyline points="1 20 1 14 7 14"/><path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15"/>',
    "clock": '<circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/>',
    "monitor": '<rect x="2" y="3" width="20" height="14" rx="2"/><line x1="8" y1="21" x2="16" y2="21"/><line x1="12" y1="17" x2="12" y2="21"/>',
    "database": '<ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3"/><path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"/>',
    "sliders": '<line x1="4" y1="21" x2="4" y2="14"/><line x1="4" y1="10" x2="4" y2="3"/><line x1="12" y1="21" x2="12" y2="12"/><line x1="12" y1="8" x2="12" y2="3"/><line x1="20" y1="21" x2="20" y2="16"/><line x1="20" y1="12" x2="20" y2="3"/><line x1="1" y1="14" x2="7" y2="14"/><line x1="9" y1="8" x2="15" y2="8"/><line x1="17" y1="16" x2="23" y2="16"/>',
}


def svg(name: str, size: int = 18, stroke: float = 1.8,
        color: str = "currentColor", extra: str = "") -> str:
    """Return inline SVG markup for a named icon."""
    paths = SVG_ICONS.get(name, "")
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 24 24" fill="none" stroke="{color}" '
        f'stroke-width="{stroke}" stroke-linecap="round" stroke-linejoin="round" '
        f'style="display:inline-block;vertical-align:-0.18em;flex-shrink:0;{extra}">'
        f'{paths}</svg>'
    )


# =============================================================================
# GLOBAL CSS
# =============================================================================

st.markdown(
    '<link rel="preconnect" href="https://fonts.googleapis.com">'
    '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
    '<link href="https://fonts.googleapis.com/css2?'
    'family=Inter:wght@400;500;600;700;800&'
    'family=JetBrains+Mono:wght@400;500;600&display=swap" '
    'rel="stylesheet">',
    unsafe_allow_html=True,
)

st.markdown("""
<style>
/* ── Design tokens ──────────────────────────────────────── */
:root {
  --bg:            #08121f;
  --bg-2:          #0a1828;
  --surface:       #0f1e33;
  --surface-2:     #142543;
  --surface-3:     #182c4d;
  --border:        rgba(125,165,220,.14);
  --border-2:      rgba(125,165,220,.22);
  --border-strong: rgba(34,211,238,.38);

  --cyan:    #22d3ee;
  --cyan-2:  #06b6d4;
  --emerald: #34d399;
  --amber:   #fbbf24;
  --danger:  #f87171;
  --violet:  #a78bfa;

  --text:    #e8eefb;
  --text-2:  #b9cbe2;
  --muted:   #7f9bbd;

  --mono: 'JetBrains Mono', ui-monospace, SFMono-Regular, monospace;
  --sans: 'Inter', system-ui, -apple-system, "Segoe UI", sans-serif;

  --r-sm: 8px;
  --r:    12px;
  --r-lg: 16px;
  --r-xl: 20px;

  --sh-sm: 0 1px 2px rgba(0,0,0,.35);
  --sh:    0 10px 26px -10px rgba(0,0,0,.55);
  --sh-lg: 0 24px 60px -22px rgba(0,0,0,.65);
  --sh-in: inset 0 1px 0 rgba(255,255,255,.04);

  --ring: 0 0 0 1px rgba(34,211,238,.55), 0 0 0 4px rgba(34,211,238,.15);
  --ease: cubic-bezier(.22,.61,.36,1);
  --ease-out: cubic-bezier(.16,.84,.44,1);
}

/* ── Base ──────────────────────────────────────────────── */
html, body,
[data-testid="stAppViewContainer"],
[data-testid="stApp"] {
  background: var(--bg) !important;
  font-family: var(--sans);
  color: var(--text);
  -webkit-font-smoothing: antialiased;
  text-rendering: optimizeLegibility;
}
[data-testid="stAppViewContainer"] > .main {
  background:
    radial-gradient(1100px 600px at 50% -15%, rgba(34,211,238,.10), transparent 55%),
    radial-gradient(900px 500px at 95% 10%, rgba(167,139,250,.07), transparent 55%),
    linear-gradient(180deg, #0a1a30 0%, #050c18 60%, #040911 100%) !important;
  background-attachment: fixed;
}
[data-testid="stMainBlockContainer"] {
  max-width: 1380px !important;
  padding: clamp(.9rem, 2vw, 1.6rem) clamp(.8rem, 2vw, 1.6rem) 4rem !important;
}

/* ── Sidebar ───────────────────────────────────────────── */
[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #0b1e3d 0%, #060f1f 100%) !important;
  border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] > div:first-child {
  padding: .85rem .8rem 1.2rem !important;
}
[data-testid="stSidebar"] hr {
  border-color: var(--border) !important;
  margin: 1rem 0 !important;
}

/* Sidebar nav buttons — styled as nav rail items */
[data-testid="stSidebar"] .stButton > button {
  justify-content: flex-start !important;
  text-align: left !important;
  border-radius: 10px !important;
  padding: .6rem .85rem !important;
  font-size: .875rem !important;
  font-weight: 500 !important;
  letter-spacing: .005em;
  transition: background .18s var(--ease), border-color .18s var(--ease),
              color .18s var(--ease), transform .12s var(--ease) !important;
  min-height: 42px !important;
}
[data-testid="stSidebar"] .stButton > button[kind="secondary"] {
  background: transparent !important;
  border: 1px solid transparent !important;
  color: var(--text-2) !important;
}
[data-testid="stSidebar"] .stButton > button[kind="secondary"]:hover {
  background: rgba(34,211,238,.07) !important;
  border-color: rgba(34,211,238,.22) !important;
  color: var(--text) !important;
}
[data-testid="stSidebar"] .stButton > button[kind="secondary"]:active {
  transform: translateY(1px);
}
[data-testid="stSidebar"] .stButton > button[kind="primary"] {
  background: linear-gradient(135deg, rgba(34,211,238,.17), rgba(52,211,153,.10)) !important;
  border: 1px solid rgba(34,211,238,.42) !important;
  color: var(--text) !important;
  box-shadow: var(--sh-in), 0 0 0 1px rgba(34,211,238,.08) !important;
}
[data-testid="stSidebar"] .stButton > button[kind="primary"]::before {
  content: "";
  position: absolute;
  left: -1px; top: 20%; bottom: 20%;
  width: 3px; border-radius: 3px;
  background: linear-gradient(180deg, var(--cyan), var(--emerald));
}

/* Streamlit default buttons elsewhere */
.stButton > button, .stDownloadButton > button {
  border-radius: 10px !important;
  font-weight: 600 !important;
  font-size: .875rem !important;
  transition: transform .12s var(--ease), box-shadow .18s var(--ease),
              background .18s var(--ease), border-color .18s var(--ease) !important;
}
.stButton > button[kind="primary"], .stDownloadButton > button[kind="primary"] {
  background: linear-gradient(135deg, #0ea5e9, #06b6d4) !important;
  border: 1px solid rgba(255,255,255,.10) !important;
  color: #031422 !important;
  box-shadow: 0 6px 18px -6px rgba(34,211,238,.55) !important;
}
.stButton > button[kind="primary"]:hover, .stDownloadButton > button[kind="primary"]:hover {
  transform: translateY(-1px);
  box-shadow: 0 10px 26px -8px rgba(34,211,238,.75) !important;
}
.stButton > button:focus-visible, .stDownloadButton > button:focus-visible,
[data-testid="stSidebar"] .stButton > button:focus-visible {
  outline: none !important;
  box-shadow: var(--ring) !important;
}

/* ── Headings ──────────────────────────────────────────── */
h1 { font-size: clamp(1.35rem, 2.6vw, 1.75rem) !important; font-weight: 800 !important; letter-spacing: -.022em !important; color: var(--text) !important; margin: .2rem 0 .35rem !important; }
h2 { font-size: clamp(1.15rem, 2.2vw, 1.4rem) !important; font-weight: 700 !important; color: var(--text) !important; letter-spacing: -.018em !important; margin: .4rem 0 .4rem !important; }
h3 { font-size: clamp(.95rem, 1.6vw, 1.05rem) !important; font-weight: 700 !important; color: var(--text) !important; }
h4 { font-size: .78rem !important; font-weight: 700 !important; color: var(--muted) !important; text-transform: uppercase; letter-spacing: .08em; margin: .8rem 0 .35rem !important; }

/* Section heading with icon */
.nl-h {
  display: flex; align-items: center; gap: .55rem;
  font-size: 1rem; font-weight: 700; color: var(--text);
  margin: .9rem 0 .55rem;
}
.nl-h .ico {
  width: 26px; height: 26px; display: grid; place-items: center;
  border-radius: 8px; background: rgba(34,211,238,.10);
  border: 1px solid rgba(34,211,238,.28); color: var(--cyan);
}

/* ── Card ──────────────────────────────────────────────── */
.nl-card {
  background: linear-gradient(180deg, rgba(20,37,67,.85), rgba(15,30,51,.85));
  border: 1px solid var(--border);
  border-radius: var(--r);
  padding: 1.05rem 1.15rem;
  height: 100%;
  box-shadow: var(--sh-sm), var(--sh-in);
  transition: border-color .2s var(--ease), transform .2s var(--ease), box-shadow .2s var(--ease);
}
.nl-card:hover {
  border-color: rgba(34,211,238,.30);
  transform: translateY(-2px);
  box-shadow: var(--sh), var(--sh-in);
}
.nl-card h3 { margin: 0 0 .4rem; display: flex; align-items: center; gap: .5rem; }
.nl-card p  { color: var(--muted); font-size: .84rem; line-height: 1.6; margin: .2rem 0 0; }
.nl-card .ic {
  width: 30px; height: 30px; display: grid; place-items: center;
  border-radius: 9px; background: rgba(34,211,238,.10);
  border: 1px solid rgba(34,211,238,.28); color: var(--cyan);
  flex-shrink: 0;
}

/* ── Metric card ───────────────────────────────────────── */
.nl-metric {
  background: linear-gradient(180deg, rgba(20,37,67,.9), rgba(15,30,51,.9));
  border: 1px solid var(--border);
  border-radius: var(--r);
  padding: 1rem 1.1rem;
  box-shadow: var(--sh-sm), var(--sh-in);
  position: relative;
  overflow: hidden;
}
.nl-metric::after {
  content: "";
  position: absolute; inset: 0;
  background: radial-gradient(300px 100px at 100% 0%, rgba(34,211,238,.10), transparent 70%);
  pointer-events: none;
}
.nl-metric-head {
  display: flex; align-items: center; justify-content: space-between;
  gap: .5rem; margin-bottom: .35rem;
}
.nl-metric-label {
  font-size: .68rem; font-weight: 700; color: var(--muted);
  text-transform: uppercase; letter-spacing: .09em;
}
.nl-metric-icon {
  width: 26px; height: 26px; display: grid; place-items: center;
  border-radius: 7px; background: rgba(34,211,238,.10);
  border: 1px solid rgba(34,211,238,.24); color: var(--cyan);
  flex-shrink: 0;
}
.nl-metric-value {
  font-size: clamp(1.25rem, 2.2vw, 1.55rem);
  font-weight: 800;
  color: var(--text);
  font-variant-numeric: tabular-nums;
  letter-spacing: -.022em;
  line-height: 1.15;
}

/* ── Responsive auto-fit grids ─────────────────────────── */
.nl-grid       { display: grid; gap: .7rem; }
.nl-grid.cols-2{ grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); }
.nl-grid.cols-3{ grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); }
.nl-grid.cols-4{ grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); }
.nl-grid.cols-5{ grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); }
.nl-grid.cols-7{ grid-template-columns: repeat(auto-fit, minmax(120px, 1fr)); }

/* ── Diagnostic result ─────────────────────────────────── */
.nl-result {
  position: relative;
  background:
    radial-gradient(600px 120px at 0% 0%, rgba(34,211,238,.16), transparent 60%),
    linear-gradient(135deg, rgba(37,99,235,.16), rgba(16,185,129,.08));
  border: 1px solid rgba(34,211,238,.42);
  border-radius: var(--r-lg);
  padding: 1.3rem 1.4rem;
  margin: .55rem 0 .8rem;
  box-shadow: var(--sh), var(--sh-in);
  overflow: hidden;
}
.nl-result::before {
  content: "";
  position: absolute; left: 0; top: 0; bottom: 0; width: 4px;
  background: linear-gradient(180deg, var(--cyan), var(--emerald));
}
.nl-result-label {
  display: flex; align-items: center; gap: .4rem;
  font-size: .66rem; font-weight: 800;
  color: var(--cyan);
  text-transform: uppercase; letter-spacing: .12em;
  margin-bottom: .55rem;
}
.nl-result-prediction {
  font-size: clamp(1.5rem, 3.2vw, 2.05rem);
  font-weight: 800;
  color: var(--text);
  letter-spacing: -.028em; line-height: 1.05;
}
.nl-result-conf {
  font-size: .9rem; color: var(--text-2);
  margin-top: .55rem;
  display: flex; align-items: center; gap: .55rem; flex-wrap: wrap;
}

/* ── Prob grid ─────────────────────────────────────────── */
.nl-prob-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
  gap: .6rem;
  margin: .55rem 0;
}
.nl-prob-card {
  background: linear-gradient(180deg, rgba(20,37,67,.9), rgba(15,30,51,.9));
  border: 1px solid var(--border);
  border-radius: var(--r);
  padding: .9rem 1rem;
  position: relative;
  box-shadow: var(--sh-sm), var(--sh-in);
}
.nl-prob-card.top {
  border-color: rgba(34,211,238,.55);
  background: linear-gradient(135deg, rgba(37,99,235,.20), rgba(16,185,129,.12));
  box-shadow: var(--sh), 0 0 0 1px rgba(34,211,238,.16), var(--sh-in);
}
.nl-prob-value {
  font-size: 1.35rem; font-weight: 800;
  color: var(--text); font-family: var(--mono);
  letter-spacing: -.02em;
}
.nl-prob-card.top .nl-prob-value { color: var(--cyan); }
.nl-prob-label { font-size: .78rem; color: var(--muted); margin-top: .3rem; font-weight: 500; }
.nl-prob-top-tag {
  position: absolute; top: .5rem; right: .55rem;
  font-size: .55rem; font-weight: 800;
  color: var(--cyan);
  text-transform: uppercase; letter-spacing: .09em;
  background: rgba(34,211,238,.15);
  border: 1px solid rgba(34,211,238,.42);
  border-radius: 999px; padding: .12rem .5rem;
}

/* ── Uncertainty card ──────────────────────────────────── */
.nl-unc {
  background: linear-gradient(135deg, rgba(139,92,246,.10), rgba(139,92,246,.04));
  border: 1px solid rgba(139,92,246,.34);
  border-radius: var(--r);
  padding: .95rem 1.1rem;
  margin: .55rem 0;
  display: flex; justify-content: space-between; align-items: center;
  gap: 1rem; flex-wrap: wrap;
  box-shadow: var(--sh-sm), var(--sh-in);
}
.nl-unc-title { font-size: .66rem; font-weight: 800; color: #b8a4ff; text-transform: uppercase; letter-spacing: .1em; }
.nl-unc-value { font-size: 1.3rem; font-weight: 800; font-family: var(--mono); letter-spacing: -.02em; }
.nl-unc-band  { font-size: .76rem; margin-top: .1rem; font-weight: 600; }

/* ── XAI agree ─────────────────────────────────────────── */
.nl-agree {
  background: linear-gradient(135deg, rgba(34,211,238,.08), rgba(34,211,238,.02));
  border: 1px solid rgba(34,211,238,.30);
  border-radius: var(--r);
  padding: .9rem 1.1rem;
  margin: .5rem 0;
  display: flex; justify-content: space-between; align-items: center;
  gap: 1rem; flex-wrap: wrap;
  box-shadow: var(--sh-sm), var(--sh-in);
}
.nl-agree-score {
  font-size: 1.5rem; font-weight: 800;
  font-family: var(--mono); letter-spacing: -.02em;
}

/* ── History row ───────────────────────────────────────── */
.nl-hist-row {
  display: flex; align-items: center; gap: 1rem;
  padding: .8rem 1rem;
  border: 1px solid var(--border);
  border-radius: var(--r);
  background: linear-gradient(180deg, rgba(20,37,67,.8), rgba(15,30,51,.8));
  margin: .4rem 0;
  box-shadow: var(--sh-sm), var(--sh-in);
  transition: border-color .18s var(--ease), transform .15s var(--ease);
}
.nl-hist-row:hover {
  border-color: rgba(34,211,238,.35);
  transform: translateX(2px);
}
.nl-hist-title { font-weight: 700; color: var(--text); font-size: .9rem; }
.nl-hist-meta  { font-size: .72rem; color: var(--muted); margin-top: .18rem; font-family: var(--mono); }

/* ── Badges ────────────────────────────────────────────── */
.nl-badge {
  display: inline-flex; align-items: center; gap: .3rem;
  padding: .22rem .65rem;
  border-radius: 999px;
  font-size: .72rem; font-weight: 700;
  font-family: var(--mono);
}
.nl-badge-hi  { background: rgba(52,211,153,.14); border: 1px solid rgba(52,211,153,.44); color: #34d399; }
.nl-badge-mid { background: rgba(251,191,36,.14);  border: 1px solid rgba(251,191,36,.44);  color: #fbbf24; }
.nl-badge-lo  { background: rgba(248,113,113,.14); border: 1px solid rgba(248,113,113,.44); color: #f87171; }

/* ── Distribution bar ──────────────────────────────────── */
.nl-dist {
  padding: .75rem .95rem;
  border: 1px solid var(--border);
  border-radius: var(--r);
  background: linear-gradient(180deg, rgba(20,37,67,.85), rgba(15,30,51,.85));
  margin: .35rem 0;
  box-shadow: var(--sh-sm), var(--sh-in);
}
.nl-dist-row { display: flex; justify-content: space-between; align-items: center; margin-bottom: .5rem; }
.nl-dist-name  { font-size: .84rem; font-weight: 700; color: var(--text); }
.nl-dist-count { font-size: .74rem; color: var(--muted); font-family: var(--mono); }
.nl-dist-track { height: 6px; border-radius: 999px; background: rgba(96,165,250,.12); overflow: hidden; }
.nl-dist-fill  {
  height: 100%; border-radius: 999px;
  background: linear-gradient(90deg, var(--cyan), var(--emerald));
  transition: width 1s var(--ease-out);
}

/* ── Engine status ─────────────────────────────────────── */
.nl-engine {
  padding: 1rem 1.15rem;
  border-radius: var(--r-lg);
  margin: .5rem 0;
  box-shadow: var(--sh-sm), var(--sh-in);
}
.nl-engine.ok  {
  background: linear-gradient(135deg, rgba(16,185,129,.10), rgba(34,211,238,.06));
  border: 1px solid rgba(16,185,129,.40);
}
.nl-engine.err {
  background: linear-gradient(135deg, rgba(239,68,68,.10), rgba(239,68,68,.04));
  border: 1px solid rgba(239,68,68,.38);
}
.nl-engine-title {
  display: flex; align-items: center; gap: .5rem;
  font-weight: 800; margin-bottom: .7rem; font-size: .95rem;
}
.nl-engine.ok  .nl-engine-title { color: var(--emerald); }
.nl-engine.err .nl-engine-title { color: var(--danger); }
.nl-engine p { font-size: .82rem; color: var(--muted); margin: .2rem 0; }

.nl-spec-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: .5rem;
  margin-top: .25rem;
}
.nl-spec-item {
  background: rgba(10,26,48,.55);
  border: 1px solid rgba(34,211,238,.20);
  border-radius: 9px;
  padding: .6rem .8rem;
}
.nl-spec-key   { font-size: .58rem; font-weight: 800; color: var(--muted); text-transform: uppercase; letter-spacing: .1em; }
.nl-spec-value { font-size: .85rem; font-weight: 700; color: var(--text); font-family: var(--mono); margin-top: .18rem; }

/* ── Empty state ───────────────────────────────────────── */
.nl-empty {
  padding: 3rem 1rem;
  text-align: center;
  border: 1.5px dashed rgba(96,165,250,.30);
  border-radius: var(--r-lg);
  color: var(--muted);
  background: rgba(15,30,51,.35);
}
.nl-empty .em-ic {
  width: 52px; height: 52px; margin: 0 auto .8rem;
  display: grid; place-items: center;
  border-radius: 50%;
  background: rgba(34,211,238,.08);
  border: 1px solid rgba(34,211,238,.28);
  color: var(--cyan);
}
.nl-empty h3 { color: var(--text); margin: .2rem 0 .35rem; }
.nl-empty p  { font-size: .88rem; margin: 0; }

/* ── Activity feed ─────────────────────────────────────── */
.nl-feed { max-height: 380px; overflow-y: auto; padding-right: .25rem; }
.nl-feed-row {
  display: flex; gap: .65rem;
  padding: .5rem .25rem;
  border-bottom: 1px solid rgba(96,165,250,.08);
  font-size: .76rem;
  align-items: flex-start;
}
.nl-feed-row:last-child { border-bottom: 0; }
.nl-feed-time { color: var(--muted); font-family: var(--mono); flex: 0 0 auto; font-size: .68rem; padding-top: 1px; }
.nl-feed-msg  { color: var(--text-2); line-height: 1.45; }
.nl-feed-row.err  .nl-feed-msg { color: var(--danger); }
.nl-feed-row.warn .nl-feed-msg { color: var(--amber); }
.nl-feed-row.ok   .nl-feed-msg { color: var(--emerald); }

/* ── Sidebar brand ─────────────────────────────────────── */
.nl-brand {
  padding: .9rem .4rem 1rem;
  border-bottom: 1px solid var(--border);
  margin-bottom: .85rem;
}
.nl-brand-top { display: flex; align-items: center; gap: .7rem; }
.nl-brand-logo {
  width: 40px; height: 40px; display: grid; place-items: center;
  border-radius: 11px;
  background: linear-gradient(135deg, rgba(34,211,238,.20), rgba(52,211,153,.14));
  border: 1px solid rgba(34,211,238,.42);
  color: var(--cyan);
  box-shadow: inset 0 1px 0 rgba(255,255,255,.06), 0 6px 20px -8px rgba(34,211,238,.55);
}
.nl-brand-name { font-size: 1.02rem; font-weight: 800; color: var(--text); letter-spacing: -.012em; line-height: 1.1; }
.nl-brand-name span { color: var(--cyan); }
.nl-brand-sub  {
  font-size: .72rem; color: var(--muted); margin-top: .35rem;
  display: flex; align-items: center; gap: .4rem;
}
.nl-brand-sub .dot {
  width: 8px; height: 8px; border-radius: 50%;
  box-shadow: 0 0 0 4px rgba(0,0,0,0);
  animation: pulse 2.4s var(--ease) infinite;
}
.nl-brand-sub .dot.ok  { background: var(--emerald); box-shadow: 0 0 0 0 rgba(52,211,153,.55); }
.nl-brand-sub .dot.err { background: var(--danger);  box-shadow: 0 0 0 0 rgba(248,113,113,.55); }
@keyframes pulse {
  0%   { box-shadow: 0 0 0 0 rgba(52,211,153,.5); }
  70%  { box-shadow: 0 0 0 8px rgba(52,211,153,0); }
  100% { box-shadow: 0 0 0 0 rgba(52,211,153,0); }
}

/* ── Scrollbar ─────────────────────────────────────────── */
::-webkit-scrollbar { width: 9px; height: 9px; }
::-webkit-scrollbar-track { background: rgba(5,14,34,.6); }
::-webkit-scrollbar-thumb {
  background: linear-gradient(180deg, rgba(34,211,238,.42), rgba(6,182,212,.32));
  border-radius: 8px;
  border: 2px solid transparent;
  background-clip: padding-box;
}
::-webkit-scrollbar-thumb:hover { background: rgba(34,211,238,.65); background-clip: padding-box; }

/* ── Disclaimer ────────────────────────────────────────── */
.nl-disclaimer {
  border-left: 3px solid var(--amber);
  background: linear-gradient(90deg, rgba(245,158,11,.10), rgba(245,158,11,.02));
  border-radius: 0 var(--r) var(--r) 0;
  padding: .8rem 1rem;
  font-size: .82rem;
  color: var(--text-2);
  margin: .7rem 0 1rem;
  line-height: 1.6;
  display: flex; gap: .7rem; align-items: flex-start;
  box-shadow: var(--sh-sm);
}
.nl-disclaimer b { color: var(--amber); }
.nl-disclaimer svg { color: var(--amber); flex-shrink: 0; margin-top: 2px; }

/* ── Summary rows ──────────────────────────────────────── */
.nl-summary-row {
  display: flex; justify-content: space-between; gap: .8rem;
  padding: .5rem 0;
  border-bottom: 1px solid rgba(96,165,250,.10);
  font-size: .82rem;
}
.nl-summary-row:last-child { border-bottom: 0; }
.nl-summary-key { color: var(--muted); font-weight: 500; }
.nl-summary-val { color: var(--text); font-family: var(--mono); font-weight: 600; text-align: right; }

/* ── Steps row ─────────────────────────────────────────── */
.nl-step {
  text-align: center;
  padding: 1rem .55rem;
  background: linear-gradient(180deg, rgba(20,37,67,.85), rgba(15,30,51,.85));
  border: 1px solid var(--border);
  border-radius: var(--r);
  height: 100%;
  box-shadow: var(--sh-sm), var(--sh-in);
  transition: transform .18s var(--ease), border-color .18s var(--ease);
}
.nl-step:hover { transform: translateY(-2px); border-color: rgba(34,211,238,.35); }
.nl-step-num {
  width: 32px; height: 32px;
  margin: 0 auto .6rem;
  display: grid; place-items: center;
  border-radius: 50%;
  background: linear-gradient(135deg, var(--cyan), var(--emerald));
  color: #05121c;
  font-size: .8rem; font-weight: 800;
  box-shadow: 0 6px 16px -6px rgba(34,211,238,.6);
}
.nl-step-title { font-size: .8rem; font-weight: 600; color: var(--text); }

/* ── Footer ────────────────────────────────────────────── */
.nl-footer {
  margin-top: 2.6rem;
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
.nl-footer strong { color: var(--text-2); font-weight: 700; }
.nl-footer .rt { display: flex; align-items: center; gap: .55rem; font-family: var(--mono); font-size: .72rem; }
.nl-footer .rt .sep { color: rgba(125,165,220,.35); }

/* ── Ticker ────────────────────────────────────────────── */
.nl-ticker {
  display: flex;
  align-items: center;
  gap: .3rem;
  overflow-x: auto;
  padding: .5rem .6rem;
  background: linear-gradient(180deg, rgba(14,40,88,.55), rgba(9,23,48,.55));
  border: 1px solid var(--border);
  border-radius: var(--r);
  margin-bottom: 1rem;
  font-size: .72rem;
  font-family: var(--mono);
  scrollbar-width: none;
  white-space: nowrap;
  box-shadow: var(--sh-sm), var(--sh-in);
}
.nl-ticker::-webkit-scrollbar { display: none; }
.nl-ticker-item {
  display: inline-flex; align-items: center; gap: .4rem;
  padding: .35rem .7rem;
  border-radius: 8px;
  color: var(--muted);
  flex: 0 0 auto;
}
.nl-ticker-item .ic {
  width: 20px; height: 20px; display: grid; place-items: center;
  border-radius: 6px; background: rgba(34,211,238,.10);
  border: 1px solid rgba(34,211,238,.24); color: var(--cyan);
  flex-shrink: 0;
}
.nl-ticker-item strong { color: var(--text); font-weight: 700; }
.nl-ticker-sep {
  width: 1px; align-self: stretch; margin: .35rem .15rem;
  background: linear-gradient(180deg, transparent, rgba(125,165,220,.24), transparent);
}

/* ── Info rows (Settings) ──────────────────────────────── */
.nl-info-row {
  display: flex; justify-content: space-between; gap: .8rem;
  padding: .55rem 0;
  border-bottom: 1px solid var(--border);
  font-size: .82rem;
}
.nl-info-row:last-child { border-bottom: 0; }
.nl-info-key { color: var(--muted); }
.nl-info-val { color: var(--text); font-family: var(--mono); font-weight: 600; text-align: right; word-break: break-word; }

/* ── Streamlit widget polish ──────────────────────────── */
[data-testid="stFileUploader"] section {
  border: 1.5px dashed rgba(96,165,250,.32) !important;
  border-radius: var(--r) !important;
  background: rgba(15,30,51,.55) !important;
  transition: border-color .2s var(--ease), background .2s var(--ease) !important;
}
[data-testid="stFileUploader"] section:hover {
  border-color: rgba(34,211,238,.55) !important;
  background: rgba(20,37,67,.7) !important;
}
[data-testid="stExpander"] {
  border: 1px solid var(--border) !important;
  border-radius: var(--r) !important;
  background: rgba(15,30,51,.55) !important;
  overflow: hidden;
}
[data-testid="stExpander"] summary { font-weight: 600 !important; }

/* ── Mobile / tablet responsiveness ────────────────────── */
@media (max-width: 900px) {
  [data-testid="stMainBlockContainer"] {
    padding: .85rem .7rem 3rem !important;
  }
  .nl-grid.cols-5, .nl-grid.cols-4 { grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); }
}
@media (max-width: 640px) {
  .nl-result-prediction { font-size: 1.45rem; }
  .nl-prob-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
  .nl-metric-value { font-size: 1.15rem; }
  .nl-step { padding: .75rem .4rem; }
  .nl-step-num { width: 26px; height: 26px; font-size: .72rem; }
  .nl-footer { flex-direction: column; align-items: flex-start; }
  [data-testid="stSidebar"] .stButton > button { font-size: .84rem !important; min-height: 40px !important; }
}

@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after {
    animation-duration: .01ms !important;
    animation-iteration-count: 1 !important;
    transition-duration: .01ms !important;
    scroll-behavior: auto !important;
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


def _st_version_tuple() -> Tuple[int, int]:
    try:
        parts = st.__version__.split(".")
        return int(parts[0]), int(parts[1])
    except Exception:
        return (0, 0)


# Material Symbols inside button icons requires Streamlit >= 1.36.
# Only use names that are guaranteed to exist in the bundled Material Symbols font.
_MATERIAL_BTN_ICONS = {
    "home", "document_scanner", "visibility", "bar_chart",
    "history", "settings", "delete_sweep", "rocket_launch",
    "play_arrow", "image", "description", "download", "restart_alt",
}


def _btn_icon(material_name: str) -> dict:
    """Return {'icon': ...} only when the running Streamlit both supports
    Material Symbols in buttons (>= 1.36) and the icon name is known-valid."""
    if _st_version_tuple() < (1, 36):
        return {}
    if material_name not in _MATERIAL_BTN_ICONS:
        return {}
    return {"icon": f":material/{material_name}:"}


def _st_width(container: bool) -> dict:
    major, minor = _st_version_tuple()
    if (major, minor) >= (1, 42):
        return {"width": "stretch" if container else "content"}
    return {"use_container_width": bool(container)}


def _W() -> dict:   return _st_width(True)
def _Wc() -> dict:  return _st_width(False)


def section_heading(title: str, icon: str) -> str:
    return (
        f'<div class="nl-h"><span class="ico">{svg(icon, 15, 2)}</span>'
        f'<span>{_e(title)}</span></div>'
    )


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

    items = [
        ("activity", "Scans",       str(total)),
        ("target",   "Avg Conf",    f"{avg_c:.1f}%"),
        ("pulse",    "Throughput",  f"{thr:.2f}/min"),
        ("check",    "Last Result", latest_pred),
        ("layers",   "Model",       model_name or "—"),
        ("cpu",      "Device",      str(DEVICE)),
    ]
    inner = ""
    for i, (ic, label, val) in enumerate(items):
        if i > 0:
            inner += '<span class="nl-ticker-sep"></span>'
        inner += (
            f'<span class="nl-ticker-item">'
            f'<span class="ic">{svg(ic, 12, 2)}</span>'
            f'{_e(label)} <strong>{_e(val)}</strong>'
            f'</span>'
        )
    st.markdown(f'<div class="nl-ticker">{inner}</div>', unsafe_allow_html=True)


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

    h = 118 + 34 * len(items) + (135 + 34 * len(items) if mc_result else 0)
    components.html(f"""
    <style>
    .lp-wrap{{padding:1rem 1.1rem;border-radius:12px;border:1px solid rgba(125,165,220,.14);
    background:linear-gradient(180deg,#142543,#0f1e33);font-family:Inter,system-ui,sans-serif;color:#e8eefb;
    box-shadow:inset 0 1px 0 rgba(255,255,255,.04)}}
    .lp-head{{font-size:.66rem;font-weight:800;color:#22d3ee;text-transform:uppercase;letter-spacing:.1em;margin-bottom:.7rem;
    display:flex;align-items:center;gap:.45rem}}
    .lp-row{{display:flex;align-items:center;gap:.65rem;margin:.35rem 0}}
    .lp-label{{min-width:125px;font-size:.76rem;color:#b9cbe2;font-weight:500}}
    .lp-track{{flex:1;height:6px;border-radius:999px;background:rgba(59,130,246,.14);overflow:hidden}}
    .lp-fill{{height:100%;width:0%;border-radius:999px;background:linear-gradient(90deg,#22d3ee,#34d399);
    transition:width 1.1s cubic-bezier(.16,.84,.44,1)}}
    .lp-mc{{background:linear-gradient(90deg,#a78bfa,#22d3ee)!important}}
    .lp-val{{min-width:52px;text-align:right;font-family:'JetBrains Mono',monospace;font-size:.72rem;color:#e8eefb;font-weight:600}}
    .lp-sec{{font-size:.6rem;font-weight:800;color:#7f9bbd;text-transform:uppercase;letter-spacing:.12em;margin:.75rem 0 .3rem}}
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
        st.markdown(
            f'<div style="color:var(--muted);font-size:.82rem;padding:.6rem .2rem;'
            f'display:flex;align-items:center;gap:.5rem">{svg("info", 14)} No activity yet.</div>',
            unsafe_allow_html=True,
        )
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
        f'<span class="rt">'
        f'{svg("cpu", 13)} PyTorch {torch.__version__.split("+")[0]}'
        f'<span class="sep">·</span>'
        f'Streamlit {st.__version__}'
        f'<span class="sep">·</span>'
        f'{"CUDA" if DEVICE.type == "cuda" else "CPU"}'
        f'</span>'
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

    dot_cls  = "ok" if engine_ok else "err"
    dot_txt  = "Engine ready" if engine_ok else "Engine offline"

    st.markdown(
        f'<div class="nl-brand">'
        f'<div class="nl-brand-top">'
        f'<div class="nl-brand-logo">{svg("brain", 22, 1.9)}</div>'
        f'<div>'
        f'<div class="nl-brand-name">NeuroLens <span>AI</span></div>'
        f'</div>'
        f'</div>'
        f'<div class="nl-brand-sub"><span class="dot {dot_cls}"></span>{dot_txt}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    nav_map = [
        ("Home",         "Home",         "home"),
        ("MRI Analysis", "MRI Analysis", "document_scanner"),
        ("XAI Lab",      "XAI Lab",      "visibility"),
        ("Dashboard",    "Dashboard",    "bar_chart"),
        ("History",      "History",      "history"),
        ("Settings",     "Settings",     "settings"),
    ]
    for page, label_txt, mat_icon in nav_map:
        active = st.session_state.nav == page
        badge  = f"  ({total_scans})" if page in ("MRI Analysis", "History") and total_scans else ""
        nav_kwargs = dict(
            key=f"nav_{page.lower().replace(' ','_')}",
            type="primary" if active else "secondary",
            **_W(),
        )
        icon_kw = _btn_icon(mat_icon)
        try:
            clicked = st.button(label_txt + badge, **icon_kw, **nav_kwargs)
        except Exception:
            # Older Streamlit or unsupported icon — retry without icon
            clicked = st.button(label_txt + badge, **nav_kwargs)
        if clicked:
            navigate_to(page)

    st.divider()

    if hist_sb:
        avg_c = float(np.mean([h["confidence"] for h in hist_sb]))
        st.markdown(
            f'<div style="font-size:.74rem;color:var(--muted);padding:.1rem .15rem .5rem;'
            f'display:flex;align-items:center;gap:.45rem">{svg("activity", 12)} '
            f'Session · <strong style="color:var(--text-2)">{total_scans} scans</strong> · avg '
            f'<strong style="color:var(--cyan)">{avg_c:.1f}%</strong></div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div style="font-size:.74rem;color:var(--muted);padding:.1rem .15rem .5rem;'
            f'display:flex;align-items:center;gap:.45rem">{svg("info", 12)} No scans this session</div>',
            unsafe_allow_html=True,
        )

    if st.button("Clear History", key="sb_clear", **_btn_icon("delete_sweep"), **_W()):
        st.session_state.sb_clear_confirm = True
    if st.session_state.get("sb_clear_confirm"):
        st.warning("Clear all session data?")
        c1, c2 = st.columns(2)
        if c1.button("Yes", key="sb_clear_yes", type="primary", **_W()):
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

    st.markdown(
        f'<h2 style="display:flex;align-items:center;gap:.6rem">'
        f'<span style="width:32px;height:32px;display:grid;place-items:center;border-radius:10px;'
        f'background:linear-gradient(135deg,rgba(34,211,238,.22),rgba(52,211,153,.14));'
        f'border:1px solid rgba(34,211,238,.42);color:#22d3ee">{svg("brain", 18, 2)}</span>'
        f'NeuroLens AI</h2>'
        f'<p style="color:var(--muted);max-width:64ch;margin:.2rem 0 1.1rem;font-size:.9rem;line-height:1.65">'
        f'Upload a brain MRI scan to receive an AI classification across four tumor types, '
        f'with Grad-CAM explainability and MC Dropout uncertainty quantification.</p>',
        unsafe_allow_html=True,
    )

    _, c, _ = st.columns([2, 1, 2])
    with c:
        if st.button("Run MRI Analysis", type="primary", key="home_cta",
                     **_btn_icon("rocket_launch"), **_W()):
            navigate_to("MRI Analysis")

    st.write("")

    # Metrics grid
    hist = st.session_state.prediction_history
    avg_lat = f"{np.mean([x.get('latency_ms',0) for x in hist]):.0f} ms" if hist else "—"

    metrics = [
        ("activity", "Scans",         str(st.session_state.live_predictions_count)),
        ("target",   "Avg Confidence",f"{st.session_state.live_avg_confidence:.1f}%"),
        ("pulse",    "Throughput",    f"{st.session_state.live_throughput:.2f}/min"),
        ("clock",    "Avg Inference", avg_lat),
    ]
    st.markdown(
        '<div class="nl-grid cols-4">' + "".join(
            f'<div class="nl-metric">'
            f'<div class="nl-metric-head">'
            f'<div class="nl-metric-label">{_e(lbl)}</div>'
            f'<div class="nl-metric-icon">{svg(ic, 14, 2)}</div>'
            f'</div>'
            f'<div class="nl-metric-value">{_e(val)}</div>'
            f'</div>'
            for ic, lbl, val in metrics
        ) + '</div>',
        unsafe_allow_html=True,
    )

    st.write("")

    # Feature cards
    features = [
        ("brain", "Brain MRI Classification", "Classifies into Glioma, Meningioma, No Tumor, or Pituitary tumor."),
        ("activity", "MC Dropout Uncertainty", "Bayesian uncertainty via repeated stochastic forward passes."),
        ("eye", "Dual XAI (Grad-CAM / ++)", "Grad-CAM and Grad-CAM++ heatmaps with Pearson agreement score."),
        ("zap", "Real-Time Inference", "Sub-100 ms on GPU; CPU fallback always available."),
    ]
    st.markdown(
        '<div class="nl-grid cols-4">' + "".join(
            f'<div class="nl-card">'
            f'<h3><span class="ic">{svg(ic, 15, 2)}</span>{_e(title)}</h3>'
            f'<p>{_e(body)}</p>'
            f'</div>'
            for ic, title, body in features
        ) + '</div>',
        unsafe_allow_html=True,
    )

    st.write("")
    st.markdown(section_heading("Analysis Pipeline", "layers"), unsafe_allow_html=True)
    steps = [("1", "Upload"), ("2", "Preprocess"), ("3", "Inference"),
             ("4", "MC Dropout"), ("5", "Grad-CAM"), ("6", "Grad-CAM++"), ("7", "Report")]
    st.markdown(
        '<div class="nl-grid cols-7">' + "".join(
            f'<div class="nl-step"><div class="nl-step-num">{n}</div>'
            f'<div class="nl-step-title">{_e(t)}</div></div>'
            for n, t in steps
        ) + '</div>',
        unsafe_allow_html=True,
    )

    st.write("")

    # Engine status
    if model_error:
        st.markdown(
            f'<div class="nl-engine err">'
            f'<div class="nl-engine-title">{svg("alert", 18, 2)} Neural Engine Unavailable</div>'
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
            f'<div class="nl-engine-title">{svg("check", 18, 2)} Neural Engine Ready</div>'
            f'<div class="nl-spec-grid">{spec_html}</div></div>',
            unsafe_allow_html=True,
        )

    if st.session_state.last_result:
        st.write("")
        st.markdown(section_heading("Last Analysis", "file"), unsafe_allow_html=True)
        render_prob_bars(st.session_state.last_result, mc_result=st.session_state.mc_result)

    render_footer()


# ═══════════════════════════════════════════════════════════════
# MRI ANALYSIS
# ═══════════════════════════════════════════════════════════════

elif nav == "MRI Analysis":
    st.markdown(
        f'<h2 style="display:flex;align-items:center;gap:.6rem">'
        f'<span style="width:32px;height:32px;display:grid;place-items:center;border-radius:10px;'
        f'background:linear-gradient(135deg,rgba(34,211,238,.22),rgba(52,211,153,.14));'
        f'border:1px solid rgba(34,211,238,.42);color:#22d3ee">{svg("scan", 18, 2)}</span>'
        f'MRI Analysis</h2>',
        unsafe_allow_html=True,
    )
    render_ticker()

    st.markdown(
        f'<div class="nl-disclaimer">{svg("shield", 18, 2)}'
        f'<div><b>Research use only.</b> This tool is not a medical device and must not be used '
        f'for clinical diagnosis or treatment decisions.</div></div>',
        unsafe_allow_html=True,
    )

    if model_error:
        st.error(f"Neural engine unavailable: {model_error}")
        st.stop()

    col_up, col_res = st.columns([1, 1.3])

    with col_up:
        st.markdown(section_heading("Upload Scan", "upload"), unsafe_allow_html=True)
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
                    if st.button("Analyze", type="primary", key="run_analysis",
                                 **_btn_icon("play_arrow"), **_W()):
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
            st.markdown(
                f'<div class="nl-result">'
                f'<div class="nl-result-label">{svg("sparkles", 13, 2)} AI Classification</div>'
                f'<div class="nl-result-prediction">{_e(result["prediction"])}</div>'
                f'<div class="nl-result-conf"><strong>{conf_v:.2f}%</strong> confidence '
                f'<span class="nl-badge nl-badge-{bc}">'
                f'{"High" if bc=="hi" else "Moderate" if bc=="mid" else "Low"}</span></div>'
                f'</div>',
                unsafe_allow_html=True,
            )

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Confidence",    f"{conf_v:.2f}%")
            m2.metric("Inference",     f"{result['latency_ms']:.1f} ms")
            m3.metric("Preprocess",    f"{result.get('preprocessing_ms',0):.1f} ms")
            m4.metric("Total",         f"{result.get('total_ms',0):.1f} ms")

            if mc_res:
                unc, band, color = mc_res["uncertainty"], mc_res["band"], mc_res["color"]
                st.markdown(
                    f'<div class="nl-unc">'
                    f'<div><div class="nl-unc-title">MC Dropout · {st.session_state.get("mc_samples", MC_SAMPLES_DEF)} passes</div>'
                    f'<div class="nl-unc-value" style="color:{color}">σ = {unc:.4f}</div>'
                    f'<div class="nl-unc-band" style="color:{color}">{_e(band)}</div></div>'
                    f'<div style="text-align:right;font-size:.78rem;color:var(--muted)">'
                    f'MC conf: <strong style="color:#22d3ee">{mc_res["confidence"]:.2f}%</strong><br>'
                    f'Prediction: <strong style="color:var(--text)">{_e(mc_res["prediction"])}</strong></div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            agree = result.get("agreement_score")
            if agree is not None:
                ac = "#34d399" if agree >= 0.7 else "#fbbf24" if agree >= 0.5 else "#f87171"
                al = "High Agreement" if agree >= 0.7 else "Moderate Agreement" if agree >= 0.5 else "Low Agreement"
                st.markdown(
                    f'<div class="nl-agree">'
                    f'<span style="font-size:.78rem;color:var(--muted);display:flex;align-items:center;gap:.45rem">'
                    f'{svg("eye", 13)} Grad-CAM ↔ Grad-CAM++ Pearson correlation</span>'
                    f'<div style="text-align:right">'
                    f'<div class="nl-agree-score" style="color:{ac}">{agree:.3f}</div>'
                    f'<div style="font-size:.7rem;color:{ac};font-weight:600">{_e(al)}</div></div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            st.markdown(section_heading("Probability Distribution", "chart"), unsafe_allow_html=True)
            grid_html = '<div class="nl-prob-grid">' + "".join(
                f'<div class="nl-prob-card {"top" if lbl == result["prediction"] else ""}">'
                f'<div class="nl-prob-value">{p:.1f}%</div>'
                f'<div class="nl-prob-label">{_e(lbl)}</div>'
                f'{"<div class=\'nl-prob-top-tag\'>Top</div>" if lbl == result["prediction"] else ""}'
                f'</div>'
                for lbl, p in result["probabilities"].items()
            ) + '</div>'
            st.markdown(grid_html, unsafe_allow_html=True)

            st.markdown(section_heading("Live Probability Stream", "activity"), unsafe_allow_html=True)
            render_prob_bars(result, mc_result=mc_res)

            if gc_png or pp_png:
                st.markdown(section_heading("Dual XAI Visualization", "eye"), unsafe_allow_html=True)
                gc_col, pp_col = st.columns(2)
                with gc_col:
                    st.markdown("**Grad-CAM** &nbsp;<span style='color:var(--muted);font-size:.75rem'>(jet · α=0.44)</span>", unsafe_allow_html=True)
                    if gc_png: st.image(gc_png, **_W())
                with pp_col:
                    st.markdown("**Grad-CAM++** &nbsp;<span style='color:var(--muted);font-size:.75rem'>(inferno · α=0.46)</span>", unsafe_allow_html=True)
                    if pp_png: st.image(pp_png, **_W())

            st.markdown(section_heading("Export", "download"), unsafe_allow_html=True)
            dl1, dl2, dl3 = st.columns(3)
            if gc_png:
                with dl1:
                    st.download_button("Grad-CAM", gc_png, "gradcam.png", "image/png",
                                       key="dl_gc", **_btn_icon("image"), **_W())
            if pp_png:
                with dl2:
                    st.download_button("Grad-CAM++", pp_png, "gradcam_pp.png", "image/png",
                                       key="dl_pp", **_btn_icon("image"), **_W())
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
                st.download_button("Report (.txt)", "\n".join(report_lines),
                                   "neurolens_report.txt", "text/plain",
                                   key="dl_report", type="primary",
                                   **_btn_icon("description"), **_W())

    render_footer()


# ═══════════════════════════════════════════════════════════════
# XAI LAB
# ═══════════════════════════════════════════════════════════════

elif nav == "XAI Lab":
    st.markdown(
        f'<h2 style="display:flex;align-items:center;gap:.6rem">'
        f'<span style="width:32px;height:32px;display:grid;place-items:center;border-radius:10px;'
        f'background:linear-gradient(135deg,rgba(34,211,238,.22),rgba(52,211,153,.14));'
        f'border:1px solid rgba(34,211,238,.42);color:#22d3ee">{svg("eye", 18, 2)}</span>'
        f'XAI Lab</h2>',
        unsafe_allow_html=True,
    )
    render_ticker()

    hist   = st.session_state.prediction_history
    gc_png = st.session_state.get("gradcam_png")
    pp_png = st.session_state.get("gradcam_pp_png")

    if not hist:
        st.markdown(
            f'<div class="nl-empty">'
            f'<div class="em-ic">{svg("eye", 22, 1.9)}</div>'
            f'<h3>No analyses yet</h3>'
            f'<p>Run an MRI analysis first to view explanations here.</p></div>',
            unsafe_allow_html=True,
        )
    else:
        result = st.session_state.last_result
        agree  = result.get("agreement_score") if result else None

        st.markdown(section_heading("Heatmaps — Most Recent Analysis", "layers"), unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Grad-CAM** &nbsp;<span style='color:var(--muted);font-size:.75rem'>(jet)</span>", unsafe_allow_html=True)
            if gc_png:
                st.image(gc_png, **_W())
                st.download_button("Download Grad-CAM", gc_png, "gradcam.png", "image/png",
                                   key="xai_gc", **_btn_icon("image"), **_W())
            else:
                st.info("Grad-CAM not available for this analysis.")
        with c2:
            st.markdown("**Grad-CAM++** &nbsp;<span style='color:var(--muted);font-size:.75rem'>(inferno)</span>", unsafe_allow_html=True)
            if pp_png:
                st.image(pp_png, **_W())
                st.download_button("Download Grad-CAM++", pp_png, "gradcam_pp.png", "image/png",
                                   key="xai_pp", **_btn_icon("image"), **_W())
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

        agree_vals = [h["agreement_score"] for h in hist if h.get("agreement_score") is not None]
        if len(agree_vals) >= 2:
            st.markdown(section_heading("Agreement Score Trend", "chart"), unsafe_allow_html=True)
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
    st.markdown(
        f'<h2 style="display:flex;align-items:center;gap:.6rem">'
        f'<span style="width:32px;height:32px;display:grid;place-items:center;border-radius:10px;'
        f'background:linear-gradient(135deg,rgba(34,211,238,.22),rgba(52,211,153,.14));'
        f'border:1px solid rgba(34,211,238,.42);color:#22d3ee">{svg("chart", 18, 2)}</span>'
        f'Dashboard</h2>',
        unsafe_allow_html=True,
    )
    render_ticker()

    dash_hist = st.session_state.prediction_history
    if not dash_hist:
        st.markdown(
            f'<div class="nl-empty">'
            f'<div class="em-ic">{svg("chart", 22, 1.9)}</div>'
            f'<h3>No data yet</h3>'
            f'<p>Run at least one MRI analysis to populate the dashboard.</p></div>',
            unsafe_allow_html=True,
        )
    else:
        total    = st.session_state.live_predictions_count
        avg_c    = st.session_state.live_avg_confidence
        unique   = len([k for k, v in st.session_state.live_class_counts.items() if v > 0])
        avg_lat  = float(np.mean([h.get("latency_ms", 0) for h in dash_hist]))
        unc_data = [h["uncertainty"] for h in dash_hist if h.get("uncertainty") is not None]
        avg_unc  = float(np.mean(unc_data)) if unc_data else None

        metrics = [
            ("activity", "Total Scans",     str(total)),
            ("target",   "Avg Confidence",  f"{avg_c:.1f}%"),
            ("layers",   "Classes Seen",    str(unique)),
            ("clock",    "Avg Latency",     f"{avg_lat:.0f} ms"),
            ("pulse",    "Avg Uncertainty", f"σ={avg_unc:.4f}" if avg_unc is not None else "—"),
        ]
        st.markdown(
            '<div class="nl-grid cols-5">' + "".join(
                f'<div class="nl-metric">'
                f'<div class="nl-metric-head">'
                f'<div class="nl-metric-label">{_e(lbl)}</div>'
                f'<div class="nl-metric-icon">{svg(ic, 14, 2)}</div>'
                f'</div>'
                f'<div class="nl-metric-value">{_e(val)}</div>'
                f'</div>'
                for ic, lbl, val in metrics
            ) + '</div>',
            unsafe_allow_html=True,
        )

        st.write("")
        col_l, col_r = st.columns([2, 1])

        with col_l:
            st.markdown(section_heading("Prediction Distribution", "chart"), unsafe_allow_html=True)
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

            st.markdown(section_heading("Confidence Trend", "activity"), unsafe_allow_html=True)
            fig = plot_confidence_trend(dash_hist)
            if fig: st.pyplot(fig, **_W()); plt.close(fig)
            else:   st.caption("Need ≥ 2 analyses.")

            st.markdown(section_heading("Inference Latency", "clock"), unsafe_allow_html=True)
            fig = plot_latency_trend(dash_hist)
            if fig: st.pyplot(fig, **_W()); plt.close(fig)
            else:   st.caption("Need ≥ 2 analyses.")

            if unc_data:
                st.markdown(section_heading("Uncertainty Trend", "pulse"), unsafe_allow_html=True)
                fig = plot_uncertainty_history(dash_hist)
                if fig: st.pyplot(fig, **_W()); plt.close(fig)

        with col_r:
            st.markdown(section_heading("Activity Feed", "database"), unsafe_allow_html=True)
            render_activity_feed()

            st.write("")
            st.markdown(section_heading("Latest Result", "file"), unsafe_allow_html=True)
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
    st.markdown(
        f'<h2 style="display:flex;align-items:center;gap:.6rem">'
        f'<span style="width:32px;height:32px;display:grid;place-items:center;border-radius:10px;'
        f'background:linear-gradient(135deg,rgba(34,211,238,.22),rgba(52,211,153,.14));'
        f'border:1px solid rgba(34,211,238,.42);color:#22d3ee">{svg("history", 18, 2)}</span>'
        f'History</h2>',
        unsafe_allow_html=True,
    )
    render_ticker()

    hist_list = st.session_state.prediction_history
    if not hist_list:
        st.markdown(
            f'<div class="nl-empty">'
            f'<div class="em-ic">{svg("history", 22, 1.9)}</div>'
            f'<h3>No history</h3>'
            f'<p>Analyses you run will appear here.</p></div>',
            unsafe_allow_html=True,
        )
    else:
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
                           key="hist_csv", **_btn_icon("download"), **_W())

    render_footer()


# ═══════════════════════════════════════════════════════════════
# SETTINGS
# ═══════════════════════════════════════════════════════════════

elif nav == "Settings":
    st.markdown(
        f'<h2 style="display:flex;align-items:center;gap:.6rem">'
        f'<span style="width:32px;height:32px;display:grid;place-items:center;border-radius:10px;'
        f'background:linear-gradient(135deg,rgba(34,211,238,.22),rgba(52,211,153,.14));'
        f'border:1px solid rgba(34,211,238,.42);color:#22d3ee">{svg("settings", 18, 2)}</span>'
        f'Settings</h2>',
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns(2)

    with c1:
        st.markdown(section_heading("Model", "brain"), unsafe_allow_html=True)
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
        st.markdown(section_heading("System", "cpu"), unsafe_allow_html=True)
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
    st.markdown(section_heading("Runtime Controls", "sliders"), unsafe_allow_html=True)

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
    st.markdown(section_heading("Technical Details", "info"), unsafe_allow_html=True)
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
    st.markdown(section_heading("Reset Session", "refresh"), unsafe_allow_html=True)
    if st.button("Reset All Session Data", type="primary", key="settings_reset",
                 **_btn_icon("restart_alt")):
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