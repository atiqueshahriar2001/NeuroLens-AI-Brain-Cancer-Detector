# =============================================================================
# NeuroLens AI — Neurodiagnostic Intelligence Platform
# Production SaaS Edition v3.0
# =============================================================================
# Architecture: CustomCNN · ResNet50 · EfficientNet-B0
# XAI: Grad-CAM + Grad-CAM++ (Dual)
# Uncertainty: MC Dropout (Monte Carlo sampling)
# Research: Uncertainty score · Reliability bands · Explanation agreement
# =============================================================================

import warnings
import time
import io
import hashlib
import platform
from importlib import metadata
from datetime import datetime
from pathlib import Path
from textwrap import dedent

import numpy as np
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image, UnidentifiedImageError

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import transforms
from torchvision.models import resnet50, efficientnet_b0

warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────────────
# HTML HELPERS — Streamlit-compat shims (old + new API)
# ─────────────────────────────────────────────────────────────────────────────
def safe_html(html: str) -> str:
    """Collapse newlines+indentation so Streamlit markdown keeps HTML as HTML."""
    return " ".join(line.strip() for line in html.strip().splitlines() if line.strip())


def _render_html(content: str, height: int = 200, scrolling: bool = False):
    """
    Render a raw HTML (with optional JS) block in an isolated frame.
    Tries the newest Streamlit API first (`st.iframe(srcdoc=...)`), then falls
    back to the legacy `components.html` for older Streamlit versions.
    """
    try:
        if hasattr(st, "iframe"):
            st.iframe(srcdoc=content, height=height, scrolling=scrolling)
            return
    except (AttributeError, TypeError):
        pass
    components.html(content, height=height, scrolling=scrolling)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NeuroLens AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
NORM_MEAN    = [0.485, 0.456, 0.406]
NORM_STD     = [0.229, 0.224, 0.225]
IMG_SIZE     = 224
MC_SAMPLES   = 20
MAX_HISTORY  = 200
CLASS_NAMES  = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]

BASE_DIR   = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "neurolens_best.pth"
if not MODEL_PATH.exists():
    for candidate in [BASE_DIR.parent / "neurolens_best.pth"]:
        if candidate.exists():
            MODEL_PATH = candidate
            break

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

test_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(NORM_MEAN, NORM_STD),
])


def uncertainty_band(uncertainty: float) -> tuple[str, str]:
    if uncertainty < 0.05:
        return "Very High Reliability", "#10b981"
    if uncertainty < 0.12:
        return "High Reliability", "#22d3ee"
    if uncertainty < 0.22:
        return "Moderate Reliability", "#f59e0b"
    return "Low Reliability", "#ef4444"


# ─────────────────────────────────────────────────────────────────────────────
# CSS — Design System v3.1 (Vercel/Linear-grade dark UI)
# ─────────────────────────────────────────────────────────────────────────────
STYLES = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Sora:wght@500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root {
    /* Surfaces */
    --bg:           #0a0e1a;
    --surface:      #0f1524;
    --surface-2:    #131a2c;
    --surface-3:    #1a2338;

    /* Borders */
    --line:         rgba(255,255,255,0.06);
    --line-strong:  rgba(255,255,255,0.12);

    /* Brand */
    --accent:       #0ea5e9;
    --accent-hi:    #38bdf8;
    --accent-soft:  rgba(14,165,233,0.10);
    --accent-line:  rgba(14,165,233,0.32);

    /* Semantic */
    --success:      #10b981;
    --warning:      #f59e0b;
    --danger:       #ef4444;
    --violet:       #8b5cf6;

    /* Text */
    --text-1:       #e8edf5;
    --text-2:       #a4b0c0;
    --text-3:       #6b7789;

    /* Type */
    --font-display: 'Sora', 'Inter', sans-serif;
    --font-body:    'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    --font-mono:    'JetBrains Mono', ui-monospace, monospace;

    /* Radius */
    --r-sm: 10px;
    --r-md: 14px;
    --r-lg: 18px;
    --r-pill: 999px;

    /* Shadow */
    --sh-1: 0 1px 2px rgba(0,0,0,.25);
    --sh-2: 0 4px 14px rgba(0,0,0,.30);
    --sh-3: 0 12px 32px rgba(0,0,0,.35);
    --sh-focus: 0 0 0 3px rgba(14,165,233,.28);

    /* Legacy aliases — keep existing inline HTML resolving */
    --bg-primary: var(--bg);
    --bg-secondary: var(--surface);
    --bg-card: var(--surface-2);
    --bg-glass: rgba(255,255,255,0.03);
    --border-subtle: var(--line);
    --border-hover: var(--accent-line);
    --accent-light: var(--accent-hi);
    --accent-glow: var(--accent-soft);
    --text-primary: var(--text-1);
    --text-secondary: var(--text-2);
    --text-muted: var(--text-3);
    --error: var(--danger);
    --warning: var(--warning);
    --success-var: var(--success);
    --radius-lg: var(--r-lg);
    --radius-md: var(--r-md);
    --radius-sm: var(--r-sm);
}

/* ── BASE ── */
* { font-family: var(--font-body); box-sizing: border-box; }
.stApp { background: var(--bg); color: var(--text-1); }
header[data-testid="stHeader"] { background: transparent; }
#MainMenu, footer { visibility: hidden; }

h1, h2, h3, .metric-value, .diagnostic-prediction,
.uncertainty-value, .hero h1, .sticky-brand-name, .footer-brand {
    font-family: var(--font-display);
    letter-spacing: -0.02em;
}

code, pre, .activity-feed, .activity-time {
    font-family: var(--font-mono);
}

/* ── SIDEBAR ── */
[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--line) !important;
    min-width: 288px !important;
    max-width: 288px !important;
    width: 288px !important;
}
[data-testid="stSidebar"] > div:first-child {
    width: 288px !important;
    padding: 1.1rem 0.75rem 1rem !important;
}
[data-testid="stSidebar"] ::-webkit-scrollbar { width: 4px; }
[data-testid="stSidebar"] ::-webkit-scrollbar-thumb {
    background: rgba(255,255,255,0.08); border-radius: 10px;
}

.sidebar-brand-title {
    font-family: var(--font-display);
    font-size: 1.15rem;
    font-weight: 700;
    color: var(--text-1);
    letter-spacing: -0.02em;
}
.sidebar-brand-subtitle {
    font-size: 0.65rem;
    color: var(--text-3);
    font-weight: 500;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin-top: 2px;
}
.sidebar-section-label {
    margin: 1.25rem 0 0.5rem;
    padding-left: 0.5rem;
    font-size: 0.62rem;
    font-weight: 700;
    letter-spacing: 0.14em;
    color: var(--text-3);
    text-transform: uppercase;
}

/* Sidebar buttons = nav items (compact) */
[data-testid="stSidebar"] .stButton { margin: 0.1rem 0 !important; }
[data-testid="stSidebar"] .stButton > button {
    width: 100% !important;
    min-height: 34px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    padding: 0.38rem 0.7rem !important;
    border-radius: var(--r-sm) !important;
    background: transparent !important;
    border: 1px solid transparent !important;
    color: var(--text-2) !important;
    font-size: 0.79rem !important;
    font-weight: 500 !important;
    transition: background 140ms ease, color 140ms ease, border-color 140ms ease !important;
    box-shadow: none !important;
}
[data-testid="stSidebar"] .stButton > button p {
    line-height: 1.15 !important;
    margin: 0 !important;
    font-size: 0.79rem !important;
    font-weight: 500 !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: rgba(255,255,255,0.035) !important;
    border-color: var(--line) !important;
    color: var(--text-1) !important;
    transform: none !important;
}
[data-testid="stSidebar"] .stButton > button:focus-visible {
    box-shadow: var(--sh-focus) !important;
    outline: none !important;
}

/* Nav active — replaces the old markdown wrapper hack */
.nav-active {
    position: relative !important;
    border-radius: var(--r-sm) !important;
    background: var(--accent-soft) !important;
    border: 1px solid var(--accent-line) !important;
    box-shadow: none !important;
    margin: -0.1rem 0 !important;
    padding: 0 !important;
    line-height: 0 !important;
}
.nav-active::before {
    content: "";
    position: absolute;
    left: 0; top: 22%;
    width: 2px; height: 56%;
    border-radius: 0 2px 2px 0;
    background: var(--accent);
}
.nav-active + [data-testid="stButton"] > button {
    background: transparent !important;
    border: none !important;
    color: var(--text-1) !important;
    font-weight: 600 !important;
}

.sidebar-active-indicator {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    height: 34px;
    padding: 0 0.75rem;
    margin: 0.1rem 0 0.6rem;
    border-radius: var(--r-sm);
    background: rgba(255,255,255,0.025);
    border: 1px solid var(--line);
    color: var(--text-2);
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.02em;
}
.sidebar-active-dot {
    flex: 0 0 6px; width: 6px; height: 6px;
    border-radius: 50%;
    background: var(--success);
}
.sidebar-active-text { line-height: 1; }

/* Sidebar engine panel */
.sidebar-engine-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.5rem;
    margin-top: 0.6rem;
}
.sidebar-mini-stat {
    padding: 0.55rem 0.65rem;
    border-radius: var(--r-sm);
    background: rgba(255,255,255,0.02);
    border: 1px solid var(--line);
}
.sidebar-mini-label {
    font-size: 0.58rem;
    color: var(--text-3);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-weight: 600;
}
.sidebar-mini-value {
    margin-top: 0.25rem;
    font-family: var(--font-mono);
    font-size: 0.78rem;
    font-weight: 600;
    color: var(--text-1);
    overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
}
.sidebar-footer {
    margin-top: 1rem;
    padding: 0.7rem 0.8rem;
    border-radius: var(--r-sm);
    background: rgba(245,158,11,0.05);
    border: 1px solid rgba(245,158,11,0.18);
    font-size: 0.68rem;
    color: var(--text-2);
    text-align: center;
    line-height: 1.5;
}

/* ── STICKY HEADER ── */
.sticky-header {
    position: fixed; top: 0.5rem;
    left: calc(288px + 0.5rem); right: 0.5rem;
    z-index: 9999;
    display: flex; align-items: center; gap: 0.85rem;
    padding: 0.6rem 0.9rem;
    border-radius: var(--r-md);
    background: rgba(15,21,36,0.85);
    backdrop-filter: blur(20px) saturate(160%);
    -webkit-backdrop-filter: blur(20px) saturate(160%);
    border: 1px solid var(--line);
    box-shadow: var(--sh-2);
    transition: border-color 200ms ease;
}
.sticky-header:hover { border-color: var(--line-strong); }

.sticky-brand { display: flex; align-items: center; gap: 0.55rem; flex-shrink: 0; }
.sticky-brand-logo {
    position: relative;
    width: 36px; height: 36px;
    border-radius: 10px;
    background: var(--accent-soft);
    border: 1px solid var(--accent-line);
    display: flex; align-items: center; justify-content: center;
    font-size: 1.1rem;
}
.sticky-brand-pulse {
    position: absolute; top: -2px; right: -2px;
    width: 8px; height: 8px;
    border-radius: 50%;
    background: var(--success);
    border: 2px solid var(--surface);
    animation: brand-pulse 2.4s ease-in-out infinite;
}
@keyframes brand-pulse {
    0%,100% { opacity: 1; }
    50%     { opacity: 0.4; }
}
.sticky-brand-text { display: flex; flex-direction: column; gap: 1px; }
.sticky-brand-name {
    font-size: 0.98rem; font-weight: 700; color: var(--text-1);
    letter-spacing: -0.015em; line-height: 1.1;
}
.sticky-brand-version {
    font-size: 0.6rem; color: var(--text-3); font-weight: 500;
    letter-spacing: 0.06em; text-transform: uppercase;
}

.sticky-divider { width: 1px; height: 22px; background: var(--line); flex-shrink: 0; }

.sticky-ticker {
    flex: 1; overflow: hidden;
    border-radius: var(--r-sm);
    background: rgba(255,255,255,0.02);
    padding: 0.4rem 0.75rem;
    min-width: 0;
    border: 1px solid var(--line);
}
.sticky-ticker-inner {
    display: flex; align-items: center; gap: 1rem;
    white-space: nowrap; overflow: hidden;
    color: var(--text-2);
    font-family: var(--font-mono);
    font-size: 0.72rem; font-weight: 500;
    letter-spacing: 0.01em;
    mask-image: linear-gradient(to right, #000 0%, #000 92%, transparent 100%);
    -webkit-mask-image: linear-gradient(to right, #000 0%, #000 92%, transparent 100%);
}
.tick-badge {
    padding: 0.15rem 0.5rem; border-radius: 6px;
    background: rgba(16,185,129,0.12);
    color: #34d399;
    border: 1px solid rgba(16,185,129,0.28);
    font-size: 0.65rem; font-weight: 700; letter-spacing: 0.08em;
    font-family: var(--font-body);
}
.tick-item { color: var(--text-2); font-family: var(--font-mono); font-size: 0.72rem; }
.tick-item b { color: var(--text-1); font-weight: 600; }
.tick-idle { color: var(--text-3); font-style: italic; }

.sticky-status {
    display: inline-flex; align-items: center; gap: 0.4rem;
    padding: 0.3rem 0.65rem; border-radius: var(--r-sm);
    font-size: 0.66rem; font-weight: 700; letter-spacing: 0.08em;
    white-space: nowrap; flex-shrink: 0;
}
.status-ok {
    background: rgba(16,185,129,0.10);
    color: #34d399;
    border: 1px solid rgba(16,185,129,0.24);
}
.status-err {
    background: rgba(239,68,68,0.10);
    color: #f87171;
    border: 1px solid rgba(239,68,68,0.24);
}
.sticky-status-dot {
    width: 6px; height: 6px; border-radius: 50%;
    background: currentColor;
}

.sticky-page {
    display: inline-flex; align-items: center; gap: 0.45rem;
    padding: 0.35rem 0.7rem; border-radius: var(--r-sm);
    background: transparent;
    border: 1px solid var(--line);
    font-size: 0.76rem; font-weight: 600; color: var(--text-1);
    white-space: nowrap; flex-shrink: 0;
}
.sticky-page-dot {
    width: 6px; height: 6px; border-radius: 50%;
    background: var(--accent);
}

.main .block-container {
    padding-top: 5.75rem !important;
    padding-left: 1.5rem !important;
    padding-right: 1.5rem !important;
    max-width: 1320px;
}

/* ── UNIVERSAL CARD PRIMITIVES ── */
.info-card, .metric-card, .uncertainty-card, .xai-card,
.probability-card, .dist-card, .latest-card, .thumbnail-card,
.chart-container, .empty-state, .diagnostic-panel {
    background: var(--surface-2);
    border: 1px solid var(--line);
    border-radius: var(--r-lg);
    box-shadow: var(--sh-1);
    transition: border-color 160ms ease, background 160ms ease, transform 160ms ease;
}

/* ── HERO ── */
.hero {
    padding: 3rem 2rem;
    border-radius: var(--r-lg);
    text-align: center;
    margin-bottom: 1.5rem;
    background: var(--surface-2);
    border: 1px solid var(--line);
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: "";
    position: absolute; inset: 0;
    background:
      radial-gradient(circle at 20% 0%, rgba(14,165,233,0.08), transparent 55%),
      radial-gradient(circle at 80% 100%, rgba(14,165,233,0.05), transparent 55%);
    pointer-events: none;
}
.hero h1 {
    font-size: 3rem;
    font-weight: 800;
    letter-spacing: -0.035em;
    margin-bottom: 0.75rem;
    color: var(--text-1);
    position: relative;
}
.hero p {
    color: var(--text-2);
    font-size: 1rem;
    line-height: 1.6;
    max-width: 620px;
    margin: 0 auto;
    position: relative;
}
.hero-badge {
    display: inline-flex; align-items: center; gap: 0.4rem;
    padding: 0.35rem 0.75rem;
    border-radius: var(--r-pill);
    background: var(--accent-soft);
    border: 1px solid var(--accent-line);
    font-size: 0.7rem; font-weight: 600;
    color: var(--accent-hi);
    margin-bottom: 1rem;
    position: relative;
}

/* ── DIAGNOSTIC PANEL (result hero) ── */
.diagnostic-panel {
    padding: 2rem 2.25rem;
    border-radius: var(--r-lg);
    background: var(--surface-2);
    border: 1px solid var(--accent-line);
    box-shadow: var(--sh-2);
    margin-top: 1.5rem;
    position: relative;
    overflow: hidden;
}
.diagnostic-panel::before {
    content: "";
    position: absolute;
    left: 0; top: 0; bottom: 0;
    width: 3px;
    background: var(--accent);
}
.diagnostic-header {
    display: flex; justify-content: space-between; align-items: center;
    margin-bottom: 1rem;
    padding-bottom: 0.75rem;
    border-bottom: 1px solid var(--line);
}
.diagnostic-title {
    font-size: 0.7rem; color: var(--text-3);
    text-transform: uppercase; letter-spacing: 0.12em; font-weight: 700;
}
.diagnostic-prediction {
    font-size: 2.75rem;
    font-weight: 800;
    color: var(--text-1);
    letter-spacing: -0.03em;
    line-height: 1.05;
}
.diagnostic-confidence {
    font-family: var(--font-mono);
    font-size: 1rem; font-weight: 500;
    color: var(--accent-hi);
    margin-top: 0.5rem;
    letter-spacing: 0;
}

/* ── BADGES ── */
.medical-badge {
    display: inline-flex; align-items: center; gap: 0.3rem;
    padding: 0.28rem 0.65rem;
    border-radius: 6px;
    font-size: 0.68rem; font-weight: 600;
    letter-spacing: 0.02em;
}
.badge-research { background: rgba(245,158,11,0.10); color: #fbbf24; border: 1px solid rgba(245,158,11,0.24); }
.badge-high     { background: rgba(16,185,129,0.10); color: #34d399; border: 1px solid rgba(16,185,129,0.24); }
.badge-moderate { background: rgba(245,158,11,0.10); color: #fbbf24; border: 1px solid rgba(245,158,11,0.24); }
.badge-low      { background: rgba(239,68,68,0.10);  color: #f87171; border: 1px solid rgba(239,68,68,0.24); }

/* ── METRIC CARDS ── */
.metric-card {
    padding: 1.25rem;
    border-radius: var(--r-md);
    background: var(--surface-2);
    border: 1px solid var(--line);
    text-align: left;
}
.metric-card:hover {
    border-color: var(--line-strong);
    background: var(--surface-3);
}
.metric-icon {
    font-size: 1.1rem;
    margin-bottom: 0.5rem;
    opacity: 0.85;
}
.metric-value {
    font-size: 2rem;
    font-weight: 700;
    color: var(--text-1);
    margin: 0.15rem 0;
    letter-spacing: -0.025em;
    font-variant-numeric: tabular-nums;
}
.metric-label {
    font-size: 0.66rem;
    color: var(--text-3);
    text-transform: uppercase;
    letter-spacing: 0.1em;
    font-weight: 600;
}

/* ── INFO CARD ── */
.info-card {
    padding: 1.5rem;
    min-height: 140px;
    border-radius: var(--r-lg);
    background: var(--surface-2);
    border: 1px solid var(--line);
}
.info-card:hover { border-color: var(--line-strong); transform: translateY(-1px); }
.info-card h3 {
    margin-top: 0;
    margin-bottom: 0.5rem;
    color: var(--text-1);
    font-size: 0.95rem;
    font-weight: 600;
    font-family: var(--font-display);
    letter-spacing: -0.01em;
}
.info-card p {
    color: var(--text-2);
    line-height: 1.55;
    font-size: 0.85rem;
    margin: 0.25rem 0;
}

/* ── UNCERTAINTY CARD ── */
.uncertainty-card {
    padding: 1.25rem 1.5rem;
    border-radius: var(--r-md);
    background: var(--surface-2);
    border: 1px solid var(--line);
    border-left: 3px solid var(--violet);
    margin-top: 1rem;
    transition: border-left-color 300ms ease, background 300ms ease;
}
.uncertainty-title {
    font-size: 0.68rem;
    color: var(--violet);
    text-transform: uppercase;
    letter-spacing: 0.12em;
    font-weight: 700;
    margin-bottom: 0.6rem;
    opacity: 0.9;
}
.uncertainty-value {
    font-family: var(--font-display);
    font-size: 1.6rem;
    font-weight: 700;
    letter-spacing: -0.02em;
    font-variant-numeric: tabular-nums;
    line-height: 1.1;
}
.uncertainty-band {
    font-size: 0.82rem;
    font-weight: 600;
    margin-top: 0.2rem;
    opacity: 0.9;
}

/* ── PROBABILITY GRID ── */
.probability-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    gap: 0.75rem;
    margin-top: 1rem;
}
.probability-card {
    padding: 1rem;
    border-radius: var(--r-md);
    background: var(--surface-2);
    border: 1px solid var(--line);
    text-align: left;
    position: relative;
    overflow: hidden;
}
.probability-card:hover {
    border-color: var(--line-strong);
    background: var(--surface-3);
}
.probability-value {
    font-family: var(--font-display);
    font-size: 1.5rem;
    font-weight: 700;
    color: var(--text-1);
    margin-bottom: 0.25rem;
    letter-spacing: -0.02em;
    font-variant-numeric: tabular-nums;
}
.probability-label {
    font-size: 0.75rem;
    color: var(--text-2);
    font-weight: 500;
}
.probability-card.predicted {
    border-color: var(--accent-line);
    background: var(--accent-soft);
}
.probability-card.predicted::before {
    content: "";
    position: absolute;
    left: 0; top: 0; bottom: 0;
    width: 3px;
    background: var(--accent);
}
.probability-card.predicted .probability-value { color: var(--accent-hi); }

/* ── CHARTS ── */
.chart-container {
    padding: 1.25rem;
    border-radius: var(--r-md);
    background: var(--surface-2);
    border: 1px solid var(--line);
    margin-top: 1rem;
}

/* ── THUMBNAILS (history rows) ── */
.thumbnail-card {
    display: flex; align-items: center; gap: 0.85rem;
    padding: 0.85rem 1rem;
    border-radius: var(--r-md);
    background: var(--surface-2);
    border: 1px solid var(--line);
    margin-bottom: 0.5rem;
}
.thumbnail-card:hover {
    border-color: var(--line-strong);
    background: var(--surface-3);
}
.thumbnail-img {
    width: 44px; height: 44px;
    border-radius: var(--r-sm);
    background: var(--accent-soft);
    border: 1px solid var(--accent-line);
    display: flex; align-items: center; justify-content: center;
    font-size: 1.15rem;
    flex-shrink: 0;
}
.thumbnail-info { flex: 1; min-width: 0; }
.thumbnail-title {
    font-family: var(--font-display);
    font-size: 0.92rem;
    font-weight: 600;
    color: var(--text-1);
    margin-bottom: 0.15rem;
    letter-spacing: -0.01em;
}
.thumbnail-meta {
    font-family: var(--font-mono);
    font-size: 0.7rem;
    color: var(--text-3);
}
.confidence-badge {
    display: inline-block;
    padding: 0.28rem 0.6rem;
    border-radius: 6px;
    font-family: var(--font-mono);
    font-size: 0.76rem;
    font-weight: 600;
}
.confidence-high   { background: rgba(16,185,129,0.10); color: #34d399; border: 1px solid rgba(16,185,129,0.22); }
.confidence-medium { background: rgba(245,158,11,0.10); color: #fbbf24; border: 1px solid rgba(245,158,11,0.22); }
.confidence-low    { background: rgba(239,68,68,0.10);  color: #f87171; border: 1px solid rgba(239,68,68,0.22); }

/* ── EMPTY STATES ── */
.empty-state {
    padding: 3rem 2rem;
    border-radius: var(--r-lg);
    background: var(--surface-2);
    border: 1px dashed var(--line-strong);
    text-align: center;
}
.empty-state-icon { font-size: 2.25rem; margin-bottom: 1rem; opacity: 0.5; }
.empty-state-title {
    font-family: var(--font-display);
    font-size: 1rem;
    font-weight: 600;
    color: var(--text-1);
    margin-bottom: 0.4rem;
    letter-spacing: -0.01em;
}
.empty-state-text {
    font-size: 0.85rem;
    color: var(--text-2);
    max-width: 380px;
    margin: 0 auto;
    line-height: 1.55;
}

/* ── ACTIVITY FEED ── */
.activity-feed {
    border-radius: var(--r-md);
    border: 1px solid var(--line);
    background: rgba(0,0,0,0.20);
    max-height: 340px;
    overflow-y: auto;
    font-family: var(--font-mono);
    font-size: 0.72rem;
}
.activity-row {
    display: flex; gap: 0.7rem;
    padding: 0.4rem 0.8rem;
    border-bottom: 1px solid rgba(255,255,255,0.03);
}
.activity-row:last-child { border-bottom: none; }
.activity-time { color: var(--text-3); min-width: 62px; }
.activity-msg  { color: var(--text-2); }
.activity-success .activity-msg { color: #6ee7b7; }
.activity-warn    .activity-msg { color: #fbbf24; }
.activity-error   .activity-msg { color: #fca5a5; }

/* ── DISTRIBUTION CARDS ── */
.dist-card {
    padding: 0.85rem 1rem;
    border-radius: var(--r-md);
    background: var(--surface-2);
    border: 1px solid var(--line);
    margin-bottom: 0.5rem;
}
.dist-card:hover { border-color: var(--line-strong); }
.dist-header {
    display: flex; justify-content: space-between; align-items: center;
    margin-bottom: 0.5rem;
}
.dist-name { font-size: 0.85rem; font-weight: 600; color: var(--text-1); }
.dist-count {
    font-family: var(--font-mono);
    font-size: 0.75rem;
    color: var(--accent-hi);
    font-weight: 500;
}
.dist-bar-bg {
    height: 4px;
    border-radius: 2px;
    background: rgba(255,255,255,0.05);
    overflow: hidden;
}
.dist-bar-fill {
    height: 100%;
    border-radius: 2px;
    background: var(--accent);
    transition: width 600ms cubic-bezier(.4,0,.2,1);
}

/* ── LATEST RESULT CARD ── */
.latest-card {
    padding: 0.75rem 1.15rem;
    border-radius: var(--r-md);
    background: var(--surface-2);
    border: 1px solid var(--line);
}
.latest-row {
    display: flex; justify-content: space-between; align-items: center;
    padding: 0.55rem 0;
    border-bottom: 1px solid var(--line);
}
.latest-row:last-child { border-bottom: none; }
.latest-key { font-size: 0.78rem; color: var(--text-2); font-weight: 500; }
.latest-val {
    font-family: var(--font-mono);
    font-size: 0.78rem;
    color: var(--text-1);
    font-weight: 500;
}

/* ── ELEGANT UPLOAD HERO ── */
.upload-hero {
    position: relative;
    margin: 1rem 0 .75rem;
    padding: 1.85rem 2rem 1.5rem;
    border-radius: var(--r-lg);
    background: var(--surface-2);
    border: 1px solid var(--line);
    overflow: hidden;
    text-align: center;
    box-shadow: var(--sh-1);
}
.upload-hero::before {
    content: "";
    position: absolute;
    top: -80%; left: 50%;
    transform: translateX(-50%);
    width: 520px; height: 520px;
    background: radial-gradient(circle, rgba(14,165,233,0.14), transparent 62%);
    pointer-events: none;
}
.upload-icon {
    position: relative;
    width: 60px; height: 60px;
    margin: 0 auto .9rem;
    border-radius: 16px;
    background: linear-gradient(135deg, rgba(14,165,233,0.22), rgba(14,165,233,0.04));
    border: 1px solid var(--accent-line);
    display: flex; align-items: center; justify-content: center;
    color: var(--accent-hi);
    box-shadow: 0 10px 26px rgba(14,165,233,0.14);
}
.upload-icon svg { width: 26px; height: 26px; }
.upload-title {
    position: relative;
    font-family: var(--font-display);
    font-size: 1.18rem;
    font-weight: 700;
    color: var(--text-1);
    letter-spacing: -0.02em;
    margin-bottom: .3rem;
}
.upload-subtitle {
    position: relative;
    font-size: .84rem;
    color: var(--text-2);
    margin-bottom: .95rem;
    line-height: 1.5;
}
.upload-badges {
    position: relative;
    display: flex; justify-content: center;
    gap: .4rem; flex-wrap: wrap;
}
.upload-badge {
    padding: .26rem .62rem;
    border-radius: 6px;
    background: rgba(14,165,233,0.08);
    border: 1px solid rgba(14,165,233,0.22);
    font-family: var(--font-mono);
    font-size: .64rem;
    font-weight: 600;
    color: var(--accent-hi);
    letter-spacing: .08em;
}
.upload-badge-muted {
    background: rgba(255,255,255,0.03);
    border-color: var(--line);
    color: var(--text-3);
}
.upload-footnote {
    display: flex; align-items: center; justify-content: center;
    gap: .45rem;
    margin-top: .85rem;
    font-size: .72rem;
    color: var(--text-3);
    letter-spacing: .01em;
}
.upload-footnote-dot {
    width: 5px; height: 5px;
    border-radius: 50%;
    background: var(--success);
    display: inline-block;
    box-shadow: 0 0 8px rgba(16,185,129,0.55);
}

/* ── EXPORT PANEL ── */
.export-panel-head {
    display: flex;
    align-items: center;
    gap: 0.85rem;
    padding: 1.15rem 1.35rem;
    border-radius: var(--r-lg) var(--r-lg) 0 0;
    background: var(--surface-2);
    border: 1px solid var(--line);
    border-bottom: 1px solid var(--line);
    margin-top: 1.5rem;
    position: relative;
    overflow: hidden;
}
.export-panel-head::before {
    content: "";
    position: absolute;
    top: -60%; left: -5%;
    width: 320px; height: 320px;
    background: radial-gradient(circle, rgba(14,165,233,0.10), transparent 62%);
    pointer-events: none;
}
.export-head-icon {
    position: relative;
    width: 42px; height: 42px;
    border-radius: 12px;
    background: linear-gradient(135deg, rgba(14,165,233,0.22), rgba(14,165,233,0.04));
    border: 1px solid var(--accent-line);
    display: flex; align-items: center; justify-content: center;
    font-size: 1.15rem;
    flex-shrink: 0;
    box-shadow: 0 8px 20px rgba(14,165,233,0.14);
}
.export-head-text { display: flex; flex-direction: column; gap: 2px; position: relative; }
.export-head-title {
    font-family: var(--font-display);
    font-size: 1rem;
    font-weight: 700;
    color: var(--text-1);
    letter-spacing: -0.015em;
    line-height: 1.2;
}
.export-head-sub {
    font-size: 0.74rem;
    color: var(--text-3);
    letter-spacing: 0.01em;
}
.export-head-badge {
    margin-left: auto;
    padding: 0.28rem 0.65rem;
    border-radius: var(--r-pill);
    background: rgba(16,185,129,0.10);
    border: 1px solid rgba(16,185,129,0.24);
    color: #34d399;
    font-family: var(--font-mono);
    font-size: 0.62rem;
    font-weight: 700;
    letter-spacing: 0.1em;
}

.export-item-label {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    font-family: var(--font-mono);
    font-size: 0.63rem;
    color: var(--text-3);
    text-transform: uppercase;
    letter-spacing: 0.12em;
    font-weight: 700;
    margin: 0 0 0.55rem 0;
    padding-left: 0.15rem;
}
.export-item-label::before {
    content: "";
    width: 3px; height: 10px;
    border-radius: 1px;
    background: var(--accent);
    opacity: 0.9;
    flex-shrink: 0;
}

/* ── STREAMLIT NATIVE OVERRIDES ── */
[data-testid="stFileUploader"] {
    background: linear-gradient(180deg, rgba(14,165,233,0.035), rgba(14,165,233,0.005));
    border-radius: var(--r-lg);
    padding: 1.15rem 1rem;
    border: 1px dashed var(--line-strong);
    transition: border-color 200ms ease, background 200ms ease;
}
[data-testid="stFileUploader"]:hover {
    border-color: var(--accent-line);
    background: linear-gradient(180deg, rgba(14,165,233,0.07), rgba(14,165,233,0.015));
}
[data-testid="stFileUploader"] small { color: var(--text-3) !important; }
[data-testid="stFileUploader"] button {
    border-radius: var(--r-sm) !important;
    background: var(--accent) !important;
    color: #041018 !important;
    border: 1px solid var(--accent) !important;
    font-weight: 600 !important;
    font-size: 0.78rem !important;
    min-height: 32px !important;
    padding: 0.4rem 0.85rem !important;
    transition: background 140ms ease, border-color 140ms ease !important;
}
[data-testid="stFileUploader"] button:hover {
    background: var(--accent-hi) !important;
    border-color: var(--accent-hi) !important;
}

[data-testid="stExpander"] {
    border-radius: var(--r-md) !important;
    border: 1px solid var(--line) !important;
    background: var(--surface-2) !important;
}
[data-testid="stExpander"]:hover { border-color: var(--line-strong) !important; }

/* ── COMPACT BUTTONS (all pages) ── */
.stButton > button {
    border-radius: var(--r-sm) !important;
    font-weight: 600 !important;
    font-size: 0.78rem !important;
    padding: 0.42rem 0.9rem !important;
    min-height: 32px !important;
    height: auto !important;
    line-height: 1.2 !important;
    transition: background 140ms ease, border-color 140ms ease, box-shadow 140ms ease !important;
    box-shadow: none !important;
}
.stButton > button p {
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    margin: 0 !important;
    line-height: 1.2 !important;
}
.stButton > button:hover { transform: none !important; box-shadow: var(--sh-1) !important; }
.stButton > button:focus-visible {
    box-shadow: var(--sh-focus) !important;
    outline: none !important;
}

/* Primary = slightly larger but still compact */
.stButton > button[kind="primary"] {
    padding: 0.5rem 1.1rem !important;
    min-height: 40px !important;
    background: var(--accent) !important;
    border: 1px solid var(--accent) !important;
    color: #041018 !important;
}
.stButton > button[kind="primary"] p {
    font-size: 0.82rem !important;
    font-weight: 600 !important;
}
.stButton > button[kind="primary"]:hover {
    background: var(--accent-hi) !important;
    border-color: var(--accent-hi) !important;
}

/* ── COMPACT DOWNLOAD BUTTONS ── */
[data-testid="stDownloadButton"] { margin-bottom: 0.25rem !important; }
[data-testid="stDownloadButton"] > button {
    width: 240px !important;
    min-width: 240px !important;
    min-height: 40px !important;
    padding: 0.55rem 0.9rem !important;
    border-radius: var(--r-sm) !important;
    background: var(--surface-3) !important;
    border: 1px solid var(--line) !important;
    color: var(--text-1) !important;
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    letter-spacing: -0.005em !important;
    transition: background 180ms ease, border-color 180ms ease,
                color 180ms ease, box-shadow 180ms ease,
                transform 180ms ease !important;
    box-shadow: none !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    gap: 0.45rem !important;
}
[data-testid="stDownloadButton"] > button p {
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    margin: 0 !important;
    line-height: 1.2 !important;
}
[data-testid="stDownloadButton"] > button svg {
    width: 14px !important;
    height: 14px !important;
}
[data-testid="stDownloadButton"] > button:hover {
    background: var(--accent-soft) !important;
    border-color: var(--accent-line) !important;
    color: var(--accent-hi) !important;
    box-shadow: 0 6px 16px rgba(14,165,233,0.14) !important;
    transform: translateY(-1px) !important;
}
[data-testid="stDownloadButton"] > button:focus-visible {
    box-shadow: var(--sh-focus) !important;
    outline: none !important;
}
[data-testid="stDownloadButton"] > button[kind="primary"] {
    background: linear-gradient(135deg, var(--accent), #0284c7) !important;
    border: 1px solid var(--accent) !important;
    color: #041018 !important;
    min-height: 42px !important;
    padding: 0.6rem 1rem !important;
}
[data-testid="stDownloadButton"] > button[kind="primary"] p {
    font-size: 0.82rem !important;
    font-weight: 600 !important;
}
[data-testid="stDownloadButton"] > button[kind="primary"]:hover {
    background: linear-gradient(135deg, var(--accent-hi), var(--accent)) !important;
    border-color: var(--accent-hi) !important;
    color: #041018 !important;
    box-shadow: 0 8px 20px rgba(14,165,233,0.30) !important;
    transform: translateY(-1px) !important;
}

[data-testid="stMetricValue"] {
    font-family: var(--font-display) !important;
    font-size: 1.5rem !important;
    font-weight: 700 !important;
    letter-spacing: -0.02em !important;
    font-variant-numeric: tabular-nums !important;
}
[data-testid="stMetricLabel"] {
    font-size: 0.68rem !important;
    color: var(--text-3) !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
    font-weight: 600 !important;
}
.stProgress > div > div > div > div {
    background: var(--accent) !important;
    border-radius: 4px !important;
}
.stAlert { border-radius: var(--r-sm) !important; }
hr { border-color: var(--line) !important; margin: 1.25rem 0 !important; }
.stImage {
    border-radius: var(--r-md) !important;
    overflow: hidden !important;
    border: 1px solid var(--line) !important;
}
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    background: transparent;
    border-bottom: 1px solid var(--line);
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    border-radius: var(--r-sm) var(--r-sm) 0 0 !important;
    padding: 0.6rem 1rem !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    color: var(--text-2) !important;
}
.stTabs [aria-selected="true"] {
    color: var(--text-1) !important;
    border-bottom: 2px solid var(--accent) !important;
}

/* ── XAI CARD ── */
.xai-card {
    padding: 1.25rem 1.5rem;
    border-radius: var(--r-md);
    background: var(--surface-2);
    border: 1px solid var(--line);
    margin-top: 1rem;
}
.xai-title {
    font-size: 0.68rem;
    color: var(--accent-hi);
    text-transform: uppercase;
    letter-spacing: 0.12em;
    font-weight: 700;
    margin-bottom: 0.5rem;
}
.xai-text {
    font-size: 0.85rem;
    color: var(--text-2);
    line-height: 1.6;
}

/* ── FOOTER ── */
.app-footer {
    display: flex;
    justify-content: space-between;
    align-items: stretch;
    gap: 1.5rem;
    padding: 1.25rem 1.5rem;
    border-radius: var(--r-lg);
    background: var(--surface-2);
    border: 1px solid var(--line);
    margin-top: 2.5rem;
    font-size: 0.8rem;
    color: var(--text-2);
    flex-wrap: wrap;
}
.footer-col { display: flex; flex-direction: column; gap: 0.5rem; }
.footer-col-brand { flex: 1; min-width: 200px; }
.footer-col-stack { flex: 1.2; min-width: 200px; }
.footer-col-stats { flex: 0 0 auto; min-width: 240px; }

.footer-brand-row { display: flex; align-items: center; gap: 0.6rem; }
.footer-brand-logo {
    width: 32px; height: 32px;
    border-radius: var(--r-sm);
    background: var(--accent-soft);
    border: 1px solid var(--accent-line);
    display: flex; align-items: center; justify-content: center;
    font-size: 1rem;
}
.footer-brand { font-size: 0.92rem; font-weight: 700; color: var(--text-1); letter-spacing: -0.015em; }
.footer-tagline {
    font-size: 0.66rem; color: var(--text-3);
    letter-spacing: 0.04em; font-weight: 500;
}
.footer-copy {
    font-family: var(--font-mono);
    font-size: 0.68rem;
    color: var(--text-3);
    line-height: 1.6;
    margin-top: 0.35rem;
}

.footer-col-title {
    font-size: 0.6rem;
    color: var(--text-3);
    text-transform: uppercase;
    letter-spacing: 0.14em;
    font-weight: 700;
    margin-bottom: 0.25rem;
}
.footer-badges { display: flex; flex-wrap: wrap; gap: 0.4rem; }
.badge-tech {
    padding: 0.25rem 0.6rem;
    border-radius: 6px;
    background: rgba(255,255,255,0.03);
    border: 1px solid var(--line);
    color: var(--text-2);
    font-family: var(--font-mono);
    font-size: 0.66rem;
    font-weight: 500;
    transition: border-color 140ms ease, color 140ms ease;
}
.badge-tech:hover { border-color: var(--line-strong); color: var(--text-1); }

.footer-stats { display: flex; gap: 1.5rem; margin-top: 0.25rem; }
.footer-stat { display: flex; flex-direction: column; gap: 0.15rem; }
.footer-stat-val {
    font-family: var(--font-display);
    font-size: 0.95rem;
    font-weight: 700;
    color: var(--text-1);
    display: inline-flex; align-items: center; gap: 0.4rem;
    letter-spacing: -0.015em;
    font-variant-numeric: tabular-nums;
}
.footer-stat-lbl {
    font-size: 0.58rem;
    color: var(--text-3);
    text-transform: uppercase;
    letter-spacing: 0.12em;
    font-weight: 700;
}
.footer-dot {
    width: 6px; height: 6px;
    border-radius: 50%;
    background: var(--success);
    display: inline-block;
}

/* ── COPYRIGHT ── */
.copyright-line {
    text-align: center;
    padding: 1.5rem 0.5rem 0.5rem;
    font-size: 0.75rem;
    color: var(--text-3);
    letter-spacing: 0.02em;
    line-height: 1.7;
}
.copyright-brand { color: var(--text-2); font-weight: 600; }
.copyright-sep { color: var(--text-3); margin: 0 0.5rem; }
.copyright-dev-label { font-size: 0.72rem; color: var(--text-3); }
.copyright-dev-name  { color: var(--text-1); font-weight: 600; }
.copyright-dev-amp   { color: var(--text-3); margin: 0 0.4rem; }

/* ── DISCLAIMER ── */
.disclaimer {
    margin-top: 1.5rem;
    padding: 1rem 1.25rem;
    border-radius: var(--r-md);
    background: rgba(245,158,11,0.05);
    border: 1px solid rgba(245,158,11,0.18);
    color: var(--text-2);
    font-size: 0.82rem;
    line-height: 1.65;
}

/* ── RESPONSIVE ── */
@media (max-width: 768px) {
    .sticky-header {
        left: 0.4rem !important; right: 0.4rem !important;
        padding: 0.5rem 0.7rem !important;
        gap: 0.5rem !important;
    }
    .sticky-ticker, .sticky-brand-version { display: none; }
    .sticky-status { display: none; }
    .sticky-brand-logo { width: 32px; height: 32px; font-size: 1rem; }
    .sticky-brand-name { font-size: 0.88rem; }
    [data-testid="stSidebar"] {
        min-width: 100% !important; max-width: 100% !important;
    }
    .main .block-container { padding-top: 4.5rem !important; }
    .hero h1 { font-size: 1.85rem !important; }
    .hero { padding: 2rem 1.25rem; }
    .probability-grid { grid-template-columns: 1fr 1fr !important; }
    .app-footer { flex-direction: column; gap: 1rem; }
    .footer-col-brand, .footer-col-stack, .footer-col-stats {
        flex: 1 1 100% !important; min-width: 0 !important;
    }
    .diagnostic-prediction { font-size: 2rem !important; }
    .upload-hero { padding: 1.5rem 1.25rem 1.25rem; }
    .upload-title { font-size: 1.05rem; }
    .upload-subtitle { font-size: 0.78rem; }
    .upload-icon { width: 52px; height: 52px; }
    .upload-icon svg { width: 22px; height: 22px; }
    .export-panel-head { padding: 1rem 1rem; gap: 0.65rem; }
    .export-head-badge { display: none; }
    .export-head-icon { width: 38px; height: 38px; font-size: 1rem; }
    .export-head-title { font-size: 0.92rem; }
}

@media (min-width: 769px) and (max-width: 1024px) {
    .sticky-header { left: calc(260px + 0.4rem) !important; }
    [data-testid="stSidebar"] {
        min-width: 260px !important;
        max-width: 260px !important;
        width: 260px !important;
    }
    [data-testid="stSidebar"] > div:first-child { width: 260px !important; }
}

@media (prefers-reduced-motion: reduce) {
    *, *::before, *::after {
        animation-duration: 0.01ms !important;
        animation-iteration-count: 1 !important;
        transition-duration: 0.01ms !important;
    }
}


/* ============================================================
   UNIVERSAL BUTTON CENTER ALIGNMENT
   ============================================================ */

/* General Streamlit buttons */
.stButton {
    display: flex !important;
    justify-content: center !important;
    align-items: center !important;
}

.stButton > button {
    margin-left: auto !important;
    margin-right: auto !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    text-align: center !important;
}

.stButton > button p {
    width: 100% !important;
    text-align: center !important;
}

/* Primary buttons */
.stButton > button[kind="primary"] {
    margin-left: auto !important;
    margin-right: auto !important;
    justify-content: center !important;
    text-align: center !important;
}

/* Sidebar navigation buttons */
[data-testid="stSidebar"] .stButton {
    display: flex !important;
    justify-content: center !important;
    align-items: center !important;
    width: 100% !important;
}

[data-testid="stSidebar"] .stButton > button {
    width: 100% !important;
    margin-left: auto !important;
    margin-right: auto !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    text-align: center !important;
}

[data-testid="stSidebar"] .stButton > button p {
    width: 100% !important;
    text-align: center !important;
}

/* Download buttons */
[data-testid="stDownloadButton"] {
    display: flex !important;
    justify-content: center !important;
    align-items: center !important;
}

[data-testid="stDownloadButton"] > button {
    width: 240px !important;      /* Changed from 100% to 240px */
    min-width: 240px !important;  /* Changed from 100% to 240px */
    height: 40px !important;
    min-height: 40px !important;
    padding: 4px 10px !important;
    margin-left: auto !important;
    margin-right: auto !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    text-align: center !important;
}

[data-testid="stDownloadButton"] > button p {
    text-align: center !important;
}

/* Primary download buttons */
[data-testid="stDownloadButton"] > button[kind="primary"] {
    margin-left: auto !important;
    margin-right: auto !important;
    justify-content: center !important;
    text-align: center !important;
}

/* File uploader button */
[data-testid="stFileUploader"] {
    text-align: center !important;
}

[data-testid="stFileUploader"] button {
    margin-left: auto !important;
    margin-right: auto !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    text-align: center !important;
}

/* Home CTA / Run AI Analysis */
.stButton:has(button[key="home_cta"]),
.stButton:has(button[key="analyze_btn"]) {
    display: flex !important;
    justify-content: center !important;
    align-items: center !important;
    width: 100% !important;
}

.stButton:has(button[key="home_cta"]) > button,
.stButton:has(button[key="analyze_btn"]) > button {
    margin-left: auto !important;
    margin-right: auto !important;
    justify-content: center !important;
    text-align: center !important;
}

/* Settings Reset / History Clear */
.stButton:has(button[key="settings_reset"]),
.stButton:has(button[key="clear_hist_btn"]),
.stButton:has(button[key="sb_clear"]),
.stButton:has(button[key="sb_reset"]) {
    display: flex !important;
    justify-content: center !important;
    align-items: center !important;
    width: 100% !important;
}

.stButton:has(button[key="settings_reset"]) > button,
.stButton:has(button[key="clear_hist_btn"]) > button,
.stButton:has(button[key="sb_clear"]) > button,
.stButton:has(button[key="sb_reset"]) > button {
    margin-left: auto !important;
    margin-right: auto !important;
    justify-content: center !important;
    text-align: center !important;
}

/* Confirm / Cancel buttons */
.stButton:has(button[key="confirm_hist"]),
.stButton:has(button[key="cancel_hist"]),
.stButton:has(button[key="sb_clear_yes"]),
.stButton:has(button[key="sb_clear_no"]),
.stButton:has(button[key="sb_reset_yes"]),
.stButton:has(button[key="sb_reset_no"]),
.stButton:has(button[key="settings_confirm_yes"]),
.stButton:has(button[key="settings_confirm_no"]) {
    display: flex !important;
    justify-content: center !important;
    align-items: center !important;
}

.stButton:has(button[key="confirm_hist"]) > button,
.stButton:has(button[key="cancel_hist"]) > button,
.stButton:has(button[key="sb_clear_yes"]) > button,
.stButton:has(button[key="sb_clear_no"]) > button,
.stButton:has(button[key="sb_reset_yes"]) > button,
.stButton:has(button[key="sb_reset_no"]) > button,
.stButton:has(button[key="settings_confirm_yes"]) > button,
.stButton:has(button[key="settings_confirm_no"]) > button {
    margin-left: auto !important;
    margin-right: auto !important;
    justify-content: center !important;
    text-align: center !important;
}

/* Buttons inside Streamlit columns */
[data-testid="stHorizontalBlock"] .stButton {
    display: flex !important;
    justify-content: center !important;
    align-items: center !important;
}

[data-testid="stHorizontalBlock"] .stButton > button {
    margin-left: auto !important;
    margin-right: auto !important;
    justify-content: center !important;
    text-align: center !important;
}

/* Keep button labels centered even with width="stretch/content" */
.stButton > button[style*="width"] {
    justify-content: center !important;
    text-align: center !important;
}

</style>
"""

st.markdown(STYLES, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# MODEL ARCHITECTURES
# ─────────────────────────────────────────────────────────────────────────────

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, pool=True):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
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
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)


def build_resnet50(num_classes):
    model = resnet50(weights=None)
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 256),
        nn.BatchNorm1d(256), nn.ReLU(inplace=True), nn.Dropout(0.35),
        nn.Linear(256, num_classes),
    )
    return model


def build_efficientnet_b0(num_classes):
    model = efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(nn.Dropout(0.35), nn.Linear(in_features, num_classes))
    return model


# ─────────────────────────────────────────────────────────────────────────────
# MODEL LOADING
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource
def load_model(model_path):
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    try:
        checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=False)
    except TypeError:
        checkpoint = torch.load(model_path, map_location=DEVICE)
    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint must be a dictionary.")

    checkpoint_classes = checkpoint.get("class_names", CLASS_NAMES)
    normalized_classes = [str(name).strip().casefold().replace(" ", "") for name in checkpoint_classes]
    expected_classes = [name.casefold().replace(" ", "") for name in CLASS_NAMES]
    if normalized_classes != expected_classes:
        raise ValueError(
            "Unexpected class order in checkpoint. "
            f"Expected {CLASS_NAMES}; found {checkpoint_classes}."
        )
    class_names = CLASS_NAMES

    num_classes    = checkpoint.get("num_classes", len(class_names))
    best_model_name = checkpoint.get("best_model_name", "CustomCNN")
    if best_model_name == "Custom CNN":
        best_model_name = "CustomCNN"
    if best_model_name not in {"CustomCNN", "ResNet50", "EfficientNet-B0"}:
        raise ValueError(f"Unsupported architecture: {best_model_name}")

    if best_model_name == "ResNet50":
        model = build_resnet50(num_classes)
    elif best_model_name == "EfficientNet-B0":
        model = build_efficientnet_b0(num_classes)
    else:
        model = CustomCNN(num_classes)

    state_dict = checkpoint.get("model_state_dict", checkpoint)
    clean_sd = {k[len("module."):] if k.startswith("module.") else k: v for k, v in state_dict.items()}
    model.load_state_dict(clean_sd, strict=True)
    model.to(DEVICE)
    model.eval()
    return model, class_names, best_model_name


# ─────────────────────────────────────────────────────────────────────────────
# INFERENCE
# ─────────────────────────────────────────────────────────────────────────────

def predict_image(image, model, class_names):
    if model is None:
        raise RuntimeError("Neural engine unavailable.")
    model.eval()
    t0 = time.perf_counter()
    tensor = test_transforms(image).unsqueeze(0).to(DEVICE)
    preprocess_ms = (time.perf_counter() - t0) * 1000

    if DEVICE.type == "cuda":
        torch.cuda.synchronize(DEVICE)
    t1 = time.perf_counter()
    with torch.inference_mode():
        outputs = model(tensor)
        probs = F.softmax(outputs, dim=1)[0].detach().cpu().numpy()
    if DEVICE.type == "cuda":
        torch.cuda.synchronize(DEVICE)
    inference_ms = (time.perf_counter() - t1) * 1000

    idx = int(np.argmax(probs))
    return (
        class_names[idx],
        float(probs[idx] * 100),
        {class_names[i]: float(probs[i] * 100) for i in range(len(class_names))},
        preprocess_ms,
        inference_ms,
    )


def mc_dropout_predict(image, model, class_names, n_samples: int = MC_SAMPLES):
    if model is None:
        return None

    def _enable_dropout(m):
        if isinstance(m, (nn.Dropout, nn.Dropout2d)):
            m.train()

    model.eval()
    model.apply(_enable_dropout)

    tensor = test_transforms(image).unsqueeze(0).to(DEVICE)
    mc_preds = []

    with torch.no_grad():
        for _ in range(n_samples):
            out = model(tensor)
            p   = F.softmax(out, dim=1)[0].detach().cpu().numpy()
            mc_preds.append(p)

    model.eval()

    mc_preds   = np.stack(mc_preds)
    mean_probs = mc_preds.mean(axis=0)
    std_probs  = mc_preds.std(axis=0)

    pred_idx     = int(np.argmax(mean_probs))
    uncertainty  = float(std_probs[pred_idx])
    band, color  = uncertainty_band(uncertainty)

    return {
        "mean_probs":  {class_names[i]: float(mean_probs[i] * 100) for i in range(len(class_names))},
        "std_probs":   {class_names[i]: float(std_probs[i] * 100)  for i in range(len(class_names))},
        "uncertainty": uncertainty,
        "band":        band,
        "color":       color,
        "prediction":  class_names[pred_idx],
        "confidence":  float(mean_probs[pred_idx] * 100),
    }


# ─────────────────────────────────────────────────────────────────────────────
# GRAD-CAM + GRAD-CAM++
# ─────────────────────────────────────────────────────────────────────────────

def _get_target_layer(model, model_name):
    if model_name == "ResNet50":
        return model.layer4[-1].conv3
    elif model_name == "EfficientNet-B0":
        return model.features[-1]
    else:
        return model.features[4].block[0]


def _cam_to_heatmap(cam_raw, original_np):
    cam = np.maximum(cam_raw, 0)
    cam -= cam.min()
    if cam.max() > 0:
        cam /= cam.max()
    return cam


def generate_gradcam(image, model, model_name):
    if model is None:
        raise RuntimeError("Neural engine unavailable.")
    model.eval()
    activations, gradients = [], []
    target_layer = _get_target_layer(model, model_name)
    fwd = target_layer.register_forward_hook(lambda m, i, o: activations.append(o.detach()))
    bwd = target_layer.register_full_backward_hook(lambda m, gi, go: gradients.append(go[0].detach()))
    fig = None
    try:
        tensor = test_transforms(image).unsqueeze(0).to(DEVICE)
        model.zero_grad()
        output = model(tensor)
        idx = int(output.argmax(dim=1).item())
        output[0, idx].backward()

        if not activations or not gradients:
            raise RuntimeError("Grad-CAM hooks failed.")

        activation = activations[0][0]
        gradient   = gradients[0][0]
        weights    = gradient.mean(dim=(1, 2), keepdim=True)
        cam_raw    = (weights * activation).sum(dim=0).detach().cpu().numpy()
        cam        = _cam_to_heatmap(cam_raw, None)

        original = np.asarray(image).astype(np.float32) / 255.0
        fig, ax = plt.subplots(figsize=(5, 5), dpi=120)
        fig.patch.set_alpha(0)
        ax.imshow(original)
        ax.imshow(cam, cmap="jet", alpha=0.44,
                  extent=(0, original.shape[1], original.shape[0], 0))
        ax.axis("off")
        fig.tight_layout(pad=0)
        return fig
    except Exception:
        if fig is not None:
            plt.close(fig)
        raise
    finally:
        try: fwd.remove()
        except Exception: pass
        try: bwd.remove()
        except Exception: pass


def generate_gradcam_pp(image, model, model_name):
    if model is None:
        raise RuntimeError("Neural engine unavailable.")
    model.eval()
    activations, gradients = [], []
    target_layer = _get_target_layer(model, model_name)
    fwd = target_layer.register_forward_hook(lambda m, i, o: activations.append(o.detach()))
    bwd = target_layer.register_full_backward_hook(lambda m, gi, go: gradients.append(go[0].detach()))
    fig = None
    try:
        tensor = test_transforms(image).unsqueeze(0).to(DEVICE)
        model.zero_grad()
        output = model(tensor)
        idx = int(output.argmax(dim=1).item())
        score = output[0, idx]
        score.backward(retain_graph=True)

        if not activations or not gradients:
            raise RuntimeError("Grad-CAM++ hooks failed.")

        activation = activations[0][0].cpu().numpy()
        gradient   = gradients[0][0].cpu().numpy()

        alpha_num   = gradient ** 2
        alpha_denom = 2 * gradient ** 2 + \
                      (activation * gradient ** 3).sum(axis=(1, 2), keepdims=True) + 1e-8
        alpha       = alpha_num / alpha_denom
        relu_grad   = np.maximum(gradient, 0)
        weights     = (alpha * relu_grad).sum(axis=(1, 2))

        cam_raw = np.zeros(activation.shape[1:], dtype=np.float32)
        for w, a in zip(weights, activation):
            cam_raw += w * a
        cam = _cam_to_heatmap(cam_raw, None)

        original = np.asarray(image).astype(np.float32) / 255.0
        fig, ax = plt.subplots(figsize=(5, 5), dpi=120)
        fig.patch.set_alpha(0)
        ax.imshow(original)
        ax.imshow(cam, cmap="inferno", alpha=0.46,
                  extent=(0, original.shape[1], original.shape[0], 0))
        ax.axis("off")
        fig.tight_layout(pad=0)
        return fig
    except Exception:
        if fig is not None:
            plt.close(fig)
        raise
    finally:
        try: fwd.remove()
        except Exception: pass
        try: bwd.remove()
        except Exception: pass


def explanation_agreement(image, model, model_name):
    hook_handles = []
    try:
        model.eval()
        activations_gc, gradients_gc = [], []
        activations_pp, gradients_pp = [], []
        target_layer = _get_target_layer(model, model_name)

        fwd1 = target_layer.register_forward_hook(lambda m, i, o: activations_gc.append(o.detach()))
        bwd1 = target_layer.register_full_backward_hook(lambda m, gi, go: gradients_gc.append(go[0].detach()))
        hook_handles.extend((fwd1, bwd1))

        tensor = test_transforms(image).unsqueeze(0).to(DEVICE)
        model.zero_grad()
        out = model(tensor)
        idx = int(out.argmax(dim=1).item())
        out[0, idx].backward(retain_graph=True)
        fwd1.remove(); bwd1.remove()

        if not activations_gc or not gradients_gc:
            return None

        act_gc = activations_gc[0][0]
        grd_gc = gradients_gc[0][0]
        w_gc   = grd_gc.mean(dim=(1, 2), keepdim=True)
        cam_gc = F.relu((w_gc * act_gc).sum(dim=0)).detach().cpu().numpy()
        cam_gc /= (cam_gc.max() + 1e-8)

        fwd2 = target_layer.register_forward_hook(lambda m, i, o: activations_pp.append(o.detach()))
        bwd2 = target_layer.register_full_backward_hook(lambda m, gi, go: gradients_pp.append(go[0].detach()))
        hook_handles.extend((fwd2, bwd2))

        model.zero_grad()
        out2 = model(tensor)
        out2[0, idx].backward()
        fwd2.remove(); bwd2.remove()

        if not activations_pp or not gradients_pp:
            return None

        act_pp  = activations_pp[0][0].cpu().numpy()
        grd_pp  = gradients_pp[0][0].cpu().numpy()
        an      = grd_pp ** 2
        ad      = 2 * grd_pp ** 2 + (act_pp * grd_pp ** 3).sum(axis=(1,2), keepdims=True) + 1e-8
        alpha   = an / ad
        w_pp    = (alpha * np.maximum(grd_pp, 0)).sum(axis=(1, 2))
        cam_pp  = np.zeros(act_pp.shape[1:], dtype=np.float32)
        for w, a in zip(w_pp, act_pp):
            cam_pp += w * a
        cam_pp = np.maximum(cam_pp, 0)
        cam_pp /= (cam_pp.max() + 1e-8)

        flat_gc = cam_gc.flatten()
        flat_pp = cam_pp.flatten()
        if flat_gc.std() < 1e-8 or flat_pp.std() < 1e-8:
            return None
        corr = float(np.corrcoef(flat_gc, flat_pp)[0, 1])
        return max(0.0, min(1.0, corr))
    except Exception:
        return None
    finally:
        for handle in hook_handles:
            handle.remove()


# ─────────────────────────────────────────────────────────────────────────────
# CHARTING
# ─────────────────────────────────────────────────────────────────────────────

def _dark_fig(w=6, h=2.8):
    fig, ax = plt.subplots(figsize=(w, h))
    fig.patch.set_facecolor("#131a2c")
    ax.set_facecolor("#131a2c")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#1f2937")
    ax.tick_params(colors="#a4b0c0", labelsize=8)
    return fig, ax


def plot_confidence_trend(history):
    if len(history) < 2:
        return None
    fig, ax = _dark_fig()
    confs = [h["confidence"] for h in history]
    xs    = list(range(1, len(history) + 1))
    ax.plot(xs, confs, marker="o", lw=2, ms=4, color="#0ea5e9")
    ax.fill_between(xs, confs, alpha=0.10, color="#0ea5e9")
    ax.set_ylim(0, 105)
    ax.set_xlabel("Analysis #", color="#a4b0c0", fontsize=9)
    ax.set_ylabel("Confidence %", color="#a4b0c0", fontsize=9)
    ax.grid(True, alpha=0.10, linestyle="--")
    fig.tight_layout(pad=0.5)
    return fig


def plot_latency_trend(history):
    if len(history) < 2:
        return None
    fig, ax = _dark_fig()
    lats = [h.get("latency_ms", 0) for h in history]
    ax.plot(range(1, len(history) + 1), lats, marker="o", lw=2, color="#0ea5e9")
    ax.set_xlabel("Analysis #", color="#a4b0c0", fontsize=9)
    ax.set_ylabel("Latency ms", color="#a4b0c0", fontsize=9)
    ax.grid(True, alpha=0.10, linestyle="--")
    fig.tight_layout(pad=0.5)
    return fig


def plot_uncertainty_history(history):
    unc_data = [(i+1, h.get("uncertainty", 0)) for i, h in enumerate(history) if "uncertainty" in h]
    if len(unc_data) < 2:
        return None
    fig, ax = _dark_fig()
    xs, ys = zip(*unc_data)
    ax.plot(xs, ys, marker="s", lw=2, ms=4, color="#8b5cf6")
    ax.fill_between(xs, ys, alpha=0.10, color="#8b5cf6")
    ax.axhline(0.12, color="#f59e0b", lw=1, ls="--", alpha=0.55, label="Moderate threshold")
    ax.axhline(0.22, color="#ef4444", lw=1, ls="--", alpha=0.55, label="Low reliability threshold")
    ax.set_xlabel("Analysis #", color="#a4b0c0", fontsize=9)
    ax.set_ylabel("MC Uncertainty σ", color="#a4b0c0", fontsize=9)
    ax.grid(True, alpha=0.10, linestyle="--")
    ax.legend(fontsize=7, labelcolor="#a4b0c0", facecolor="#131a2c", edgecolor="#1f2937")
    fig.tight_layout(pad=0.5)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────

defaults = {
    "nav":                    "🏠 Home",
    "last_result":            None,
    "last_image":             None,
    "gradcam_image":          None,
    "gradcam_pp_image":       None,
    "mc_result":              None,
    "agreement_score":        None,
    "prediction_history":     [],
    "activity_log":           [],
    "live_session_start":     None,
    "live_predictions_count": 0,
    "live_avg_confidence":    0.0,
    "live_class_counts":      {},
    "live_throughput":        0.0,
    "live_last_confidence":   0.0,
    "live_latency_ms":        0.0,
    "live_inference_running": False,
}

for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v


def clear_prediction_history():
    for fig_key in ("gradcam_image", "gradcam_pp_image"):
        old = st.session_state.get(fig_key)
        if old is not None:
            plt.close(old)
    st.session_state.prediction_history    = []
    st.session_state.last_result           = None
    st.session_state.last_image            = None
    st.session_state.gradcam_image         = None
    st.session_state.gradcam_pp_image      = None
    st.session_state.mc_result             = None
    st.session_state.agreement_score       = None
    st.session_state.live_predictions_count = 0
    st.session_state.live_avg_confidence   = 0.0
    st.session_state.live_last_confidence  = 0.0
    st.session_state.live_class_counts     = {}
    st.session_state.live_throughput       = 0.0
    st.session_state.live_latency_ms       = 0.0


def log_activity(message, level="info"):
    ts = datetime.now().strftime("%H:%M:%S")
    st.session_state.activity_log.append({"timestamp": ts, "message": message, "level": level})
    if len(st.session_state.activity_log) > 60:
        st.session_state.activity_log = st.session_state.activity_log[-60:]


def update_live_stats(result, latency_ms):
    history = st.session_state.prediction_history
    st.session_state.live_predictions_count = len(history)
    if history:
        confs = [h["confidence"] for h in history]
        st.session_state.live_avg_confidence  = float(np.mean(confs))
        st.session_state.live_last_confidence = float(confs[-1])
    st.session_state.live_latency_ms = float(latency_ms)
    counts = {}
    for h in history:
        counts[h["prediction"]] = counts.get(h["prediction"], 0) + 1
    st.session_state.live_class_counts = counts
    if st.session_state.live_session_start:
        elapsed = (datetime.now() - st.session_state.live_session_start).total_seconds()
        if elapsed > 0:
            st.session_state.live_throughput = len(history) / elapsed * 60.0


# ─────────────────────────────────────────────────────────────────────────────
# MODEL INIT
# ─────────────────────────────────────────────────────────────────────────────

model = None
class_names = None
model_name  = None
model_error = None

try:
    model, class_names, model_name = load_model(MODEL_PATH)
except Exception as exc:
    model_error = str(exc)


# ─────────────────────────────────────────────────────────────────────────────
# NEURAL ANIMATION (retained for reuse)
# ─────────────────────────────────────────────────────────────────────────────

def render_neural_animation():
    _render_html("""
    <div id="nw" style="position:relative;width:100%;height:160px;overflow:hidden;border-radius:20px;
    background:radial-gradient(circle at 50% 50%,rgba(14,165,233,.12),rgba(10,14,26,.15) 55%,rgba(10,14,26,.7));
    border:1px solid rgba(56,189,248,.2)">
    <canvas id="nc" style="width:100%;height:100%;display:block"></canvas>
    <div style="position:absolute;left:18px;bottom:12px;display:flex;align-items:center;gap:8px;
    font:700 11px Inter,sans-serif;letter-spacing:.14em;color:#7dd3fc">
    <div style="width:8px;height:8px;border-radius:50%;background:#22d3ee"></div>LIVE NEURAL NETWORK</div></div>
    <script>
    const c=document.getElementById('nc'),ctx=c.getContext('2d');let ps=[];
    function rs(){const r=c.getBoundingClientRect(),d=window.devicePixelRatio||1;
    c.width=r.width*d;c.height=r.height*d;ctx.setTransform(d,0,0,d,0,0);
    ps=Array.from({length:Math.max(22,Math.floor(r.width/30))},()=>({x:Math.random()*r.width,y:Math.random()*r.height,vx:(Math.random()-.5)*.32,vy:(Math.random()-.5)*.32,r:1.4+Math.random()*2}))}
    function dr(){const r=c.getBoundingClientRect();ctx.clearRect(0,0,r.width,r.height);
    for(const p of ps){p.x+=p.vx;p.y+=p.vy;if(p.x<0||p.x>r.width)p.vx*=-1;if(p.y<0||p.y>r.height)p.vy*=-1}
    for(let i=0;i<ps.length;i++)for(let j=i+1;j<ps.length;j++){const a=ps[i],b=ps[j],d=Math.hypot(a.x-b.x,a.y-b.y);
    if(d<110){ctx.strokeStyle=`rgba(56,189,248,${(1-d/110)*.22})`;ctx.lineWidth=.65;ctx.beginPath();ctx.moveTo(a.x,a.y);ctx.lineTo(b.x,b.y);ctx.stroke()}}
    for(const p of ps){ctx.beginPath();ctx.arc(p.x,p.y,p.r,0,Math.PI*2);ctx.fillStyle='#38bdf8';ctx.fill()}
    requestAnimationFrame(dr)}
    window.addEventListener('resize',rs);rs();dr();
    </script>""", height=160, scrolling=False)


# ─────────────────────────────────────────────────────────────────────────────
# LIVE TICKER
# ─────────────────────────────────────────────────────────────────────────────

def render_live_ticker():
    counts = st.session_state.live_class_counts or {}
    items  = list(counts.items())
    text   = " · ".join(f"<b>{n}</b>: {v}" for n, v in items) if items else "<b>Awaiting first scan</b>"
    unc_str = ""
    mc = st.session_state.mc_result
    if mc:
        band  = mc.get("band", "")
        u_val = mc.get("uncertainty", 0)
        unc_str = f" · Uncertainty σ={u_val:.3f} ({band})"
    _render_html(f"""
    <style>
    .tw{{overflow:hidden;border-radius:11px;border:1px solid rgba(255,255,255,0.06);
    background:rgba(19,26,44,0.9);padding:.55rem 0;}}
    .tt{{display:inline-block;white-space:nowrap;padding-left:100%;
    animation:mq 26s linear infinite;color:#a4b0c0;font:500 .8rem 'JetBrains Mono',monospace;letter-spacing:.02em}}
    .lp{{display:inline-block;width:6px;height:6px;border-radius:50%;background:#0ea5e9;margin-right:9px;vertical-align:middle}}
    @keyframes mq{{0%{{transform:translateX(0);}}100%{{transform:translateX(-100%);}}}}
    </style>
    <div class="tw"><div class="tt"><span class="lp"></span>LIVE ·
    {text} · Throughput: {st.session_state.live_throughput:.2f}/min ·
    Last Conf: {st.session_state.live_last_confidence:.1f}% ·
    Avg Conf: {st.session_state.live_avg_confidence:.1f}%{unc_str}
    </div></div>""", height=40)


# ─────────────────────────────────────────────────────────────────────────────
# STICKY HEADER
# ─────────────────────────────────────────────────────────────────────────────

def render_sticky_header():
    counts = st.session_state.live_class_counts or {}
    if counts:
        ticker_items = " ".join(
            f"<span class='tick-item'><b>{k}</b> · {v}</span>" for k, v in counts.items()
        )
    else:
        ticker_items = "<span class='tick-item tick-idle'>Awaiting first scan…</span>"

    mc     = st.session_state.mc_result
    nav    = st.session_state.nav
    eng    = model_name or "—"

    engine_ok   = model_error is None and MODEL_PATH.exists()
    status_cls  = "status-ok" if engine_ok else "status-err"
    status_text = "ONLINE" if engine_ok else "OFFLINE"
    device_tag  = "GPU" if DEVICE.type == "cuda" else "CPU"

    mc_str = f"<span class='tick-item'>Uncertainty σ=<b>{mc['uncertainty']:.3f}</b></span>" if mc else ""

    st.markdown(safe_html(f"""
    <div class="sticky-header">
        <div class="sticky-brand">
            <div class="sticky-brand-logo">
                🧠
                <span class="sticky-brand-pulse"></span>
            </div>
            <div class="sticky-brand-text">
                <div class="sticky-brand-name">NeuroLens AI</div>
                <div class="sticky-brand-version">v3.0 · Production</div>
            </div>
        </div>

        <div class="sticky-divider"></div>

        <div class="sticky-ticker">
            <div class="sticky-ticker-inner">
                <span class="tick-badge">LIVE</span>
                {ticker_items}
                <span class="tick-item">Engine: <b>{eng}</b></span>
                <span class="tick-item">Device: <b>{device_tag}</b></span>
                {mc_str}
            </div>
        </div>

        <div class="sticky-divider"></div>

        <div class="sticky-status {status_cls}">
            <span class="sticky-status-dot"></span>
            <span>{status_text}</span>
        </div>

        <div class="sticky-page">
            <span class="sticky-page-dot"></span>
            {nav}
        </div>
    </div>"""), unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# ACTIVITY FEED
# ─────────────────────────────────────────────────────────────────────────────

def render_activity_feed():
    log = st.session_state.activity_log[-14:][::-1]
    if not log:
        st.markdown("<div style='color:var(--text-muted);font-size:.84rem'>No activity yet.</div>", unsafe_allow_html=True)
        return
    rows = "".join(
        f"<div class='activity-row activity-{e['level']}'>"
        f"<span class='activity-time'>{e['timestamp']}</span>"
        f"<span class='activity-msg'>{e['message']}</span></div>"
        for e in log
    )
    st.markdown(f"<div class='activity-feed'>{rows}</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# LIVE PROBABILITY BARS
# ─────────────────────────────────────────────────────────────────────────────

def render_live_probability_animation(result, mc_result=None):
    items = list(result["probabilities"].items())
    top   = result["prediction"]
    conf  = result["confidence"]

    mc_rows = ""
    if mc_result:
        mc_items = list(mc_result["mean_probs"].items())
        mc_std   = mc_result["std_probs"]
        mc_rows = "".join(
            f'<div class="lp-row"><div class="lp-label">{n} <span style="color:#8b5cf6;font-family:\'JetBrains Mono\',monospace">±{mc_std.get(n,0):.1f}%</span></div>'
            f'<div class="lp-track"><div class="lp-fill lp-mc" style="width:0%" data-target="{p:.2f}"></div></div>'
            f'<div class="lp-val">{p:.1f}%</div></div>'
            for n, p in mc_items
        )

    bars = "".join(
        f'<div class="lp-row"><div class="lp-label">{n}</div>'
        f'<div class="lp-track"><div class="lp-fill" style="width:0%" data-target="{p:.2f}"></div></div>'
        f'<div class="lp-val">{p:.1f}%</div></div>'
        for n, p in items
    )

    unc_html = ""
    if mc_result:
        band  = mc_result["band"]
        uval  = mc_result["uncertainty"]
        color = mc_result["color"]
        unc_html = f'<div style="margin-top:.85rem;padding:.65rem .85rem;border-radius:10px;background:rgba(139,92,246,0.06);border:1px solid rgba(139,92,246,0.20)"><span style="font-size:.66rem;color:#8b5cf6;text-transform:uppercase;letter-spacing:.12em;font-weight:700;font-family:\'Inter\',sans-serif">MC Uncertainty</span><span style="float:right;font-weight:700;color:{color};font-family:\'JetBrains Mono\',monospace;font-size:.82rem">σ={uval:.4f} · {band}</span></div>'

    _render_html(f"""
    <style>
    .live-prob{{padding:1.25rem 1.5rem;border-radius:16px;border:1px solid rgba(255,255,255,0.06);
    background:#131a2c;font-family:Inter,sans-serif;color:#e8edf5}}
    .live-prob-head{{display:flex;justify-content:space-between;align-items:center;margin-bottom:.85rem;
    padding-bottom:.6rem;border-bottom:1px solid rgba(255,255,255,0.06)}}
    .live-prob-title{{font-size:.68rem;color:#38bdf8;text-transform:uppercase;letter-spacing:.12em;font-weight:700}}
    .live-prob-pred{{font-family:Sora,sans-serif;font-size:.95rem;font-weight:700;color:#e8edf5;letter-spacing:-.01em}}
    .lp-row{{display:flex;align-items:center;gap:.75rem;margin:.45rem 0}}
    .lp-label{{min-width:140px;font-size:.8rem;color:#a4b0c0;font-weight:500}}
    .lp-track{{flex:1;height:6px;border-radius:999px;background:rgba(255,255,255,0.05);overflow:hidden;position:relative}}
    .lp-fill{{height:100%;width:0%;background:#0ea5e9;border-radius:999px;
    transition:width 1.2s cubic-bezier(.2,.8,.2,1)}}
    .lp-mc{{background:#8b5cf6 !important}}
    .lp-val{{min-width:60px;text-align:right;font-family:'JetBrains Mono',monospace;font-weight:600;color:#e8edf5;font-size:.78rem}}
    .lp-section{{font-size:.62rem;color:#6b7789;text-transform:uppercase;letter-spacing:.12em;font-weight:700;margin:.9rem 0 .35rem}}
    </style>
    <div class="live-prob">
        <div class="live-prob-head">
            <span class="live-prob-title">Live Probability Stream</span>
            <span class="live-prob-pred">{top} · {conf:.1f}%</span>
        </div>
        <div class="lp-section">Standard Inference</div>
        {bars}
        {"<div class='lp-section'>MC Dropout (Bayesian)</div>" + mc_rows if mc_result else ""}
        {unc_html}
    </div>
    <script>
    requestAnimationFrame(()=>{{
        document.querySelectorAll('.lp-fill').forEach(el=>{{
            const t=parseFloat(el.getAttribute('data-target'));
            requestAnimationFrame(()=>{{el.style.width=t+'%';}});
        }});
    }});
    </script>""", height=60 + 40 * len(items) + (40 * len(items) + 100 if mc_result else 0))


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    history    = st.session_state.prediction_history
    total_scans= len(history)
    avg_conf   = float(np.mean([x["confidence"] for x in history])) if history else 0.0
    last_pred  = history[-1]["prediction"] if history else "No analysis yet"
    active_nav = st.session_state.nav

    st.markdown(safe_html("""
    <div style="padding:.3rem .4rem 1.1rem">
        <div class="sidebar-brand-title">🧠 NeuroLens AI</div>
        <div class="sidebar-brand-subtitle">Neurodiagnostic Intelligence Platform · v3.0</div>
    </div>"""), unsafe_allow_html=True)

    engine_online = MODEL_PATH.exists() and model_error is None
    st.markdown(safe_html(f"""
    <div class="sidebar-active-indicator">
        <span class="sidebar-active-dot"></span>
        <span class="sidebar-active-text">ACTIVE: {active_nav}</span>
    </div>"""), unsafe_allow_html=True)

    groups = [
        ("MAIN",      ["🏠 Home", "🔬 MRI Analysis", "📊 Dashboard"]),
        ("ANALYTICS", ["🕘 History", "🔥 Grad-CAM", "🎯 XAI Lab"]),
        ("SYSTEM",    ["⚙️ Settings"]),
    ]

    for group_name, items in groups:
        st.markdown(f'<div class="sidebar-section-label">{group_name}</div>', unsafe_allow_html=True)
        for item in items:
            label = f"▶ {item}" if active_nav == item else item
            if active_nav == item:
                st.markdown('<div class="nav-active">', unsafe_allow_html=True)
            if st.button(label, key=f"nav_{item}", width="stretch"):
                st.session_state.nav = item
                st.rerun()
            if active_nav == item:
                st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")
    eng_status = "🟢 ENGINE READY" if engine_online else "🔴 MODEL MISSING"
    mc_badge   = ""
    if st.session_state.mc_result:
        mc = st.session_state.mc_result
        mc_badge = f"<br><div style='margin-top:.4rem;font-size:.68rem;color:#8b5cf6;font-family:\"JetBrains Mono\",monospace'>σ={mc['uncertainty']:.4f} · {mc['band']}</div>"

    st.markdown(safe_html(f"""
    <div style="margin-top:.5rem;padding:.8rem;border-radius:12px;background:rgba(255,255,255,.02);border:1px solid rgba(255,255,255,.06)">
        <div class="sidebar-mini-label">NEURAL ENGINE</div>
        <div style="margin-top:.3rem;font-size:.82rem;font-weight:700;color:{'#10b981' if engine_online else '#ef4444'}">{eng_status}</div>
        {mc_badge}
        <div class="sidebar-engine-grid" style="margin-top:.65rem">
            <div class="sidebar-mini-stat"><div class="sidebar-mini-label">Device</div><div class="sidebar-mini-value">{DEVICE}</div></div>
            <div class="sidebar-mini-stat"><div class="sidebar-mini-label">Scans</div><div class="sidebar-mini-value">{total_scans}</div></div>
            <div class="sidebar-mini-stat"><div class="sidebar-mini-label">Architecture</div><div class="sidebar-mini-value">{model_name or '—'}</div></div>
            <div class="sidebar-mini-stat"><div class="sidebar-mini-label">Avg Conf</div><div class="sidebar-mini-value">{avg_conf:.0f}%</div></div>
        </div>
    </div>"""), unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section-label">QUICK ACTIONS</div>', unsafe_allow_html=True)
    if st.button("🗑️ Clear History", key="sb_clear", width="stretch"):
        st.session_state.confirm_clear = True
    if st.session_state.get("confirm_clear", False):
        st.caption("Clear all session records?")
        c1, c2 = st.columns(2)
        if c1.button("Confirm", key="sb_clear_yes", width="stretch"):
            clear_prediction_history()
            st.session_state.confirm_clear = False
            st.rerun()
        if c2.button("Cancel", key="sb_clear_no", width="stretch"):
            st.session_state.confirm_clear = False

    if st.button("🔄 Reset Session", key="sb_reset", width="stretch"):
        st.session_state.confirm_reset = True
    if st.session_state.get("confirm_reset", False):
        st.caption("Reset entire session?")
        c1, c2 = st.columns(2)
        if c1.button("Confirm", key="sb_reset_yes", width="stretch"):
            for fig_key in ("gradcam_image", "gradcam_pp_image"):
                old = st.session_state.get(fig_key)
                if old is not None:
                    plt.close(old)
            for k, v in defaults.items():
                st.session_state[k] = v
            st.rerun()
        if c2.button("Cancel", key="sb_reset_no", width="stretch"):
            st.session_state.confirm_reset = False

    st.markdown(safe_html("""
    <div class="sidebar-footer">
        ⚠️ Research Prototype<br>
        Not for clinical use. Predictions do not constitute medical diagnoses.
    </div>"""), unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# ROUTER
# ─────────────────────────────────────────────────────────────────────────────

render_sticky_header()
nav = st.session_state.nav


# ══════════════════════════════════════════════════════════════════════════════
# HOME
# ══════════════════════════════════════════════════════════════════════════════

if nav == "🏠 Home":
    render_live_ticker()

    st.markdown(safe_html(dedent("""
    <div class="hero">
        <div class="hero-badge">🧠 AI-Powered Brain MRI Analysis · Research Prototype</div>
        <h1>NeuroLens AI</h1>
        <p>Deep Learning · Dual XAI (Grad-CAM + Grad-CAM++) · MC Dropout Uncertainty Estimation · Neuroimaging</p>
    </div>""")), unsafe_allow_html=True)

    # ── Centered CTA button (medium-small, narrower columns) ─────────────
    _cta_l, _cta_c, _cta_r = st.columns([1.6, 1, 1.6])
    with _cta_c:
        if st.button("🔬 Analyze MRI Scan", type="primary", key="home_cta", width="stretch"):
            st.session_state.nav = "🔬 MRI Analysis"
            st.rerun()

    # ── Centered caption ─────────────────────────────────────────────────
    st.markdown(
        "<div style='text-align:center;color:var(--text-3);"
        "font-size:.82rem;margin-top:.55rem;line-height:1.5'>"
        "Research prototype for education and research. "
        "Model predictions are not medical diagnoses."
        "</div>",
        unsafe_allow_html=True,
    )

    h2 = st.session_state.prediction_history
    hc1, hc2, hc3, hc4 = st.columns(4)
    lm = [
        ("📡", st.session_state.live_predictions_count, "Live Scans"),
        ("🎯", f"{st.session_state.live_avg_confidence:.1f}%", "Avg. Confidence"),
        ("⚡", f"{st.session_state.live_throughput:.2f}/m", "Throughput"),
        ("⏱️", f"{np.mean([x.get('latency_ms', 0) for x in h2]):.0f} ms" if h2 else "—", "Avg. Inference"),
    ]
    for col, (icon, val, lbl) in zip([hc1, hc2, hc3, hc4], lm):
        with col:
            st.markdown(f'<div class="metric-card"><div class="metric-icon">{icon}</div><div class="metric-value">{val}</div><div class="metric-label">{lbl}</div></div>', unsafe_allow_html=True)

    st.write("")
    c1, c2, c3, c4 = st.columns(4)
    cards = [
        ("🔬", "Brain MRI Classification", "Four-class classification: Glioma, Meningioma, No Tumor, Pituitary using state-of-the-art CNN architectures."),
        ("🎯", "MC Dropout Uncertainty", "Bayesian uncertainty estimation via Monte Carlo Dropout. Reliability bands tell you how confident the model really is."),
        ("🔥", "Dual XAI (Grad-CAM++)", "Side-by-side Grad-CAM and Grad-CAM++ visualizations with a score showing how similar their heatmaps are."),
        ("⚡", "Real-Time Inference", "Detailed timing breakdown: preprocessing, inference, and XAI generation with CUDA-accurate latency measurement."),
    ]
    for col, (icon, title, desc) in zip([c1, c2, c3, c4], cards):
        with col:
            st.markdown(f'<div class="info-card"><div style="font-size:1.6rem;margin-bottom:.6rem;opacity:.9">{icon}</div><h3>{title}</h3><p>{desc}</p></div>', unsafe_allow_html=True)

    st.write("")
    st.subheader("How NeuroLens Works")
    steps = [
        ("1", "Upload MRI", "JPG, PNG, WEBP"),
        ("2", "Preprocess", "Resize → Normalize"),
        ("3", "Inference", "Deep CNN forward pass"),
        ("4", "MC Dropout", "20 stochastic passes → σ"),
        ("5", "Dual XAI", "Grad-CAM + Grad-CAM++"),
        ("6", "Agreement", "Explanation Agreement Score"),
        ("7", "Report", "Download TXT/PNG"),
    ]
    step_cols = st.columns(len(steps))
    for col, (num, title, desc) in zip(step_cols, steps):
        with col:
            st.markdown(safe_html(f"""
            <div style="text-align:center;padding:.9rem .5rem;border-radius:14px;background:#131a2c;border:1px solid rgba(255,255,255,.06)">
                <div style="font-family:Sora,sans-serif;font-size:1.3rem;font-weight:700;color:#38bdf8">{num}</div>
                <div style="font-size:.78rem;font-weight:600;color:var(--text-primary);margin:.3rem 0">{title}</div>
                <div style="font-size:.66rem;color:var(--text-muted)">{desc}</div>
            </div>"""), unsafe_allow_html=True)

    st.write("")
    if model_error:
        st.markdown(safe_html(f"""
        <div class="info-card" style="border-color:rgba(239,68,68,.3);background:rgba(239,68,68,.05)">
            <h3 style="color:var(--error)">⚠️ Neural Engine Unavailable</h3>
            <p>Expected: <code>{MODEL_PATH}</code></p>
        </div>"""), unsafe_allow_html=True)
        with st.expander("Technical Details"):
            st.code(model_error)
    else:
        st.markdown(safe_html(f"""
        <div class="info-card" style="border-color:rgba(16,185,129,.3); background:rgba(16,185,129,.05)">
            <div style="display: flex; align-items: center; gap: 0.6rem; margin-bottom: 1rem;">
                <span style="font-size: 1.2rem;">✅</span>
                <h3 style="color:var(--success-var); margin: 0; font-size: 1.1rem;">Neural Engine Ready</h3>
            </div>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 1rem;">
                <div style="background: rgba(0,0,0,0.15); padding: 0.75rem 1rem; border-radius: 10px; border: 1px solid rgba(16,185,129,0.2);">
                    <div style="font-size: 0.65rem; color: #6b7789; text-transform: uppercase; letter-spacing: 0.08em; font-weight: 600;">Architecture</div>
                    <div style="font-size: 0.95rem; font-weight: 700; color: #e8edf5; margin-top: 0.3rem; font-family: 'JetBrains Mono', monospace;">{model_name}</div>
                </div>
                <div style="background: rgba(0,0,0,0.15); padding: 0.75rem 1rem; border-radius: 10px; border: 1px solid rgba(16,185,129,0.2);">
                    <div style="font-size: 0.65rem; color: #6b7789; text-transform: uppercase; letter-spacing: 0.08em; font-weight: 600;">Device</div>
                    <div style="font-size: 0.95rem; font-weight: 700; color: #e8edf5; margin-top: 0.3rem; font-family: 'JetBrains Mono', monospace;">{DEVICE}</div>
                </div>
                <div style="background: rgba(0,0,0,0.15); padding: 0.75rem 1rem; border-radius: 10px; border: 1px solid rgba(16,185,129,0.2);">
                    <div style="font-size: 0.65rem; color: #6b7789; text-transform: uppercase; letter-spacing: 0.08em; font-weight: 600;">Classes</div>
                    <div style="font-size: 0.9rem; font-weight: 700; color: #38bdf8; margin-top: 0.3rem;">{', '.join(class_names)}</div>
                </div>
                <div style="background: rgba(0,0,0,0.15); padding: 0.75rem 1rem; border-radius: 10px; border: 1px solid rgba(16,185,129,0.2);">
                    <div style="font-size: 0.65rem; color: #6b7789; text-transform: uppercase; letter-spacing: 0.08em; font-weight: 600;">MC Dropout & XAI</div>
                    <div style="font-size: 0.9rem; font-weight: 700; color: #e8edf5; margin-top: 0.3rem;">{MC_SAMPLES} passes · Grad-CAM + Grad-CAM++</div>
                </div>
            </div>
        </div>"""), unsafe_allow_html=True)

    if st.session_state.last_result:
        st.write("")
        st.subheader("Last Analysis — Live Probability Stream")
        render_live_probability_animation(
            st.session_state.last_result,
            mc_result=st.session_state.mc_result
        )


# ══════════════════════════════════════════════════════════════════════════════
# MRI ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

elif nav == "🔬 MRI Analysis":
    st.title("🔬 MRI Diagnostic Analysis")
    st.caption("Upload a brain MRI scan for AI-powered classification with uncertainty estimation and dual explainability.")
    render_live_ticker()

    if model_error:
        st.error("🚨 Neural engine unavailable. Cannot run inference.")
        st.info(f"Expected model: `{MODEL_PATH}`")
        with st.expander("Technical Details"):
            st.code(model_error)
    else:
        # ═══════════════════════════════════════════════════════════
        # ELEGANT UPLOAD ZONE
        # ═══════════════════════════════════════════════════════════
        st.markdown(safe_html("""
        <div class="upload-hero">
            <div class="upload-icon">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor"
                     stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
                    <polyline points="17 8 12 3 7 8"/>
                    <line x1="12" y1="3" x2="12" y2="15"/>
                </svg>
            </div>
            <div class="upload-title">Upload Brain MRI Scan</div>
            <div class="upload-subtitle">
                Drop your scan below or click to browse your files
            </div>
            <div class="upload-badges">
                <span class="upload-badge">JPG</span>
                <span class="upload-badge">JPEG</span>
                <span class="upload-badge">PNG</span>
                <span class="upload-badge">WEBP</span>
                <span class="upload-badge upload-badge-muted">≤ 200 MB</span>
            </div>
        </div>
        """), unsafe_allow_html=True)

        _up_l, _up_c, _up_r = st.columns([1, 2, 1])
        with _up_c:
            uploaded_file = st.file_uploader(
                "Upload Brain MRI Scan (JPG · PNG · WEBP)",
                type=["jpg", "jpeg", "png", "webp"],
                key="mri_uploader",
                label_visibility="collapsed",
            )
            st.markdown(safe_html("""
            <div class="upload-footnote">
                <span class="upload-footnote-dot"></span>
                Files are processed locally · Nothing is uploaded to external servers
            </div>
            """), unsafe_allow_html=True)

        image_id = None
        if uploaded_file is not None:
            upload_bytes = uploaded_file.getvalue()
            image_id     = hashlib.sha256(upload_bytes).hexdigest()

            prev = st.session_state.last_result
            if prev is not None and prev.get("image_id") != image_id:
                for fk in ("gradcam_image", "gradcam_pp_image"):
                    old = st.session_state.get(fk)
                    if old is not None:
                        plt.close(old)
                st.session_state.last_result     = None
                st.session_state.last_image      = None
                st.session_state.gradcam_image   = None
                st.session_state.gradcam_pp_image= None
                st.session_state.mc_result       = None
                st.session_state.agreement_score = None

            try:
                image = Image.open(io.BytesIO(upload_bytes)).convert("RGB")
            except (UnidentifiedImageError, OSError, ValueError):
                st.error("⚠️ Invalid image. Please upload a valid JPG, PNG, or WEBP file.")
                image = None

            if image is not None:
                col1, col2 = st.columns([1, 2])

                with col1:
                    st.markdown("### MRI Preview")
                    st.image(image, width="stretch")
                    st.caption(f"Original: {image.width}×{image.height}px · Processed: {IMG_SIZE}×{IMG_SIZE}px · RGB · Normalized")

                with col2:
                    st.markdown("### Analysis Configuration")
                    cfg1, cfg2, cfg3 = st.columns(3)
                    cfg1.metric("Architecture", model_name or "—")
                    cfg2.metric("Device", str(DEVICE).upper())
                    cfg3.metric("Classes", len(class_names))

                    run_mc = st.checkbox("Enable MC Dropout Uncertainty (20 passes)", value=True, key="run_mc_cb")
                    run_xai = st.checkbox("Enable Dual XAI (Grad-CAM + Grad-CAM++)", value=True, key="run_xai_cb")
                    run_agreement = st.checkbox("Compute Explanation Agreement Score", value=True, key="run_agree_cb")

                    # Medium-small primary CTA — sizes to content
                    if st.button("🔍 Run AI Analysis", type="primary", width="content", key="analyze_btn"):
                        if st.session_state.live_session_start is None:
                            st.session_state.live_session_start = datetime.now()

                        st.session_state.live_inference_running = True
                        status = st.empty()
                        prog   = st.empty()

                        try:
                            total_t0 = time.perf_counter()
                            stages   = [
                                "Image Loaded", "Preprocessing", "Normalization",
                                "Tensor Preparation", "Neural Network Inference",
                                "Probability Calculation",
                            ]
                            if run_mc:
                                stages.append("MC Dropout Estimation")
                            if run_xai:
                                stages += ["Grad-CAM Generation", "Grad-CAM++ Generation"]
                            if run_agreement:
                                stages.append("Explanation Agreement Score")
                            stages.append("Report Generation")

                            total_stages = len(stages)
                            for i, stage in enumerate(stages[:-4]):
                                status.info(f"⏳ {stage}…")
                                prog.progress((i + 1) / total_stages)

                            log_activity("MRI uploaded; preprocessing started", "info")
                            (predicted_class, confidence, probability_dict,
                             preprocessing_ms, inference_ms) = predict_image(image, model, class_names)
                            log_activity(f"Inference complete: {predicted_class} ({confidence:.1f}%)", "success")

                            step_offset = 4
                            mc_result   = None
                            if run_mc:
                                status.info("⏳ MC Dropout Uncertainty Estimation…")
                                prog.progress((step_offset + 1) / total_stages)
                                step_offset += 1
                                try:
                                    mc_result = mc_dropout_predict(image, model, class_names, MC_SAMPLES)
                                    log_activity(f"MC Dropout: σ={mc_result['uncertainty']:.4f} ({mc_result['band']})", "info")
                                except Exception:
                                    log_activity("MC Dropout failed", "warn")

                            gradcam_fig  = None
                            gradcam_ms   = 0.0
                            gradcam_pp_fig = None
                            gradcam_pp_ms  = 0.0

                            if run_xai:
                                status.info("⏳ Generating Grad-CAM…")
                                prog.progress((step_offset + 1) / total_stages)
                                step_offset += 1
                                try:
                                    t_gc = time.perf_counter()
                                    gradcam_fig = generate_gradcam(image, model, model_name)
                                    gradcam_ms  = (time.perf_counter() - t_gc) * 1000
                                    log_activity("Grad-CAM generated", "info")
                                except Exception:
                                    log_activity("Grad-CAM failed", "warn")

                                status.info("⏳ Generating Grad-CAM++…")
                                prog.progress((step_offset + 1) / total_stages)
                                step_offset += 1
                                try:
                                    t_pp = time.perf_counter()
                                    gradcam_pp_fig = generate_gradcam_pp(image, model, model_name)
                                    gradcam_pp_ms  = (time.perf_counter() - t_pp) * 1000
                                    log_activity("Grad-CAM++ generated", "info")
                                except Exception:
                                    log_activity("Grad-CAM++ failed", "warn")

                            agree_score = None
                            if run_agreement and run_xai:
                                status.info("⏳ Computing Explanation Agreement Score…")
                                prog.progress((step_offset + 1) / total_stages)
                                step_offset += 1
                                try:
                                    agree_score = explanation_agreement(image, model, model_name)
                                    if agree_score is not None:
                                        log_activity(f"Explanation Agreement Score: {agree_score:.3f}", "info")
                                except Exception:
                                    log_activity("Agreement score failed", "warn")

                            total_ms = (time.perf_counter() - total_t0) * 1000
                            status.info("⏳ Building report…")
                            prog.progress(1.0)

                            result = {
                                "prediction":      predicted_class,
                                "confidence":      confidence,
                                "probabilities":   probability_dict,
                                "model":           model_name,
                                "timestamp":       datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                "latency_ms":      inference_ms,
                                "preprocessing_ms":preprocessing_ms,
                                "gradcam_ms":      gradcam_ms if gradcam_fig else None,
                                "gradcam_pp_ms":   gradcam_pp_ms if gradcam_pp_fig else None,
                                "total_ms":        total_ms,
                                "gradcam_available": gradcam_fig is not None,
                                "gradcam_pp_available": gradcam_pp_fig is not None,
                                "image_id":        image_id,
                                "uncertainty":     mc_result["uncertainty"] if mc_result else None,
                                "mc_band":         mc_result["band"] if mc_result else None,
                                "agreement_score": agree_score,
                            }

                            for fk, new_fig in [("gradcam_image", gradcam_fig), ("gradcam_pp_image", gradcam_pp_fig)]:
                                old = st.session_state.get(fk)
                                if old is not None:
                                    plt.close(old)
                                st.session_state[fk] = new_fig

                            st.session_state.last_result     = result
                            st.session_state.last_image      = image.copy()
                            st.session_state.mc_result       = mc_result
                            st.session_state.agreement_score = agree_score

                            st.session_state.prediction_history.append(result)
                            st.session_state.prediction_history = st.session_state.prediction_history[-MAX_HISTORY:]
                            update_live_stats(result, inference_ms)
                            log_activity("Analysis report generated", "success")

                            status.success(f"✅ {predicted_class} · {confidence:.2f}% · {inference_ms:.0f} ms inference")
                            prog.empty()

                        except Exception as exc:
                            msg = "CUDA out of memory. Try CPU inference." if "out of memory" in str(exc).lower() \
                                  else "Analysis failed. Check image and model."
                            status.error(msg)
                            with st.expander("Technical Details"):
                                st.code(str(exc))
                            log_activity("Analysis failed", "error")
                        finally:
                            st.session_state.live_inference_running = False

        if (image_id is not None
                and st.session_state.last_result is not None
                and st.session_state.last_result.get("image_id") == image_id):

            result   = st.session_state.last_result
            mc_res   = st.session_state.mc_result
            conf_val = result["confidence"]
            conf_lbl = "High" if conf_val >= 80 else "Moderate" if conf_val >= 60 else "Low"
            conf_cls = "high" if conf_val >= 80 else "moderate" if conf_val >= 60 else "low"

            st.markdown(safe_html(f"""
            <div class="diagnostic-panel">
                <div class="diagnostic-header">
                    <span class="diagnostic-title">AI Classification Result</span>
                    <span class="medical-badge badge-research">RESEARCH PROTOTYPE</span>
                </div>
                <div class="diagnostic-prediction">{result['prediction']}</div>
                <div class="diagnostic-confidence">{conf_val:.2f}% model confidence</div>
                <div style="margin-top:.85rem">
                    <span class="medical-badge badge-{conf_cls}">{conf_lbl} Confidence</span>
                </div>
            </div>"""), unsafe_allow_html=True)

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Model Confidence", f"{conf_val:.2f}%")
            m2.metric("Inference Latency", f"{result['latency_ms']:.1f} ms")
            m3.metric("Preprocessing", f"{result.get('preprocessing_ms', 0):.1f} ms")
            m4.metric("Total Time", f"{result.get('total_ms', 0):.1f} ms")

            if mc_res:
                unc   = mc_res["uncertainty"]
                band  = mc_res["band"]
                color = mc_res["color"]
                st.markdown(safe_html(f"""
                <div class="uncertainty-card">
                    <div class="uncertainty-title">MC Dropout Uncertainty Estimation ({MC_SAMPLES} passes)</div>
                    <div style="display:flex;justify-content:space-between;align-items:center">
                        <div>
                            <div class="uncertainty-value" style="color:{color}">σ = {unc:.4f}</div>
                            <div class="uncertainty-band" style="color:{color}">{band}</div>
                        </div>
                        <div style="text-align:right;font-size:.8rem;color:var(--text-secondary)">
                            MC Confidence: <b style="color:var(--accent-light);font-family:'JetBrains Mono',monospace">{mc_res['confidence']:.2f}%</b><br>
                            Prediction: <b>{mc_res['prediction']}</b>
                        </div>
                    </div>
                </div>"""), unsafe_allow_html=True)

            agree = result.get("agreement_score")
            if agree is not None:
                a_color = "#10b981" if agree >= 0.7 else "#f59e0b" if agree >= 0.5 else "#ef4444"
                a_label = "High Agreement" if agree >= 0.7 else "Moderate Agreement" if agree >= 0.5 else "Low Agreement"
                st.markdown(safe_html(f"""
                <div class="xai-card">
                    <div class="xai-title">Explanation Agreement Score</div>
                    <div style="display:flex;justify-content:space-between;align-items:center">
                        <div class="xai-text">
                            Pearson correlation between Grad-CAM and Grad-CAM++ heatmaps.<br>
                            A high score (&gt;0.70) means both methods highlight similar regions. Similarity alone does not verify that an explanation is correct.
                        </div>
                        <div style="text-align:right;min-width:90px">
                            <div style="font-family:'Sora',sans-serif;font-size:1.5rem;font-weight:700;color:{a_color};letter-spacing:-.02em">{agree:.3f}</div>
                            <div style="font-size:.7rem;color:{a_color};font-weight:600">{a_label}</div>
                        </div>
                    </div>
                </div>"""), unsafe_allow_html=True)

            st.warning("⚠️ NeuroLens AI is a research prototype for educational use. Model outputs are not medical diagnoses and must not replace evaluation by a qualified healthcare professional.")

            st.write("")
            st.markdown("### Probability Distribution")
            prob_html = '<div class="probability-grid">' + "".join(
                f'<div class="probability-card {"predicted" if label == result["prediction"] else ""}">'
                f'<div class="probability-value">{prob:.1f}%</div>'
                f'<div class="probability-label">{label}</div>'
                f'{"<div style=\"font-size:.62rem;color:var(--accent-light);margin-top:.3rem;font-weight:700;letter-spacing:.08em\">▲ TOP</div>" if label == result["prediction"] else ""}'
                f'</div>'
                for label, prob in result["probabilities"].items()
            ) + '</div>'
            st.markdown(prob_html, unsafe_allow_html=True)

            st.write("")
            st.markdown("### Live Probability Stream")
            render_live_probability_animation(result, mc_result=mc_res)

            if st.session_state.gradcam_image is not None or st.session_state.gradcam_pp_image is not None:
                st.write("")
                st.markdown("### Dual XAI Visualization")
                gc_col, pp_col = st.columns(2)
                with gc_col:
                    st.markdown("**Grad-CAM**")
                    if st.session_state.gradcam_image:
                        st.pyplot(st.session_state.gradcam_image, width="stretch")
                        st.caption("Grad-CAM · Jet colormap · α=0.44")
                with pp_col:
                    st.markdown("**Grad-CAM++ (Sharper)**")
                    if st.session_state.gradcam_pp_image:
                        st.pyplot(st.session_state.gradcam_pp_image, width="stretch")
                        st.caption("Grad-CAM++ · Inferno colormap · α=0.46")

                st.markdown(safe_html("""
                <div class="xai-card">
                    <div class="xai-title">Why This Prediction?</div>
                    <div class="xai-text">
                        Highlighted regions represent image areas that contributed more strongly to the model's classification.
                        Grad-CAM uses weighted class activations; Grad-CAM++ uses second-order gradients for sharper localization.
                        Neither method provides medically certified tumor localization. Use for research interpretation only.
                    </div>
                </div>"""), unsafe_allow_html=True)

            # ─────────────────────────────────────────────────
            # UNIFIED EXPORT PANEL
            # ─────────────────────────────────────────────────
            gc_buf = None
            if st.session_state.gradcam_image is not None:
                _b = io.BytesIO()
                st.session_state.gradcam_image.savefig(
                    _b, format="png", bbox_inches="tight", dpi=160
                )
                gc_buf = _b.getvalue()

            pp_buf = None
            if st.session_state.gradcam_pp_image is not None:
                _b2 = io.BytesIO()
                st.session_state.gradcam_pp_image.savefig(
                    _b2, format="png", bbox_inches="tight", dpi=160
                )
                pp_buf = _b2.getvalue()

            st.markdown(safe_html("""
            <div class="export-panel-head">
                <div class="export-head-icon">📥</div>
                <div class="export-head-text">
                    <div class="export-head-title">Export Results</div>
                    <div class="export-head-sub">Download your AI analysis outputs</div>
                </div>
                <div class="export-head-badge">READY</div>
            </div>
            """), unsafe_allow_html=True)

            has_gc = gc_buf is not None
            has_pp = pp_buf is not None

            if has_gc and has_pp:
                dc1, dc2 = st.columns(2)
                with dc1:
                    st.markdown('<div class="export-item-label">Grad-CAM · PNG</div>', unsafe_allow_html=True)
                    st.download_button(
                        "⬇️  Download Grad-CAM",
                        gc_buf, "gradcam.png", "image/png",
                        key="dl_gc", width="stretch",
                    )
                with dc2:
                    st.markdown('<div class="export-item-label">Grad-CAM++ · PNG</div>', unsafe_allow_html=True)
                    st.download_button(
                        "⬇️  Download Grad-CAM++",
                        pp_buf, "gradcam_pp.png", "image/png",
                        key="dl_pp", width="stretch",
                    )
            elif has_gc:
                st.markdown('<div class="export-item-label">Grad-CAM · PNG</div>', unsafe_allow_html=True)
                st.download_button(
                    "⬇️  Download Grad-CAM",
                    gc_buf, "gradcam.png", "image/png",
                    key="dl_gc", width="stretch",
                )
            elif has_pp:
                st.markdown('<div class="export-item-label">Grad-CAM++ · PNG</div>', unsafe_allow_html=True)
                st.download_button(
                    "⬇️  Download Grad-CAM++",
                    pp_buf, "gradcam_pp.png", "image/png",
                    key="dl_pp", width="stretch",
                )

            st.markdown(
                '<div class="export-item-label" style="margin-top:1rem">Analysis Report · TXT</div>',
                unsafe_allow_html=True,
            )
            st.download_button(
                "📄  Download Analysis Report",
                "\n".join([
                    "═══════════════════════════════════════════════════════",
                    "  NeuroLens AI — MRI Analysis Report",
                    "═══════════════════════════════════════════════════════",
                    f"Timestamp           : {result['timestamp']}",
                    f"Model Architecture  : {result['model']}",
                    f"Inference Device    : {DEVICE}",
                    "─────────────────────────────────────────────────────",
                    "  AI CLASSIFICATION RESULT",
                    "─────────────────────────────────────────────────────",
                    f"Model Prediction    : {result['prediction']}",
                    f"Model Confidence    : {result['confidence']:.4f}%",
                    "  Class Probabilities:",
                    *[f"    {k:<16}: {v:.4f}%" for k, v in result["probabilities"].items()],
                    "─────────────────────────────────────────────────────",
                    "  UNCERTAINTY ESTIMATION (MC Dropout)",
                    "─────────────────────────────────────────────────────",
                    f"MC Uncertainty σ    : {result.get('uncertainty', 'N/A')}",
                    f"Reliability Band    : {result.get('mc_band', 'N/A')}",
                    "─────────────────────────────────────────────────────",
                    "  EXPLAINABILITY",
                    "─────────────────────────────────────────────────────",
                    f"Grad-CAM Available  : {'Yes' if result.get('gradcam_available') else 'No'}",
                    f"Grad-CAM++ Available: {'Yes' if result.get('gradcam_pp_available') else 'No'}",
                    "Agreement Score     : " + (f"{result.get('agreement_score'):.4f}" if result.get('agreement_score') is not None else 'N/A'),
                    "─────────────────────────────────────────────────────",
                    "  TIMING",
                    "─────────────────────────────────────────────────────",
                    f"Preprocessing       : {result.get('preprocessing_ms', 0):.2f} ms",
                    f"Inference           : {result['latency_ms']:.2f} ms",
                    f"Grad-CAM            : {result.get('gradcam_ms') or 0:.2f} ms",
                    f"Grad-CAM++          : {result.get('gradcam_pp_ms') or 0:.2f} ms",
                    f"Total               : {result.get('total_ms', 0):.2f} ms",
                    "═══════════════════════════════════════════════════════",
                    "  DISCLAIMER",
                    "─────────────────────────────────────────────────────",
                    "  NeuroLens AI is an AI research prototype intended for",
                    "  educational and research purposes only. Model outputs",
                    "  do NOT constitute medical diagnoses and must NOT replace",
                    "  evaluation by a qualified healthcare professional.",
                    "═══════════════════════════════════════════════════════",
                ]),
                "neurolens_report.txt",
                "text/plain",
                key="dl_report",
                width="stretch",
                type="primary",
            )

    st.write("")


# ══════════════════════════════════════════════════════════════════════════════
# DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════

elif nav == "📊 Dashboard":
    st.title("📊 Neurodiagnostic Dashboard")
    st.caption("Session analytics from completed MRI analyses")
    render_live_ticker()

    if st.session_state.live_session_start is None:
        st.session_state.live_session_start = datetime.now()

    history = st.session_state.prediction_history

    if not history:
        st.markdown(safe_html("""
        <div class="empty-state">
            <div class="empty-state-icon">📊</div>
            <div class="empty-state-title">Awaiting live diagnostic data</div>
            <div class="empty-state-text">Run an MRI analysis to populate the dashboard.</div>
        </div>"""), unsafe_allow_html=True)
    else:
        total    = st.session_state.live_predictions_count
        avg_c    = st.session_state.live_avg_confidence
        unique   = len(st.session_state.live_class_counts)
        thru     = st.session_state.live_throughput
        avg_lat  = float(np.mean([h.get("latency_ms", 0) for h in history]))
        unc_data = [h["uncertainty"] for h in history if h.get("uncertainty") is not None]
        avg_unc  = float(np.mean(unc_data)) if unc_data else None
        agree_data = [h["agreement_score"] for h in history if h.get("agreement_score") is not None]
        avg_agree = float(np.mean(agree_data)) if agree_data else None

        m1, m2, m3, m4, m5 = st.columns(5)
        mets = [
            ("📈", total, "Total Scans"),
            ("🎯", f"{avg_c:.1f}%", "Avg. Confidence"),
            ("🧬", unique, "Classes Seen"),
            ("⚡", f"{avg_lat:.0f} ms", "Avg. Latency"),
            ("🎲", f"σ={avg_unc:.4f}" if avg_unc is not None else "—", "Avg. Uncertainty"),
        ]
        for col, (icon, val, lbl) in zip([m1, m2, m3, m4, m5], mets):
            with col:
                st.markdown(f'<div class="metric-card"><div class="metric-icon">{icon}</div><div class="metric-value">{val}</div><div class="metric-label">{lbl}</div></div>', unsafe_allow_html=True)

        if avg_agree is not None:
            st.markdown(safe_html(f"""
            <div style="margin:.75rem 0;padding:.85rem 1.25rem;border-radius:14px;background:#131a2c;border:1px solid rgba(14,165,233,.22);display:flex;justify-content:space-between;align-items:center">
                <span style="font-size:.72rem;color:var(--accent-light);font-weight:700;text-transform:uppercase;letter-spacing:.1em">Average Explanation Agreement Score</span>
                <span style="font-family:'Sora',sans-serif;font-size:1.3rem;font-weight:700;color:var(--text-primary);letter-spacing:-.02em">{avg_agree:.3f}</span>
            </div>"""), unsafe_allow_html=True)

        st.write("")
        col_left, col_right = st.columns([2, 1])

        with col_left:
            st.subheader("Prediction Distribution")
            counts  = st.session_state.live_class_counts
            max_cnt = max(counts.values()) if counts else 1
            for label, count in counts.items():
                pct      = (count / max_cnt) * 100
                live_pct = (count / total) * 100 if total else 0
                st.markdown(safe_html(f"""
                <div class="dist-card">
                    <div class="dist-header">
                        <div class="dist-name">{label}</div>
                        <div class="dist-count">{count} · {live_pct:.1f}%</div>
                    </div>
                    <div class="dist-bar-bg"><div class="dist-bar-fill" style="width:{pct:.0f}%"></div></div>
                </div>"""), unsafe_allow_html=True)

            st.write("")
            st.subheader("Model Confidence Trend")
            cf = plot_confidence_trend(history)
            if cf:
                st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                st.pyplot(cf, width="stretch")
                st.markdown('</div>', unsafe_allow_html=True)
                plt.close(cf)
            else:
                st.caption("Need ≥2 analyses.")

            st.subheader("Inference Latency Trend")
            lf = plot_latency_trend(history)
            if lf:
                st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                st.pyplot(lf, width="stretch")
                st.markdown('</div>', unsafe_allow_html=True)
                plt.close(lf)
            else:
                st.caption("Need ≥2 analyses.")

            if unc_data:
                st.subheader("MC Dropout Uncertainty Trend")
                uf = plot_uncertainty_history(history)
                if uf:
                    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                    st.pyplot(uf, width="stretch")
                    st.markdown('</div>', unsafe_allow_html=True)
                    plt.close(uf)

        with col_right:
            st.subheader("Live Activity Feed")
            render_activity_feed()

            st.write("")
            st.subheader("Latest Result")
            latest = history[-1]
            elapsed = "—"
            if st.session_state.live_session_start:
                el = datetime.now() - st.session_state.live_session_start
                elapsed = f"{int(el.total_seconds())}s"
            rows = [
                ("Prediction",      latest["prediction"]),
                ("Confidence",      f"{latest['confidence']:.2f}%"),
                ("Architecture",    latest["model"]),
                ("Inference",       f"{latest.get('latency_ms', 0):.0f} ms"),
                ("Uncertainty σ",   f"{latest['uncertainty']:.4f}" if latest.get("uncertainty") is not None else "—"),
                ("Agreement",       f"{latest['agreement_score']:.3f}" if latest.get("agreement_score") is not None else "—"),
                ("Session Time",    elapsed),
                ("Analyzed At",     latest["timestamp"]),
            ]
            card_html = '<div class="latest-card">' + "".join(
                f'<div class="latest-row"><div class="latest-key">{k}</div><div class="latest-val">{v}</div></div>'
                for k, v in rows
            ) + '</div>'
            st.markdown(card_html, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# HISTORY
# ══════════════════════════════════════════════════════════════════════════════

elif nav == "🕘 History":
    st.title("🕘 Analysis History")
    st.caption("Review all session AI diagnostic reports")
    render_live_ticker()

    history = st.session_state.prediction_history

    if not history:
        st.markdown(safe_html("""
        <div class="empty-state">
            <div class="empty-state-icon">🕘</div>
            <div class="empty-state-title">No diagnostic history</div>
            <div class="empty-state-text">Analysis reports appear here after running MRI scans.</div>
        </div>"""), unsafe_allow_html=True)
    else:
        fc1, fc2, fc3, fc4 = st.columns([1, 1, 1, 2])
        pred_filter = fc1.selectbox("Prediction", ["All"] + CLASS_NAMES)
        conf_filter = fc2.selectbox("Confidence", ["All", "High (≥80%)", "Moderate (60–79%)", "Low (<60%)"])
        sort_order  = fc3.selectbox("Sort", ["Newest first", "Oldest first"])
        search_q    = fc4.text_input("Search")

        filtered = []
        for rec in history:
            c = rec["confidence"]
            if pred_filter != "All" and rec["prediction"] != pred_filter: continue
            if conf_filter == "High (≥80%)" and c < 80: continue
            if conf_filter == "Moderate (60–79%)" and not (60 <= c < 80): continue
            if conf_filter == "Low (<60%)" and c >= 60: continue
            q = search_q.strip().lower()
            if q and q not in rec["model"].lower() and q not in rec["prediction"].lower(): continue
            filtered.append(rec)

        if sort_order == "Newest first":
            filtered = list(reversed(filtered))

        if not filtered:
            st.info("No records match these filters.")
        else:
            st.caption(f"Showing {len(filtered)} of {len(history)} records")
            for item in filtered:
                conf = item["confidence"]
                cc   = "confidence-high" if conf >= 80 else "confidence-medium" if conf >= 60 else "confidence-low"
                unc_tag = f" · σ={item['uncertainty']:.4f}" if item.get("uncertainty") is not None else ""
                agree_tag = f" · Agreement={item['agreement_score']:.3f}" if item.get("agreement_score") is not None else ""
                st.markdown(safe_html(f"""
                <div class="thumbnail-card">
                    <div class="thumbnail-img">🧠</div>
                    <div class="thumbnail-info">
                        <div class="thumbnail-title">{item['prediction']}</div>
                        <div class="thumbnail-meta">{item['timestamp']} · {item['model']}{unc_tag}{agree_tag}</div>
                    </div>
                    <div><span class="confidence-badge {cc}">{conf:.1f}%</span></div>
                </div>"""), unsafe_allow_html=True)
                with st.expander(f"Details — {item['timestamp']}"):
                    d1, d2 = st.columns(2)
                    with d1:
                        st.write(f"**Prediction:** {item['prediction']}")
                        st.write(f"**Confidence:** {item['confidence']:.4f}%")
                        st.write(f"**Architecture:** {item['model']}")
                        st.write(f"**Inference:** {item.get('latency_ms', 0):.1f} ms")
                    with d2:
                        if item.get("uncertainty") is not None:
                            st.write(f"**MC Uncertainty σ:** {item['uncertainty']:.6f}")
                            st.write(f"**Reliability:** {item.get('mc_band', '—')}")
                        if item.get("agreement_score") is not None:
                            st.write(f"**Agreement Score:** {item['agreement_score']:.4f}")
                        st.write("**Class Probabilities:**")
                        for lbl, prob in item["probabilities"].items():
                            st.write(f"  · {lbl}: {prob:.4f}%")

        st.write("")
        if st.button("🗑️ Clear History", key="clear_hist_btn", width="content"):
            st.session_state.confirm_clear_hist = True
        if st.session_state.get("confirm_clear_hist", False):
            st.warning("Clear all records from this session?")
            cc1, cc2, _ = st.columns([1, 1, 4])
            if cc1.button("Confirm", key="confirm_hist", width="stretch"):
                clear_prediction_history()
                st.session_state.confirm_clear_hist = False
                st.rerun()
            if cc2.button("Cancel", key="cancel_hist", width="stretch"):
                st.session_state.confirm_clear_hist = False


# ══════════════════════════════════════════════════════════════════════════════
# GRAD-CAM PAGE
# ══════════════════════════════════════════════════════════════════════════════

elif nav == "🔥 Grad-CAM":
    st.title("🔥 Grad-CAM Explainability")
    st.caption("Visualize which MRI regions influenced the model's classification decision")
    render_live_ticker()

    if st.session_state.gradcam_image is None and st.session_state.gradcam_pp_image is None:
        st.markdown(safe_html("""
        <div class="empty-state">
            <div class="empty-state-icon">🔥</div>
            <div class="empty-state-title">No Grad-CAM visualization yet</div>
            <div class="empty-state-text">Run an MRI analysis with XAI enabled to generate heatmaps.</div>
        </div>"""), unsafe_allow_html=True)
    else:
        orig_col, gc_col, pp_col = st.columns(3)
        with orig_col:
            st.subheader("Original MRI")
            if st.session_state.last_image:
                st.image(st.session_state.last_image, width="stretch")
        with gc_col:
            st.subheader("Grad-CAM (Jet)")
            if st.session_state.gradcam_image:
                st.pyplot(st.session_state.gradcam_image, width="stretch")
        with pp_col:
            st.subheader("Grad-CAM++ (Inferno)")
            if st.session_state.gradcam_pp_image:
                st.pyplot(st.session_state.gradcam_pp_image, width="stretch")

        st.markdown("**Heatmap influence:** Low (dark) ░░░▒▒▒████ High (bright)")

        agree = st.session_state.agreement_score
        if agree is not None:
            a_color = "#10b981" if agree >= 0.7 else "#f59e0b" if agree >= 0.5 else "#ef4444"
            a_label = "High Agreement" if agree >= 0.7 else "Moderate" if agree >= 0.5 else "Low Agreement"
            st.markdown(safe_html(f"""
            <div class="xai-card">
                <div class="xai-title">Explanation Agreement Score</div>
                <div style="display:flex;justify-content:space-between;align-items:center">
                    <div class="xai-text">
                        Correlation between Grad-CAM and Grad-CAM++ attention regions.<br>
                        High scores (&gt;0.70) indicate similar heatmaps, but do not establish that the explanation is correct.
                    </div>
                    <div style="text-align:right">
                        <div style="font-family:'Sora',sans-serif;font-size:1.5rem;font-weight:700;color:{a_color};letter-spacing:-.02em">{agree:.3f}</div>
                        <div style="font-size:.7rem;color:{a_color};font-weight:600">{a_label}</div>
                    </div>
                </div>
            </div>"""), unsafe_allow_html=True)

        st.markdown(safe_html("""
        <div class="xai-card">
            <div class="xai-title">About These Visualizations</div>
            <div class="xai-text">
                <b>Grad-CAM</b> computes weighted class activation maps from the last convolutional layer's gradients.
                Highlighted regions contributed more strongly to the model's output.<br><br>
                <b>Grad-CAM++</b> uses second-order gradients for sharper, more precise saliency localization, particularly
                useful when multiple regions of the same class appear in the scan.<br><br>
                ⚠️ These visualizations explain the <em>model's</em> decision, not the ground-truth anatomy.
                They do not constitute medically certified tumor localization.
            </div>
        </div>"""), unsafe_allow_html=True)

        if st.session_state.last_result:
            res = st.session_state.last_result
            c1, c2 = st.columns(2)
            c1.metric("Prediction", res["prediction"])
            c2.metric("Confidence", f"{res['confidence']:.2f}%")

        dl1, dl2 = st.columns(2)
        if st.session_state.gradcam_image:
            buf = io.BytesIO()
            st.session_state.gradcam_image.savefig(buf, format="png", bbox_inches="tight", dpi=160)
            dl1.download_button("⬇️ Grad-CAM PNG", buf.getvalue(), "gradcam.png", "image/png", key="gradcam_dl", width="stretch")
        if st.session_state.gradcam_pp_image:
            buf2 = io.BytesIO()
            st.session_state.gradcam_pp_image.savefig(buf2, format="png", bbox_inches="tight", dpi=160)
            dl2.download_button("⬇️ Grad-CAM++ PNG", buf2.getvalue(), "gradcam_pp.png", "image/png", key="gradcam_pp_dl", width="stretch")


# ══════════════════════════════════════════════════════════════════════════════
# XAI LAB
# ══════════════════════════════════════════════════════════════════════════════

elif nav == "🎯 XAI Lab":
    st.title("🎯 XAI Research Lab")
    st.caption("Deep-dive into explainability, uncertainty, and model behavior analysis")
    render_live_ticker()

    history = st.session_state.prediction_history

    if not history:
        st.markdown(safe_html("""
        <div class="empty-state">
            <div class="empty-state-icon">🎯</div>
            <div class="empty-state-title">No XAI data yet</div>
            <div class="empty-state-text">Run analyses with MC Dropout and dual XAI enabled to populate this lab.</div>
        </div>"""), unsafe_allow_html=True)
    else:
        st.subheader("Uncertainty Distribution")
        unc_data = [(i+1, h["uncertainty"]) for i, h in enumerate(history) if h.get("uncertainty") is not None]
        if len(unc_data) >= 2:
            fig = plot_uncertainty_history(history)
            if fig:
                st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                st.pyplot(fig, width="stretch")
                st.markdown('</div>', unsafe_allow_html=True)
                plt.close(fig)

        bands = {}
        for h in history:
            b = h.get("mc_band")
            if b:
                bands[b] = bands.get(b, 0) + 1
        if bands:
            st.write("")
            st.subheader("Reliability Band Distribution")
            bc1, bc2, bc3, bc4 = st.columns(4)
            band_cols = {"Very High Reliability": (bc1, "#10b981"),
                         "High Reliability":      (bc2, "#0ea5e9"),
                         "Moderate Reliability":  (bc3, "#f59e0b"),
                         "Low Reliability":       (bc4, "#ef4444")}
            for band, (col, color) in band_cols.items():
                cnt = bands.get(band, 0)
                with col:
                    st.markdown(safe_html(f"""
                    <div class="metric-card" style="border-color:{color}30">
                        <div class="metric-value" style="color:{color};font-size:1.7rem">{cnt}</div>
                        <div class="metric-label">{band.replace(' Reliability', '')}</div>
                    </div>"""), unsafe_allow_html=True)

        agree_hist = [(i+1, h["agreement_score"]) for i, h in enumerate(history) if h.get("agreement_score") is not None]
        if len(agree_hist) >= 2:
            st.write("")
            st.subheader("Explanation Agreement Score Over Time")
            fig2, ax2 = _dark_fig()
            xs, ys = zip(*agree_hist)
            ax2.plot(xs, ys, marker="D", lw=2, ms=4, color="#0ea5e9")
            ax2.fill_between(xs, ys, alpha=0.08, color="#0ea5e9")
            ax2.axhline(0.7, color="#10b981", lw=1, ls="--", alpha=0.6, label="High agreement threshold")
            ax2.axhline(0.5, color="#f59e0b", lw=1, ls="--", alpha=0.6, label="Moderate threshold")
            ax2.set_ylim(0, 1.05)
            ax2.set_xlabel("Analysis #", color="#a4b0c0", fontsize=9)
            ax2.set_ylabel("Agreement Score", color="#a4b0c0", fontsize=9)
            ax2.legend(fontsize=7, labelcolor="#a4b0c0", facecolor="#131a2c", edgecolor="#1f2937")
            ax2.grid(True, alpha=0.10, linestyle="--")
            fig2.tight_layout(pad=0.5)
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.pyplot(fig2, width="stretch")
            st.markdown('</div>', unsafe_allow_html=True)
            plt.close(fig2)

        st.write("")
        st.subheader("Per-Class Uncertainty Analysis")
        class_unc = {c: [] for c in CLASS_NAMES}
        for h in history:
            if h.get("uncertainty") is not None and h.get("prediction") in class_unc:
                class_unc[h["prediction"]].append(h["uncertainty"])
        rows = []
        for cls, vals in class_unc.items():
            if vals:
                rows.append({
                    "Class": cls,
                    "Count": len(vals),
                    "Mean σ": f"{np.mean(vals):.6f}",
                    "Min σ":  f"{np.min(vals):.6f}",
                    "Max σ":  f"{np.max(vals):.6f}",
                })
        if rows:
            import pandas as pd
            df = pd.DataFrame(rows)
            st.dataframe(df, width="stretch", hide_index=True)
        else:
            st.caption("No per-class uncertainty data yet.")

        with st.expander("📖 Methodology Reference"):
            st.markdown("""
            **MC Dropout Uncertainty Estimation**
            - Gal & Ghahramani (2016): Dropout as Bayesian approximation
            - Enables Dropout layers during inference for stochastic forward passes
            - **σ < 0.05**: Very High Reliability  ·  **σ < 0.12**: High Reliability
            - **σ < 0.22**: Moderate Reliability  ·  **σ ≥ 0.22**: Low Reliability

            **Grad-CAM (Selvaraju et al., 2017)**
            - Class Activation Mapping via gradient-weighted pooling over last conv layer
            - Weights = mean of gradient spatial dimensions

            **Grad-CAM++ (Chattopadhay et al., 2018)**
            - Second-order gradient weighting for sharper, more precise saliency
            - Better handles multiple occurrences of the same class in one image

            **Explanation Agreement Score**
            - Pearson correlation between flattened Grad-CAM and Grad-CAM++ heatmaps
            - Score ≥ 0.70: High agreement — consistent explanation
            - Score ≥ 0.50: Moderate agreement
            - Score < 0.50: Low agreement — treat XAI results with caution
            """)


# ══════════════════════════════════════════════════════════════════════════════
# SETTINGS
# ══════════════════════════════════════════════════════════════════════════════

elif nav == "⚙️ Settings":
    st.title("⚙️ System Settings")
    st.caption("Neural diagnostic engine configuration and session management")

    c1, c2 = st.columns(2)
    with c1:
        try: tv = metadata.version("torchvision")
        except Exception: tv = "—"
        st.markdown(safe_html(f"""
        <div class="info-card">
            <h3>Neural Engine</h3>
            <p><b>Architecture:</b> {model_name or 'Unavailable'}</p>
            <p><b>Status:</b> {'✅ Loaded' if not model_error else '❌ Failed'}</p>
            <p><b>Device:</b> {DEVICE}</p>
            <p><b>Input Resolution:</b> {IMG_SIZE} × {IMG_SIZE}</p>
            <p><b>Classes:</b> {', '.join(CLASS_NAMES)}</p>
            <p><b>Checkpoint:</b> {MODEL_PATH.name if MODEL_PATH.exists() else 'Not found'}</p>
            <p><b>MC Dropout Passes:</b> {MC_SAMPLES}</p>
        </div>"""), unsafe_allow_html=True)

    with c2:
        st.markdown(safe_html(f"""
        <div class="info-card">
            <h3>System Information</h3>
            <p><b>PyTorch:</b> {torch.__version__}</p>
            <p><b>Torchvision:</b> {tv}</p>
            <p><b>Streamlit:</b> {st.__version__}</p>
            <p><b>Python:</b> {platform.python_version()}</p>
            <p><b>CUDA Available:</b> {torch.cuda.is_available()}</p>
            <p><b>CUDA Device:</b> {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}</p>
            <p><b>XAI Methods:</b> Grad-CAM · Grad-CAM++</p>
            <p><b>Uncertainty:</b> MC Dropout ({MC_SAMPLES} passes)</p>
        </div>"""), unsafe_allow_html=True)

        show_tech = st.toggle("Show technical details", key="show_tech")
        if show_tech:
            with st.expander("Checkpoint and runtime", expanded=True):
                st.write(f"Checkpoint path: `{MODEL_PATH}`")
                st.write(f"Device: `{DEVICE}` · Input: `{IMG_SIZE}×{IMG_SIZE}`")
                st.write(f"Normalization mean: `{NORM_MEAN}`")
                st.write(f"Normalization std: `{NORM_STD}`")
                if model_error:
                    st.code(model_error)

        st.write("")
        st.markdown('<div class="disclaimer"><b>⚠️ Reset Session</b><br>Permanently clears all diagnostic reports and session data.</div>', unsafe_allow_html=True)
        if st.button("🧹 Reset Session", type="primary", width="content", key="settings_reset"):
            st.session_state.settings_confirm_reset = True
        if st.session_state.get("settings_confirm_reset", False):
            st.warning("Reset all session results?")
            rc1, rc2, _ = st.columns([1, 1, 4])
            if rc1.button("Confirm", key="settings_confirm_yes", width="stretch"):
                for fk in ("gradcam_image", "gradcam_pp_image"):
                    old = st.session_state.get(fk)
                    if old is not None:
                        plt.close(old)
                for k, v in defaults.items():
                    st.session_state[k] = v
                st.session_state.settings_confirm_reset = False
                st.success("Session cleared.")
                st.rerun()
            if rc2.button("Cancel", key="settings_confirm_no", width="stretch"):
                st.session_state.settings_confirm_reset = False


# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────

year       = datetime.now().year
build_time = datetime.now().strftime("%Y-%m-%d %H:%M UTC")
total_s    = st.session_state.live_predictions_count
avg_c_f    = st.session_state.live_avg_confidence
eng_ok     = model_error is None and MODEL_PATH.exists()

torch_ver = torch.__version__.split('+')[0]
st_ver    = st.__version__

st.markdown(safe_html(f"""
<div class="app-footer">
    <div class="footer-col footer-col-brand">
        <div class="footer-brand-row">
            <div class="footer-brand-logo">🧠</div>
            <div>
                <div class="footer-brand">NeuroLens AI</div>
                <div class="footer-tagline">Neurodiagnostic Intelligence Platform</div>
            </div>
        </div>
        <div class="footer-copy">
            © {year} NeuroLens AI<br>
            Build {build_time}
        </div>
    </div>

    <div class="footer-col footer-col-stack">
        <div class="footer-col-title">Tech Stack</div>
        <div class="footer-badges">
            <span class="badge-tech">PyTorch {torch_ver}</span>
            <span class="badge-tech">Streamlit {st_ver}</span>
            <span class="badge-tech">Grad-CAM</span>
            <span class="badge-tech">Grad-CAM++</span>
            <span class="badge-tech">MC Dropout</span>
            <span class="badge-tech">{model_name or 'CustomCNN'}</span>
        </div>
    </div>

    <div class="footer-col footer-col-stats">
        <div class="footer-col-title">Session</div>
        <div class="footer-stats">
            <div class="footer-stat">
                <div class="footer-stat-val">{total_s}</div>
                <div class="footer-stat-lbl">Scans</div>
            </div>
            <div class="footer-stat">
                <div class="footer-stat-val">{avg_c_f:.1f}%</div>
                <div class="footer-stat-lbl">Avg Conf</div>
            </div>
            <div class="footer-stat">
                <div class="footer-stat-val"><span class="footer-dot"></span> {'Online' if eng_ok else 'Offline'}</div>
                <div class="footer-stat-lbl">Status</div>
            </div>
        </div>
    </div>
</div>"""), unsafe_allow_html=True)

st.markdown(safe_html(f"""
<div class="copyright-line">
    <span class="copyright-brand">© {year} NeuroLens AI</span>
    <span class="copyright-sep">·</span>
    All Rights Reserved
    <br>
    <span class="copyright-dev-label">
        Developed by
        <span class="copyright-dev-name">MD. Atique Shahriar</span>
        <span class="copyright-dev-amp">&amp;</span>
        <span class="copyright-dev-name">Aronna Das</span>
    </span>
</div>"""), unsafe_allow_html=True)