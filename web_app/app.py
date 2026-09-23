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
# CSS
# ─────────────────────────────────────────────────────────────────────────────
STYLES = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root {
    --bg-primary:    #0b1120;
    --bg-secondary:  #111827;
    --bg-card:       rgba(17, 24, 39, 0.80);
    --bg-glass:      rgba(255, 255, 255, 0.03);
    --border-subtle: rgba(255, 255, 255, 0.07);
    --border-hover:  rgba(14, 165, 233, 0.45);
    --accent:        #0ea5e9;
    --accent-light:  #38bdf8;
    --accent-glow:   rgba(14, 165, 233, 0.15);
    --text-primary:  #f1f5f9;
    --text-secondary:#94a3b8;
    --text-muted:    #64748b;
    --success:       #10b981;
    --warning:       #f59e0b;
    --error:         #ef4444;
    --purple:        #8b5cf6;
    --cyan:          #06b6d4;
    --gradient-1:    linear-gradient(135deg, #0ea5e9 0%, #06b6d4 100%);
    --gradient-purple: linear-gradient(135deg, #8b5cf6 0%, #a78bfa 100%);
    --shadow-glow:   0 0 28px rgba(14, 165, 233, 0.18);
    --radius-lg:     20px;
    --radius-md:     14px;
    --radius-sm:     10px;
}

* { font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; box-sizing: border-box; }
.stApp { background: var(--bg-primary); color: var(--text-primary); }
header[data-testid="stHeader"] { background: transparent; }
#MainMenu, footer { visibility: hidden; }

/* ── SIDEBAR ── */
[data-testid="stSidebar"] {
    background: var(--bg-secondary) !important;
    border-right: 1px solid var(--border-subtle) !important;
    min-width: 300px !important; max-width: 300px !important; width: 300px !important;
}
[data-testid="stSidebar"] > div:first-child {
    width: 300px !important; padding: 1.2rem 0.75rem 1rem !important;
}
[data-testid="stSidebar"] ::-webkit-scrollbar { width: 4px; }
[data-testid="stSidebar"] ::-webkit-scrollbar-thumb { background: rgba(14,165,233,.25); border-radius: 10px; }

.sidebar-brand-title {
    font-size: 1.35rem; font-weight: 800;
    background: var(--gradient-1);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
    letter-spacing: -0.02em;
}
.sidebar-brand-subtitle { font-size: 0.65rem; color: var(--text-muted); font-weight: 500; letter-spacing: 0.07em; text-transform: uppercase; }
.sidebar-section-label {
    margin: 1rem 0 0.4rem; padding-left: 0.2rem;
    font-size: 0.65rem; font-weight: 800; letter-spacing: 0.14em;
    color: var(--text-muted); text-transform: uppercase;
}
[data-testid="stSidebar"] .stButton { margin: 0.18rem 0 !important; }
[data-testid="stSidebar"] .stButton > button {
    width: 100% !important; min-height: 44px !important;
    display: flex !important; align-items: center !important; justify-content: flex-start !important;
    padding: 0.6rem 0.85rem !important; border-radius: 12px !important;
    background: rgba(255,255,255,.025) !important; border: 1px solid rgba(255,255,255,.055) !important;
    color: var(--text-secondary) !important; font-size: 0.85rem !important; font-weight: 600 !important;
    transition: all 0.22s ease !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: rgba(14,165,233,.10) !important; border-color: rgba(14,165,233,.35) !important;
    color: var(--text-primary) !important; transform: translateX(4px) !important;
    box-shadow: 0 4px 14px rgba(14,165,233,.10) !important;
}
[data-testid="stSidebar"] .stButton > button p {
    line-height: 1 !important;
    margin: 0 !important;
    display: flex; align-items: center; gap: 0.35rem;
}
.nav-active {
    position: relative !important; border-radius: 12px !important;
    background: linear-gradient(135deg, rgba(14,165,233,.20), rgba(6,182,212,.07)) !important;
    border: 1px solid rgba(14,165,233,.42) !important;
    box-shadow: 0 0 22px rgba(14,165,233,.12) !important;
}
.nav-active::before {
    content: ""; position: absolute; left: 0; top: 20%; width: 3px; height: 60%;
    border-radius: 0 4px 4px 0; background: var(--accent-light); box-shadow: 0 0 10px var(--accent);
}

.sidebar-active-indicator {
    display: flex;
    align-items: center;
    justify-content: flex-start;
    gap: 0.6rem;
    height: 40px;
    padding: 0 0.85rem;
    margin: 0.2rem 0 0.5rem;
    border-radius: 10px;
    background: var(--accent-glow);
    border: 1px solid var(--border-hover);
    color: var(--accent-light);
    font-size: 0.75rem;
    font-weight: 700;
    line-height: 1;
    letter-spacing: 0.02em;
    box-shadow: var(--shadow-glow);
}
.sidebar-active-dot {
    flex: 0 0 8px;
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: var(--success);
    box-shadow: 0 0 8px var(--success);
    align-self: center;
}
.sidebar-active-text {
    display: inline-flex;
    align-items: center;
    height: 100%;
    line-height: 1;
    vertical-align: middle;
}

.sidebar-engine-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 0.45rem; margin-top: 0.75rem; }
.sidebar-mini-stat { padding: 0.6rem; border-radius: 9px; background: rgba(255,255,255,.025); border: 1px solid var(--border-subtle); }
.sidebar-mini-label { font-size: 0.6rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: .07em; }
.sidebar-mini-value { margin-top: 0.2rem; font-size: 0.78rem; font-weight: 700; color: var(--text-primary); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.sidebar-footer { margin-top: 1rem; padding: 0.75rem; border-radius: 12px; background: rgba(245,158,11,.06); border: 1px solid rgba(245,158,11,.18); font-size: .7rem; color: var(--text-secondary); text-align: center; line-height: 1.45; }

/* ── STICKY HEADER (redesigned) ── */
.sticky-header {
    position: fixed; top: 0.5rem; left: calc(300px + 0.4rem); right: 0.4rem; z-index: 9999;
    display: flex; align-items: center; gap: 0.85rem; padding: 0.7rem 1rem;
    border-radius: 16px;
    background: linear-gradient(135deg, rgba(11,17,32,.92), rgba(17,24,39,.82));
    backdrop-filter: blur(24px) saturate(160%); -webkit-backdrop-filter: blur(24px) saturate(160%);
    border: 1px solid rgba(56,189,248,.20);
    box-shadow: 0 8px 32px rgba(0,0,0,.4), 0 0 24px rgba(14,165,233,.06), inset 0 1px 0 rgba(255,255,255,.04);
    transition: all 0.3s ease;
}

.sticky-brand { display: flex; align-items: center; gap: 0.6rem; padding-right: 0.85rem; flex-shrink: 0; }
.sticky-brand-logo {
    position: relative;
    width: 42px; height: 42px; border-radius: 11px;
    background: var(--gradient-1);
    display: flex; align-items: center; justify-content: center;
    font-size: 1.35rem;
    box-shadow: 0 0 18px rgba(14,165,233,.5), inset 0 1px 0 rgba(255,255,255,.2);
}
.sticky-brand-pulse {
    position: absolute; top: -3px; right: -3px;
    width: 11px; height: 11px; border-radius: 50%;
    background: var(--success); border: 2px solid #0b1120;
    box-shadow: 0 0 8px var(--success);
    animation: brand-pulse 2s infinite;
}
@keyframes brand-pulse {
    0%, 100% { box-shadow: 0 0 8px var(--success); }
    50%      { box-shadow: 0 0 14px var(--success), 0 0 0 4px rgba(16,185,129,.22); }
}
.sticky-brand-text { display: flex; flex-direction: column; gap: 0.1rem; }
.sticky-brand-name {
    font-size: 1.05rem; font-weight: 800; color: var(--text-primary);
    letter-spacing: -0.015em; line-height: 1.1;
}
.sticky-brand-version {
    font-size: 0.6rem; color: var(--accent-light); font-weight: 700;
    letter-spacing: .12em; text-transform: uppercase; opacity: 0.85;
}

.sticky-divider { width: 1px; height: 26px; background: rgba(255,255,255,.08); flex-shrink: 0; }

.sticky-ticker {
    flex: 1; overflow: hidden; border-radius: 10px;
    border: 1px solid rgba(56,189,248,.16);
    background: linear-gradient(90deg, rgba(14,165,233,.06), rgba(6,182,212,.02));
    padding: 0.5rem 0; min-width: 0;
}
.sticky-ticker-inner {
    display: inline-flex; align-items: center; gap: 1.2rem;
    white-space: nowrap; padding-left: 100%;
    color: #7dd3fc; font: 600 0.78rem Inter, sans-serif;
    letter-spacing: .03em;
    animation: ticker-scroll 30s linear infinite;
}
.tick-badge {
    padding: 0.2rem 0.55rem; border-radius: 6px;
    background: rgba(16,185,129,.15); color: #34d399;
    border: 1px solid rgba(16,185,129,.3);
    font-size: 0.68rem; font-weight: 800; letter-spacing: 0.1em;
    margin-right: 0.3rem;
}
.tick-item { color: #94a3b8; }
.tick-item b { color: #e2e8f0; font-weight: 700; }
.tick-idle { color: #64748b; font-style: italic; }
@keyframes ticker-scroll { 0% { transform: translateX(0); } 100% { transform: translateX(-100%); } }

.sticky-status {
    display: inline-flex; align-items: center; gap: 0.4rem;
    padding: 0.35rem 0.75rem; border-radius: 8px;
    font-size: 0.68rem; font-weight: 800; letter-spacing: 0.08em;
    white-space: nowrap; flex-shrink: 0;
}
.status-ok {
    background: rgba(16,185,129,.10); color: #34d399;
    border: 1px solid rgba(16,185,129,.28);
}
.status-err {
    background: rgba(239,68,68,.10); color: #f87171;
    border: 1px solid rgba(239,68,68,.28);
}
.sticky-status-dot {
    width: 6px; height: 6px; border-radius: 50%;
    background: currentColor; box-shadow: 0 0 6px currentColor;
}

.sticky-page {
    display: inline-flex; align-items: center; gap: 0.5rem;
    padding: 0.4rem 0.85rem; border-radius: 9px;
    background: rgba(14,165,233,.10);
    border: 1px solid rgba(56,189,248,.30);
    font-size: 0.82rem; font-weight: 700; color: var(--text-primary);
    white-space: nowrap; flex-shrink: 0;
}
.sticky-page-dot {
    width: 7px; height: 7px; border-radius: 50%;
    background: var(--success); box-shadow: 0 0 8px var(--success);
    animation: dot-pulse 1.6s infinite;
}
@keyframes dot-pulse { 0%,100% { opacity: 1; } 50% { opacity: .35; } }

.sticky-clock {
    display: flex; flex-direction: column; align-items: flex-end;
    font-family: 'JetBrains Mono', monospace; line-height: 1.05;
    white-space: nowrap; padding-left: 0.2rem; flex-shrink: 0;
}
.sticky-clock-time { font-size: 0.95rem; font-weight: 700; color: var(--text-primary); letter-spacing: .03em; }
.sticky-clock-date { font-size: 0.6rem; color: var(--text-muted); letter-spacing: .08em; text-transform: uppercase; font-weight: 600; }

.main .block-container { padding-top: 6.5rem !important; padding-left: 1.5rem !important; padding-right: 1.5rem !important; }

/* ── CARDS ── */
.info-card {
    padding: 1.5rem; min-height: 150px; border-radius: var(--radius-lg);
    background: var(--bg-card); border: 1px solid var(--border-subtle);
    transition: all 0.25s ease;
}
.info-card:hover { border-color: var(--border-hover); box-shadow: var(--shadow-glow); transform: translateY(-2px); }
.info-card h3 { margin-top: 0; color: var(--text-primary); font-size: 1rem; font-weight: 700; }
.info-card p { color: var(--text-secondary); line-height: 1.6; font-size: 0.88rem; margin: 0.25rem 0; }

.metric-card { padding: 1.4rem; border-radius: var(--radius-lg); background: var(--bg-card); border: 1px solid var(--border-subtle); text-align: center; transition: all 0.25s ease; }
.metric-card:hover { border-color: var(--border-hover); box-shadow: var(--shadow-glow); transform: translateY(-2px); }
.metric-value { font-size: 2.2rem; font-weight: 800; color: var(--text-primary); margin: 0.4rem 0; }
.metric-label { font-size: 0.7rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: .1em; font-weight: 600; }
.metric-icon { font-size: 1.8rem; margin-bottom: 0.35rem; }

.hero { padding: 3rem 2.5rem; border-radius: 24px; text-align: center; margin-bottom: 1.75rem;
    background: linear-gradient(135deg, rgba(14,165,233,.10) 0%, rgba(6,182,212,.05) 50%, rgba(139,92,246,.08) 100%);
    border: 1px solid rgba(56,189,248,.22); position: relative; overflow: hidden; }
.hero::before { content: ""; position: absolute; top: -50%; left: -50%; width: 200%; height: 200%; background: radial-gradient(circle at 60% 40%, rgba(14,165,233,.08), transparent 60%); pointer-events: none; }
.hero h1 { font-size: 3rem; font-weight: 900; letter-spacing: -0.03em; margin-bottom: .5rem; background: var(--gradient-1); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; }
.hero p { color: var(--text-secondary); font-size: 1.05rem; line-height: 1.7; max-width: 680px; margin: 0 auto; }
.hero-badge { display: inline-flex; align-items: center; gap: .4rem; padding: .4rem .9rem; border-radius: 8px; background: var(--accent-glow); border: 1px solid var(--border-hover); font-size: .72rem; font-weight: 700; color: var(--accent-light); margin-bottom: .85rem; }

.diagnostic-panel { padding: 1.75rem; border-radius: 22px; background: var(--bg-card); border: 1px solid var(--border-hover); box-shadow: var(--shadow-glow); margin-top: 1.5rem; }
.diagnostic-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem; padding-bottom: .75rem; border-bottom: 1px solid var(--border-subtle); }
.diagnostic-title { font-size: .78rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: .1em; font-weight: 700; }
.diagnostic-prediction { font-size: 2rem; font-weight: 900; color: var(--text-primary); letter-spacing: -0.02em; }
.diagnostic-confidence { font-size: 1.2rem; font-weight: 800; color: var(--accent-light); margin-top: .2rem; }

.medical-badge { display: inline-flex; align-items: center; gap: .3rem; padding: .3rem .75rem; border-radius: 6px; font-size: .72rem; font-weight: 700; }
.badge-research { background: rgba(245,158,11,.12); color: var(--warning); border: 1px solid rgba(245,158,11,.3); }
.badge-high { background: rgba(16,185,129,.12); color: var(--success); border: 1px solid rgba(16,185,129,.3); }
.badge-moderate { background: rgba(245,158,11,.12); color: var(--warning); border: 1px solid rgba(245,158,11,.3); }
.badge-low { background: rgba(239,68,68,.12); color: var(--error); border: 1px solid rgba(239,68,68,.3); }

.uncertainty-card { padding: 1.25rem 1.5rem; border-radius: var(--radius-md); background: rgba(139,92,246,.08); border: 1px solid rgba(139,92,246,.25); margin-top: 1.2rem; }
.uncertainty-title { font-size: .72rem; color: #a78bfa; text-transform: uppercase; letter-spacing: .1em; font-weight: 700; margin-bottom: .6rem; }
.uncertainty-value { font-size: 1.5rem; font-weight: 800; }
.uncertainty-band { font-size: .8rem; font-weight: 600; margin-top: .15rem; }

.probability-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr)); gap: .75rem; margin-top: 1rem; }
.probability-card { padding: 1rem; border-radius: 12px; background: var(--bg-glass); border: 1px solid var(--border-subtle); text-align: center; transition: all .2s ease; }
.probability-card:hover { border-color: var(--border-hover); transform: translateY(-2px); }
.probability-value { font-size: 1.5rem; font-weight: 800; color: var(--accent-light); margin-bottom: .2rem; }
.probability-label { font-size: .72rem; color: var(--text-secondary); font-weight: 500; }
.probability-card.predicted { border-color: rgba(14,165,233,.5); background: rgba(14,165,233,.08); }
.probability-card.predicted .probability-value { color: var(--accent); }

.chart-container { padding: 1.25rem; border-radius: 16px; background: var(--bg-card); border: 1px solid var(--border-subtle); margin-top: 1rem; }

.thumbnail-card { display: flex; align-items: center; gap: .85rem; padding: .9rem 1rem; border-radius: 14px; background: var(--bg-card); border: 1px solid var(--border-subtle); transition: all .2s ease; margin-bottom: .6rem; }
.thumbnail-card:hover { border-color: var(--border-hover); box-shadow: 0 4px 16px rgba(14,165,233,.08); }
.thumbnail-img { width: 54px; height: 54px; border-radius: 10px; background: var(--bg-glass); display: flex; align-items: center; justify-content: center; font-size: 1.4rem; border: 1px solid var(--border-subtle); flex-shrink: 0; }
.thumbnail-info { flex: 1; min-width: 0; }
.thumbnail-title { font-size: .9rem; font-weight: 700; color: var(--text-primary); margin-bottom: .18rem; }
.thumbnail-meta { font-size: .72rem; color: var(--text-muted); }
.confidence-badge { display: inline-block; padding: .3rem .7rem; border-radius: 8px; font-size: .8rem; font-weight: 700; }
.confidence-high { background: rgba(16,185,129,.12); color: #10b981; border: 1px solid rgba(16,185,129,.3); }
.confidence-medium { background: rgba(245,158,11,.12); color: #f59e0b; border: 1px solid rgba(245,158,11,.3); }
.confidence-low { background: rgba(239,68,68,.12); color: #ef4444; border: 1px solid rgba(239,68,68,.3); }

.empty-state { padding: 3.5rem 2rem; border-radius: var(--radius-lg); background: var(--bg-card); border: 1px solid var(--border-subtle); text-align: center; }
.empty-state-icon { font-size: 3rem; margin-bottom: 1rem; opacity: .55; }
.empty-state-title { font-size: 1.1rem; font-weight: 700; color: var(--text-primary); margin-bottom: .5rem; }
.empty-state-text { font-size: .88rem; color: var(--text-secondary); }

.activity-feed { border-radius: 12px; border: 1px solid rgba(255,255,255,.06); background: rgba(11,17,32,.65); max-height: 340px; overflow-y: auto; font-family: 'JetBrains Mono', monospace; font-size: .76rem; }
.activity-row { display: flex; gap: .7rem; padding: .42rem .8rem; border-bottom: 1px solid rgba(255,255,255,.04); }
.activity-row:last-child { border-bottom: none; }
.activity-time { color: #64748b; min-width: 68px; }
.activity-msg { color: #cbd5e1; }
.activity-success .activity-msg { color: #6ee7b7; }
.activity-warn    .activity-msg { color: #fbbf24; }
.activity-error   .activity-msg { color: #fca5a5; }

.dist-card { padding: .9rem 1.1rem; border-radius: 12px; background: var(--bg-card); border: 1px solid var(--border-subtle); margin-bottom: .65rem; transition: all .2s ease; }
.dist-card:hover { border-color: var(--border-hover); }
.dist-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: .45rem; }
.dist-name { font-size: .88rem; font-weight: 700; color: var(--text-primary); }
.dist-count { font-size: .78rem; color: var(--accent-light); font-weight: 700; }
.dist-bar-bg { height: 6px; border-radius: 3px; background: rgba(255,255,255,.06); overflow: hidden; }
.dist-bar-fill { height: 100%; border-radius: 3px; background: var(--gradient-1); transition: width .5s cubic-bezier(.4,0,.2,1); }

.latest-card { padding: 1.25rem; border-radius: 16px; background: var(--bg-card); border: 1px solid var(--border-subtle); }
.latest-row { display: flex; justify-content: space-between; align-items: center; padding: .65rem 0; border-bottom: 1px solid var(--border-subtle); }
.latest-row:last-child { border-bottom: none; }
.latest-key { font-size: .82rem; color: var(--text-secondary); font-weight: 500; }
.latest-val { font-size: .82rem; color: var(--text-primary); font-weight: 700; }

[data-testid="stFileUploader"] { background: var(--bg-card); border-radius: 16px; padding: 1.25rem; border: 2px dashed var(--border-subtle); }
[data-testid="stFileUploader"]:hover { border-color: var(--accent); background: var(--accent-glow); }

[data-testid="stExpander"] { border-radius: 14px !important; border: 1px solid var(--border-subtle) !important; background: var(--bg-card) !important; }

.disclaimer { margin-top: 1.5rem; padding: 1rem 1.25rem; border-radius: 12px; background: rgba(245,158,11,.07); border: 1px solid rgba(245,158,11,.22); color: var(--text-secondary); font-size: .84rem; line-height: 1.65; }

.stButton > button { border-radius: 12px !important; font-weight: 700 !important; transition: all .2s ease !important; }
.stButton > button:hover { transform: translateY(-1px); }
[data-testid="stMetricValue"] { font-size: 1.65rem !important; font-weight: 800 !important; }
[data-testid="stMetricLabel"] { font-size: .72rem !important; color: var(--text-muted) !important; text-transform: uppercase !important; letter-spacing: .05em !important; }
.stProgress > div > div > div > div { background: var(--gradient-1) !important; border-radius: 4px !important; }
.stAlert { border-radius: 12px !important; }
hr { border-color: var(--border-subtle) !important; margin: 1.25rem 0 !important; }
.stImage { border-radius: 16px !important; overflow: hidden !important; border: 1px solid var(--border-subtle) !important; }

.xai-card { padding: 1.25rem 1.5rem; border-radius: var(--radius-md); background: rgba(14,165,233,.06); border: 1px solid rgba(14,165,233,.22); margin-top: 1.2rem; }
.xai-title { font-size: .72rem; color: var(--accent-light); text-transform: uppercase; letter-spacing: .1em; font-weight: 700; margin-bottom: .5rem; }
.xai-text { font-size: .85rem; color: var(--text-secondary); line-height: 1.65; }

/* ── APP FOOTER (redesigned) ── */
.app-footer {
    display: flex;
    justify-content: space-between;
    align-items: stretch;
    gap: 1.5rem;
    padding: 1.25rem 1.5rem;
    border-radius: 18px;
    background: linear-gradient(135deg, rgba(17,24,39,.85), rgba(11,17,32,.75));
    border: 1px solid var(--border-subtle);
    margin-top: 2rem;
    font-size: .8rem;
    color: var(--text-secondary);
    flex-wrap: wrap;
}

.footer-col { display: flex; flex-direction: column; gap: 0.5rem; }
.footer-col-brand { flex: 1; min-width: 220px; }
.footer-col-stack { flex: 1.2; min-width: 220px; }
.footer-col-stats { flex: 0 0 auto; min-width: 260px; }

.footer-brand-row { display: flex; align-items: center; gap: 0.65rem; }
.footer-brand-logo {
    width: 36px; height: 36px; border-radius: 9px;
    background: var(--gradient-1);
    display: flex; align-items: center; justify-content: center;
    font-size: 1.1rem;
    box-shadow: 0 0 14px rgba(14,165,233,.4);
}
.footer-brand {
    font-weight: 800; color: var(--text-primary);
    font-size: 0.95rem; letter-spacing: -0.01em;
}
.footer-tagline {
    font-size: 0.68rem; color: var(--text-muted);
    letter-spacing: 0.05em; font-weight: 500;
}
.footer-copy { font-size: 0.7rem; color: var(--text-muted); line-height: 1.5; margin-top: 0.3rem; }

.footer-col-title {
    font-size: 0.62rem; color: var(--text-muted);
    text-transform: uppercase; letter-spacing: 0.14em;
    font-weight: 800; margin-bottom: 0.2rem;
}
.footer-badges { display: flex; flex-wrap: wrap; gap: 0.4rem; }
.badge-tech {
    padding: 0.28rem 0.65rem; border-radius: 7px;
    background: rgba(56,189,248,.08);
    border: 1px solid rgba(56,189,248,.18);
    color: #7dd3fc; font-size: 0.68rem; font-weight: 700;
    letter-spacing: 0.02em;
    transition: all .2s ease;
}
.badge-tech:hover {
    background: rgba(56,189,248,.15);
    border-color: rgba(56,189,248,.35);
    transform: translateY(-1px);
}

.footer-stats { display: flex; gap: 1.2rem; margin-top: 0.2rem; }
.footer-stat { display: flex; flex-direction: column; gap: 0.15rem; }
.footer-stat-val {
    font-size: 0.95rem; font-weight: 800; color: var(--text-primary);
    display: inline-flex; align-items: center; gap: 0.35rem;
    letter-spacing: -0.01em;
}
.footer-stat-lbl {
    font-size: 0.6rem; color: var(--text-muted);
    text-transform: uppercase; letter-spacing: 0.1em;
    font-weight: 700;
}
.footer-dot {
    width: 7px; height: 7px; border-radius: 50%;
    background: var(--success); box-shadow: 0 0 8px var(--success);
    display: inline-block;
}

/* ── RESPONSIVE ── */
@media (max-width: 768px) {
    .sticky-header { left: .3rem !important; right: .3rem !important; padding: .55rem .75rem !important; gap: 0.5rem !important; }
    .sticky-ticker, .sticky-brand-version, .sticky-clock { display: none; }
    .sticky-status { display: none; }
    .sticky-brand-logo { width: 36px; height: 36px; font-size: 1.1rem; }
    .sticky-brand-name { font-size: 0.92rem; }
    [data-testid="stSidebar"] { min-width: 100% !important; max-width: 100% !important; }
    .main .block-container { padding-top: 5rem !important; }
    .hero h1 { font-size: 1.85rem !important; }
    .probability-grid { grid-template-columns: 1fr 1fr !important; }
    .app-footer { flex-direction: column; gap: 1rem; }
    .footer-col-brand, .footer-col-stack, .footer-col-stats { flex: 1 1 100% !important; min-width: 0 !important; }
}
@media (min-width: 769px) and (max-width: 1024px) {
    .sticky-header { left: calc(260px + .3rem) !important; }
    [data-testid="stSidebar"] { min-width: 260px !important; max-width: 260px !important; width: 260px !important; }
    [data-testid="stSidebar"] > div:first-child { width: 260px !important; }
}
@media (prefers-reduced-motion: reduce) {
    *, *::before, *::after { animation-duration: .01ms !important; transition-duration: .01ms !important; }
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
    fig.patch.set_facecolor("#111827")
    ax.set_facecolor("#111827")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#374151")
    ax.tick_params(colors="#94a3b8", labelsize=8)
    return fig, ax


def plot_confidence_trend(history):
    if len(history) < 2:
        return None
    fig, ax = _dark_fig()
    confs = [h["confidence"] for h in history]
    xs    = list(range(1, len(history) + 1))
    ax.plot(xs, confs, marker="o", lw=2, ms=4, color="#6366f1")
    ax.fill_between(xs, confs, alpha=0.12, color="#6366f1")
    ax.set_ylim(0, 105)
    ax.set_xlabel("Analysis #", color="#94a3b8", fontsize=9)
    ax.set_ylabel("Confidence %", color="#94a3b8", fontsize=9)
    ax.grid(True, alpha=0.12, linestyle="--")
    fig.tight_layout(pad=0.5)
    return fig


def plot_latency_trend(history):
    if len(history) < 2:
        return None
    fig, ax = _dark_fig()
    lats = [h.get("latency_ms", 0) for h in history]
    ax.plot(range(1, len(history) + 1), lats, marker="o", lw=2, color="#06b6d4")
    ax.set_xlabel("Analysis #", color="#94a3b8", fontsize=9)
    ax.set_ylabel("Latency ms", color="#94a3b8", fontsize=9)
    ax.grid(True, alpha=0.12, linestyle="--")
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
    ax.set_xlabel("Analysis #", color="#94a3b8", fontsize=9)
    ax.set_ylabel("MC Uncertainty σ", color="#94a3b8", fontsize=9)
    ax.grid(True, alpha=0.12, linestyle="--")
    ax.legend(fontsize=7, labelcolor="#94a3b8", facecolor="#111827", edgecolor="#374151")
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
# NEURAL ANIMATION
# ─────────────────────────────────────────────────────────────────────────────

def render_neural_animation():
    _render_html("""
    <div id="nw" style="position:relative;width:100%;height:160px;overflow:hidden;border-radius:20px;
    background:radial-gradient(circle at 50% 50%,rgba(14,165,233,.12),rgba(11,17,32,.15) 55%,rgba(11,17,32,.7));
    border:1px solid rgba(56,189,248,.2)">
    <canvas id="nc" style="width:100%;height:100%;display:block"></canvas>
    <div style="position:absolute;left:18px;bottom:12px;display:flex;align-items:center;gap:8px;
    font:700 11px Inter,sans-serif;letter-spacing:.14em;color:#7dd3fc;text-shadow:0 0 12px rgba(56,189,248,.8)">
    <div style="width:8px;height:8px;border-radius:50%;background:#22d3ee;
    box-shadow:0 0 0 0 rgba(34,211,238,.6);animation:p 1.5s infinite"></div>LIVE NEURAL NETWORK</div></div>
    <style>@keyframes p{70%{box-shadow:0 0 0 10px rgba(34,211,238,0)}100%{box-shadow:0 0 0 0 rgba(34,211,238,0)}}</style>
    <script>
    const c=document.getElementById('nc'),ctx=c.getContext('2d');let ps=[];
    function rs(){const r=c.getBoundingClientRect(),d=window.devicePixelRatio||1;
    c.width=r.width*d;c.height=r.height*d;ctx.setTransform(d,0,0,d,0,0);
    ps=Array.from({length:Math.max(22,Math.floor(r.width/30))},()=>({x:Math.random()*r.width,y:Math.random()*r.height,vx:(Math.random()-.5)*.32,vy:(Math.random()-.5)*.32,r:1.4+Math.random()*2}))}
    function dr(){const r=c.getBoundingClientRect();ctx.clearRect(0,0,r.width,r.height);
    for(const p of ps){p.x+=p.vx;p.y+=p.vy;if(p.x<0||p.x>r.width)p.vx*=-1;if(p.y<0||p.y>r.height)p.vy*=-1}
    for(let i=0;i<ps.length;i++)for(let j=i+1;j<ps.length;j++){const a=ps[i],b=ps[j],d=Math.hypot(a.x-b.x,a.y-b.y);
    if(d<110){ctx.strokeStyle=`rgba(56,189,248,${(1-d/110)*.24})`;ctx.lineWidth=.65;ctx.beginPath();ctx.moveTo(a.x,a.y);ctx.lineTo(b.x,b.y);ctx.stroke()}}
    for(const p of ps){ctx.beginPath();ctx.arc(p.x,p.y,p.r,0,Math.PI*2);ctx.fillStyle='#38bdf8';ctx.shadowBlur=10;ctx.shadowColor='#0ea5e9';ctx.fill()}
    ctx.shadowBlur=0;requestAnimationFrame(dr)}
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
    @keyframes mq{{0%{{transform:translateX(100%);}}100%{{transform:translateX(-100%);}}}}
    .tw{{overflow:hidden;border-radius:11px;border:1px solid rgba(56,189,248,.28);
    background:linear-gradient(90deg,rgba(14,165,233,.10),rgba(6,182,212,.03));padding:.55rem 0;
    box-shadow:0 0 18px rgba(14,165,233,.07)}}
    .tt{{display:inline-block;white-space:nowrap;padding-left:100%;
    animation:mq 26s linear infinite;color:#7dd3fc;font:600 .82rem Inter,sans-serif;letter-spacing:.04em}}
    .lp{{display:inline-block;width:7px;height:7px;border-radius:50%;background:#22d3ee;margin-right:9px;
    box-shadow:0 0 0 0 rgba(34,211,238,.7);animation:tp 1.4s infinite;vertical-align:middle}}
    @keyframes tp{{70%{{box-shadow:0 0 0 9px rgba(34,211,238,0);}}100%{{box-shadow:0 0 0 0 rgba(34,211,238,0);}}}}
    </style>
    <div class="tw"><div class="tt"><span class="lp"></span>LIVE ·
    {text} · Throughput: {st.session_state.live_throughput:.2f}/min ·
    Last Conf: {st.session_state.live_last_confidence:.1f}% ·
    Avg Conf: {st.session_state.live_avg_confidence:.1f}%{unc_str}
    </div></div>""", height=44)


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
    now    = datetime.now()
    clock_time = now.strftime("%H:%M:%S")
    clock_date = now.strftime("%b %d, %Y").upper()

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
                <span class="tick-badge">⚡ LIVE</span>
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

        <div class="sticky-clock">
            <div class="sticky-clock-time">{clock_time}</div>
            <div class="sticky-clock-date">{clock_date}</div>
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
            f'<div class="lp-row"><div class="lp-label">{n} <span style="color:#8b5cf6">±{mc_std.get(n,0):.1f}%</span></div>'
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
        unc_html = f'<div style="margin-top:.75rem;padding:.6rem .85rem;border-radius:9px;background:rgba(139,92,246,.08);border:1px solid rgba(139,92,246,.22)"><span style="font-size:.7rem;color:#a78bfa;text-transform:uppercase;letter-spacing:.1em;font-weight:700">MC Uncertainty</span><span style="float:right;font-weight:800;color:{color}">σ={uval:.4f} · {band}</span></div>'

    _render_html(f"""
    <style>
    .live-prob{{padding:1.2rem 1.5rem;border-radius:18px;border:1px solid rgba(56,189,248,.32);
    background:linear-gradient(135deg,rgba(14,165,233,.09),rgba(6,182,212,.03));font-family:Inter,sans-serif;color:#e2e8f0}}
    .live-prob-head{{display:flex;justify-content:space-between;align-items:center;margin-bottom:.85rem;
    padding-bottom:.55rem;border-bottom:1px solid rgba(255,255,255,.06)}}
    .live-prob-title{{font-size:.72rem;color:#7dd3fc;text-transform:uppercase;letter-spacing:.12em;font-weight:700}}
    .live-prob-pred{{font-size:1rem;font-weight:800;color:#f8fafc}}
    .lp-row{{display:flex;align-items:center;gap:.7rem;margin:.4rem 0}}
    .lp-label{{min-width:130px;font-size:.8rem;color:#cbd5e1;font-weight:600}}
    .lp-track{{flex:1;height:9px;border-radius:999px;background:rgba(255,255,255,.06);overflow:hidden;position:relative}}
    .lp-fill{{height:100%;width:0%;background:linear-gradient(90deg,#0ea5e9,#06b6d4);border-radius:999px;
    box-shadow:0 0 10px rgba(14,165,233,.5);transition:width 1.4s cubic-bezier(.2,.8,.2,1)}}
    .lp-mc{{background:linear-gradient(90deg,#8b5cf6,#a78bfa) !important;box-shadow:0 0 10px rgba(139,92,246,.5) !important}}
    .lp-val{{min-width:55px;text-align:right;font-weight:700;color:#38bdf8;font-size:.82rem}}
    .lp-section{{font-size:.68rem;color:#64748b;text-transform:uppercase;letter-spacing:.1em;font-weight:700;margin:.85rem 0 .3rem}}
    </style>
    <div class="live-prob">
        <div class="live-prob-head">
            <span class="live-prob-title">⚡ Live Probability Stream</span>
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
        mc_badge = f"<br><div style='margin-top:.4rem;font-size:.68rem;color:#a78bfa'>σ={mc['uncertainty']:.4f} · {mc['band']}</div>"

    st.markdown(safe_html(f"""
    <div style="margin-top:.5rem;padding:.8rem;border-radius:12px;background:rgba(255,255,255,.025);border:1px solid rgba(255,255,255,.06)">
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
        if c1.button("Confirm", key="sb_clear_yes"):
            clear_prediction_history()
            st.session_state.confirm_clear = False
            st.rerun()
        if c2.button("Cancel", key="sb_clear_no"):
            st.session_state.confirm_clear = False

    if st.button("🔄 Reset Session", key="sb_reset", width="stretch"):
        st.session_state.confirm_reset = True
    if st.session_state.get("confirm_reset", False):
        st.caption("Reset entire session?")
        c1, c2 = st.columns(2)
        if c1.button("Confirm", key="sb_reset_yes"):
            for fig_key in ("gradcam_image", "gradcam_pp_image"):
                old = st.session_state.get(fig_key)
                if old is not None:
                    plt.close(old)
            for k, v in defaults.items():
                st.session_state[k] = v
            st.rerun()
        if c2.button("Cancel", key="sb_reset_no"):
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
    render_neural_animation()
    render_live_ticker()

    st.markdown(safe_html(dedent("""
    <div class="hero">
        <div class="hero-badge">🧠 AI-Powered Brain MRI Analysis · Research Prototype</div>
        <h1>NeuroLens AI</h1>
        <p>Deep Learning · Dual XAI (Grad-CAM + Grad-CAM++) · MC Dropout Uncertainty Estimation · Neuroimaging</p>
    </div>""")), unsafe_allow_html=True)

    if st.button("🔬 Analyze MRI Scan", type="primary", key="home_cta"):
        st.session_state.nav = "🔬 MRI Analysis"
        st.rerun()
    st.caption("Research prototype for education and research. Model predictions are not medical diagnoses.")

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
            st.markdown(f'<div class="info-card"><div style="font-size:2rem;margin-bottom:.65rem">{icon}</div><h3>{title}</h3><p>{desc}</p></div>', unsafe_allow_html=True)

    st.write("")
    st.subheader("How NeuroLens Works")
    steps = [
        ("1", "Upload MRI", "JPG, PNG, WEBP"),
        ("2", "Preprocess", "Resize → Normalize"),
        ("3", "Inference", "Deep CNN forward pass"),
        ("4", "MC Dropout", "20 stochastic passes → uncertainty σ"),
        ("5", "Dual XAI", "Grad-CAM + Grad-CAM++"),
        ("6", "Agreement", "Explanation Agreement Score"),
        ("7", "Report", "Download TXT/PNG"),
    ]
    step_cols = st.columns(len(steps))
    for col, (num, title, desc) in zip(step_cols, steps):
        with col:
            st.markdown(safe_html(f"""
            <div style="text-align:center;padding:.85rem .5rem;border-radius:14px;background:rgba(14,165,233,.06);border:1px solid rgba(14,165,233,.18)">
                <div style="font-size:1.4rem;font-weight:900;color:var(--accent-light)">{num}</div>
                <div style="font-size:.78rem;font-weight:700;color:var(--text-primary);margin:.2rem 0">{title}</div>
                <div style="font-size:.68rem;color:var(--text-muted)">{desc}</div>
            </div>"""), unsafe_allow_html=True)

    st.write("")
    if model_error:
        st.markdown(safe_html(f"""
        <div class="info-card" style="border-color:rgba(239,68,68,.3);background:rgba(239,68,68,.06)">
            <h3 style="color:var(--error)">⚠️ Neural Engine Unavailable</h3>
            <p>Expected: <code>{MODEL_PATH}</code></p>
        </div>"""), unsafe_allow_html=True)
        with st.expander("Technical Details"):
            st.code(model_error)
    else:
        st.markdown(safe_html(f"""
        <div class="info-card" style="border-color:rgba(16,185,129,.3);background:rgba(16,185,129,.06)">
            <h3 style="color:var(--success)">✅ Neural Engine Ready</h3>
            <p>Architecture: <b>{model_name}</b> · Device: <b>{DEVICE}</b></p>
            <p>Classes: <b style="color:var(--accent-light)">{', '.join(class_names)}</b></p>
            <p>MC Dropout Passes: <b>{MC_SAMPLES}</b> · Dual XAI: <b>Grad-CAM + Grad-CAM++</b></p>
        </div>"""), unsafe_allow_html=True)

    if st.session_state.last_result:
        st.write("")
        st.subheader("⚡ Last Analysis — Live Probability Stream")
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
        uploaded_file = st.file_uploader(
            "Upload Brain MRI Scan (JPG · PNG · WEBP)",
            type=["jpg", "jpeg", "png", "webp"],
            key="mri_uploader",
        )

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

                    if st.button("🔍 Run AI Analysis", type="primary", width="stretch", key="analyze_btn"):
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
                    <span class="diagnostic-title">🩺 AI Classification Result</span>
                    <span class="medical-badge badge-research">RESEARCH PROTOTYPE</span>
                </div>
                <div class="diagnostic-prediction">{result['prediction']}</div>
                <div class="diagnostic-confidence">{conf_val:.2f}% model confidence</div>
                <div style="margin-top:.75rem">
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
                    <div class="uncertainty-title">🎯 MC Dropout Uncertainty Estimation ({MC_SAMPLES} passes)</div>
                    <div style="display:flex;justify-content:space-between;align-items:center">
                        <div>
                            <div class="uncertainty-value" style="color:{color}">σ = {unc:.4f}</div>
                            <div class="uncertainty-band" style="color:{color}">{band}</div>
                        </div>
                        <div style="text-align:right;font-size:.82rem;color:var(--text-secondary)">
                            MC Confidence: <b style="color:var(--accent-light)">{mc_res['confidence']:.2f}%</b><br>
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
                    <div class="xai-title">🔗 Explanation Agreement Score</div>
                    <div style="display:flex;justify-content:space-between;align-items:center">
                        <div class="xai-text">
                            Pearson correlation between Grad-CAM and Grad-CAM++ heatmaps.<br>
                            A high score (&gt;0.70) means both methods highlight similar regions. Similarity alone does not verify that an explanation is correct.
                        </div>
                        <div style="text-align:right;min-width:90px">
                            <div style="font-size:1.5rem;font-weight:800;color:{a_color}">{agree:.3f}</div>
                            <div style="font-size:.72rem;color:{a_color};font-weight:700">{a_label}</div>
                        </div>
                    </div>
                </div>"""), unsafe_allow_html=True)

            st.warning("⚠️ NeuroLens AI is a research prototype for educational use. Model outputs are not medical diagnoses and must not replace evaluation by a qualified healthcare professional.")

            st.write("")
            st.markdown("### 📊 Probability Distribution")
            prob_html = '<div class="probability-grid">' + "".join(
                f'<div class="probability-card {"predicted" if label == result["prediction"] else ""}">'
                f'<div class="probability-value">{prob:.1f}%</div>'
                f'<div class="probability-label">{label}</div>'
                f'{"<div style=\"font-size:.65rem;color:var(--accent);margin-top:.2rem;font-weight:700\">▲ TOP</div>" if label == result["prediction"] else ""}'
                f'</div>'
                for label, prob in result["probabilities"].items()
            ) + '</div>'
            st.markdown(prob_html, unsafe_allow_html=True)

            st.write("")
            st.markdown("### ⚡ Live Probability Stream")
            render_live_probability_animation(result, mc_result=mc_res)

            if st.session_state.gradcam_image is not None or st.session_state.gradcam_pp_image is not None:
                st.write("")
                st.markdown("### 🔥 Dual XAI Visualization")
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
                    <div class="xai-title">🔥 Why This Prediction?</div>
                    <div class="xai-text">
                        Highlighted regions represent image areas that contributed more strongly to the model's classification.
                        Grad-CAM uses weighted class activations; Grad-CAM++ uses second-order gradients for sharper localization.
                        Neither method provides medically certified tumor localization. Use for research interpretation only.
                    </div>
                </div>"""), unsafe_allow_html=True)

                cam_col, pp_col2 = st.columns(2)
                if st.session_state.gradcam_image:
                    buf = io.BytesIO()
                    st.session_state.gradcam_image.savefig(buf, format="png", bbox_inches="tight", dpi=160)
                    cam_col.download_button("⬇️ Grad-CAM PNG", buf.getvalue(), "gradcam.png", "image/png", key="dl_gc")
                if st.session_state.gradcam_pp_image:
                    buf2 = io.BytesIO()
                    st.session_state.gradcam_pp_image.savefig(buf2, format="png", bbox_inches="tight", dpi=160)
                    pp_col2.download_button("⬇️ Grad-CAM++ PNG", buf2.getvalue(), "gradcam_pp.png", "image/png", key="dl_pp")

            lines = [
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
            ]
            st.write("")
            st.download_button(
                "📄 Download Analysis Report (TXT)",
                "\n".join(lines),
                "neurolens_report.txt",
                "text/plain",
                key="dl_report",
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
            <div style="margin:.75rem 0;padding:.85rem 1.25rem;border-radius:12px;background:rgba(14,165,233,.07);border:1px solid rgba(14,165,233,.22);display:flex;justify-content:space-between;align-items:center">
                <span style="font-size:.78rem;color:var(--accent-light);font-weight:700;text-transform:uppercase;letter-spacing:.08em">Average Explanation Agreement Score</span>
                <span style="font-size:1.35rem;font-weight:800;color:var(--text-primary)">{avg_agree:.3f}</span>
            </div>"""), unsafe_allow_html=True)

        st.write("")
        col_left, col_right = st.columns([2, 1])

        with col_left:
            st.subheader("📊 Prediction Distribution")
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
            st.subheader("📈 Model Confidence Trend")
            cf = plot_confidence_trend(history)
            if cf:
                st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                st.pyplot(cf, width="stretch")
                st.markdown('</div>', unsafe_allow_html=True)
                plt.close(cf)
            else:
                st.caption("Need ≥2 analyses.")

            st.subheader("⏱️ Inference Latency Trend")
            lf = plot_latency_trend(history)
            if lf:
                st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                st.pyplot(lf, width="stretch")
                st.markdown('</div>', unsafe_allow_html=True)
                plt.close(lf)
            else:
                st.caption("Need ≥2 analyses.")

            if unc_data:
                st.subheader("🎯 MC Dropout Uncertainty Trend")
                uf = plot_uncertainty_history(history)
                if uf:
                    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                    st.pyplot(uf, width="stretch")
                    st.markdown('</div>', unsafe_allow_html=True)
                    plt.close(uf)

        with col_right:
            st.subheader("📡 Live Activity Feed")
            render_activity_feed()

            st.write("")
            st.subheader("🧠 Latest Result")
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
        if st.button("🗑️ Clear History", key="clear_hist_btn"):
            st.session_state.confirm_clear_hist = True
        if st.session_state.get("confirm_clear_hist", False):
            st.warning("Clear all records from this session?")
            cc1, cc2 = st.columns(2)
            if cc1.button("Confirm", key="confirm_hist"):
                clear_prediction_history()
                st.session_state.confirm_clear_hist = False
                st.rerun()
            if cc2.button("Cancel", key="cancel_hist"):
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
                <div class="xai-title">🔗 Explanation Agreement Score</div>
                <div style="display:flex;justify-content:space-between;align-items:center">
                    <div class="xai-text">
                        Correlation between Grad-CAM and Grad-CAM++ attention regions.<br>
                        High scores (&gt;0.70) indicate similar heatmaps, but do not establish that the explanation is correct.
                    </div>
                    <div style="text-align:right">
                        <div style="font-size:1.6rem;font-weight:900;color:{a_color}">{agree:.3f}</div>
                        <div style="font-size:.72rem;color:{a_color};font-weight:700">{a_label}</div>
                    </div>
                </div>
            </div>"""), unsafe_allow_html=True)

        st.markdown(safe_html("""
        <div class="xai-card">
            <div class="xai-title">🧠 About These Visualizations</div>
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
            dl1.download_button("⬇️ Grad-CAM PNG", buf.getvalue(), "gradcam.png", "image/png", key="gradcam_dl")
        if st.session_state.gradcam_pp_image:
            buf2 = io.BytesIO()
            st.session_state.gradcam_pp_image.savefig(buf2, format="png", bbox_inches="tight", dpi=160)
            dl2.download_button("⬇️ Grad-CAM++ PNG", buf2.getvalue(), "gradcam_pp.png", "image/png", key="gradcam_pp_dl")


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
        st.subheader("📊 Uncertainty Distribution")
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
            st.subheader("🏷️ Reliability Band Distribution")
            bc1, bc2, bc3, bc4 = st.columns(4)
            band_cols = {"Very High Reliability": (bc1, "#10b981"),
                         "High Reliability":      (bc2, "#22d3ee"),
                         "Moderate Reliability":  (bc3, "#f59e0b"),
                         "Low Reliability":       (bc4, "#ef4444")}
            for band, (col, color) in band_cols.items():
                cnt = bands.get(band, 0)
                with col:
                    st.markdown(safe_html(f"""
                    <div class="metric-card" style="border-color:{color}30">
                        <div class="metric-value" style="color:{color};font-size:1.8rem">{cnt}</div>
                        <div class="metric-label">{band.replace(' Reliability', '')}</div>
                    </div>"""), unsafe_allow_html=True)

        agree_hist = [(i+1, h["agreement_score"]) for i, h in enumerate(history) if h.get("agreement_score") is not None]
        if len(agree_hist) >= 2:
            st.write("")
            st.subheader("🔗 Explanation Agreement Score Over Time")
            fig2, ax2 = _dark_fig()
            xs, ys = zip(*agree_hist)
            ax2.plot(xs, ys, marker="D", lw=2, ms=4, color="#0ea5e9")
            ax2.fill_between(xs, ys, alpha=0.08, color="#0ea5e9")
            ax2.axhline(0.7, color="#10b981", lw=1, ls="--", alpha=0.6, label="High agreement threshold")
            ax2.axhline(0.5, color="#f59e0b", lw=1, ls="--", alpha=0.6, label="Moderate threshold")
            ax2.set_ylim(0, 1.05)
            ax2.set_xlabel("Analysis #", color="#94a3b8", fontsize=9)
            ax2.set_ylabel("Agreement Score", color="#94a3b8", fontsize=9)
            ax2.legend(fontsize=7, labelcolor="#94a3b8", facecolor="#111827", edgecolor="#374151")
            ax2.grid(True, alpha=0.12, linestyle="--")
            fig2.tight_layout(pad=0.5)
            st.markdown('<div class="chart-container">', unsafe_allow_html=True)
            st.pyplot(fig2, width="stretch")
            st.markdown('</div>', unsafe_allow_html=True)
            plt.close(fig2)

        st.write("")
        st.subheader("📋 Per-Class Uncertainty Analysis")
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
            <h3>📋 Neural Engine</h3>
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
            <h3>🖥️ System Information</h3>
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
        if st.button("🧹 Reset Session", type="primary", width="stretch", key="settings_reset"):
            st.session_state.settings_confirm_reset = True
        if st.session_state.get("settings_confirm_reset", False):
            st.warning("Reset all session results?")
            rc1, rc2 = st.columns(2)
            if rc1.button("Confirm", key="settings_confirm_yes"):
                for fk in ("gradcam_image", "gradcam_pp_image"):
                    old = st.session_state.get(fk)
                    if old is not None:
                        plt.close(old)
                for k, v in defaults.items():
                    st.session_state[k] = v
                st.session_state.settings_confirm_reset = False
                st.success("Session cleared.")
                st.rerun()
            if rc2.button("Cancel", key="settings_confirm_no"):
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
            © {year} NeuroLens Research<br>
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

st.caption("NeuroLens AI is a research prototype for educational and research purposes. Predictions are not medical diagnoses and must not replace evaluation by a qualified healthcare professional.")