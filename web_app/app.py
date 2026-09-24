# =============================================================================
# NeuroLens AI — Neurodiagnostic Intelligence Platform
# Production SaaS Edition v3.1 — Enhanced UI + Premium Sidebar v5.0
# =============================================================================

import warnings
import time
import io
import copy
import hashlib
import platform
from importlib import metadata
from datetime import datetime
from pathlib import Path

import numpy as np
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


# ─────────────────────────────────────────────────────────────────────────────
# SVG ICON LIBRARY — replaces all emoji
# ─────────────────────────────────────────────────────────────────────────────
class Icons:
    """Inline SVG icons. All accept size and color args."""

    @staticmethod
    def _wrap(path_d: str, size=18, color="currentColor", viewbox="0 0 24 24", extra="") -> str:
        return (
            f'<svg width="{size}" height="{size}" viewBox="{viewbox}" fill="none" '
            f'stroke="{color}" stroke-width="1.65" stroke-linecap="round" '
            f'stroke-linejoin="round" style="display:inline-block;vertical-align:middle;flex-shrink:0" {extra}>'
            f'{path_d}</svg>'
        )

    @staticmethod
    def brain(size=18, color="currentColor"):
        return Icons._wrap(
            '<path d="M9.5 2A2.5 2.5 0 0 1 12 4.5v15a2.5 2.5 0 0 1-4.96-.46 2.5 2.5 0 0 1-1.7-3.3A3 3 0 0 1 3 13V9a3 3 0 0 1 2.34-2.92 2.5 2.5 0 0 1 4.16-4.08Z"/>'
            '<path d="M14.5 2A2.5 2.5 0 0 0 12 4.5v15a2.5 2.5 0 0 0 4.96-.46 2.5 2.5 0 0 0 1.7-3.3A3 3 0 0 0 21 13V9a3 3 0 0 0-2.34-2.92 2.5 2.5 0 0 0-4.16-4.08Z"/>',
            size, color
        )

    @staticmethod
    def home(size=18, color="currentColor"):
        return Icons._wrap(
            '<path d="m3 9 9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z"/>'
            '<polyline points="9 22 9 12 15 12 15 22"/>',
            size, color
        )

    @staticmethod
    def microscope(size=18, color="currentColor"):
        return Icons._wrap(
            '<path d="M6 18h8"/><path d="M3 22h18"/>'
            '<path d="M14 22a7 7 0 1 0 0-14h-1"/>'
            '<path d="M9 14h2"/><path d="M9 12a2 2 0 0 1-2-2V6h6v4a2 2 0 0 1-2 2Z"/>'
            '<path d="M12 6V3a1 1 0 0 0-1-1H9a1 1 0 0 0-1 1v3"/>',
            size, color
        )

    @staticmethod
    def chart(size=18, color="currentColor"):
        return Icons._wrap(
            '<line x1="18" y1="20" x2="18" y2="10"/>'
            '<line x1="12" y1="20" x2="12" y2="4"/>'
            '<line x1="6" y1="20" x2="6" y2="14"/>'
            '<line x1="2" y1="20" x2="22" y2="20"/>',
            size, color
        )

    @staticmethod
    def history(size=18, color="currentColor"):
        return Icons._wrap(
            '<path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8"/>'
            '<path d="M3 3v5h5"/>'
            '<path d="M12 7v5l4 2"/>',
            size, color
        )

    @staticmethod
    def heatmap(size=18, color="currentColor"):
        return Icons._wrap(
            '<circle cx="12" cy="12" r="3"/>'
            '<path d="M12 2v3"/><path d="M12 19v3"/>'
            '<path d="m4.22 4.22 2.12 2.12"/><path d="m17.66 17.66 2.12 2.12"/>'
            '<path d="M2 12h3"/><path d="M19 12h3"/>'
            '<path d="m4.22 19.78 2.12-2.12"/><path d="m17.66 6.34 2.12-2.12"/>',
            size, color
        )

    @staticmethod
    def lab(size=18, color="currentColor"):
        return Icons._wrap(
            '<path d="M14.5 2v17.5c0 1.4-1.1 2.5-2.5 2.5h0c-1.4 0-2.5-1.1-2.5-2.5V2"/>'
            '<path d="M8.5 2h7"/>'
            '<path d="M14.5 16h-5"/>'
            '<path d="m8.5 13 5.5 3"/>',
            size, color
        )

    @staticmethod
    def settings(size=18, color="currentColor"):
        return Icons._wrap(
            '<path d="M12.22 2h-.44a2 2 0 0 0-2 2v.18a2 2 0 0 1-1 1.73l-.43.25a2 2 0 0 1-2 0l-.15-.08a2 2 0 0 0-2.73.73l-.22.38a2 2 0 0 0 .73 2.73l.15.1a2 2 0 0 1 1 1.72v.51a2 2 0 0 1-1 1.74l-.15.09a2 2 0 0 0-.73 2.73l.22.38a2 2 0 0 0 2.73.73l.15-.08a2 2 0 0 1 2 0l.43.25a2 2 0 0 1 1 1.73V20a2 2 0 0 0 2 2h.44a2 2 0 0 0 2-2v-.18a2 2 0 0 1 1-1.73l.43-.25a2 2 0 0 1 2 0l.15.08a2 2 0 0 0 2.73-.73l.22-.39a2 2 0 0 0-.73-2.73l-.15-.08a2 2 0 0 1-1-1.74v-.5a2 2 0 0 1 1-1.74l.15-.09a2 2 0 0 0 .73-2.73l-.22-.38a2 2 0 0 0-2.73-.73l-.15.08a2 2 0 0 1-2 0l-.43-.25a2 2 0 0 1-1-1.73V4a2 2 0 0 0-2-2z"/>'
            '<circle cx="12" cy="12" r="3"/>',
            size, color
        )

    @staticmethod
    def upload(size=24, color="currentColor"):
        return Icons._wrap(
            '<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>'
            '<polyline points="17 8 12 3 7 8"/>'
            '<line x1="12" y1="3" x2="12" y2="15"/>',
            size, color
        )

    @staticmethod
    def check_circle(size=16, color="#10b981"):
        return Icons._wrap(
            '<path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/>'
            '<polyline points="22 4 12 14.01 9 11.01"/>',
            size, color
        )

    @staticmethod
    def alert_triangle(size=16, color="#f59e0b"):
        return Icons._wrap(
            '<path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"/>'
            '<line x1="12" y1="9" x2="12" y2="13"/>'
            '<line x1="12" y1="17" x2="12.01" y2="17"/>',
            size, color
        )

    @staticmethod
    def x_circle(size=16, color="#ef4444"):
        return Icons._wrap(
            '<circle cx="12" cy="12" r="10"/>'
            '<line x1="15" y1="9" x2="9" y2="15"/>'
            '<line x1="9" y1="9" x2="15" y2="15"/>',
            size, color
        )

    @staticmethod
    def download(size=16, color="currentColor"):
        return Icons._wrap(
            '<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>'
            '<polyline points="7 10 12 15 17 10"/>'
            '<line x1="12" y1="15" x2="12" y2="3"/>',
            size, color
        )

    @staticmethod
    def activity(size=16, color="currentColor"):
        return Icons._wrap(
            '<polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>',
            size, color
        )

    @staticmethod
    def cpu(size=16, color="currentColor"):
        return Icons._wrap(
            '<rect x="4" y="4" width="16" height="16" rx="2" ry="2"/>'
            '<rect x="9" y="9" width="6" height="6"/>'
            '<line x1="9" y1="1" x2="9" y2="4"/><line x1="15" y1="1" x2="15" y2="4"/>'
            '<line x1="9" y1="20" x2="9" y2="23"/><line x1="15" y1="20" x2="15" y2="23"/>'
            '<line x1="20" y1="9" x2="23" y2="9"/><line x1="20" y1="14" x2="23" y2="14"/>'
            '<line x1="1" y1="9" x2="4" y2="9"/><line x1="1" y1="14" x2="4" y2="14"/>',
            size, color
        )

    @staticmethod
    def layers(size=16, color="currentColor"):
        return Icons._wrap(
            '<polygon points="12 2 2 7 12 12 22 7 12 2"/>'
            '<polyline points="2 17 12 22 22 17"/>'
            '<polyline points="2 12 12 17 22 12"/>',
            size, color
        )

    @staticmethod
    def zap(size=16, color="currentColor"):
        return Icons._wrap(
            '<polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/>',
            size, color
        )

    @staticmethod
    def target(size=16, color="currentColor"):
        return Icons._wrap(
            '<circle cx="12" cy="12" r="10"/>'
            '<circle cx="12" cy="12" r="6"/>'
            '<circle cx="12" cy="12" r="2"/>',
            size, color
        )

    @staticmethod
    def shield(size=16, color="currentColor"):
        return Icons._wrap(
            '<path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>',
            size, color
        )

    @staticmethod
    def eye(size=16, color="currentColor"):
        return Icons._wrap(
            '<path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/>'
            '<circle cx="12" cy="12" r="3"/>',
            size, color
        )

    @staticmethod
    def info(size=16, color="currentColor"):
        return Icons._wrap(
            '<circle cx="12" cy="12" r="10"/>'
            '<line x1="12" y1="16" x2="12" y2="12"/>'
            '<line x1="12" y1="8" x2="12.01" y2="8"/>',
            size, color
        )

    @staticmethod
    def scan(size=20, color="currentColor"):
        return Icons._wrap(
            '<path d="M3 7V5a2 2 0 0 1 2-2h2"/>'
            '<path d="M17 3h2a2 2 0 0 1 2 2v2"/>'
            '<path d="M21 17v2a2 2 0 0 1-2 2h-2"/>'
            '<path d="M7 21H5a2 2 0 0 1-2-2v-2"/>'
            '<line x1="7" y1="12" x2="17" y2="12"/>',
            size, color
        )

    @staticmethod
    def pulse(size=16, color="currentColor"):
        return Icons._wrap(
            '<polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>',
            size, color
        )

    @staticmethod
    def flask(size=16, color="currentColor"):
        return Icons._wrap(
            '<path d="M9 3h6l1 6-4 8H8L4 9l1-6Z"/>'
            '<path d="M6 9h12"/>',
            size, color
        )


# ─────────────────────────────────────────────────────────────────────────────
# HTML HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def safe_html(html: str) -> str:
    return " ".join(line.strip() for line in html.strip().splitlines() if line.strip())


def _render_html(content: str, height: int = 200, scrolling: bool = False):
    """Render arbitrary HTML via components.html."""
    components.html(content, height=height, scrolling=scrolling)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NeuroLens AI",
    page_icon=":material/neurology:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# STREAMLIT WIDTH COMPATIBILITY HELPER
# ─────────────────────────────────────────────────────────────────────────────
def _st_width_arg(container: bool) -> dict:
    """
    Return the correct width kwarg dict for the current Streamlit version.
    - Streamlit >= 1.37: uses new width="stretch" / "content" API.
    - Streamlit  < 1.37: falls back to legacy use_container_width bool.
    """
    try:
        parts = st.__version__.split(".")[:2]
        major, minor = int(parts[0]), int(parts[1])
        if (major, minor) >= (1, 37):
            return {"width": "stretch" if container else "content"}
    except Exception:
        pass
    return {"use_container_width": bool(container)}


def _stretch() -> dict:
    """Container-full width for buttons, images, dataframes, etc."""
    return _st_width_arg(True)


def _content() -> dict:
    """Content-fit width (default behaviour)."""
    return _st_width_arg(False)


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
NORM_MEAN   = [0.485, 0.456, 0.406]
NORM_STD    = [0.229, 0.224, 0.225]
IMG_SIZE    = 224
MC_SAMPLES  = 20
MAX_HISTORY = 200
CLASS_NAMES = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]

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
# CSS — Design System v4.0 (Enhanced)
# ─────────────────────────────────────────────────────────────────────────────
STYLES = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Sora:wght@500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

:root {
    --bg:         #060b18;
    --surface:    #0b1120;
    --surface-2:  #0f172a;
    --surface-3:  #162032;
    --surface-4:  #1d2d44;
    --line:       rgba(255,255,255,0.055);
    --line-md:    rgba(255,255,255,0.09);
    --line-hi:    rgba(255,255,255,0.14);
    --accent:        #0891b2;
    --accent-hi:     #22d3ee;
    --accent-soft:   rgba(8,145,178,0.09);
    --accent-line:   rgba(8,145,178,0.28);
    --accent-glow:   rgba(8,145,178,0.18);
    --success:    #059669;
    --success-hi: #34d399;
    --warning:    #d97706;
    --warning-hi: #fbbf24;
    --danger:     #dc2626;
    --danger-hi:  #f87171;
    --violet:     #7c3aed;
    --violet-hi:  #a78bfa;
    --text-1: #e2e8f0;
    --text-2: #94a3b8;
    --text-3: #4b5869;
    --text-4: #2d3a4a;
    --font-display: 'Sora', 'Inter', sans-serif;
    --font-body:    'Inter', -apple-system, sans-serif;
    --font-mono:    'JetBrains Mono', ui-monospace, monospace;
    --r-xs:   5px;
    --r-sm:   8px;
    --r-md:   12px;
    --r-lg:   18px;
    --r-xl:   24px;
    --r-pill: 999px;
    --sb-w:   270px;
    --hd-h:   54px;
    --sh-1:  0 1px 3px rgba(0,0,0,.3),  0 1px 2px rgba(0,0,0,.2);
    --sh-2:  0 4px 16px rgba(0,0,0,.4), 0 2px 6px rgba(0,0,0,.25);
    --sh-3:  0 16px 40px rgba(0,0,0,.5);
    --sh-accent: 0 0 0 3px rgba(8,145,178,0.25);
    --sh-inset:  inset 0 1px 0 rgba(255,255,255,0.05);
}

*, *::before, *::after { box-sizing: border-box; }
body { -webkit-font-smoothing: antialiased; -moz-osx-font-smoothing: grayscale; }
.stApp { background: var(--bg); color: var(--text-1); }
header[data-testid="stHeader"] { background: transparent; }
#MainMenu, footer { visibility: hidden; }
h1, h2, h3, h4 { font-family: var(--font-display); letter-spacing: -0.025em; }
code, pre, .mono { font-family: var(--font-mono); }

[data-testid="stSidebar"] {
    background: var(--surface) !important;
    border-right: 1px solid var(--line) !important;
    min-width: var(--sb-w) !important;
    max-width: var(--sb-w) !important;
    width: var(--sb-w) !important;
}
[data-testid="stSidebar"] > div:first-child {
    width: var(--sb-w) !important;
    padding: 0 !important;
}
@keyframes pulse-dot { 0%,100%{opacity:1;} 50%{opacity:.35;} }

.sticky-header {
    position: fixed; top: 0.45rem;
    left: calc(var(--sb-w) + 0.5rem); right: 0.5rem;
    z-index: 9999;
    display: flex; align-items: center; gap: 0.5rem;
    padding: 0 0.75rem;
    height: var(--hd-h);
    border-radius: var(--r-md);
    background: rgba(11,17,32,0.9);
    backdrop-filter: blur(20px) saturate(180%);
    -webkit-backdrop-filter: blur(20px) saturate(180%);
    border: 1px solid var(--line-md);
    box-shadow: var(--sh-2), var(--sh-inset);
}
.hd-brand { display: flex; align-items: center; gap: 0.5rem; flex-shrink: 0; }
.hd-brand-icon {
    width: 30px; height: 30px; border-radius: 8px;
    background: var(--accent-soft); border: 1px solid var(--accent-line);
    display: flex; align-items: center; justify-content: center;
    position: relative; flex-shrink: 0;
}
.hd-brand-dot {
    position: absolute; top: -2px; right: -2px;
    width: 6px; height: 6px; border-radius: 50%;
    background: var(--success-hi); border: 1.5px solid var(--surface);
    animation: pulse-dot 2.5s ease-in-out infinite;
}
.hd-brand-name {
    font-family: var(--font-display); font-size: 0.85rem;
    font-weight: 700; color: var(--text-1); letter-spacing: -0.015em;
}
.hd-divider { width: 1px; height: 18px; background: var(--line); flex-shrink: 0; }
.hd-ticker {
    flex: 1; min-width: 0;
    height: 28px; border-radius: var(--r-xs);
    background: rgba(255,255,255,0.02);
    border: 1px solid var(--line);
    display: flex; align-items: center;
    padding: 0 0.6rem; overflow: hidden;
}
.hd-ticker-inner {
    display: flex; align-items: center; gap: 0.6rem;
    white-space: nowrap; overflow: hidden;
    font-family: var(--font-mono); font-size: 0.66rem;
    color: var(--text-2); font-weight: 500;
    mask-image: linear-gradient(to right, #000 0%, #000 90%, transparent 100%);
    -webkit-mask-image: linear-gradient(to right, #000 0%, #000 90%, transparent 100%);
}
.hd-live-badge {
    padding: 0.1rem 0.42rem; border-radius: 4px;
    background: rgba(5,150,105,0.12); color: var(--success-hi);
    border: 1px solid rgba(5,150,105,0.28);
    font-size: 0.56rem; font-weight: 700; letter-spacing: 0.1em;
    font-family: var(--font-body); flex-shrink: 0;
}
.hd-tick { color: var(--text-2); font-size: 0.66rem; }
.hd-tick b { color: var(--text-1); font-weight: 600; }
.hd-tick-idle { color: var(--text-3); font-style: italic; }
.hd-status {
    display: inline-flex; align-items: center; gap: 0.32rem;
    padding: 0.2rem 0.5rem; border-radius: var(--r-xs);
    font-size: 0.58rem; font-weight: 700; letter-spacing: 0.08em;
    white-space: nowrap; flex-shrink: 0; height: 24px;
}
.hd-status-ok {
    background: rgba(5,150,105,0.09); color: var(--success-hi);
    border: 1px solid rgba(5,150,105,0.22);
}
.hd-status-err {
    background: rgba(220,38,38,0.09); color: var(--danger-hi);
    border: 1px solid rgba(220,38,38,0.22);
}
.hd-status-dot { width: 5px; height: 5px; border-radius: 50%; background: currentColor; }
.hd-page {
    display: inline-flex; align-items: center; gap: 0.32rem;
    padding: 0.2rem 0.5rem; border-radius: var(--r-xs);
    background: transparent; border: 1px solid var(--line);
    font-size: 0.68rem; font-weight: 600; color: var(--text-1);
    white-space: nowrap; flex-shrink: 0; height: 24px;
}
.hd-page-dot { width: 5px; height: 5px; border-radius: 50%; background: var(--accent); }

.main .block-container {
    padding-top: 5rem !important;
    padding-left: 1.5rem !important;
    padding-right: 1.5rem !important;
    max-width: 1380px;
}

.card {
    background: var(--surface-2);
    border: 1px solid var(--line);
    border-radius: var(--r-lg);
    box-shadow: var(--sh-1), var(--sh-inset);
    transition: border-color 150ms ease, box-shadow 150ms ease;
}
.card:hover { border-color: var(--line-md); }
.card-sm {
    background: var(--surface-2); border: 1px solid var(--line);
    border-radius: var(--r-md); box-shadow: var(--sh-1);
}

.hero {
    padding: 3rem 2rem 2.5rem;
    border-radius: var(--r-xl);
    text-align: center;
    margin-bottom: 1.5rem;
    background: var(--surface-2);
    border: 1px solid var(--line);
    position: relative;
    overflow: hidden;
    box-shadow: var(--sh-1), var(--sh-inset);
}
.hero::before {
    content: "";
    position: absolute; inset: 0;
    background:
      radial-gradient(ellipse at 15% 0%, rgba(8,145,178,0.12), transparent 50%),
      radial-gradient(ellipse at 85% 100%, rgba(8,145,178,0.07), transparent 50%);
    pointer-events: none;
}
.hero-eyebrow {
    display: inline-flex; align-items: center; gap: 0.4rem;
    padding: 0.28rem 0.7rem; border-radius: var(--r-pill);
    background: var(--accent-soft); border: 1px solid var(--accent-line);
    font-size: 0.63rem; font-weight: 600; color: var(--accent-hi);
    margin-bottom: 1rem; position: relative; letter-spacing: 0.02em;
}
.hero h1 {
    font-family: var(--font-display);
    font-size: 2.75rem; font-weight: 800;
    letter-spacing: -0.04em; line-height: 1.05;
    color: var(--text-1); margin: 0 0 0.75rem; position: relative;
}
.hero .hero-sub,
.hero .hero-sub p,
.hero p.hero-sub,
[data-testid="stMarkdownContainer"] .hero-sub,
[data-testid="stMarkdownContainer"] .hero-sub p,
[data-testid="stMarkdownContainer"] p.hero-sub {
    color: var(--text-2) !important;
    font-size: 0.9rem !important;
    line-height: 1.65 !important;
    max-width: 580px !important;
    margin-left: auto !important;
    margin-right: auto !important;
    padding: 0 1rem !important;
    position: relative !important;
    font-weight: 400 !important;
    text-align: center !important;
    display: block !important;
    width: 100% !important;
}
.hero-divider {
    width: 48px; height: 2px; border-radius: 1px;
    background: linear-gradient(to right, var(--accent), transparent);
    margin: 1rem auto 0; position: relative;
}

.metric-card {
    padding: 1.1rem 1.2rem;
    border-radius: var(--r-md);
    background: var(--surface-2);
    border: 1px solid var(--line);
    box-shadow: var(--sh-1), var(--sh-inset);
    transition: border-color 150ms ease;
}
.metric-card:hover { border-color: var(--line-md); }
.metric-icon-wrap {
    width: 30px; height: 30px; border-radius: 8px;
    background: var(--accent-soft); border: 1px solid var(--accent-line);
    display: flex; align-items: center; justify-content: center;
    margin-bottom: 0.6rem;
}
.metric-value {
    font-family: var(--font-display);
    font-size: 1.65rem; font-weight: 700;
    color: var(--text-1); letter-spacing: -0.03em;
    line-height: 1.1; margin-bottom: 0.2rem;
    font-variant-numeric: tabular-nums;
}
.metric-label {
    font-size: 0.6rem; color: var(--text-3);
    font-weight: 600; letter-spacing: 0.08em;
}

.info-card {
    padding: 1.25rem 1.4rem;
    border-radius: var(--r-lg);
    background: var(--surface-2);
    border: 1px solid var(--line);
    box-shadow: var(--sh-1), var(--sh-inset);
    transition: border-color 150ms ease, transform 150ms ease;
    height: 100%;
}
.info-card:hover { border-color: var(--line-md); transform: translateY(-1px); }
.info-card-icon {
    width: 36px; height: 36px; border-radius: 9px;
    background: var(--accent-soft); border: 1px solid var(--accent-line);
    display: flex; align-items: center; justify-content: center;
    margin-bottom: 0.75rem;
}
.info-card h3 {
    margin: 0 0 0.5rem; color: var(--text-1);
    font-size: 0.88rem; font-weight: 600;
    letter-spacing: -0.01em;
}
.info-card p {
    color: var(--text-2); line-height: 1.6; font-size: 0.78rem; margin: 0;
}

.diagnostic-panel {
    padding: 2rem 2.25rem;
    border-radius: var(--r-xl);
    background: var(--surface-2);
    border: 1px solid var(--accent-line);
    box-shadow: var(--sh-2), var(--sh-inset), 0 0 0 1px rgba(8,145,178,0.05);
    margin-top: 1.5rem;
    position: relative;
    overflow: hidden;
}
.diagnostic-panel::before {
    content: "";
    position: absolute;
    left: 0; top: 0; bottom: 0; width: 3px;
    background: linear-gradient(to bottom, var(--accent), transparent 80%);
}
.diag-header {
    display: flex; justify-content: space-between; align-items: flex-start;
    margin-bottom: 1rem; padding-bottom: 0.85rem;
    border-bottom: 1px solid var(--line);
}
.diag-label {
    font-size: 0.6rem; color: var(--text-3); font-weight: 700;
    text-transform: uppercase; letter-spacing: 0.12em;
}
.diag-prediction {
    font-family: var(--font-display);
    font-size: 2.5rem; font-weight: 800;
    color: var(--text-1); letter-spacing: -0.04em; line-height: 1.05;
    margin: 0.25rem 0;
}
.diag-confidence {
    font-family: var(--font-mono);
    font-size: 0.9rem; font-weight: 500;
    color: var(--accent-hi); letter-spacing: 0;
}

.badge {
    display: inline-flex; align-items: center; gap: 0.3rem;
    padding: 0.22rem 0.55rem; border-radius: 6px;
    font-size: 0.62rem; font-weight: 600; letter-spacing: 0.02em;
}
.badge-research { background: rgba(217,119,6,0.09); color: var(--warning-hi); border: 1px solid rgba(217,119,6,0.22); }
.badge-high     { background: rgba(5,150,105,0.09); color: var(--success-hi); border: 1px solid rgba(5,150,105,0.22); }
.badge-moderate { background: rgba(217,119,6,0.09); color: var(--warning-hi); border: 1px solid rgba(217,119,6,0.22); }
.badge-low      { background: rgba(220,38,38,0.09); color: var(--danger-hi);  border: 1px solid rgba(220,38,38,0.22); }

.uncertainty-card {
    padding: 1.1rem 1.4rem;
    border-radius: var(--r-md);
    background: rgba(124,58,237,0.04);
    border: 1px solid rgba(124,58,237,0.18);
    border-left: 3px solid var(--violet);
    margin-top: 1rem;
}
.unc-title {
    font-size: 0.6rem; color: var(--violet-hi);
    text-transform: uppercase; letter-spacing: 0.12em;
    font-weight: 700; margin-bottom: 0.5rem;
}
.unc-value {
    font-family: var(--font-display);
    font-size: 1.5rem; font-weight: 700;
    letter-spacing: -0.02em; font-variant-numeric: tabular-nums;
}
.unc-band { font-size: 0.78rem; font-weight: 600; margin-top: 0.12rem; }

.prob-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(148px, 1fr));
    gap: 0.65rem; margin-top: 1rem;
}
.prob-card {
    padding: 0.9rem 1rem;
    border-radius: var(--r-md);
    background: var(--surface-2);
    border: 1px solid var(--line);
    position: relative; overflow: hidden;
    transition: border-color 150ms ease;
}
.prob-card:hover { border-color: var(--line-md); }
.prob-card.is-top {
    background: rgba(8,145,178,0.06);
    border-color: var(--accent-line);
}
.prob-card.is-top::before {
    content: "";
    position: absolute; left: 0; top: 0; bottom: 0; width: 3px;
    background: var(--accent);
}
.prob-value {
    font-family: var(--font-display); font-size: 1.4rem; font-weight: 700;
    color: var(--text-1); letter-spacing: -0.025em;
    font-variant-numeric: tabular-nums; margin-bottom: 0.18rem;
}
.prob-card.is-top .prob-value { color: var(--accent-hi); }
.prob-label { font-size: 0.7rem; color: var(--text-2); font-weight: 500; }
.prob-top-tag {
    font-size: 0.55rem; color: var(--accent); font-weight: 700;
    letter-spacing: 0.1em; margin-top: 0.2rem;
}

.xai-card {
    padding: 1.1rem 1.4rem;
    border-radius: var(--r-md);
    background: rgba(8,145,178,0.03);
    border: 1px solid rgba(8,145,178,0.14);
    margin-top: 1rem;
}
.xai-title {
    font-size: 0.6rem; color: var(--accent-hi);
    text-transform: uppercase; letter-spacing: 0.12em;
    font-weight: 700; margin-bottom: 0.5rem;
}
.xai-text { font-size: 0.8rem; color: var(--text-2); line-height: 1.65; }

.thumb-card {
    display: flex; align-items: center; gap: 0.85rem;
    padding: 0.8rem 1rem;
    border-radius: var(--r-md);
    background: var(--surface-2);
    border: 1px solid var(--line);
    margin-bottom: 0.4rem;
    transition: border-color 150ms ease;
}
.thumb-card:hover { border-color: var(--line-md); }
.thumb-icon {
    width: 42px; height: 42px; border-radius: var(--r-sm);
    background: var(--accent-soft); border: 1px solid var(--accent-line);
    display: flex; align-items: center; justify-content: center; flex-shrink: 0;
}
.thumb-info { flex: 1; min-width: 0; }
.thumb-title {
    font-family: var(--font-display); font-size: 0.88rem;
    font-weight: 600; color: var(--text-1); letter-spacing: -0.01em;
}
.thumb-meta {
    font-family: var(--font-mono); font-size: 0.63rem; color: var(--text-3);
    margin-top: 0.15rem;
}
.conf-badge {
    display: inline-block; padding: 0.22rem 0.55rem;
    border-radius: 6px; font-family: var(--font-mono);
    font-size: 0.7rem; font-weight: 600;
}
.conf-hi  { background: rgba(5,150,105,0.09); color: var(--success-hi); border: 1px solid rgba(5,150,105,0.22); }
.conf-mid { background: rgba(217,119,6,0.09);  color: var(--warning-hi); border: 1px solid rgba(217,119,6,0.22); }
.conf-lo  { background: rgba(220,38,38,0.09);  color: var(--danger-hi);  border: 1px solid rgba(220,38,38,0.22); }

.dist-card {
    padding: 0.75rem 0.9rem;
    border-radius: var(--r-md);
    background: var(--surface-2); border: 1px solid var(--line);
    margin-bottom: 0.4rem;
    transition: border-color 150ms ease;
}
.dist-card:hover { border-color: var(--line-md); }
.dist-row { display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.4rem; }
.dist-name { font-size: 0.8rem; font-weight: 600; color: var(--text-1); }
.dist-count { font-family: var(--font-mono); font-size: 0.7rem; color: var(--accent-hi); }
.dist-track { height: 4px; border-radius: 2px; background: rgba(255,255,255,0.05); overflow: hidden; }
.dist-fill { height: 100%; border-radius: 2px; background: var(--accent); transition: width 600ms ease; }

.upload-hero {
    padding: 2rem 2rem 1.75rem;
    border-radius: var(--r-xl);
    background: var(--surface-2);
    border: 1px solid var(--line);
    overflow: hidden; text-align: center;
    position: relative;
    box-shadow: var(--sh-1), var(--sh-inset);
    margin: 0.75rem 0 0.65rem;
}
.upload-hero::before {
    content: "";
    position: absolute; top: -100px; left: 50%; transform: translateX(-50%);
    width: 500px; height: 500px;
    background: radial-gradient(circle, rgba(8,145,178,0.1), transparent 65%);
    pointer-events: none;
}
.upload-icon-wrap {
    position: relative;
    width: 56px; height: 56px; border-radius: 14px;
    background: linear-gradient(135deg, rgba(8,145,178,0.2), rgba(8,145,178,0.04));
    border: 1px solid var(--accent-line);
    display: flex; align-items: center; justify-content: center;
    margin: 0 auto 0.85rem;
    box-shadow: 0 8px 24px rgba(8,145,178,0.15);
}
.upload-title {
    position: relative; font-family: var(--font-display);
    font-size: 1.1rem; font-weight: 700; color: var(--text-1);
    letter-spacing: -0.02em; margin-bottom: 0.3rem;
}
.upload-sub {
    position: relative; font-size: 0.78rem; color: var(--text-2);
    margin-bottom: 0.85rem; line-height: 1.5;
}
.upload-formats {
    position: relative; display: flex; justify-content: center; gap: 0.35rem; flex-wrap: wrap;
}
.fmt-badge {
    padding: 0.2rem 0.5rem; border-radius: 5px;
    background: rgba(8,145,178,0.07); border: 1px solid rgba(8,145,178,0.2);
    font-family: var(--font-mono); font-size: 0.58rem; font-weight: 600;
    color: var(--accent-hi); letter-spacing: 0.06em;
}
.fmt-badge-muted {
    background: rgba(255,255,255,0.02); border-color: var(--line); color: var(--text-3);
}
.upload-note {
    position: relative; display: flex; align-items: center; justify-content: center;
    gap: 0.4rem; margin-top: 0.7rem;
    font-size: 0.66rem; color: var(--text-3);
}
.upload-note-dot {
    width: 5px; height: 5px; border-radius: 50%;
    background: var(--success-hi); display: inline-block;
    box-shadow: 0 0 8px rgba(52,211,153,0.55);
}

.act-feed {
    border-radius: var(--r-md); border: 1px solid var(--line);
    background: rgba(0,0,0,0.25); max-height: 300px; overflow-y: auto;
    font-family: var(--font-mono); font-size: 0.67rem;
}
.act-row {
    display: flex; gap: 0.6rem; padding: 0.32rem 0.65rem;
    border-bottom: 1px solid rgba(255,255,255,0.03);
}
.act-row:last-child { border-bottom: none; }
.act-time { color: var(--text-3); min-width: 56px; flex-shrink: 0; }
.act-msg { color: var(--text-2); }
.act-success .act-msg { color: #6ee7b7; }
.act-warn    .act-msg { color: #fbbf24; }
.act-error   .act-msg { color: #fca5a5; }

.latest-card {
    border-radius: var(--r-md); border: 1px solid var(--line);
    background: var(--surface-2); overflow: hidden;
}
.latest-row {
    display: flex; justify-content: space-between; align-items: center;
    padding: 0.45rem 0.85rem; border-bottom: 1px solid var(--line);
}
.latest-row:last-child { border-bottom: none; }
.latest-key { font-size: 0.7rem; color: var(--text-2); font-weight: 500; }
.latest-val { font-family: var(--font-mono); font-size: 0.7rem; color: var(--text-1); font-weight: 500; }

.empty-state {
    padding: 3rem 2rem;
    border-radius: var(--r-xl);
    background: var(--surface-2);
    border: 1px dashed var(--line-md);
    text-align: center;
}
.empty-icon {
    width: 52px; height: 52px; border-radius: 14px;
    background: rgba(255,255,255,0.03); border: 1px solid var(--line);
    display: flex; align-items: center; justify-content: center;
    margin: 0 auto 1rem; opacity: 0.5;
}
.empty-title {
    font-family: var(--font-display); font-size: 1rem; font-weight: 600;
    color: var(--text-1); margin-bottom: 0.4rem; letter-spacing: -0.01em;
}
.empty-text {
    font-size: 0.8rem; color: var(--text-2);
    max-width: 360px; margin: 0 auto; line-height: 1.6;
}

.chart-wrap {
    padding: 1rem; border-radius: var(--r-md);
    background: var(--surface-2); border: 1px solid var(--line);
    margin-top: 0.85rem; box-shadow: var(--sh-1);
}

.export-head {
    display: flex; align-items: center; gap: 0.85rem;
    padding: 1.1rem 1.4rem;
    border-radius: var(--r-lg) var(--r-lg) 0 0;
    background: var(--surface-2);
    border: 1px solid var(--line); border-bottom: none;
    margin-top: 1.5rem; position: relative; overflow: hidden;
}
.export-head::before {
    content: "";
    position: absolute; top: -80px; left: -20px;
    width: 280px; height: 280px;
    background: radial-gradient(circle, rgba(8,145,178,0.08), transparent 60%);
    pointer-events: none;
}
.export-head-icon {
    position: relative; width: 40px; height: 40px; border-radius: 10px;
    background: var(--accent-soft); border: 1px solid var(--accent-line);
    display: flex; align-items: center; justify-content: center; flex-shrink: 0;
    box-shadow: 0 8px 20px rgba(8,145,178,0.12);
}
.export-head-title {
    font-family: var(--font-display); font-size: 0.95rem; font-weight: 700;
    color: var(--text-1); letter-spacing: -0.015em; position: relative;
}
.export-head-sub { font-size: 0.67rem; color: var(--text-3); position: relative; }
.export-head-badge {
    margin-left: auto; padding: 0.22rem 0.55rem;
    border-radius: var(--r-pill); position: relative;
    background: rgba(5,150,105,0.09); border: 1px solid rgba(5,150,105,0.22);
    color: var(--success-hi); font-family: var(--font-mono);
    font-size: 0.58rem; font-weight: 700; letter-spacing: 0.1em;
}
.export-item-label {
    display: flex; align-items: center; justify-content: center; gap: 0.4rem;
    font-family: var(--font-mono); font-size: 0.58rem;
    color: var(--text-3); text-transform: uppercase; letter-spacing: 0.12em;
    font-weight: 700; margin: 0 0 0.45rem; text-align: center; width: 100%;
}
.export-item-label::before {
    content: ""; width: 3px; height: 9px; border-radius: 1px;
    background: var(--accent); opacity: 0.9; flex-shrink: 0;
}

.step-card {
    padding: 0.8rem 0.6rem;
    border-radius: var(--r-md);
    background: var(--surface-2); border: 1px solid var(--line);
    text-align: center;
    transition: border-color 150ms ease;
}
.step-card:hover { border-color: var(--line-md); }
.step-num {
    font-family: var(--font-display); font-size: 1.15rem; font-weight: 700;
    color: var(--accent-hi); line-height: 1;
}
.step-title { font-size: 0.72rem; font-weight: 600; color: var(--text-1); margin: 0.25rem 0 0.1rem; }
.step-desc { font-size: 0.6rem; color: var(--text-3); line-height: 1.4; }

[data-testid="stFileUploader"] {
    background: linear-gradient(180deg, rgba(8,145,178,0.03), rgba(8,145,178,0.005));
    border-radius: var(--r-lg); padding: 1rem 0.9rem;
    border: 1px dashed var(--line-md);
    transition: border-color 200ms ease, background 200ms ease;
}
[data-testid="stFileUploader"]:hover {
    border-color: var(--accent-line);
    background: linear-gradient(180deg, rgba(8,145,178,0.06), rgba(8,145,178,0.01));
}
[data-testid="stFileUploader"] small { color: var(--text-3) !important; font-size: 0.7rem !important; }
[data-testid="stFileUploader"] button {
    border-radius: var(--r-sm) !important;
    background: var(--accent) !important; color: #050e1a !important;
    border: 1px solid var(--accent) !important; font-weight: 600 !important;
    font-size: 0.73rem !important; min-height: 30px !important;
    padding: 0.32rem 0.75rem !important;
}
[data-testid="stFileUploader"] button:hover {
    background: var(--accent-hi) !important; border-color: var(--accent-hi) !important;
}

[data-testid="stExpander"] {
    border-radius: var(--r-md) !important; border: 1px solid var(--line) !important;
    background: var(--surface-2) !important;
}
[data-testid="stExpander"]:hover { border-color: var(--line-md) !important; }

.stButton > button {
    border-radius: var(--r-sm) !important; font-weight: 600 !important;
    font-size: 0.78rem !important; padding: 0.4rem 0.9rem !important;
    min-height: 34px !important; line-height: 1.15 !important;
    transition: all 130ms ease !important; box-shadow: none !important;
}
.stButton > button p { font-size: 0.78rem !important; font-weight: 600 !important; margin: 0 !important; }
.stButton > button:hover { transform: none !important; box-shadow: var(--sh-1) !important; }
.stButton > button:focus-visible { box-shadow: var(--sh-accent) !important; outline: none !important; }
.stButton > button[kind="primary"] {
    padding: 0.45rem 1.1rem !important; min-height: 38px !important;
    background: var(--accent) !important;
    border: 1px solid var(--accent) !important; color: #050e1a !important;
}
.stButton > button[kind="primary"]:hover {
    background: var(--accent-hi) !important; border-color: var(--accent-hi) !important;
}
.stButton { display: flex !important; justify-content: center !important; align-items: center !important; }
.stButton > button { margin-left: auto !important; margin-right: auto !important; justify-content: center !important; }

[data-testid="stDownloadButton"] {
    display: flex !important; justify-content: center !important;
    align-items: center !important; margin-bottom: 0.85rem !important;
}
[data-testid="stDownloadButton"] > button {
    width: 220px !important; min-width: 220px !important;
    min-height: 36px !important; padding: 0.45rem 0.85rem !important;
    border-radius: var(--r-sm) !important;
    background: var(--surface-3) !important; border: 1px solid var(--line) !important;
    color: var(--text-1) !important; font-size: 0.74rem !important;
    font-weight: 600 !important; display: flex !important;
    align-items: center !important; justify-content: center !important;
    transition: all 150ms ease !important; box-shadow: none !important;
}
[data-testid="stDownloadButton"] > button p { font-size: 0.74rem !important; font-weight: 600 !important; margin: 0 !important; }
[data-testid="stDownloadButton"] > button:hover {
    background: var(--accent-soft) !important;
    border-color: var(--accent-line) !important; color: var(--accent-hi) !important;
    box-shadow: 0 4px 14px rgba(8,145,178,0.12) !important; transform: translateY(-1px) !important;
}
[data-testid="stDownloadButton"] > button[kind="primary"] {
    background: var(--accent) !important; border-color: var(--accent) !important;
    color: #050e1a !important; min-height: 40px !important;
}
[data-testid="stDownloadButton"] > button[kind="primary"]:hover {
    background: var(--accent-hi) !important; border-color: var(--accent-hi) !important;
    color: #050e1a !important; box-shadow: 0 6px 18px rgba(8,145,178,0.28) !important;
}

[data-testid="stMetricValue"] {
    font-family: var(--font-display) !important; font-size: 1.3rem !important;
    font-weight: 700 !important; letter-spacing: -0.02em !important;
}
[data-testid="stMetricLabel"] {
    font-size: 0.6rem !important; color: var(--text-3) !important;
    letter-spacing: 0.08em !important; font-weight: 600 !important;
}
.stProgress > div > div > div > div { background: var(--accent) !important; border-radius: 4px !important; }
.stAlert { border-radius: var(--r-sm) !important; padding: 0.6rem 0.85rem !important; }
hr { border-color: var(--line) !important; margin: 1rem 0 !important; }
.stImage { border-radius: var(--r-md) !important; overflow: hidden !important; border: 1px solid var(--line) !important; }
.stTabs [data-baseweb="tab-list"] {
    gap: 4px; background: transparent; border-bottom: 1px solid var(--line);
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important; border-radius: var(--r-sm) var(--r-sm) 0 0 !important;
    padding: 0.45rem 0.8rem !important; font-size: 0.78rem !important;
    font-weight: 500 !important; color: var(--text-2) !important;
}
.stTabs [aria-selected="true"] { color: var(--text-1) !important; border-bottom: 2px solid var(--accent) !important; }
[data-testid="stCheckbox"] { padding: 0.15rem 0 !important; }
[data-testid="stCheckbox"] label { font-size: 0.78rem !important; }
[data-testid="stToggle"] label { font-size: 0.78rem !important; }

.app-footer {
    position: relative; margin-top: 3rem;
    display: grid; grid-template-columns: 1.5fr 1.4fr 1fr; gap: 1.5rem;
    padding: 2rem;
    border-radius: var(--r-xl);
    background: var(--surface-2);
    border: 1px solid var(--line);
    overflow: hidden; box-shadow: var(--sh-2), var(--sh-inset);
}
.app-footer::before {
    content: "";
    position: absolute; top: 0; left: 0; right: 0; height: 1px;
    background: linear-gradient(90deg, transparent, rgba(8,145,178,0.45), rgba(124,58,237,0.35), transparent);
}
.footer-brand-lockup { display: flex; align-items: center; gap: 0.75rem; margin-bottom: 0.65rem; }
.footer-brand-icon {
    width: 42px; height: 42px; border-radius: 11px;
    background: var(--accent-soft); border: 1px solid var(--accent-line);
    display: flex; align-items: center; justify-content: center; flex-shrink: 0;
    box-shadow: 0 6px 18px rgba(8,145,178,0.12);
}
.footer-brand-name {
    font-family: var(--font-display); font-size: 1rem; font-weight: 800;
    color: var(--text-1); letter-spacing: -0.02em;
}
.footer-brand-tag { font-size: 0.58rem; color: var(--text-3); letter-spacing: 0.05em; margin-top: 1px; }
.footer-desc { font-size: 0.74rem; color: var(--text-2); line-height: 1.6; max-width: 300px; }
.footer-copy {
    font-family: var(--font-mono); font-size: 0.62rem; color: var(--text-3);
    padding-top: 0.55rem; border-top: 1px dashed var(--line); margin-top: 0.55rem;
}
.footer-col-title {
    display: flex; align-items: center; gap: 0.4rem;
    font-family: var(--font-mono); font-size: 0.56rem; color: var(--text-3);
    font-weight: 700; letter-spacing: 0.14em; margin-bottom: 0.65rem;
}
.footer-col-title::before {
    content: ""; width: 2px; height: 9px; border-radius: 1px;
    background: var(--accent); flex-shrink: 0;
}
.footer-tech-badges { display: flex; flex-wrap: wrap; gap: 0.4rem; }
.ft-badge {
    display: inline-flex; align-items: center; gap: 0.28rem;
    padding: 0.28rem 0.55rem; border-radius: 6px;
    background: rgba(255,255,255,0.02); border: 1px solid var(--line);
    color: var(--text-2); font-family: var(--font-mono); font-size: 0.62rem;
    font-weight: 500; transition: all 150ms ease;
}
.ft-badge:hover {
    border-color: var(--accent-line); color: var(--accent-hi);
    background: var(--accent-soft); transform: translateY(-1px);
}
.ft-badge-dot { width: 4px; height: 4px; border-radius: 50%; background: var(--accent); flex-shrink: 0; }
.footer-stats { display: grid; grid-template-columns: 1fr 1fr; gap: 0.5rem; }
.f-stat {
    padding: 0.65rem 0.75rem; border-radius: 9px;
    background: rgba(0,0,0,0.2); border: 1px solid var(--line);
    transition: border-color 150ms ease;
}
.f-stat:hover { border-color: var(--line-md); }
.f-stat-wide { grid-column: 1 / -1; }
.f-stat-val {
    font-family: var(--font-display); font-size: 1rem; font-weight: 700;
    color: var(--text-1); display: flex; align-items: center; gap: 0.35rem;
    letter-spacing: -0.02em; font-variant-numeric: tabular-nums;
}
.f-stat-lbl { font-size: 0.52rem; color: var(--text-3); letter-spacing: 0.12em; font-weight: 700; margin-top: 0.1rem; }
.f-dot { width: 6px; height: 6px; border-radius: 50%; background: var(--success-hi); box-shadow: 0 0 7px rgba(52,211,153,0.55); animation: pulse-dot 2.2s ease-in-out infinite; }
.f-dot-off { background: var(--danger-hi); box-shadow: 0 0 7px rgba(248,113,113,0.55); }

.copyright-line {
    text-align: center; padding: 1rem 0.5rem 0.5rem;
    font-size: 0.68rem; color: var(--text-3); line-height: 1.9;
}

.disclaimer {
    margin-top: 1.25rem; padding: 0.85rem 1.1rem; border-radius: var(--r-md);
    background: rgba(217,119,6,0.05); border: 1px solid rgba(217,119,6,0.16);
    color: var(--text-2); font-size: 0.78rem; line-height: 1.65;
    display: flex; gap: 0.65rem; align-items: flex-start;
}

@media (max-width: 1280px) {
    :root { --sb-w: 252px; }
    .main .block-container { padding-left: 1.1rem !important; padding-right: 1.1rem !important; }
}
@media (max-width: 1024px) {
    :root { --sb-w: 235px; }
    .sticky-header { left: calc(var(--sb-w) + 0.4rem) !important; }
    .app-footer { grid-template-columns: 1fr 1fr !important; }
    .app-footer > div:last-child { grid-column: 1 / -1 !important; }
}
@media (max-width: 768px) {
    :root { --sb-w: 100%; }
    [data-testid="stSidebar"] { min-width: 100% !important; max-width: 100% !important; }
    .sticky-header { left: 0.35rem !important; right: 0.35rem !important; }
    .hd-ticker { display: none; }
    .main .block-container { padding-top: 4.5rem !important; }
    .hero h1 { font-size: 1.75rem !important; }
    .app-footer { grid-template-columns: 1fr !important; padding: 1.5rem; }
    [data-testid="stDownloadButton"] > button { width: 100% !important; min-width: 0 !important; }
}
@media (max-width: 480px) {
    .prob-grid { grid-template-columns: 1fr 1fr !important; }
}
@media (prefers-reduced-motion: reduce) {
    *, *::before, *::after { animation-duration: 0.01ms !important; transition-duration: 0.01ms !important; }
}
</style>
"""
st.markdown(STYLES, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR v5.0 — Premium Redesign
# ─────────────────────────────────────────────────────────────────────────────
SIDEBAR_STYLES = """
<style>
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, var(--surface) 0%, #080d1c 55%, var(--bg) 100%) !important;
    border-right: 1px solid var(--line) !important;
    min-width: var(--sb-w) !important;
    max-width: var(--sb-w) !important;
    width: var(--sb-w) !important;
    position: relative;
    overflow: hidden;
}
[data-testid="stSidebar"]::before {
    content: "";
    position: absolute;
    inset: 0;
    background-image: radial-gradient(circle, rgba(255,255,255,0.03) 1px, transparent 1px);
    background-size: 22px 22px;
    pointer-events: none;
    z-index: 0;
    opacity: 0.55;
    mask-image: linear-gradient(180deg, #000 0%, #000 65%, transparent 100%);
    -webkit-mask-image: linear-gradient(180deg, #000 0%, #000 65%, transparent 100%);
}
[data-testid="stSidebar"] > div:first-child {
    width: var(--sb-w) !important;
    padding: 0 !important;
    position: relative;
    z-index: 1;
}
[data-testid="stSidebar"] ::-webkit-scrollbar { width: 3px; height: 3px; }
[data-testid="stSidebar"] ::-webkit-scrollbar-track { background: transparent; }
[data-testid="stSidebar"] ::-webkit-scrollbar-thumb {
    background: var(--line-md); border-radius: 10px;
    transition: background 180ms ease;
}
[data-testid="stSidebar"] ::-webkit-scrollbar-thumb:hover {
    background: var(--accent-line);
}

.sb-brand {
    padding: 1.15rem 1rem 0.9rem;
    border-bottom: 1px solid var(--line);
    display: flex; align-items: center; gap: 0.75rem;
    position: relative;
    overflow: hidden;
}
.sb-brand::after {
    content: "";
    position: absolute;
    bottom: -50px; left: -30px;
    width: 220px; height: 100px;
    background: radial-gradient(ellipse, rgba(8,145,178,0.12), transparent 70%);
    pointer-events: none;
    z-index: 0;
}
.sb-brand-icon {
    width: 40px; height: 40px; border-radius: 11px; flex-shrink: 0;
    background: linear-gradient(135deg, rgba(8,145,178,0.22), rgba(8,145,178,0.03));
    border: 1px solid var(--accent-line);
    display: flex; align-items: center; justify-content: center;
    position: relative;
    box-shadow: 0 0 22px rgba(8,145,178,0.18), inset 0 1px 0 rgba(255,255,255,0.08);
    z-index: 1;
    transition: box-shadow 220ms ease;
}
.sb-brand:hover .sb-brand-icon {
    box-shadow: 0 0 30px rgba(34,211,238,0.28), inset 0 1px 0 rgba(255,255,255,0.12);
}
.sb-brand-pulse {
    position: absolute; top: -2px; right: -2px;
    width: 9px; height: 9px; border-radius: 50%;
    background: var(--success-hi);
    border: 2px solid var(--surface);
    box-shadow: 0 0 8px rgba(52,211,153,0.85);
    animation: pulse-dot 2.5s ease-in-out infinite;
}
.sb-brand-text { min-width: 0; flex: 1; position: relative; z-index: 1; }
.sb-brand-name {
    font-family: var(--font-display);
    font-size: 0.95rem; font-weight: 700;
    color: var(--text-1); letter-spacing: -0.025em; line-height: 1.1;
    display: flex; align-items: center; gap: 0.35rem;
    flex-wrap: nowrap;
}
.sb-brand-name .sb-brand-ai { color: var(--accent-hi); font-weight: 800; }
.sb-brand-ver {
    display: inline-flex; align-items: center;
    padding: 0.08rem 0.36rem; border-radius: 4px;
    background: var(--accent-soft); border: 1px solid var(--accent-line);
    font-family: var(--font-mono); font-size: 0.5rem; font-weight: 700;
    color: var(--accent-hi); letter-spacing: 0.04em;
    transform: translateY(-1px);
    margin-left: 0.15rem;
}
.sb-brand-sub {
    font-size: 0.6rem; color: var(--text-3); font-weight: 500;
    letter-spacing: 0.05em; margin-top: 3px;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}

.sb-session {
    margin: 0.7rem 0.85rem 0;
    padding: 0.5rem 0.7rem;
    border-radius: var(--r-sm);
    background: linear-gradient(135deg, rgba(255,255,255,0.028), rgba(255,255,255,0.008));
    border: 1px solid var(--line);
    display: flex; align-items: center; gap: 0.5rem;
    font-size: 0.64rem; color: var(--text-2); font-weight: 500;
    transition: border-color 180ms ease;
}
.sb-session:hover { border-color: var(--line-md); }
.sb-session-dot {
    width: 6px; height: 6px; border-radius: 50%; flex-shrink: 0;
    animation: pulse-dot 2.5s ease-in-out infinite;
}
.sb-session-txt {
    flex: 1; min-width: 0; color: var(--text-2);
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
    font-weight: 500;
}
.sb-session-time {
    font-family: var(--font-mono); font-size: 0.58rem;
    color: var(--text-3); font-weight: 600;
    padding: 0.08rem 0.38rem; border-radius: 4px;
    background: rgba(255,255,255,0.02); border: 1px solid var(--line);
    flex-shrink: 0;
}

.sb-nav-group {
    padding: 0.85rem 0.95rem 0.35rem;
    font-size: 0.55rem; font-weight: 700;
    color: var(--text-4); letter-spacing: 0.15em;
    text-transform: uppercase;
    display: flex; align-items: center; gap: 0.55rem;
}
.sb-nav-group::after {
    content: ""; flex: 1; height: 1px;
    background: linear-gradient(to right, var(--line), transparent 90%);
}

[data-testid="stSidebar"] [class*="st-key-nav_"] {
    margin: 1px 0.6rem !important;
    width: auto !important;
    display: block !important;
}
[data-testid="stSidebar"] [class*="st-key-nav_"] button {
    width: 100% !important; min-height: 38px !important; height: 38px !important;
    display: flex !important; align-items: center !important;
    justify-content: flex-start !important;
    padding: 0 0.7rem !important;
    border-radius: 9px !important;
    background: transparent !important;
    border: 1px solid transparent !important;
    color: var(--text-2) !important;
    font-size: 0.78rem !important; font-weight: 500 !important;
    line-height: 1 !important;
    transition: background 180ms ease, border-color 180ms ease,
                color 180ms ease, transform 180ms cubic-bezier(.2,.8,.2,1),
                box-shadow 180ms ease !important;
    box-shadow: none !important;
    position: relative;
    overflow: hidden;
    margin: 0 !important;
}
[data-testid="stSidebar"] [class*="st-key-nav_"] button p {
    margin: 0 !important; line-height: 1 !important;
    font-size: 0.78rem !important; font-weight: 500 !important;
    transition: color 180ms ease !important;
}
[data-testid="stSidebar"] [class*="st-key-nav_"] button:hover {
    background: rgba(255,255,255,0.04) !important;
    border-color: var(--line) !important;
    color: var(--text-1) !important;
    transform: translateX(2px) !important;
}
[data-testid="stSidebar"] [class*="st-key-nav_"] button:hover p { color: var(--text-1) !important; }
[data-testid="stSidebar"] [class*="st-key-nav_"] button:focus-visible {
    box-shadow: var(--sh-accent) !important; outline: none !important;
}
[data-testid="stSidebar"] .stButton { display: block !important; width: auto !important; }

[data-testid="stSidebar"] [class*="st-key-nav_"] button::before {
    content: "";
    display: inline-block;
    width: 15px;
    height: 15px;
    margin-right: 9px;
    flex-shrink: 0;
    background-color: currentColor;
    -webkit-mask-repeat: no-repeat;
    -webkit-mask-position: center;
    -webkit-mask-size: contain;
    mask-repeat: no-repeat;
    mask-position: center;
    mask-size: contain;
    opacity: 0.85;
    transition: opacity 130ms ease, background-color 180ms ease;
}
[data-testid="stSidebar"] [class*="st-key-nav_"] button:hover::before { opacity: 1; }

[data-testid="stSidebar"] .st-key-nav_home button::before {
    -webkit-mask-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%23000' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='m3 9 9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z'/%3E%3Cpolyline points='9 22 9 12 15 12 15 22'/%3E%3C/svg%3E");
            mask-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%23000' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='m3 9 9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z'/%3E%3Cpolyline points='9 22 9 12 15 12 15 22'/%3E%3C/svg%3E");
}
[data-testid="stSidebar"] .st-key-nav_mri_analysis button::before {
    -webkit-mask-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%23000' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M6 18h8'/%3E%3Cpath d='M3 22h18'/%3E%3Cpath d='M14 22a7 7 0 1 0 0-14h-1'/%3E%3Cpath d='M9 14h2'/%3E%3Cpath d='M9 12a2 2 0 0 1-2-2V6h6v4a2 2 0 0 1-2 2Z'/%3E%3Cpath d='M12 6V3a1 1 0 0 0-1-1H9a1 1 0 0 0-1 1v3'/%3E%3C/svg%3E");
            mask-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%23000' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M6 18h8'/%3E%3Cpath d='M3 22h18'/%3E%3Cpath d='M14 22a7 7 0 1 0 0-14h-1'/%3E%3Cpath d='M9 14h2'/%3E%3Cpath d='M9 12a2 2 0 0 1-2-2V6h6v4a2 2 0 0 1-2 2Z'/%3E%3Cpath d='M12 6V3a1 1 0 0 0-1-1H9a1 1 0 0 0-1 1v3'/%3E%3C/svg%3E");
}
[data-testid="stSidebar"] .st-key-nav_dashboard button::before {
    -webkit-mask-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%23000' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cline x1='18' y1='20' x2='18' y2='10'/%3E%3Cline x1='12' y1='20' x2='12' y2='4'/%3E%3Cline x1='6' y1='20' x2='6' y2='14'/%3E%3Cline x1='2' y1='20' x2='22' y2='20'/%3E%3C/svg%3E");
            mask-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%23000' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cline x1='18' y1='20' x2='18' y2='10'/%3E%3Cline x1='12' y1='20' x2='12' y2='4'/%3E%3Cline x1='6' y1='20' x2='6' y2='14'/%3E%3Cline x1='2' y1='20' x2='22' y2='20'/%3E%3C/svg%3E");
}
[data-testid="stSidebar"] .st-key-nav_history button::before {
    -webkit-mask-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%23000' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8'/%3E%3Cpath d='M3 3v5h5'/%3E%3Cpath d='M12 7v5l4 2'/%3E%3C/svg%3E");
            mask-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%23000' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8'/%3E%3Cpath d='M3 3v5h5'/%3E%3Cpath d='M12 7v5l4 2'/%3E%3C/svg%3E");
}
[data-testid="stSidebar"] .st-key-nav_gradcam button::before {
    -webkit-mask-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%23000' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Ccircle cx='12' cy='12' r='3'/%3E%3Cpath d='M12 2v3'/%3E%3Cpath d='M12 19v3'/%3E%3Cpath d='m4.22 4.22 2.12 2.12'/%3E%3Cpath d='m17.66 17.66 2.12 2.12'/%3E%3Cpath d='M2 12h3'/%3E%3Cpath d='M19 12h3'/%3E%3Cpath d='m4.22 19.78 2.12-2.12'/%3E%3Cpath d='m17.66 6.34 2.12-2.12'/%3E%3C/svg%3E");
            mask-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%23000' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Ccircle cx='12' cy='12' r='3'/%3E%3Cpath d='M12 2v3'/%3E%3Cpath d='M12 19v3'/%3E%3Cpath d='m4.22 4.22 2.12 2.12'/%3E%3Cpath d='m17.66 17.66 2.12 2.12'/%3E%3Cpath d='M2 12h3'/%3E%3Cpath d='M19 12h3'/%3E%3Cpath d='m4.22 19.78 2.12-2.12'/%3E%3Cpath d='m17.66 6.34 2.12-2.12'/%3E%3C/svg%3E");
}
[data-testid="stSidebar"] .st-key-nav_xailab button::before {
    -webkit-mask-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%23000' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M14.5 2v17.5c0 1.4-1.1 2.5-2.5 2.5h0c-1.4 0-2.5-1.1-2.5-2.5V2'/%3E%3Cpath d='M8.5 2h7'/%3E%3Cpath d='M14.5 16h-5'/%3E%3Cpath d='m8.5 13 5.5 3'/%3E%3C/svg%3E");
            mask-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%23000' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M14.5 2v17.5c0 1.4-1.1 2.5-2.5 2.5h0c-1.4 0-2.5-1.1-2.5-2.5V2'/%3E%3Cpath d='M8.5 2h7'/%3E%3Cpath d='M14.5 16h-5'/%3E%3Cpath d='m8.5 13 5.5 3'/%3E%3C/svg%3E");
}
[data-testid="stSidebar"] .st-key-nav_settings button::before {
    -webkit-mask-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%23000' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M12.22 2h-.44a2 2 0 0 0-2 2v.18a2 2 0 0 1-1 1.73l-.43.25a2 2 0 0 1-2 0l-.15-.08a2 2 0 0 0-2.73.73l-.22.38a2 2 0 0 0 .73 2.73l.15.1a2 2 0 0 1 1 1.72v.51a2 2 0 0 1-1 1.74l-.15.09a2 2 0 0 0-.73 2.73l.22.38a2 2 0 0 0 2.73.73l.15-.08a2 2 0 0 1 2 0l.43.25a2 2 0 0 1 1 1.73V20a2 2 0 0 0 2 2h.44a2 2 0 0 0 2-2v-.18a2 2 0 0 1 1-1.73l.43-.25a2 2 0 0 1 2 0l.15.08a2 2 0 0 0 2.73-.73l.22-.39a2 2 0 0 0-.73-2.73l-.15-.08a2 2 0 0 1-1-1.74v-.5a2 2 0 0 1 1-1.74l.15-.09a2 2 0 0 0 .73-2.73l-.22-.38a2 2 0 0 0-2.73-.73l-.15.08a2 2 0 0 1-2 0l-.43-.25a2 2 0 0 1-1-1.73V4a2 2 0 0 0-2-2z'/%3E%3Ccircle cx='12' cy='12' r='3'/%3E%3C/svg%3E");
            mask-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%23000' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M12.22 2h-.44a2 2 0 0 0-2 2v.18a2 2 0 0 1-1 1.73l-.43.25a2 2 0 0 1-2 0l-.15-.08a2 2 0 0 0-2.73.73l-.22.38a2 2 0 0 0 .73 2.73l.15.1a2 2 0 0 1 1 1.72v.51a2 2 0 0 1-1 1.74l-.15.09a2 2 0 0 0-.73 2.73l.22.38a2 2 0 0 0 2.73.73l.15-.08a2 2 0 0 1 2 0l.43.25a2 2 0 0 1 1 1.73V20a2 2 0 0 0 2 2h.44a2 2 0 0 0 2-2v-.18a2 2 0 0 1 1-1.73l.43-.25a2 2 0 0 1 2 0l.15.08a2 2 0 0 0 2.73-.73l.22-.39a2 2 0 0 0-.73-2.73l-.15-.08a2 2 0 0 1-1-1.74v-.5a2 2 0 0 1 1-1.74l.15-.09a2 2 0 0 0 .73-2.73l-.22-.38a2 2 0 0 0-2.73-.73l-.15.08a2 2 0 0 1-2 0l-.43-.25a2 2 0 0 1-1-1.73V4a2 2 0 0 0-2-2z'/%3E%3Ccircle cx='12' cy='12' r='3'/%3E%3C/svg%3E");
}

.sb-engine {
    margin: 0.75rem 0.85rem 0;
    padding: 0.85rem 0.9rem;
    border-radius: var(--r-md);
    background: linear-gradient(180deg, rgba(0,0,0,0.28), rgba(0,0,0,0.16));
    border: 1px solid var(--line);
    transition: border-color 200ms ease, box-shadow 200ms ease;
    position: relative;
    overflow: hidden;
}
.sb-engine:hover {
    border-color: var(--accent-line);
    box-shadow: 0 0 22px rgba(8,145,178,0.08);
}
.sb-engine-head {
    display: flex; align-items: center; justify-content: space-between;
    margin-bottom: 0.55rem;
}
.sb-engine-title {
    font-size: 0.56rem; font-weight: 700;
    color: var(--text-3); letter-spacing: 0.14em;
    text-transform: uppercase;
}
.sb-engine-live {
    display: inline-flex; align-items: center; gap: 0.3rem;
    padding: 0.12rem 0.42rem; border-radius: 4px;
    background: rgba(5,150,105,0.1); color: var(--success-hi);
    border: 1px solid rgba(5,150,105,0.25);
    font-size: 0.5rem; font-weight: 800; letter-spacing: 0.12em;
    font-family: var(--font-body);
}
.sb-engine-live-dot {
    width: 4px; height: 4px; border-radius: 50%;
    background: currentColor;
    animation: pulse-dot 2s ease-in-out infinite;
    box-shadow: 0 0 5px currentColor;
}
.sb-engine-status {
    display: flex; align-items: center; gap: 0.45rem;
    font-size: 0.72rem; font-weight: 600;
    margin-bottom: 0.6rem;
}
.sb-engine-confbar {
    display: flex; align-items: center; gap: 0.5rem;
    margin-bottom: 0.45rem;
}
.sb-engine-confbar-track {
    flex: 1; height: 4px; border-radius: 999px;
    background: rgba(255,255,255,0.05); overflow: hidden;
}
.sb-engine-confbar-fill {
    height: 100%; border-radius: 999px;
    background: linear-gradient(90deg, var(--accent), var(--accent-hi));
    transition: width 600ms cubic-bezier(.2,.8,.2,1);
}
.sb-engine-confbar-lbl {
    font-family: var(--font-mono); font-size: 0.6rem;
    color: var(--text-2); font-weight: 600;
    min-width: 34px; text-align: right;
}
.sb-engine-grid {
    display: grid; grid-template-columns: 1fr 1fr; gap: 0.4rem;
    margin-top: 0.5rem;
}
.sb-mini {
    padding: 0.45rem 0.55rem;
    border-radius: 7px;
    background: rgba(255,255,255,0.02);
    border: 1px solid var(--line);
    transition: border-color 150ms ease;
    overflow: hidden;
    min-width: 0;
}
.sb-mini:hover { border-color: var(--line-md); }
.sb-mini-lbl {
    font-size: 0.5rem; color: var(--text-3);
    font-weight: 700; letter-spacing: 0.1em;
}
.sb-mini-val {
    font-family: var(--font-mono); font-size: 0.68rem; font-weight: 600;
    margin-top: 0.15rem;
    white-space: nowrap; overflow: hidden; text-overflow: ellipsis;
}
.sb-mini-white  .sb-mini-val { color: var(--text-1); }
.sb-mini-cyan   .sb-mini-val { color: var(--accent-hi); }
.sb-mini-violet .sb-mini-val { color: var(--violet-hi); }

[data-testid="stSidebar"] [class*="st-key-sb_clear"] button,
[data-testid="stSidebar"] [class*="st-key-sb_reset"] button {
    min-height: 34px !important; height: 34px !important;
    display: flex !important; align-items: center !important;
    justify-content: flex-start !important;
    padding: 0 0.7rem !important;
    border-radius: 8px !important;
    background: rgba(255,255,255,0.02) !important;
    border: 1px solid var(--line) !important;
    color: var(--text-2) !important;
    font-size: 0.72rem !important; font-weight: 500 !important;
    transition: all 180ms ease !important;
    box-shadow: none !important;
    position: relative;
    margin: 0 0.6rem !important;
    width: calc(100% - 1.2rem) !important;
}
[data-testid="stSidebar"] [class*="st-key-sb_clear"] button p,
[data-testid="stSidebar"] [class*="st-key-sb_reset"] button p {
    margin: 0 !important;
    font-size: 0.72rem !important;
    font-weight: 500 !important;
    transition: color 180ms ease !important;
}
[data-testid="stSidebar"] [class*="st-key-sb_clear"]:not([class*="yes"]):not([class*="no"]) button:hover {
    border-color: rgba(217,119,6,0.4) !important;
    color: var(--warning-hi) !important;
    background: rgba(217,119,6,0.06) !important;
}
[data-testid="stSidebar"] [class*="st-key-sb_clear"]:not([class*="yes"]):not([class*="no"]) button:hover p {
    color: var(--warning-hi) !important;
}
[data-testid="stSidebar"] [class*="st-key-sb_reset"]:not([class*="yes"]):not([class*="no"]) button:hover {
    border-color: rgba(220,38,38,0.4) !important;
    color: var(--danger-hi) !important;
    background: rgba(220,38,38,0.06) !important;
}
[data-testid="stSidebar"] [class*="st-key-sb_reset"]:not([class*="yes"]):not([class*="no"]) button:hover p {
    color: var(--danger-hi) !important;
}

.sb-confirm {
    margin: 0.35rem 0.85rem 0;
    padding: 0.65rem 0.75rem;
    border-radius: var(--r-sm) var(--r-sm) 0 0;
    background: rgba(217,119,6,0.06);
    border: 1px solid rgba(217,119,6,0.22);
    border-bottom: none;
}
.sb-confirm-title {
    display: flex; align-items: center; gap: 0.35rem;
    font-size: 0.66rem; font-weight: 700;
    color: var(--warning-hi); letter-spacing: 0.02em;
    margin-bottom: 0.2rem;
}
.sb-confirm-text {
    font-size: 0.6rem; color: var(--text-3); line-height: 1.45;
}
.sb-confirm-danger {
    background: rgba(220,38,38,0.05);
    border-color: rgba(220,38,38,0.22);
}
.sb-confirm-danger .sb-confirm-title { color: var(--danger-hi); }
[data-testid="stSidebar"] [class*="st-key-sb_clear_yes"] button,
[data-testid="stSidebar"] [class*="st-key-sb_clear_no"] button,
[data-testid="stSidebar"] [class*="st-key-sb_reset_yes"] button,
[data-testid="stSidebar"] [class*="st-key-sb_reset_no"] button {
    border-radius: 0 0 var(--r-sm) var(--r-sm) !important;
    background: rgba(255,255,255,0.02) !important;
    border: 1px solid var(--line) !important;
    border-top: none !important;
    font-size: 0.66rem !important;
    min-height: 30px !important;
    height: 30px !important;
    color: var(--text-2) !important;
    padding: 0 !important;
    width: 100% !important;
    margin: 0 !important;
}
[data-testid="stSidebar"] [class*="st-key-sb_clear_yes"] button p,
[data-testid="stSidebar"] [class*="st-key-sb_clear_no"] button p,
[data-testid="stSidebar"] [class*="st-key-sb_reset_yes"] button p,
[data-testid="stSidebar"] [class*="st-key-sb_reset_no"] button p {
    font-size: 0.66rem !important;
    font-weight: 600 !important;
}
[data-testid="stSidebar"] [class*="st-key-sb_clear_yes"] button:hover {
    color: var(--warning-hi) !important;
    background: rgba(217,119,6,0.1) !important;
}
[data-testid="stSidebar"] [class*="st-key-sb_reset_yes"] button:hover {
    color: var(--danger-hi) !important;
    background: rgba(220,38,38,0.1) !important;
}
[data-testid="stSidebar"] [data-testid="column"] { padding: 0 !important; }

.sb-warning {
    margin: 0.7rem 0.85rem 0.9rem;
    padding: 0.6rem 0.75rem 0.6rem 0.65rem;
    border-radius: var(--r-sm);
    background: rgba(217,119,6,0.035);
    border: 1px solid rgba(217,119,6,0.14);
    border-left: 2px solid var(--warning);
    font-size: 0.62rem; color: var(--text-2); line-height: 1.55;
    display: flex; gap: 0.5rem; align-items: flex-start;
}
.sb-warning svg { flex-shrink: 0; margin-top: 1px; }

[data-testid="stSidebar"] hr {
    margin: 0.55rem 0.85rem !important;
    border-color: var(--line) !important;
    border-width: 1px 0 0 !important;
}
</style>
"""
st.markdown(SIDEBAR_STYLES, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# MODEL ARCHITECTURES
# ─────────────────────────────────────────────────────────────────────────────

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, pool=True):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
        ]
        if pool: layers.append(nn.MaxPool2d(2))
        self.block = nn.Sequential(*layers)
    def forward(self, x): return self.block(x)


class CustomCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(3, 32, pool=True), ConvBlock(32, 64, pool=True),
            ConvBlock(64, 128, pool=True), ConvBlock(128, 256, pool=False),
            ConvBlock(256, 256, pool=True), nn.Dropout2d(0.3),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Linear(256, 256), nn.BatchNorm1d(256),
            nn.ReLU(inplace=True), nn.Dropout(0.35), nn.Linear(256, num_classes),
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
        x = self.features(x); x = self.pool(x); return self.classifier(x)


def build_resnet50(num_classes):
    model = resnet50(weights=None)
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 256), nn.BatchNorm1d(256),
        nn.ReLU(inplace=True), nn.Dropout(0.35), nn.Linear(256, num_classes),
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
    normalized_classes = [str(n).strip().casefold().replace(" ", "") for n in checkpoint_classes]
    expected_classes   = [n.casefold().replace(" ", "") for n in CLASS_NAMES]
    if normalized_classes != expected_classes:
        raise ValueError(f"Unexpected class order. Expected {CLASS_NAMES}; found {checkpoint_classes}.")

    num_classes     = checkpoint.get("num_classes", len(CLASS_NAMES))
    best_model_name = checkpoint.get("best_model_name", "CustomCNN")
    if best_model_name == "Custom CNN": best_model_name = "CustomCNN"
    if best_model_name not in {"CustomCNN", "ResNet50", "EfficientNet-B0"}:
        raise ValueError(f"Unsupported architecture: {best_model_name}")

    if best_model_name == "ResNet50": model = build_resnet50(num_classes)
    elif best_model_name == "EfficientNet-B0": model = build_efficientnet_b0(num_classes)
    else: model = CustomCNN(num_classes)

    state_dict = checkpoint.get("model_state_dict", checkpoint)
    clean_sd   = {k[len("module."):] if k.startswith("module.") else k: v for k, v in state_dict.items()}
    model.load_state_dict(clean_sd, strict=True)
    model.to(DEVICE); model.eval()
    return model, CLASS_NAMES, best_model_name


# ─────────────────────────────────────────────────────────────────────────────
# INFERENCE
# ─────────────────────────────────────────────────────────────────────────────

def predict_image(image, model, class_names):
    if model is None: raise RuntimeError("Neural engine unavailable.")
    model.eval()
    t0 = time.perf_counter()
    tensor = test_transforms(image).unsqueeze(0).to(DEVICE)
    preprocess_ms = (time.perf_counter() - t0) * 1000

    if DEVICE.type == "cuda": torch.cuda.synchronize(DEVICE)
    t1 = time.perf_counter()
    with torch.inference_mode():
        probs = F.softmax(model(tensor), dim=1)[0].detach().cpu().numpy()
    if DEVICE.type == "cuda": torch.cuda.synchronize(DEVICE)
    inference_ms = (time.perf_counter() - t1) * 1000

    idx = int(np.argmax(probs))
    return (
        class_names[idx], float(probs[idx] * 100),
        {class_names[i]: float(probs[i] * 100) for i in range(len(class_names))},
        preprocess_ms, inference_ms,
    )


def mc_dropout_predict(image, model, class_names, n_samples=MC_SAMPLES):
    """MC Dropout. Restores model.eval() even if an exception occurs."""
    if model is None:
        return None

    def _enable_dropout(m):
        if isinstance(m, (nn.Dropout, nn.Dropout2d)):
            m.train()

    model.eval()
    try:
        model.apply(_enable_dropout)
        tensor = test_transforms(image).unsqueeze(0).to(DEVICE)
        mc_preds = []
        with torch.no_grad():
            for _ in range(n_samples):
                mc_preds.append(F.softmax(model(tensor), dim=1)[0].detach().cpu().numpy())
    finally:
        model.eval()

    mc_preds   = np.stack(mc_preds)
    mean_probs = mc_preds.mean(axis=0)
    std_probs  = mc_preds.std(axis=0)
    pred_idx   = int(np.argmax(mean_probs))
    uncertainty = float(std_probs[pred_idx])
    band, color = uncertainty_band(uncertainty)
    return {
        "mean_probs":  {class_names[i]: float(mean_probs[i]*100) for i in range(len(class_names))},
        "std_probs":   {class_names[i]: float(std_probs[i]*100)  for i in range(len(class_names))},
        "uncertainty": uncertainty, "band": band, "color": color,
        "prediction":  class_names[pred_idx], "confidence": float(mean_probs[pred_idx]*100),
    }


# ─────────────────────────────────────────────────────────────────────────────
# GRAD-CAM + GRAD-CAM++
# ─────────────────────────────────────────────────────────────────────────────

def _get_target_layer(model, model_name):
    if model_name == "ResNet50": return model.layer4[-1].conv3
    elif model_name == "EfficientNet-B0": return model.features[-1]
    else: return model.features[4].block[0]


def _cam_to_heatmap(cam_raw, _):
    cam = np.maximum(cam_raw, 0)
    cam -= cam.min()
    if cam.max() > 0: cam /= cam.max()
    return cam


def generate_gradcam(image, model, model_name):
    if model is None: raise RuntimeError("Neural engine unavailable.")
    model.eval()
    activations, gradients = [], []
    tl  = _get_target_layer(model, model_name)
    fwd = tl.register_forward_hook(lambda m,i,o: activations.append(o.detach()))
    bwd = tl.register_full_backward_hook(lambda m,gi,go: gradients.append(go[0].detach()))
    fig = None
    try:
        tensor = test_transforms(image).unsqueeze(0).to(DEVICE)
        model.zero_grad(); out = model(tensor)
        out[0, int(out.argmax(dim=1).item())].backward()
        if not activations or not gradients: raise RuntimeError("Hooks failed.")
        act = activations[0][0]; grd = gradients[0][0]
        w   = grd.mean(dim=(1,2), keepdim=True)
        cam = _cam_to_heatmap((w * act).sum(dim=0).detach().cpu().numpy(), None)
        orig = np.asarray(image).astype(np.float32) / 255.0
        fig, ax = plt.subplots(figsize=(5,5), dpi=120); fig.patch.set_alpha(0)
        ax.imshow(orig); ax.imshow(cam, cmap="jet", alpha=0.44, extent=(0,orig.shape[1],orig.shape[0],0))
        ax.axis("off"); fig.tight_layout(pad=0)
        return fig
    except Exception:
        if fig: plt.close(fig)
        raise
    finally:
        try: fwd.remove()
        except Exception: pass
        try: bwd.remove()
        except Exception: pass


def generate_gradcam_pp(image, model, model_name):
    if model is None: raise RuntimeError("Neural engine unavailable.")
    model.eval()
    activations, gradients = [], []
    tl  = _get_target_layer(model, model_name)
    fwd = tl.register_forward_hook(lambda m,i,o: activations.append(o.detach()))
    bwd = tl.register_full_backward_hook(lambda m,gi,go: gradients.append(go[0].detach()))
    fig = None
    try:
        tensor = test_transforms(image).unsqueeze(0).to(DEVICE)
        model.zero_grad(); out = model(tensor)
        out[0, int(out.argmax(dim=1).item())].backward(retain_graph=True)
        if not activations or not gradients: raise RuntimeError("Hooks failed.")
        act = activations[0][0].cpu().numpy(); grd = gradients[0][0].cpu().numpy()
        an  = grd ** 2
        ad = 2*grd**2 + (act*grd**3).sum(axis=(1,2),keepdims=True) + 1e-8
        alpha = an/ad
        w = (alpha * np.maximum(grd,0)).sum(axis=(1,2))
        cam_raw = np.zeros(act.shape[1:], dtype=np.float32)
        for ww,aa in zip(w,act): cam_raw += ww * aa
        cam  = _cam_to_heatmap(cam_raw, None)
        orig = np.asarray(image).astype(np.float32) / 255.0
        fig, ax = plt.subplots(figsize=(5,5), dpi=120); fig.patch.set_alpha(0)
        ax.imshow(orig); ax.imshow(cam, cmap="inferno", alpha=0.46, extent=(0,orig.shape[1],orig.shape[0],0))
        ax.axis("off"); fig.tight_layout(pad=0)
        return fig
    except Exception:
        if fig: plt.close(fig)
        raise
    finally:
        try: fwd.remove()
        except Exception: pass
        try: bwd.remove()
        except Exception: pass


def explanation_agreement(image, model, model_name):
    """Pearson correlation between Grad-CAM and Grad-CAM++ heatmaps."""
    if model is None:
        return None
    model.eval()
    handles = []
    try:
        tl = _get_target_layer(model, model_name)
        acts_gc, grds_gc, acts_pp, grds_pp = [], [], [], []
        fwd1 = tl.register_forward_hook(lambda m,i,o: acts_gc.append(o.detach()))
        bwd1 = tl.register_full_backward_hook(lambda m,gi,go: grds_gc.append(go[0].detach()))
        handles.extend((fwd1, bwd1))

        tensor = test_transforms(image).unsqueeze(0).to(DEVICE)
        model.zero_grad()
        out = model(tensor)
        idx = int(out.argmax(dim=1).item())
        out[0, idx].backward(retain_graph=True)

        fwd1.remove(); bwd1.remove()

        if not acts_gc or not grds_gc:
            return None

        act_gc = acts_gc[0][0]; grd_gc = grds_gc[0][0]
        w_gc   = grd_gc.mean(dim=(1,2), keepdim=True)
        cam_gc = F.relu((w_gc * act_gc).sum(dim=0)).detach().cpu().numpy()
        cam_gc /= (cam_gc.max() + 1e-8)

        fwd2 = tl.register_forward_hook(lambda m,i,o: acts_pp.append(o.detach()))
        bwd2 = tl.register_full_backward_hook(lambda m,gi,go: grds_pp.append(go[0].detach()))
        handles.extend((fwd2, bwd2))

        model.zero_grad()
        out2 = model(tensor)
        out2[0, idx].backward()
        fwd2.remove(); bwd2.remove()

        if not acts_pp or not grds_pp:
            return None

        act_pp = acts_pp[0][0].cpu().numpy()
        grd_pp = grds_pp[0][0].cpu().numpy()
        an  = grd_pp**2
        ad  = 2*grd_pp**2 + (act_pp*grd_pp**3).sum(axis=(1,2),keepdims=True) + 1e-8
        w_pp = (an/ad * np.maximum(grd_pp,0)).sum(axis=(1,2))
        cam_pp = np.zeros(act_pp.shape[1:], dtype=np.float32)
        for ww, aa in zip(w_pp, act_pp):
            cam_pp += ww * aa
        cam_pp = np.maximum(cam_pp, 0)
        cam_pp /= (cam_pp.max() + 1e-8)

        f1, f2 = cam_gc.flatten(), cam_pp.flatten()
        if f1.std() < 1e-8 or f2.std() < 1e-8:
            return None
        corr = float(np.corrcoef(f1, f2)[0, 1])
        if not np.isfinite(corr):
            return None
        return float(max(0.0, min(1.0, corr)))
    except Exception:
        return None
    finally:
        for h in handles:
            try: h.remove()
            except Exception: pass


# ─────────────────────────────────────────────────────────────────────────────
# CHARTING
# ─────────────────────────────────────────────────────────────────────────────

def _dark_fig(w=6, h=2.8):
    fig, ax = plt.subplots(figsize=(w, h))
    bg = "#0f172a"
    fig.patch.set_facecolor(bg); ax.set_facecolor(bg)
    for side in ("top", "right"): ax.spines[side].set_visible(False)
    for side in ("left", "bottom"): ax.spines[side].set_color("#1e2d42")
    ax.tick_params(colors="#94a3b8", labelsize=8)
    return fig, ax


def plot_confidence_trend(history):
    if len(history) < 2: return None
    fig, ax = _dark_fig()
    confs = [h["confidence"] for h in history]
    xs = list(range(1, len(history) + 1))
    ax.plot(xs, confs, marker="o", lw=2, ms=4, color="#0891b2")
    ax.fill_between(xs, confs, alpha=0.09, color="#0891b2")
    ax.set_ylim(0, 105)
    ax.set_xlabel("Analysis #", color="#94a3b8", fontsize=9)
    ax.set_ylabel("Confidence %", color="#94a3b8", fontsize=9)
    ax.grid(True, alpha=0.09, ls="--")
    fig.tight_layout(pad=0.5)
    return fig


def plot_latency_trend(history):
    if len(history) < 2: return None
    fig, ax = _dark_fig()
    lats = [h.get("latency_ms", 0) for h in history]
    ax.plot(range(1, len(history) + 1), lats, marker="o", lw=2, color="#0891b2")
    ax.set_xlabel("Analysis #", color="#94a3b8", fontsize=9)
    ax.set_ylabel("Latency ms", color="#94a3b8", fontsize=9)
    ax.grid(True, alpha=0.09, ls="--")
    fig.tight_layout(pad=0.5)
    return fig


def plot_uncertainty_history(history):
    data = [(i + 1, h["uncertainty"]) for i, h in enumerate(history) if "uncertainty" in h]
    if len(data) < 2: return None
    fig, ax = _dark_fig()
    xs, ys = zip(*data)
    ax.plot(xs, ys, marker="s", lw=2, ms=4, color="#7c3aed")
    ax.fill_between(xs, ys, alpha=0.08, color="#7c3aed")
    ax.axhline(0.12, color="#d97706", lw=1, ls="--", alpha=0.55, label="Moderate threshold")
    ax.axhline(0.22, color="#dc2626", lw=1, ls="--", alpha=0.55, label="Low reliability")
    ax.set_xlabel("Analysis #", color="#94a3b8", fontsize=9)
    ax.set_ylabel("MC Uncertainty σ", color="#94a3b8", fontsize=9)
    ax.grid(True, alpha=0.09, ls="--")
    ax.legend(fontsize=7, labelcolor="#94a3b8", facecolor="#0f172a", edgecolor="#1e2d42")
    fig.tight_layout(pad=0.5)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE — deepcopy to avoid shared mutable references
# ─────────────────────────────────────────────────────────────────────────────

DEFAULTS = {
    "nav": "Home", "last_result": None, "last_image": None,
    "gradcam_image": None, "gradcam_pp_image": None,
    "mc_result": None, "agreement_score": None,
    "prediction_history": [], "activity_log": [],
    "live_session_start": None, "live_predictions_count": 0,
    "live_avg_confidence": 0.0, "live_class_counts": {},
    "live_throughput": 0.0, "live_last_confidence": 0.0,
    "live_latency_ms": 0.0, "live_inference_running": False,
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = copy.deepcopy(v)


def clear_prediction_history():
    for fk in ("gradcam_image", "gradcam_pp_image"):
        old = st.session_state.get(fk)
        if old: plt.close(old)
    for k in ["prediction_history", "last_result", "last_image",
              "gradcam_image", "gradcam_pp_image", "mc_result", "agreement_score",
              "live_predictions_count", "live_avg_confidence", "live_last_confidence",
              "live_class_counts", "live_throughput", "live_latency_ms"]:
        st.session_state[k] = copy.deepcopy(DEFAULTS[k])


def log_activity(message, level="info"):
    ts = datetime.now().strftime("%H:%M:%S")
    log = list(st.session_state.activity_log)
    log.append({"timestamp": ts, "message": message, "level": level})
    if len(log) > 60:
        log = log[-60:]
    st.session_state.activity_log = log


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
        el = (datetime.now() - st.session_state.live_session_start).total_seconds()
        if el > 0:
            st.session_state.live_throughput = len(history) / el * 60.0


# ─────────────────────────────────────────────────────────────────────────────
# MODEL INIT
# ─────────────────────────────────────────────────────────────────────────────
model = class_names = model_name = model_error = None
try:
    model, class_names, model_name = load_model(MODEL_PATH)
except Exception as exc:
    model_error = str(exc)


# ─────────────────────────────────────────────────────────────────────────────
# NAV ITEMS CONFIG
# ─────────────────────────────────────────────────────────────────────────────
NAV_GROUPS = [
    ("Main", [
        ("Home",         Icons.home(16, "#94a3b8"),       "Home"),
        ("MRI Analysis", Icons.microscope(16, "#94a3b8"), "MRI Analysis"),
        ("Dashboard",    Icons.chart(16, "#94a3b8"),      "Dashboard"),
    ]),
    ("Analytics", [
        ("History",  Icons.history(16, "#94a3b8"),  "History"),
        ("Grad-CAM", Icons.heatmap(16, "#94a3b8"),  "Grad-CAM"),
        ("XAI Lab",  Icons.lab(16, "#94a3b8"),      "XAI Lab"),
    ]),
    ("System", [
        ("Settings", Icons.settings(16, "#94a3b8"), "Settings"),
    ]),
]

PAGE_LABELS = {
    "Home": "Home", "MRI Analysis": "MRI Analysis",
    "Dashboard": "Dashboard", "History": "History",
    "Grad-CAM": "Grad-CAM", "XAI Lab": "XAI Lab", "Settings": "Settings",
}


# ─────────────────────────────────────────────────────────────────────────────
# LIVE TICKER
# ─────────────────────────────────────────────────────────────────────────────

def render_live_ticker():
    counts = st.session_state.live_class_counts or {}
    items  = list(counts.items())
    text   = " · ".join(f"<b>{n}</b>: {v}" for n, v in items) if items else "<b>Awaiting first scan</b>"
    mc  = st.session_state.mc_result
    unc = f" · σ={mc['uncertainty']:.3f} ({mc['band']})" if mc else ""
    _render_html(f"""
    <style>
    .tw{{overflow:hidden;border-radius:8px;border:1px solid rgba(255,255,255,0.06);
    background:rgba(15,23,42,0.9);padding:.4rem 0;}}
    .tt{{display:inline-block;white-space:nowrap;padding-left:100%;
    animation:mq 28s linear infinite;color:#94a3b8;font:500 .72rem 'JetBrains Mono',monospace}}
    .lp{{display:inline-block;width:5px;height:5px;border-radius:50%;background:#0891b2;margin-right:8px;vertical-align:middle}}
    @keyframes mq{{0%{{transform:translateX(0);}}100%{{transform:translateX(-100%);}}}}
    </style>
    <div class="tw"><div class="tt"><span class="lp"></span>LIVE ·
    {text} · Throughput: {st.session_state.live_throughput:.2f}/min ·
    Last Conf: {st.session_state.live_last_confidence:.1f}% ·
    Avg Conf: {st.session_state.live_avg_confidence:.1f}%{unc}
    </div></div>""", height=33)


# ─────────────────────────────────────────────────────────────────────────────
# STICKY HEADER
# ─────────────────────────────────────────────────────────────────────────────

def render_sticky_header():
    counts = st.session_state.live_class_counts or {}
    if counts:
        items_html = " ".join(f"<span class='hd-tick'><b>{k}</b>: {v}</span>" for k, v in counts.items())
    else:
        items_html = "<span class='hd-tick hd-tick-idle'>Awaiting first scan…</span>"

    mc  = st.session_state.mc_result
    nav = PAGE_LABELS.get(st.session_state.nav, st.session_state.nav)
    eng = model_name or "—"
    eng_ok = model_error is None and MODEL_PATH.exists()
    st_cls = "hd-status-ok" if eng_ok else "hd-status-err"
    st_txt = "Online" if eng_ok else "Offline"
    dev_tag = "GPU" if DEVICE.type == "cuda" else "CPU"
    mc_str = f"<span class='hd-tick'>σ=<b>{mc['uncertainty']:.3f}</b></span>" if mc else ""

    brain_svg = Icons.brain(18, "#22d3ee")

    st.markdown(safe_html(f"""
    <div class="sticky-header">
        <div class="hd-brand">
            <div class="hd-brand-icon">
                {brain_svg}
                <span class="hd-brand-dot"></span>
            </div>
            <span class="hd-brand-name">NeuroLens AI</span>
        </div>
        <div class="hd-divider"></div>
        <div class="hd-ticker">
            <div class="hd-ticker-inner">
                <span class="hd-live-badge">LIVE</span>
                {items_html}
                <span class="hd-tick">Engine: <b>{eng}</b></span>
                <span class="hd-tick">Device: <b>{dev_tag}</b></span>
                {mc_str}
            </div>
        </div>
        <div class="hd-divider"></div>
        <div class="hd-status {st_cls}">
            <span class="hd-status-dot"></span>{st_txt}
        </div>
        <div class="hd-page">
            <span class="hd-page-dot"></span>{nav}
        </div>
    </div>"""), unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# LIVE PROBABILITY BARS
# ─────────────────────────────────────────────────────────────────────────────

def render_live_probability_animation(result, mc_result=None):
    items = list(result["probabilities"].items())
    top   = result["prediction"]
    conf  = result["confidence"]

    mc_rows = ""
    if mc_result:
        mc_std = mc_result["std_probs"]
        mc_rows = "".join(
            f'<div class="lp-row"><div class="lp-label">{n}'
            f'<span style="color:#a78bfa;font-family:\'JetBrains Mono\',monospace;font-size:.7rem"> ±{mc_std.get(n,0):.1f}%</span></div>'
            f'<div class="lp-track"><div class="lp-fill lp-mc" data-target="{p:.2f}" style="width:0%"></div></div>'
            f'<div class="lp-val">{p:.1f}%</div></div>'
            for n, p in mc_result["mean_probs"].items()
        )

    bars = "".join(
        f'<div class="lp-row"><div class="lp-label">{n}</div>'
        f'<div class="lp-track"><div class="lp-fill" data-target="{p:.2f}" style="width:0%"></div></div>'
        f'<div class="lp-val">{p:.1f}%</div></div>'
        for n, p in items
    )

    unc_html = ""
    if mc_result:
        band = mc_result["band"]; uval = mc_result["uncertainty"]; color = mc_result["color"]
        unc_html = (
            f'<div style="margin-top:.65rem;padding:.5rem .75rem;border-radius:8px;'
            f'background:rgba(124,58,237,0.05);border:1px solid rgba(124,58,237,0.18)">'
            f'<span style="font-size:.58rem;color:#a78bfa;text-transform:uppercase;letter-spacing:.12em;font-weight:700;font-family:\'Inter\',sans-serif">MC Uncertainty</span>'
            f'<span style="float:right;font-weight:700;color:{color};font-family:\'JetBrains Mono\',monospace;font-size:.74rem">σ={uval:.4f} · {band}</span></div>'
        )

    h = 55 + 34 * len(items) + (34 * len(items) + 80 if mc_result else 0)
    _render_html(f"""
    <style>
    .lp-wrap{{padding:1rem 1.25rem;border-radius:14px;border:1px solid rgba(255,255,255,0.06);background:#0f172a;font-family:Inter,sans-serif;color:#e2e8f0}}
    .lp-head{{display:flex;justify-content:space-between;align-items:center;margin-bottom:.7rem;padding-bottom:.5rem;border-bottom:1px solid rgba(255,255,255,0.055)}}
    .lp-head-title{{font-size:.58rem;color:#22d3ee;text-transform:uppercase;letter-spacing:.12em;font-weight:700}}
    .lp-head-pred{{font-family:Sora,sans-serif;font-size:.85rem;font-weight:700;color:#e2e8f0;letter-spacing:-.01em}}
    .lp-row{{display:flex;align-items:center;gap:.6rem;margin:.35rem 0}}
    .lp-label{{min-width:132px;font-size:.74rem;color:#94a3b8;font-weight:500}}
    .lp-track{{flex:1;height:5px;border-radius:999px;background:rgba(255,255,255,0.05);overflow:hidden}}
    .lp-fill{{height:100%;width:0%;background:#0891b2;border-radius:999px;transition:width 1.1s cubic-bezier(.2,.8,.2,1)}}
    .lp-mc{{background:#7c3aed!important}}
    .lp-val{{min-width:52px;text-align:right;font-family:'JetBrains Mono',monospace;font-weight:600;color:#e2e8f0;font-size:.72rem}}
    .lp-sec{{font-size:.56rem;color:#4b5869;text-transform:uppercase;letter-spacing:.12em;font-weight:700;margin:.65rem 0 .28rem}}
    </style>
    <div class="lp-wrap">
        <div class="lp-head">
            <span class="lp-head-title">Live Probability Stream</span>
            <span class="lp-head-pred">{top} · {conf:.1f}%</span>
        </div>
        <div class="lp-sec">Standard Inference</div>
        {bars}
        {"<div class='lp-sec'>MC Dropout · Bayesian</div>" + mc_rows if mc_result else ""}
        {unc_html}
    </div>
    <script>
    requestAnimationFrame(()=>{{
        document.querySelectorAll('.lp-fill').forEach(el=>{{
            const t=parseFloat(el.getAttribute('data-target'));
            requestAnimationFrame(()=>{{el.style.width=t+'%';}});
        }});
    }});
    </script>""", height=h)


# ─────────────────────────────────────────────────────────────────────────────
# ACTIVITY FEED
# ─────────────────────────────────────────────────────────────────────────────

def render_activity_feed():
    log = st.session_state.activity_log[-14:][::-1]
    if not log:
        st.markdown("<div style='color:#4b5869;font-size:.78rem;padding:.5rem'>No activity yet.</div>", unsafe_allow_html=True)
        return
    rows = "".join(
        f"<div class='act-row act-{e['level']}'>"
        f"<span class='act-time'>{e['timestamp']}</span>"
        f"<span class='act-msg'>{e['message']}</span></div>"
        for e in log
    )
    st.markdown(f"<div class='act-feed'>{rows}</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    history     = st.session_state.prediction_history
    total_scans = len(history)
    avg_conf    = float(np.mean([x["confidence"] for x in history])) if history else 0.0
    active_nav  = st.session_state.nav
    engine_ok   = MODEL_PATH.exists() and model_error is None

    if st.session_state.live_session_start:
        elapsed = int((datetime.now() - st.session_state.live_session_start).total_seconds())
        if elapsed < 60:
            uptime = f"{elapsed}s"
        elif elapsed < 3600:
            uptime = f"{elapsed // 60}m"
        else:
            hrs  = elapsed // 3600
            mins = (elapsed % 3600) // 60
            uptime = f"{hrs}h{mins}m"
    else:
        uptime = "—"

    # ── Brand ──
    brain_large = Icons.brain(20, "#22d3ee")
    st.markdown(safe_html(f"""
    <div class="sb-brand">
        <div class="sb-brand-icon">
            {brain_large}
            <span class="sb-brand-pulse"></span>
        </div>
        <div class="sb-brand-text">
            <div class="sb-brand-name">NeuroLens <span class="sb-brand-ai">AI</span><span class="sb-brand-ver">v3.1</span></div>
            <div class="sb-brand-sub">Neurodiagnostic Intelligence</div>
        </div>
    </div>"""), unsafe_allow_html=True)

    # ── Session pill ──
    dot_color = "var(--success-hi)" if engine_ok else "var(--danger-hi)"
    st.markdown(safe_html(f"""
    <div class="sb-session">
        <span class="sb-session-dot" style="background:{dot_color};box-shadow:0 0 8px {dot_color}"></span>
        <span class="sb-session-txt">{active_nav}</span>
        <span class="sb-session-time">{uptime}</span>
    </div>"""), unsafe_allow_html=True)

    # ── Nav badges ──
    badge_css_parts = []
    if total_scans > 0:
        badge_css_parts.append(f"""
        [data-testid="stSidebar"] .st-key-nav_mri_analysis button::after {{
            content: "{total_scans}";
            position: absolute;
            right: 0.6rem;
            top: 50%;
            transform: translateY(-50%);
            padding: 0.05rem 0.42rem;
            border-radius: 999px;
            background: rgba(8,145,178,0.10);
            color: #22d3ee;
            border: 1px solid rgba(8,145,178,0.28);
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.56rem;
            font-weight: 700;
            line-height: 1.4;
            letter-spacing: 0.02em;
        }}
        [data-testid="stSidebar"] .st-key-nav_history button::after {{
            content: "{total_scans}";
            position: absolute;
            right: 0.6rem;
            top: 50%;
            transform: translateY(-50%);
            padding: 0.05rem 0.42rem;
            border-radius: 999px;
            background: rgba(8,145,178,0.10);
            color: #22d3ee;
            border: 1px solid rgba(8,145,178,0.28);
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.56rem;
            font-weight: 700;
            line-height: 1.4;
        }}
        """)

    agree_vals = [h.get("agreement_score") for h in history if h.get("agreement_score") is not None]
    if agree_vals:
        avg_agree = float(np.mean(agree_vals))
        badge_css_parts.append(f"""
        [data-testid="stSidebar"] .st-key-nav_xailab button::after {{
            content: "{avg_agree:.2f}";
            position: absolute;
            right: 0.6rem;
            top: 50%;
            transform: translateY(-50%);
            padding: 0.05rem 0.42rem;
            border-radius: 999px;
            background: rgba(124,58,237,0.10);
            color: #a78bfa;
            border: 1px solid rgba(124,58,237,0.28);
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.56rem;
            font-weight: 700;
            line-height: 1.4;
        }}
        """)

    if badge_css_parts:
        st.markdown(f"<style>{''.join(badge_css_parts)}</style>", unsafe_allow_html=True)

    # ── Nav groups ──
    active_css_parts = []
    for group_name, items in NAV_GROUPS:
        st.markdown(f'<div class="sb-nav-group">{group_name}</div>', unsafe_allow_html=True)
        for label, icon_svg, key in items:
            is_active = active_nav == key
            slug = key.lower().replace("-", "").replace(" ", "_")

            if st.button(label, key=f"nav_{slug}", **_stretch()):
                st.session_state.nav = key
                st.rerun()

            if is_active:
                active_css_parts.append(f"""
                [data-testid="stSidebar"] .st-key-nav_{slug} button {{
                    background: linear-gradient(90deg, rgba(8,145,178,0.13), rgba(8,145,178,0.02)) !important;
                    border-color: rgba(8,145,178,0.30) !important;
                    color: #22d3ee !important;
                    font-weight: 600 !important;
                    box-shadow: inset 2px 0 0 0 #0891b2, 0 0 14px rgba(8,145,178,0.10) !important;
                }}
                [data-testid="stSidebar"] .st-key-nav_{slug} button p {{
                    color: #22d3ee !important;
                    font-weight: 600 !important;
                }}
                [data-testid="stSidebar"] .st-key-nav_{slug} button::before {{
                    opacity: 1 !important;
                    background-color: #22d3ee !important;
                }}
                """)

    if active_css_parts:
        st.markdown(f"<style>{''.join(active_css_parts)}</style>", unsafe_allow_html=True)

    # ── Divider ──
    st.markdown(
        '<div style="height:1px;background:var(--line);margin:0.6rem 0.85rem"></div>',
        unsafe_allow_html=True,
    )

    # ── Neural Engine Panel ──
    eng_color = "var(--success-hi)" if engine_ok else "var(--danger-hi)"
    eng_label = "Engine Ready" if engine_ok else "Engine Offline"
    dev_tag   = "CUDA" if DEVICE.type == "cuda" else "CPU"
    dev_color = "#22d3ee" if DEVICE.type == "cuda" else "#94a3b8"

    live_chip = (
        '<div class="sb-engine-live"><span class="sb-engine-live-dot"></span>LIVE</div>'
        if engine_ok else ''
    )

    mc_row = ""
    if st.session_state.mc_result:
        mc = st.session_state.mc_result
        mc_row = f"""
        <div class="sb-engine-confbar" style="margin-top:0.35rem;margin-bottom:0">
            <span style="font-size:0.55rem;color:#a78bfa;font-weight:700;letter-spacing:0.1em;min-width:42px">σ MC</span>
            <span style="font-family:'JetBrains Mono',monospace;font-size:0.62rem;color:#a78bfa;font-weight:600">{mc['uncertainty']:.4f}</span>
            <span style="margin-left:auto;font-size:0.55rem;color:#a78bfa;font-weight:600">{mc['band'].replace(' Reliability','')}</span>
        </div>
        """

    conf_pct = max(0.0, min(100.0, avg_conf))
    st.markdown(safe_html(f"""
    <div class="sb-engine">
        <div class="sb-engine-head">
            <span class="sb-engine-title">Neural Engine</span>
            {live_chip}
        </div>
        <div class="sb-engine-status">
            <span style="width:6px;height:6px;border-radius:50%;background:{eng_color};box-shadow:0 0 8px {eng_color};flex-shrink:0;display:inline-block"></span>
            <span style="color:{eng_color}">{eng_label}</span>
            <span style="margin-left:auto;font-size:0.55rem;color:{dev_color};font-weight:700;font-family:'JetBrains Mono',monospace;padding:0.05rem 0.4rem;border-radius:4px;border:1px solid {dev_color}30;background:{dev_color}0f;letter-spacing:0.08em">{dev_tag}</span>
        </div>
        <div class="sb-engine-confbar">
            <span style="font-size:0.55rem;color:var(--text-3);font-weight:700;letter-spacing:0.1em;min-width:42px">CONF</span>
            <div class="sb-engine-confbar-track">
                <div class="sb-engine-confbar-fill" style="width:{conf_pct:.1f}%"></div>
            </div>
            <span class="sb-engine-confbar-lbl">{conf_pct:.0f}%</span>
        </div>
        {mc_row}
        <div class="sb-engine-grid">
            <div class="sb-mini sb-mini-white"><div class="sb-mini-lbl">SCANS</div><div class="sb-mini-val">{total_scans}</div></div>
            <div class="sb-mini sb-mini-cyan"><div class="sb-mini-lbl">DEVICE</div><div class="sb-mini-val">{dev_tag}</div></div>
            <div class="sb-mini sb-mini-violet" style="grid-column:1 / -1"><div class="sb-mini-lbl">MODEL</div><div class="sb-mini-val">{model_name or '—'}</div></div>
        </div>
    </div>"""), unsafe_allow_html=True)

    # ── Quick Actions ──
    st.markdown('<div class="sb-nav-group">Quick Actions</div>', unsafe_allow_html=True)

    if st.button("Clear History", key="sb_clear", **_stretch()):
        st.session_state.confirm_clear = True
    if st.session_state.get("confirm_clear", False):
        st.markdown(safe_html(f"""
        <div class="sb-confirm">
            <div class="sb-confirm-title">{Icons.alert_triangle(12, "#fbbf24")} Clear all records?</div>
            <div class="sb-confirm-text">All session diagnostic history will be removed.</div>
        </div>"""), unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        if c1.button("Confirm", key="sb_clear_yes", **_stretch()):
            clear_prediction_history()
            st.session_state.confirm_clear = False
            st.rerun()
        if c2.button("Cancel", key="sb_clear_no", **_stretch()):
            st.session_state.confirm_clear = False

    if st.button("Reset Session", key="sb_reset", **_stretch()):
        st.session_state.confirm_reset = True
    if st.session_state.get("confirm_reset", False):
        st.markdown(safe_html(f"""
        <div class="sb-confirm sb-confirm-danger">
            <div class="sb-confirm-title">{Icons.alert_triangle(12, "#f87171")} Reset entire session?</div>
            <div class="sb-confirm-text">This clears all results, history, and logs.</div>
        </div>"""), unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        if c1.button("Confirm", key="sb_reset_yes", **_stretch()):
            for fk in ("gradcam_image", "gradcam_pp_image"):
                old = st.session_state.get(fk)
                if old: plt.close(old)
            for k, v in DEFAULTS.items():
                st.session_state[k] = copy.deepcopy(v)
            st.rerun()
        if c2.button("Cancel", key="sb_reset_no", **_stretch()):
            st.session_state.confirm_reset = False

    # ── Footer warning ──
    alert_svg = Icons.alert_triangle(13, "#d97706")
    st.markdown(safe_html(f"""
    <div class="sb-warning">
        {alert_svg}
        <span>Research prototype only. Not for clinical use.</span>
    </div>"""), unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# ROUTER
# ─────────────────────────────────────────────────────────────────────────────
render_sticky_header()
nav = st.session_state.nav


# ══════════════════════════════════════════════════════════════════════════════
# HOME
# ══════════════════════════════════════════════════════════════════════════════

if nav == "Home":
    render_live_ticker()

    scan_lg = Icons.scan(18, "#22d3ee")
    st.markdown(safe_html(f"""
    <div class="hero">
        <div class="hero-eyebrow">
            {scan_lg}
            AI-Powered Brain MRI Analysis · Research Prototype
        </div>
        <h1>NeuroLens AI</h1>
        <p class="hero-sub">Deep Learning neuroimaging with dual XAI explainability, Bayesian uncertainty estimation, and real-time inference diagnostics.</p>
        <div class="hero-divider"></div>
    </div>"""), unsafe_allow_html=True)

    _l, _c, _r = st.columns([1.8, 1, 1.8])
    with _c:
        if st.button("Run MRI Analysis", type="primary", key="home_cta", **_stretch()):
            st.session_state.nav = "MRI Analysis"
            st.rerun()

    st.markdown(
        "<div style='text-align:center;color:#4b5869;font-size:.74rem;margin-top:.5rem;line-height:1.5'>"
        "Research prototype for education and research. Predictions are not medical diagnoses."
        "</div>", unsafe_allow_html=True
    )

    st.write("")
    h2 = st.session_state.prediction_history
    mc1, mc2, mc3, mc4 = st.columns(4)
    pulse_svg = Icons.pulse(18, "#22d3ee")
    tgt_svg   = Icons.target(18, "#22d3ee")
    zap_svg   = Icons.zap(18, "#22d3ee")
    cpu_svg   = Icons.cpu(18, "#22d3ee")
    metrics = [
        (pulse_svg, st.session_state.live_predictions_count, "Live Scans"),
        (tgt_svg,   f"{st.session_state.live_avg_confidence:.1f}%", "Avg. Confidence"),
        (zap_svg,   f"{st.session_state.live_throughput:.2f}/m", "Throughput"),
        (cpu_svg,   f"{np.mean([x.get('latency_ms',0) for x in h2]):.0f} ms" if h2 else "—", "Avg. Inference"),
    ]
    for col, (icon, val, lbl) in zip([mc1, mc2, mc3, mc4], metrics):
        with col:
            st.markdown(safe_html(f"""
            <div class="metric-card">
                <div class="metric-icon-wrap">{icon}</div>
                <div class="metric-value">{val}</div>
                <div class="metric-label">{lbl}</div>
            </div>"""), unsafe_allow_html=True)

    st.write("")
    c1, c2, c3, c4 = st.columns(4)
    micro_svg = Icons.microscope(22, "#22d3ee")
    shld_svg  = Icons.shield(22, "#22d3ee")
    eye_svg   = Icons.eye(22, "#22d3ee")
    zap2_svg  = Icons.zap(22, "#22d3ee")
    cards = [
        (micro_svg, "Brain MRI Classification",  "Four-class classification: Glioma, Meningioma, No Tumor, Pituitary using state-of-the-art CNN architectures."),
        (shld_svg,  "MC Dropout Uncertainty",    "Bayesian uncertainty estimation via Monte Carlo Dropout. Reliability bands quantify how confident the model really is."),
        (eye_svg,   "Dual XAI (Grad-CAM++)",     "Side-by-side Grad-CAM and Grad-CAM++ visualizations with an agreement score showing heatmap consistency."),
        (zap2_svg,  "Real-Time Inference",       "Detailed timing breakdown: preprocessing, inference, and XAI generation with CUDA-accurate latency measurement."),
    ]
    for col, (icon, title, desc) in zip([c1, c2, c3, c4], cards):
        with col:
            st.markdown(safe_html(f"""
            <div class="info-card">
                <div class="info-card-icon">{icon}</div>
                <h3>{title}</h3>
                <p>{desc}</p>
            </div>"""), unsafe_allow_html=True)

    st.write("")
    st.subheader("How NeuroLens Works")
    steps = [
        ("1", "Upload MRI", "JPG, PNG, WEBP"),
        ("2", "Preprocess", "Resize · Normalize"),
        ("3", "Inference", "CNN forward pass"),
        ("4", "MC Dropout", "20 stochastic passes"),
        ("5", "Dual XAI", "Grad-CAM + Grad-CAM++"),
        ("6", "Agreement", "Explanation correlation"),
        ("7", "Report", "Download TXT / PNG"),
    ]
    scols = st.columns(len(steps))
    for col, (num, title, desc) in zip(scols, steps):
        with col:
            st.markdown(safe_html(f"""
            <div class="step-card">
                <div class="step-num">{num}</div>
                <div class="step-title">{title}</div>
                <div class="step-desc">{desc}</div>
            </div>"""), unsafe_allow_html=True)

    st.write("")
    if model_error:
        err_svg = Icons.x_circle(18, "#ef4444")
        st.markdown(safe_html(f"""
        <div class="info-card" style="border-color:rgba(220,38,38,.28);background:rgba(220,38,38,.04)">
            <div style="display:flex;align-items:center;gap:.5rem;margin-bottom:.5rem">{err_svg}
                <h3 style="color:#f87171;margin:0">Neural Engine Unavailable</h3></div>
            <p>Expected: <code style="font-family:var(--font-mono)">{MODEL_PATH}</code></p>
        </div>"""), unsafe_allow_html=True)
        with st.expander("Technical Details"):
            st.code(model_error)
    else:
        chk_svg = Icons.check_circle(18, "#34d399")
        st.markdown(safe_html(f"""
        <div class="info-card" style="border-color:rgba(5,150,105,.25);background:rgba(5,150,105,.04)">
            <div style="display:flex;align-items:center;gap:.5rem;margin-bottom:.85rem">{chk_svg}
                <h3 style="color:#34d399;margin:0">Neural Engine Ready</h3></div>
            <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:.65rem">
                <div style="background:rgba(0,0,0,.15);padding:.6rem .8rem;border-radius:9px;border:1px solid rgba(5,150,105,.18)">
                    <div style="font-size:.58rem;color:#4b5869;letter-spacing:.08em;font-weight:600;margin-bottom:.2rem">ARCHITECTURE</div>
                    <div style="font-size:.85rem;font-weight:700;color:#e2e8f0;font-family:var(--font-mono)">{model_name}</div>
                </div>
                <div style="background:rgba(0,0,0,.15);padding:.6rem .8rem;border-radius:9px;border:1px solid rgba(5,150,105,.18)">
                    <div style="font-size:.58rem;color:#4b5869;letter-spacing:.08em;font-weight:600;margin-bottom:.2rem">DEVICE</div>
                    <div style="font-size:.85rem;font-weight:700;color:#e2e8f0;font-family:var(--font-mono)">{DEVICE}</div>
                </div>
                <div style="background:rgba(0,0,0,.15);padding:.6rem .8rem;border-radius:9px;border:1px solid rgba(5,150,105,.18)">
                    <div style="font-size:.58rem;color:#4b5869;letter-spacing:.08em;font-weight:600;margin-bottom:.2rem">CLASSES</div>
                    <div style="font-size:.8rem;font-weight:700;color:#22d3ee">{', '.join(class_names)}</div>
                </div>
                <div style="background:rgba(0,0,0,.15);padding:.6rem .8rem;border-radius:9px;border:1px solid rgba(5,150,105,.18)">
                    <div style="font-size:.58rem;color:#4b5869;letter-spacing:.08em;font-weight:600;margin-bottom:.2rem">MC DROPOUT &amp; XAI</div>
                    <div style="font-size:.8rem;font-weight:700;color:#e2e8f0">{MC_SAMPLES} passes · Grad-CAM + Grad-CAM++</div>
                </div>
            </div>
        </div>"""), unsafe_allow_html=True)

    if st.session_state.last_result:
        st.write("")
        st.subheader("Last Analysis — Live Stream")
        render_live_probability_animation(st.session_state.last_result, mc_result=st.session_state.mc_result)


# ══════════════════════════════════════════════════════════════════════════════
# MRI ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

elif nav == "MRI Analysis":
    micro_h = Icons.microscope(22, "#22d3ee")
    st.markdown(f"<h1 style='display:flex;align-items:center;gap:.5rem'>{micro_h} MRI Diagnostic Analysis</h1>", unsafe_allow_html=True)
    st.caption("Upload a brain MRI scan for AI-powered classification with uncertainty estimation and dual explainability.")
    render_live_ticker()

    if model_error:
        st.error("Neural engine unavailable. Cannot run inference.")
        st.info(f"Expected model: `{MODEL_PATH}`")
        with st.expander("Technical Details"):
            st.code(model_error)
    else:
        up_svg = Icons.upload(26, "#22d3ee")
        st.markdown(safe_html(f"""
        <div class="upload-hero">
            <div class="upload-icon-wrap">{up_svg}</div>
            <div class="upload-title">Upload Brain MRI Scan</div>
            <div class="upload-sub">Drop your scan below or click to browse</div>
            <div class="upload-formats">
                <span class="fmt-badge">JPG</span><span class="fmt-badge">JPEG</span>
                <span class="fmt-badge">PNG</span><span class="fmt-badge">WEBP</span>
                <span class="fmt-badge fmt-badge-muted">≤ 200 MB</span>
            </div>
            <div class="upload-note">
                <span class="upload-note-dot"></span>
                Processed locally — nothing uploaded externally
            </div>
        </div>"""), unsafe_allow_html=True)

        _ul, _uc, _ur = st.columns([1, 2, 1])
        with _uc:
            uploaded_file = st.file_uploader(
                "Upload Brain MRI Scan", type=["jpg", "jpeg", "png", "webp"],
                key="mri_uploader", label_visibility="collapsed",
            )

        image_id = None
        image = None
        if uploaded_file is not None:
            upload_bytes = uploaded_file.getvalue()
            image_id = hashlib.sha256(upload_bytes).hexdigest()
            prev = st.session_state.last_result
            if prev is not None and prev.get("image_id") != image_id:
                for fk in ("gradcam_image", "gradcam_pp_image"):
                    old = st.session_state.get(fk)
                    if old: plt.close(old)
                for k in ["last_result", "last_image", "gradcam_image", "gradcam_pp_image",
                          "mc_result", "agreement_score"]:
                    st.session_state[k] = None
            try:
                image = Image.open(io.BytesIO(upload_bytes)).convert("RGB")
            except (UnidentifiedImageError, OSError, ValueError):
                st.error("Invalid image. Please upload a valid JPG, PNG, or WEBP file.")
                image = None

            if image is not None:
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.markdown("#### MRI Preview")
                    st.image(image, **_stretch())
                    st.caption(f"Original: {image.width}×{image.height}px · Processed: {IMG_SIZE}×{IMG_SIZE}px")

                with col2:
                    st.markdown("#### Analysis Configuration")
                    cf1, cf2, cf3 = st.columns(3)
                    cf1.metric("Architecture", model_name or "—")
                    cf2.metric("Device", str(DEVICE).upper())
                    cf3.metric("Classes", len(class_names))

                    run_mc    = st.checkbox("Enable MC Dropout Uncertainty (20 passes)", value=True, key="run_mc_cb")
                    run_xai   = st.checkbox("Enable Dual XAI (Grad-CAM + Grad-CAM++)", value=True, key="run_xai_cb")
                    run_agree = st.checkbox("Compute Explanation Agreement Score", value=True, key="run_agree_cb")

                    if st.button("Run AI Analysis", type="primary", key="analyze_btn", **_content()):
                        if st.session_state.live_session_start is None:
                            st.session_state.live_session_start = datetime.now()
                        st.session_state.live_inference_running = True
                        status = st.empty(); prog = st.empty()
                        try:
                            total_t0 = time.perf_counter()
                            stages = ["Image Loaded", "Preprocessing", "Normalization", "Tensor Prep",
                                      "Neural Inference", "Probability Calc"]
                            if run_mc:    stages.append("MC Dropout")
                            if run_xai:   stages += ["Grad-CAM", "Grad-CAM++"]
                            if run_agree: stages.append("Agreement Score")
                            stages.append("Report")
                            total_stages = len(stages)

                            for i, stage in enumerate(stages[:-4]):
                                status.info(f"Processing: {stage}…")
                                prog.progress((i + 1) / total_stages)

                            log_activity("MRI uploaded; preprocessing started", "info")
                            predicted_class, confidence, probability_dict, preprocessing_ms, inference_ms = predict_image(image, model, class_names)
                            log_activity(f"Inference complete: {predicted_class} ({confidence:.1f}%)", "success")

                            step_off = 4
                            mc_result = None
                            if run_mc:
                                status.info("Processing: MC Dropout…")
                                prog.progress((step_off + 1) / total_stages); step_off += 1
                                try:
                                    mc_result = mc_dropout_predict(image, model, class_names, MC_SAMPLES)
                                    if mc_result:
                                        log_activity(f"MC Dropout: σ={mc_result['uncertainty']:.4f} ({mc_result['band']})", "info")
                                except Exception:
                                    log_activity("MC Dropout failed", "warn")

                            gradcam_fig = gradcam_ms = gradcam_pp_fig = gradcam_pp_ms = None
                            if run_xai:
                                status.info("Processing: Grad-CAM…")
                                prog.progress((step_off + 1) / total_stages); step_off += 1
                                try:
                                    t_gc = time.perf_counter()
                                    gradcam_fig = generate_gradcam(image, model, model_name)
                                    gradcam_ms = (time.perf_counter() - t_gc) * 1000
                                    log_activity("Grad-CAM generated", "info")
                                except Exception:
                                    log_activity("Grad-CAM failed", "warn")

                                status.info("Processing: Grad-CAM++…")
                                prog.progress((step_off + 1) / total_stages); step_off += 1
                                try:
                                    t_pp = time.perf_counter()
                                    gradcam_pp_fig = generate_gradcam_pp(image, model, model_name)
                                    gradcam_pp_ms = (time.perf_counter() - t_pp) * 1000
                                    log_activity("Grad-CAM++ generated", "info")
                                except Exception:
                                    log_activity("Grad-CAM++ failed", "warn")

                            agree_score = None
                            if run_agree and run_xai:
                                status.info("Processing: Agreement Score…")
                                prog.progress((step_off + 1) / total_stages); step_off += 1
                                try:
                                    agree_score = explanation_agreement(image, model, model_name)
                                    if agree_score is not None:
                                        log_activity(f"Agreement: {agree_score:.3f}", "info")
                                except Exception:
                                    log_activity("Agreement score failed", "warn")

                            total_ms = (time.perf_counter() - total_t0) * 1000
                            status.info("Building report…")
                            prog.progress(1.0)

                            result = {
                                "prediction": predicted_class, "confidence": confidence,
                                "probabilities": probability_dict, "model": model_name,
                                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                "latency_ms": inference_ms, "preprocessing_ms": preprocessing_ms,
                                "gradcam_ms": gradcam_ms if gradcam_fig else None,
                                "gradcam_pp_ms": gradcam_pp_ms if gradcam_pp_fig else None,
                                "total_ms": total_ms,
                                "gradcam_available": gradcam_fig is not None,
                                "gradcam_pp_available": gradcam_pp_fig is not None,
                                "image_id": image_id,
                                "uncertainty": mc_result["uncertainty"] if mc_result else None,
                                "mc_band": mc_result["band"] if mc_result else None,
                                "agreement_score": agree_score,
                            }

                            for fk, nf in [("gradcam_image", gradcam_fig), ("gradcam_pp_image", gradcam_pp_fig)]:
                                old = st.session_state.get(fk)
                                if old: plt.close(old)
                                st.session_state[fk] = nf

                            st.session_state.last_result = result
                            st.session_state.last_image = image.copy()
                            st.session_state.mc_result = mc_result
                            st.session_state.agreement_score = agree_score
                            st.session_state.prediction_history.append(result)
                            st.session_state.prediction_history = st.session_state.prediction_history[-MAX_HISTORY:]
                            update_live_stats(result, inference_ms)
                            log_activity("Analysis report generated", "success")
                            status.success(f"{predicted_class} · {confidence:.2f}% · {inference_ms:.0f} ms")
                            prog.empty()

                        except Exception as exc:
                            msg = "CUDA out of memory. Try CPU inference." if "out of memory" in str(exc).lower() else "Analysis failed. Check image and model."
                            status.error(msg)
                            with st.expander("Technical Details"):
                                st.code(str(exc))
                            log_activity("Analysis failed", "error")
                        finally:
                            st.session_state.live_inference_running = False

        if (image_id is not None and st.session_state.last_result is not None
                and st.session_state.last_result.get("image_id") == image_id):

            result = st.session_state.last_result
            mc_res = st.session_state.mc_result
            conf_v = result["confidence"]
            conf_lbl = "High" if conf_v >= 80 else "Moderate" if conf_v >= 60 else "Low"
            conf_cls = "high" if conf_v >= 80 else "moderate" if conf_v >= 60 else "low"

            shld_r = Icons.shield(16, "#94a3b8")
            st.markdown(safe_html(f"""
            <div class="diagnostic-panel">
                <div class="diag-header">
                    <span class="diag-label">AI Classification Result</span>
                    <span class="badge badge-research">{shld_r} Research Prototype</span>
                </div>
                <div class="diag-prediction">{result['prediction']}</div>
                <div class="diag-confidence">{conf_v:.2f}% model confidence</div>
                <div style="margin-top:.75rem">
                    <span class="badge badge-{conf_cls}">{conf_lbl} Confidence</span>
                </div>
            </div>"""), unsafe_allow_html=True)

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Confidence",    f"{conf_v:.2f}%")
            m2.metric("Inference",     f"{result['latency_ms']:.1f} ms")
            m3.metric("Preprocessing", f"{result.get('preprocessing_ms',0):.1f} ms")
            m4.metric("Total Time",    f"{result.get('total_ms',0):.1f} ms")

            if mc_res:
                unc = mc_res["uncertainty"]; band = mc_res["band"]; color = mc_res["color"]
                st.markdown(safe_html(f"""
                <div class="uncertainty-card">
                    <div class="unc-title">MC Dropout Uncertainty Estimation ({MC_SAMPLES} passes)</div>
                    <div style="display:flex;justify-content:space-between;align-items:center">
                        <div>
                            <div class="unc-value" style="color:{color}">σ = {unc:.4f}</div>
                            <div class="unc-band" style="color:{color}">{band}</div>
                        </div>
                        <div style="text-align:right;font-size:.74rem;color:#94a3b8">
                            MC Confidence: <b style="color:#22d3ee;font-family:var(--font-mono)">{mc_res['confidence']:.2f}%</b><br>
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
                            High score (&gt;0.70) means both methods highlight similar regions.
                        </div>
                        <div style="text-align:right;min-width:85px;margin-left:1rem">
                            <div style="font-family:var(--font-display);font-size:1.4rem;font-weight:700;color:{a_color};letter-spacing:-.02em">{agree:.3f}</div>
                            <div style="font-size:.64rem;color:{a_color};font-weight:600">{a_label}</div>
                        </div>
                    </div>
                </div>"""), unsafe_allow_html=True)

            alert_svg = Icons.alert_triangle(16, "#d97706")
            st.markdown(safe_html(f"""
            <div class="disclaimer">
                {alert_svg}
                <span>NeuroLens AI is a research prototype for educational use. Model outputs are not medical diagnoses and must not replace evaluation by a qualified healthcare professional.</span>
            </div>"""), unsafe_allow_html=True)

            st.write("")
            st.markdown("#### Probability Distribution")
            prob_html = '<div class="prob-grid">' + "".join(
                f'<div class="prob-card {"is-top" if label == result["prediction"] else ""}">'
                f'<div class="prob-value">{prob:.1f}%</div>'
                f'<div class="prob-label">{label}</div>'
                f'{"<div class=\"prob-top-tag\">Top Prediction</div>" if label == result["prediction"] else ""}'
                f'</div>' for label, prob in result["probabilities"].items()
            ) + '</div>'
            st.markdown(prob_html, unsafe_allow_html=True)

            st.write("")
            st.markdown("#### Live Probability Stream")
            render_live_probability_animation(result, mc_result=mc_res)

            if st.session_state.gradcam_image or st.session_state.gradcam_pp_image:
                st.write("")
                st.markdown("#### Dual XAI Visualization")
                gc_col, pp_col = st.columns(2)
                with gc_col:
                    st.markdown("**Grad-CAM** · Jet colormap")
                    if st.session_state.gradcam_image:
                        st.pyplot(st.session_state.gradcam_image, **_stretch())
                        st.caption("Weighted class activations · α=0.44")
                with pp_col:
                    st.markdown("**Grad-CAM++** · Inferno colormap")
                    if st.session_state.gradcam_pp_image:
                        st.pyplot(st.session_state.gradcam_pp_image, **_stretch())
                        st.caption("Second-order gradients · α=0.46")

                st.markdown(safe_html("""
                <div class="xai-card">
                    <div class="xai-title">About These Visualizations</div>
                    <div class="xai-text">
                        Highlighted regions represent image areas that contributed to the model's classification.
                        Grad-CAM uses weighted class activations; Grad-CAM++ uses second-order gradients for sharper saliency.
                        These visualizations explain the <em>model's</em> decision, not ground-truth anatomy.
                    </div>
                </div>"""), unsafe_allow_html=True)

            gc_buf = pp_buf = None
            if st.session_state.gradcam_image:
                b = io.BytesIO()
                st.session_state.gradcam_image.savefig(b, format="png", bbox_inches="tight", dpi=160)
                gc_buf = b.getvalue()
            if st.session_state.gradcam_pp_image:
                b2 = io.BytesIO()
                st.session_state.gradcam_pp_image.savefig(b2, format="png", bbox_inches="tight", dpi=160)
                pp_buf = b2.getvalue()

            dl_svg = Icons.download(16, "#22d3ee")
            st.markdown(safe_html(f"""
            <div class="export-head">
                <div class="export-head-icon">{dl_svg}</div>
                <div>
                    <div class="export-head-title">Export Results</div>
                    <div class="export-head-sub">Download AI analysis outputs</div>
                </div>
                <span class="export-head-badge">Ready</span>
            </div>"""), unsafe_allow_html=True)

            if gc_buf and pp_buf:
                dc1, dc2 = st.columns(2)
                with dc1:
                    st.markdown('<div class="export-item-label">Grad-CAM · PNG</div>', unsafe_allow_html=True)
                    st.download_button("Download Grad-CAM", gc_buf, "gradcam.png", "image/png", key="dl_gc", **_stretch())
                with dc2:
                    st.markdown('<div class="export-item-label">Grad-CAM++ · PNG</div>', unsafe_allow_html=True)
                    st.download_button("Download Grad-CAM++", pp_buf, "gradcam_pp.png", "image/png", key="dl_pp", **_stretch())
            elif gc_buf:
                st.markdown('<div class="export-item-label">Grad-CAM · PNG</div>', unsafe_allow_html=True)
                st.download_button("Download Grad-CAM", gc_buf, "gradcam.png", "image/png", key="dl_gc", **_stretch())
            elif pp_buf:
                st.markdown('<div class="export-item-label">Grad-CAM++ · PNG</div>', unsafe_allow_html=True)
                st.download_button("Download Grad-CAM++", pp_buf, "gradcam_pp.png", "image/png", key="dl_pp", **_stretch())

            st.markdown('<div class="export-item-label" style="margin-top:.85rem">Analysis Report · TXT</div>', unsafe_allow_html=True)
            st.download_button(
                "Download Analysis Report",
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
                    f"MC Uncertainty σ    : {result.get('uncertainty','N/A')}",
                    f"Reliability Band    : {result.get('mc_band','N/A')}",
                    "─────────────────────────────────────────────────────",
                    "  EXPLAINABILITY",
                    "─────────────────────────────────────────────────────",
                    f"Grad-CAM Available  : {'Yes' if result.get('gradcam_available') else 'No'}",
                    f"Grad-CAM++ Available: {'Yes' if result.get('gradcam_pp_available') else 'No'}",
                    "Agreement Score     : " + (f"{result.get('agreement_score'):.4f}" if result.get('agreement_score') is not None else 'N/A'),
                    "─────────────────────────────────────────────────────",
                    "  TIMING",
                    "─────────────────────────────────────────────────────",
                    f"Preprocessing       : {result.get('preprocessing_ms',0):.2f} ms",
                    f"Inference           : {result['latency_ms']:.2f} ms",
                    f"Grad-CAM            : {result.get('gradcam_ms') or 0:.2f} ms",
                    f"Grad-CAM++          : {result.get('gradcam_pp_ms') or 0:.2f} ms",
                    f"Total               : {result.get('total_ms',0):.2f} ms",
                    "═══════════════════════════════════════════════════════",
                    "  DISCLAIMER",
                    "─────────────────────────────────────────────────────",
                    "  NeuroLens AI is a research prototype for educational use.",
                    "  Outputs do NOT constitute medical diagnoses.",
                    "═══════════════════════════════════════════════════════",
                ]),
                "neurolens_report.txt", "text/plain", key="dl_report", type="primary", **_stretch(),
            )


# ══════════════════════════════════════════════════════════════════════════════
# DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════

elif nav == "Dashboard":
    chart_h = Icons.chart(22, "#22d3ee")
    st.markdown(f"<h1 style='display:flex;align-items:center;gap:.5rem'>{chart_h} Neurodiagnostic Dashboard</h1>", unsafe_allow_html=True)
    st.caption("Session analytics from completed MRI analyses")
    render_live_ticker()
    if st.session_state.live_session_start is None:
        st.session_state.live_session_start = datetime.now()
    history = st.session_state.prediction_history

    if not history:
        chart_svg = Icons.chart(32, "#4b5869")
        st.markdown(safe_html(f"""
        <div class="empty-state">
            <div class="empty-icon">{chart_svg}</div>
            <div class="empty-title">Awaiting diagnostic data</div>
            <div class="empty-text">Run an MRI analysis to populate the dashboard.</div>
        </div>"""), unsafe_allow_html=True)
    else:
        total   = st.session_state.live_predictions_count
        avg_c   = st.session_state.live_avg_confidence
        unique  = len(st.session_state.live_class_counts)
        avg_lat = float(np.mean([h.get("latency_ms", 0) for h in history]))
        unc_data = [h["uncertainty"] for h in history if h.get("uncertainty") is not None]
        avg_unc = float(np.mean(unc_data)) if unc_data else None
        agree_data = [h["agreement_score"] for h in history if h.get("agreement_score") is not None]
        avg_agree  = float(np.mean(agree_data)) if agree_data else None

        p_svg = Icons.pulse(18, "#22d3ee"); t_svg = Icons.target(18, "#22d3ee")
        f_svg = Icons.flask(18, "#22d3ee"); z_svg = Icons.zap(18, "#22d3ee"); sh_svg = Icons.shield(18, "#22d3ee")
        m1, m2, m3, m4, m5 = st.columns(5)
        mets = [
            (p_svg, total, "Total Scans"),
            (t_svg, f"{avg_c:.1f}%", "Avg. Confidence"),
            (f_svg, unique, "Classes Seen"),
            (z_svg, f"{avg_lat:.0f} ms", "Avg. Latency"),
            (sh_svg, f"σ={avg_unc:.4f}" if avg_unc is not None else "—", "Avg. Uncertainty"),
        ]
        for col, (icon, val, lbl) in zip([m1, m2, m3, m4, m5], mets):
            with col:
                st.markdown(safe_html(f"""
                <div class="metric-card">
                    <div class="metric-icon-wrap">{icon}</div>
                    <div class="metric-value">{val}</div>
                    <div class="metric-label">{lbl}</div>
                </div>"""), unsafe_allow_html=True)

        if avg_agree is not None:
            st.markdown(safe_html(f"""
            <div style="margin:.65rem 0;padding:.7rem 1.1rem;border-radius:10px;background:rgba(8,145,178,0.04);border:1px solid rgba(8,145,178,0.18);display:flex;justify-content:space-between;align-items:center">
                <span style="font-size:.64rem;color:#22d3ee;font-weight:700;letter-spacing:.1em">Average Explanation Agreement Score</span>
                <span style="font-family:var(--font-display);font-size:1.2rem;font-weight:700;color:#e2e8f0;letter-spacing:-.02em">{avg_agree:.3f}</span>
            </div>"""), unsafe_allow_html=True)

        st.write("")
        col_l, col_r = st.columns([2, 1])
        with col_l:
            st.subheader("Prediction Distribution")
            counts = st.session_state.live_class_counts
            mx = max(counts.values()) if counts else 1
            for label, count in counts.items():
                pct = (count / mx) * 100
                live_pct = (count / total) * 100 if total else 0
                st.markdown(safe_html(f"""
                <div class="dist-card">
                    <div class="dist-row"><div class="dist-name">{label}</div><div class="dist-count">{count} · {live_pct:.1f}%</div></div>
                    <div class="dist-track"><div class="dist-fill" style="width:{pct:.0f}%"></div></div>
                </div>"""), unsafe_allow_html=True)

            st.write("")
            st.subheader("Confidence Trend")
            cf = plot_confidence_trend(history)
            if cf:
                st.pyplot(cf, **_stretch())
                plt.close(cf)
            else:
                st.caption("Need ≥2 analyses.")

            st.subheader("Inference Latency")
            lf = plot_latency_trend(history)
            if lf:
                st.pyplot(lf, **_stretch())
                plt.close(lf)
            else:
                st.caption("Need ≥2 analyses.")

            if unc_data:
                st.subheader("Uncertainty Trend")
                uf = plot_uncertainty_history(history)
                if uf:
                    st.pyplot(uf, **_stretch())
                    plt.close(uf)

        with col_r:
            act_svg = Icons.activity(16, "#22d3ee")
            st.markdown(f"<div style='display:flex;align-items:center;gap:.4rem;margin-bottom:.5rem'>{act_svg}<b style='font-size:.88rem'>Activity Feed</b></div>", unsafe_allow_html=True)
            render_activity_feed()

            st.write("")
            st.subheader("Latest Result")
            latest = history[-1]
            el = "—"
            if st.session_state.live_session_start:
                e = datetime.now() - st.session_state.live_session_start
                el = f"{int(e.total_seconds())}s"
            rows = [
                ("Prediction",    latest["prediction"]),
                ("Confidence",    f"{latest['confidence']:.2f}%"),
                ("Architecture",  latest["model"]),
                ("Inference",     f"{latest.get('latency_ms',0):.0f} ms"),
                ("Uncertainty σ", f"{latest['uncertainty']:.4f}" if latest.get("uncertainty") is not None else "—"),
                ("Agreement",     f"{latest['agreement_score']:.3f}" if latest.get("agreement_score") is not None else "—"),
                ("Session Time",  el),
                ("Analyzed At",   latest["timestamp"]),
            ]
            st.markdown('<div class="latest-card">' + "".join(
                f'<div class="latest-row"><div class="latest-key">{k}</div><div class="latest-val">{v}</div></div>'
                for k, v in rows
            ) + '</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# HISTORY
# ══════════════════════════════════════════════════════════════════════════════

elif nav == "History":
    hist_h = Icons.history(22, "#22d3ee")
    st.markdown(f"<h1 style='display:flex;align-items:center;gap:.5rem'>{hist_h} Analysis History</h1>", unsafe_allow_html=True)
    st.caption("Review all session AI diagnostic reports")
    render_live_ticker()
    history = st.session_state.prediction_history

    if not history:
        hist_svg = Icons.history(32, "#4b5869")
        st.markdown(safe_html(f"""
        <div class="empty-state">
            <div class="empty-icon">{hist_svg}</div>
            <div class="empty-title">No diagnostic history</div>
            <div class="empty-text">Analysis reports appear here after running MRI scans.</div>
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
            scan_sm = Icons.scan(18, "#22d3ee")
            for item in filtered:
                conf = item["confidence"]
                cc = "conf-hi" if conf >= 80 else "conf-mid" if conf >= 60 else "conf-lo"
                unc = f" · σ={item['uncertainty']:.4f}" if item.get("uncertainty") is not None else ""
                ag  = f" · Agr={item['agreement_score']:.3f}" if item.get("agreement_score") is not None else ""
                st.markdown(safe_html(f"""
                <div class="thumb-card">
                    <div class="thumb-icon">{scan_sm}</div>
                    <div class="thumb-info">
                        <div class="thumb-title">{item['prediction']}</div>
                        <div class="thumb-meta">{item['timestamp']} · {item['model']}{unc}{ag}</div>
                    </div>
                    <div><span class="conf-badge {cc}">{conf:.1f}%</span></div>
                </div>"""), unsafe_allow_html=True)
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
                        for lbl, prob in item["probabilities"].items():
                            st.write(f"  · {lbl}: {prob:.4f}%")

        st.write("")
        if st.button("Clear History", key="clear_hist_btn"):
            st.session_state.confirm_clear_hist = True
        if st.session_state.get("confirm_clear_hist", False):
            st.warning("Clear all session records?")
            cc1, cc2, _ = st.columns([1, 1, 4])
            if cc1.button("Confirm", key="confirm_hist", **_stretch()):
                clear_prediction_history()
                st.session_state.confirm_clear_hist = False
                st.rerun()
            if cc2.button("Cancel", key="cancel_hist", **_stretch()):
                st.session_state.confirm_clear_hist = False


# ══════════════════════════════════════════════════════════════════════════════
# GRAD-CAM
# ══════════════════════════════════════════════════════════════════════════════

elif nav == "Grad-CAM":
    hm_h = Icons.heatmap(22, "#22d3ee")
    st.markdown(f"<h1 style='display:flex;align-items:center;gap:.5rem'>{hm_h} Grad-CAM Explainability</h1>", unsafe_allow_html=True)
    st.caption("Visualize which MRI regions influenced the model's classification")
    render_live_ticker()

    if not st.session_state.gradcam_image and not st.session_state.gradcam_pp_image:
        hm_svg = Icons.heatmap(32, "#4b5869")
        st.markdown(safe_html(f"""
        <div class="empty-state">
            <div class="empty-icon">{hm_svg}</div>
            <div class="empty-title">No Grad-CAM visualization yet</div>
            <div class="empty-text">Run an MRI analysis with XAI enabled to generate heatmaps.</div>
        </div>"""), unsafe_allow_html=True)
    else:
        oc, gc, pc = st.columns(3)
        with oc:
            st.subheader("Original MRI")
            if st.session_state.last_image:
                st.image(st.session_state.last_image, **_stretch())
        with gc:
            st.subheader("Grad-CAM · Jet")
            if st.session_state.gradcam_image:
                st.pyplot(st.session_state.gradcam_image, **_stretch())
        with pc:
            st.subheader("Grad-CAM++ · Inferno")
            if st.session_state.gradcam_pp_image:
                st.pyplot(st.session_state.gradcam_pp_image, **_stretch())

        st.markdown("<div style='font-size:.72rem;color:#4b5869;margin:.5rem 0'>Heatmap influence: Low (dark) ░░░▒▒▒████ High (bright)</div>", unsafe_allow_html=True)

        agree = st.session_state.agreement_score
        if agree is not None:
            a_color = "#10b981" if agree >= 0.7 else "#f59e0b" if agree >= 0.5 else "#ef4444"
            a_label = "High Agreement" if agree >= 0.7 else "Moderate" if agree >= 0.5 else "Low Agreement"
            st.markdown(safe_html(f"""
            <div class="xai-card">
                <div class="xai-title">Explanation Agreement Score</div>
                <div style="display:flex;justify-content:space-between;align-items:center">
                    <div class="xai-text">Correlation between Grad-CAM and Grad-CAM++ attention regions.<br>High scores (&gt;0.70) indicate consistent heatmaps.</div>
                    <div style="text-align:right;min-width:80px;margin-left:1rem">
                        <div style="font-family:var(--font-display);font-size:1.4rem;font-weight:700;color:{a_color};letter-spacing:-.02em">{agree:.3f}</div>
                        <div style="font-size:.64rem;color:{a_color};font-weight:600">{a_label}</div>
                    </div>
                </div>
            </div>"""), unsafe_allow_html=True)

        st.markdown(safe_html("""
        <div class="xai-card">
            <div class="xai-title">About These Visualizations</div>
            <div class="xai-text">
                <b>Grad-CAM</b> computes weighted class activation maps from the last convolutional layer's gradients.
                <b>Grad-CAM++</b> uses second-order gradients for sharper saliency localization.
                Both explain the model's decision, not ground-truth anatomy.
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
            dl1.download_button("Download Grad-CAM", buf.getvalue(), "gradcam.png", "image/png", key="gradcam_dl", **_stretch())
        if st.session_state.gradcam_pp_image:
            buf2 = io.BytesIO()
            st.session_state.gradcam_pp_image.savefig(buf2, format="png", bbox_inches="tight", dpi=160)
            dl2.download_button("Download Grad-CAM++", buf2.getvalue(), "gradcam_pp.png", "image/png", key="gradcam_pp_dl", **_stretch())


# ══════════════════════════════════════════════════════════════════════════════
# XAI LAB
# ══════════════════════════════════════════════════════════════════════════════

elif nav == "XAI Lab":
    lab_h = Icons.lab(22, "#22d3ee")
    st.markdown(f"<h1 style='display:flex;align-items:center;gap:.5rem'>{lab_h} XAI Research Lab</h1>", unsafe_allow_html=True)
    st.caption("Uncertainty, explainability, and model behavior analysis")
    render_live_ticker()
    history = st.session_state.prediction_history

    if not history:
        lab_svg = Icons.lab(32, "#4b5869")
        st.markdown(safe_html(f"""
        <div class="empty-state">
            <div class="empty-icon">{lab_svg}</div>
            <div class="empty-title">No XAI data yet</div>
            <div class="empty-text">Run analyses with MC Dropout and dual XAI enabled.</div>
        </div>"""), unsafe_allow_html=True)
    else:
        st.subheader("Uncertainty Distribution")
        unc_data = [(i + 1, h["uncertainty"]) for i, h in enumerate(history) if "uncertainty" in h]
        if len(unc_data) >= 2:
            uf = plot_uncertainty_history(history)
            if uf:
                st.pyplot(uf, **_stretch())
                plt.close(uf)

        bands = {}
        for h in history:
            b = h.get("mc_band")
            if b:
                bands[b] = bands.get(b, 0) + 1
        if bands:
            st.write("")
            st.subheader("Reliability Band Distribution")
            bc1, bc2, bc3, bc4 = st.columns(4)
            band_map = {
                "Very High Reliability": (bc1, "#10b981"),
                "High Reliability":      (bc2, "#0891b2"),
                "Moderate Reliability":  (bc3, "#d97706"),
                "Low Reliability":       (bc4, "#dc2626"),
            }
            for band, (col, color) in band_map.items():
                cnt = bands.get(band, 0)
                with col:
                    st.markdown(safe_html(f"""
                    <div class="metric-card" style="border-color:{color}30">
                        <div class="metric-value" style="color:{color};font-size:1.55rem">{cnt}</div>
                        <div class="metric-label">{band.replace(' Reliability','')}</div>
                    </div>"""), unsafe_allow_html=True)

        agree_hist = [(i + 1, h["agreement_score"]) for i, h in enumerate(history) if h.get("agreement_score") is not None]
        if len(agree_hist) >= 2:
            st.write("")
            st.subheader("Agreement Score Trend")
            fig2, ax2 = _dark_fig()
            xs, ys = zip(*agree_hist)
            ax2.plot(xs, ys, marker="D", lw=2, ms=4, color="#0891b2")
            ax2.fill_between(xs, ys, alpha=0.07, color="#0891b2")
            ax2.axhline(0.7, color="#059669", lw=1, ls="--", alpha=0.55, label="High agreement")
            ax2.axhline(0.5, color="#d97706", lw=1, ls="--", alpha=0.55, label="Moderate")
            ax2.set_ylim(0, 1.05)
            ax2.set_xlabel("Analysis #", color="#94a3b8", fontsize=9)
            ax2.set_ylabel("Agreement Score", color="#94a3b8", fontsize=9)
            ax2.legend(fontsize=7, labelcolor="#94a3b8", facecolor="#0f172a", edgecolor="#1e2d42")
            ax2.grid(True, alpha=0.09, ls="--")
            fig2.tight_layout(pad=0.5)
            st.pyplot(fig2, **_stretch())
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
                    "Min σ": f"{np.min(vals):.6f}",
                    "Max σ": f"{np.max(vals):.6f}",
                })
        if rows:
            import pandas as pd
            st.dataframe(pd.DataFrame(rows), hide_index=True, **_stretch())
        else:
            st.caption("No per-class uncertainty data yet.")

        with st.expander("Methodology Reference"):
            st.markdown("""
**MC Dropout Uncertainty** — Gal & Ghahramani (2016)
- Enables Dropout during inference for stochastic forward passes
- σ < 0.05: Very High · σ < 0.12: High · σ < 0.22: Moderate · σ ≥ 0.22: Low

**Grad-CAM** — Selvaraju et al. (2017)
- Class Activation Mapping via gradient-weighted pooling over last conv layer

**Grad-CAM++** — Chattopadhay et al. (2018)
- Second-order gradient weighting for sharper, more precise saliency

**Explanation Agreement Score**
- Pearson correlation between Grad-CAM and Grad-CAM++ heatmaps
- ≥ 0.70: High · ≥ 0.50: Moderate · < 0.50: Low agreement
            """)


# ══════════════════════════════════════════════════════════════════════════════
# SETTINGS
# ══════════════════════════════════════════════════════════════════════════════

elif nav == "Settings":
    set_h = Icons.settings(22, "#22d3ee")
    st.markdown(f"<h1 style='display:flex;align-items:center;gap:.5rem'>{set_h} System Settings</h1>", unsafe_allow_html=True)
    st.caption("Neural engine configuration and session management")

    c1, c2 = st.columns(2)
    with c1:
        lay_svg = Icons.layers(18, "#22d3ee")
        st.markdown(safe_html(f"""
        <div class="info-card">
            <div class="info-card-icon">{lay_svg}</div>
            <h3>Neural Engine</h3>
            <p><b>Architecture:</b> {model_name or 'Unavailable'}</p>
            <p><b>Status:</b> {"Ready" if not model_error else "Failed"}</p>
            <p><b>Device:</b> {DEVICE}</p>
            <p><b>Input Resolution:</b> {IMG_SIZE} × {IMG_SIZE}</p>
            <p><b>Classes:</b> {', '.join(CLASS_NAMES)}</p>
            <p><b>Checkpoint:</b> {MODEL_PATH.name if MODEL_PATH.exists() else 'Not found'}</p>
            <p><b>MC Dropout Passes:</b> {MC_SAMPLES}</p>
        </div>"""), unsafe_allow_html=True)

    with c2:
        cpu_svg = Icons.cpu(18, "#22d3ee")
        try:
            tv = metadata.version("torchvision")
        except Exception:
            tv = "—"
        st.markdown(safe_html(f"""
        <div class="info-card">
            <div class="info-card-icon">{cpu_svg}</div>
            <h3>System Information</h3>
            <p><b>PyTorch:</b> {torch.__version__}</p>
            <p><b>Torchvision:</b> {tv}</p>
            <p><b>Streamlit:</b> {st.__version__}</p>
            <p><b>Python:</b> {platform.python_version()}</p>
            <p><b>CUDA Available:</b> {torch.cuda.is_available()}</p>
            <p><b>CUDA Device:</b> {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}</p>
            <p><b>XAI Methods:</b> Grad-CAM · Grad-CAM++</p>
        </div>"""), unsafe_allow_html=True)

        show_tech = st.toggle("Show technical details", key="show_tech")
        if show_tech:
            with st.expander("Checkpoint & Runtime", expanded=True):
                st.write(f"Checkpoint: `{MODEL_PATH}`")
                st.write(f"Device: `{DEVICE}` · Input: `{IMG_SIZE}×{IMG_SIZE}`")
                st.write(f"Norm mean: `{NORM_MEAN}` · std: `{NORM_STD}`")
                if model_error:
                    st.code(model_error)

        st.write("")
        alert_svg = Icons.alert_triangle(16, "#d97706")
        st.markdown(safe_html(f"""
        <div class="disclaimer">
            {alert_svg}
            <span><b>Reset Session</b> — Permanently clears all diagnostic reports and session data.</span>
        </div>"""), unsafe_allow_html=True)

        if st.button("Reset Session", type="primary", key="settings_reset"):
            st.session_state.settings_confirm_reset = True
        if st.session_state.get("settings_confirm_reset", False):
            st.warning("Reset all session results?")
            rc1, rc2, _ = st.columns([1, 1, 4])
            if rc1.button("Confirm", key="settings_confirm_yes", **_stretch()):
                for fk in ("gradcam_image", "gradcam_pp_image"):
                    old = st.session_state.get(fk)
                    if old:
                        plt.close(old)
                for k, v in DEFAULTS.items():
                    st.session_state[k] = copy.deepcopy(v)
                st.session_state.settings_confirm_reset = False
                st.success("Session cleared.")
                st.rerun()
            if rc2.button("Cancel", key="settings_confirm_no", **_stretch()):
                st.session_state.settings_confirm_reset = False


# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────

year       = datetime.now().year
build_time = datetime.now().strftime("%Y-%m-%d %H:%M UTC")
total_s    = st.session_state.live_predictions_count
avg_c_f    = st.session_state.live_avg_confidence
eng_ok     = model_error is None and MODEL_PATH.exists()
torch_ver  = torch.__version__.split('+')[0]
st_ver     = st.__version__
brain_ft   = Icons.brain(22, "#22d3ee")

st.markdown(safe_html(f"""
<div class="app-footer">
    <div>
        <div class="footer-brand-lockup">
            <div class="footer-brand-icon">{brain_ft}</div>
            <div>
                <div class="footer-brand-name">NeuroLens AI</div>
                <div class="footer-brand-tag">Neurodiagnostic Intelligence</div>
            </div>
        </div>
        <div class="footer-desc">
            An advanced research platform for brain MRI classification with dual XAI explainability and Bayesian uncertainty estimation.
        </div>
        <div class="footer-copy">© {year} NeuroLens AI · All Rights Reserved</div>
    </div>

    <div>
        <div class="footer-col-title">Tech Stack</div>
        <div class="footer-tech-badges">
            <span class="ft-badge"><span class="ft-badge-dot"></span>PyTorch {torch_ver}</span>
            <span class="ft-badge"><span class="ft-badge-dot"></span>Streamlit {st_ver}</span>
            <span class="ft-badge"><span class="ft-badge-dot"></span>Grad-CAM</span>
            <span class="ft-badge"><span class="ft-badge-dot"></span>Grad-CAM++</span>
            <span class="ft-badge"><span class="ft-badge-dot"></span>MC Dropout</span>
            <span class="ft-badge"><span class="ft-badge-dot"></span>{model_name or 'CustomCNN'}</span>
        </div>
        <div class="footer-col-title" style="margin-top:.75rem">Build</div>
        <div style="font-family:var(--font-mono);font-size:.63rem;color:#4b5869">{build_time}</div>
    </div>

    <div>
        <div class="footer-col-title">Session</div>
        <div class="footer-stats">
            <div class="f-stat">
                <div class="f-stat-val">{total_s}</div>
                <div class="f-stat-lbl">Total Scans</div>
            </div>
            <div class="f-stat">
                <div class="f-stat-val">{avg_c_f:.1f}%</div>
                <div class="f-stat-lbl">Avg Confidence</div>
            </div>
            <div class="f-stat f-stat-wide">
                <div class="f-stat-val">
                    <span class="f-dot {'f-dot-off' if not eng_ok else ''}"></span>
                    {'Engine Online' if eng_ok else 'Engine Offline'}
                </div>
                <div class="f-stat-lbl">Status</div>
            </div>
        </div>
    </div>
</div>"""), unsafe_allow_html=True)

st.markdown(safe_html(f"""
<div class="copyright-line">
    <span style="color:#94a3b8;font-weight:600">© {year} NeuroLens AI</span> · All Rights Reserved
    <br>
    <span style="font-size:.66rem;color:#4b5869">
        Developed by
        <span style="color:#e2e8f0;font-weight:600">MD. Atique Shahriar</span>
        &amp;
        <span style="color:#e2e8f0;font-weight:600">Aronna Das</span>
    </span>
</div>"""), unsafe_allow_html=True)