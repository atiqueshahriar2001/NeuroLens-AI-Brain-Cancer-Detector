# =============================================================================
# NeuroLens AI — Neurodiagnostic Intelligence Platform
# Production SaaS Edition v3.7.3
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


# ─────────────────────────────────────────────────────────────────────────────
# SVG ICON LIBRARY
# ─────────────────────────────────────────────────────────────────────────────
class Icons:
    """Inline SVG icons. All accept size and color args."""

    @staticmethod
    def _wrap(path_d: str, size=18, color="currentColor",
              viewbox="0 0 24 24", extra="") -> str:
        return (
            f'<svg width="{size}" height="{size}" viewBox="{viewbox}" fill="none" '
            f'stroke="{color}" stroke-width="1.65" stroke-linecap="round" '
            f'stroke-linejoin="round" '
            f'style="display:inline-block;vertical-align:middle;flex-shrink:0" {extra}>'
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
    def bar_chart(size=18, color="currentColor"):
        return Icons._wrap(
            '<line x1="12" y1="20" x2="12" y2="10"/>'
            '<line x1="18" y1="20" x2="18" y2="4"/>'
            '<line x1="6" y1="20" x2="6" y2="16"/>',
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
    def download(size=16, color="currentColor"):
        return Icons._wrap(
            '<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>'
            '<polyline points="7 10 12 15 17 10"/>'
            '<line x1="12" y1="15" x2="12" y2="3"/>',
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
    def refresh(size=16, color="currentColor"):
        return Icons._wrap(
            '<path d="M3 12a9 9 0 0 1 9-9 9.75 9.75 0 0 1 6.74 2.74L21 8"/>'
            '<path d="M21 3v5h-5"/>'
            '<path d="M21 12a9 9 0 0 1-9 9 9.75 9.75 0 0 1-6.74-2.74L3 16"/>'
            '<path d="M8 16H3v5"/>',
            size, color
        )

    @staticmethod
    def trash(size=16, color="currentColor"):
        return Icons._wrap(
            '<polyline points="3 6 5 6 21 6"/>'
            '<path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/>',
            size, color
        )

    @staticmethod
    def activity(size=16, color="currentColor"):
        return Icons._wrap('<polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>', size, color)

    @staticmethod
    def pulse(size=16, color="currentColor"):
        return Icons._wrap('<polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/>', size, color)

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
        return Icons._wrap('<polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/>', size, color)

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
        return Icons._wrap('<path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>', size, color)

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
    def flask(size=16, color="currentColor"):
        return Icons._wrap(
            '<path d="M9 3h6l1 6-4 8H8L4 9l1-6Z"/>'
            '<path d="M6 9h12"/>',
            size, color
        )

    @staticmethod
    def trending_up(size=16, color="currentColor"):
        return Icons._wrap(
            '<polyline points="22 7 13.5 15.5 8.5 10.5 2 17"/>'
            '<polyline points="16 7 22 7 22 13"/>',
            size, color
        )

    @staticmethod
    def pie_chart(size=16, color="currentColor"):
        return Icons._wrap(
            '<path d="M21.21 15.89A10 10 0 1 1 8 2.83"/>'
            '<path d="M22 12A10 10 0 0 0 12 2v10z"/>',
            size, color
        )

    @staticmethod
    def clock(size=16, color="currentColor"):
        return Icons._wrap(
            '<circle cx="12" cy="12" r="10"/>'
            '<polyline points="12 6 12 12 16 14"/>',
            size, color
        )

    @staticmethod
    def percent(size=16, color="currentColor"):
        return Icons._wrap(
            '<line x1="19" y1="5" x2="5" y2="19"/>'
            '<circle cx="6.5" cy="6.5" r="2.5"/>'
            '<circle cx="17.5" cy="17.5" r="2.5"/>',
            size, color
        )

    @staticmethod
    def grid(size=16, color="currentColor"):
        return Icons._wrap(
            '<rect x="3" y="3" width="7" height="7"/>'
            '<rect x="14" y="3" width="7" height="7"/>'
            '<rect x="14" y="14" width="7" height="7"/>'
            '<rect x="3" y="14" width="7" height="7"/>',
            size, color
        )

    @staticmethod
    def image(size=16, color="currentColor"):
        return Icons._wrap(
            '<rect x="3" y="3" width="18" height="18" rx="2" ry="2"/>'
            '<circle cx="9" cy="9" r="2"/>'
            '<path d="m21 15-3.086-3.086a2 2 0 0 0-2.828 0L6 21"/>',
            size, color
        )

    @staticmethod
    def file_text(size=16, color="currentColor"):
        return Icons._wrap(
            '<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>'
            '<polyline points="14 2 14 8 20 8"/>'
            '<line x1="16" y1="13" x2="8" y2="13"/>'
            '<line x1="16" y1="17" x2="8" y2="17"/>'
            '<polyline points="10 9 9 9 8 9"/>',
            size, color
        )

    @staticmethod
    def list_icon(size=16, color="currentColor"):
        return Icons._wrap(
            '<line x1="8" y1="6" x2="21" y2="6"/>'
            '<line x1="8" y1="12" x2="21" y2="12"/>'
            '<line x1="8" y1="18" x2="21" y2="18"/>'
            '<line x1="3" y1="6" x2="3.01" y2="6"/>'
            '<line x1="3" y1="12" x2="3.01" y2="12"/>'
            '<line x1="3" y1="18" x2="3.01" y2="18"/>',
            size, color
        )

    @staticmethod
    def filter_icon(size=16, color="currentColor"):
        return Icons._wrap('<polygon points="22 3 2 3 10 12.46 10 19 14 21 14 12.46 22 3"/>', size, color)

    @staticmethod
    def search(size=16, color="currentColor"):
        return Icons._wrap(
            '<circle cx="11" cy="11" r="8"/>'
            '<line x1="21" y1="21" x2="16.65" y2="16.65"/>',
            size, color
        )

    @staticmethod
    def sliders(size=16, color="currentColor"):
        return Icons._wrap(
            '<line x1="4" y1="21" x2="4" y2="14"/>'
            '<line x1="4" y1="10" x2="4" y2="3"/>'
            '<line x1="12" y1="21" x2="12" y2="12"/>'
            '<line x1="12" y1="8" x2="12" y2="3"/>'
            '<line x1="20" y1="21" x2="20" y2="16"/>'
            '<line x1="20" y1="12" x2="20" y2="3"/>'
            '<line x1="1" y1="14" x2="7" y2="14"/>'
            '<line x1="9" y1="8" x2="15" y2="8"/>'
            '<line x1="17" y1="16" x2="23" y2="16"/>',
            size, color
        )

    @staticmethod
    def database(size=16, color="currentColor"):
        return Icons._wrap(
            '<ellipse cx="12" cy="5" rx="9" ry="3"/>'
            '<path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3"/>'
            '<path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"/>',
            size, color
        )

    @staticmethod
    def terminal(size=16, color="currentColor"):
        return Icons._wrap(
            '<polyline points="4 17 10 11 4 5"/>'
            '<line x1="12" y1="19" x2="20" y2="19"/>',
            size, color
        )

    @staticmethod
    def box(size=16, color="currentColor"):
        return Icons._wrap(
            '<path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"/>'
            '<polyline points="3.27 6.96 12 12.01 20.73 6.96"/>'
            '<line x1="12" y1="22.08" x2="12" y2="12"/>',
            size, color
        )

    @staticmethod
    def calendar(size=16, color="currentColor"):
        return Icons._wrap(
            '<rect x="3" y="4" width="18" height="18" rx="2" ry="2"/>'
            '<line x1="16" y1="2" x2="16" y2="6"/>'
            '<line x1="8" y1="2" x2="8" y2="6"/>'
            '<line x1="3" y1="10" x2="21" y2="10"/>',
            size, color
        )

    @staticmethod
    def sparkle(size=16, color="currentColor"):
        return Icons._wrap(
            '<path d="M12 2v6M12 16v6M4.22 4.22l4.24 4.24M15.54 15.54l4.24 4.24M2 12h6M16 12h6M4.22 19.78l4.24-4.24M15.54 8.46l4.24-4.24"/>',
            size, color
        )


# ─────────────────────────────────────────────────────────────────────────────
# HTML HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def safe_html(html: str) -> str:
    return " ".join(line.strip() for line in html.strip().splitlines() if line.strip())


def _escape_html(text) -> str:
    if text is None:
        return ""
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )


def _render_html(content: str, height: int = 200, scrolling: bool = False) -> None:
    components.html(content, height=height, scrolling=scrolling)


def _icon_header(svg_svg: str, text: str, level: int = 3) -> str:
    tag = f"h{level}"
    return (
        f"<{tag} style='display:flex;align-items:center;gap:.5rem;"
        f"flex-wrap:wrap;margin:.6rem 0 .6rem'>{svg_svg}"
        f"<span>{text}</span></{tag}>"
    )


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
# WIDTH COMPATIBILITY HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def _st_width_arg(container: bool) -> dict:
    try:
        parts = st.__version__.split(".")[:2]
        major, minor = int(parts[0]), int(parts[1])
        if (major, minor) >= (1, 42):
            return {"width": "stretch" if container else "content"}
    except Exception:
        pass
    return {"use_container_width": bool(container)}


def _stretch() -> dict:
    return _st_width_arg(True)


def _content() -> dict:
    return _st_width_arg(False)


def _stretch_pyplot() -> dict:
    return _st_width_arg(True)


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

NORM_MEAN        = [0.485, 0.456, 0.406]
NORM_STD         = [0.229, 0.224, 0.225]
IMG_SIZE         = 224
MC_SAMPLES_DEF   = 20
MAX_HISTORY      = 200
CLASS_NAMES      = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]

# ─────────────────────────────────────────────────────────────────────────────
# SAFETY LIMITS (NEW)
# ─────────────────────────────────────────────────────────────────────────────
MAX_UPLOAD_BYTES = 200 * 1024 * 1024          # 200 MB
MAX_IMAGE_PIXELS = 50_000_000                 # ~50 MP
try:
    Image.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS
except Exception:
    pass

BASE_DIR   = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "neurolens_best.pth"
if not MODEL_PATH.exists():
    for candidate in [BASE_DIR.parent / "neurolens_best.pth"]:
        if candidate.exists():
            MODEL_PATH = candidate
            break

DEFAULT_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

test_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(NORM_MEAN, NORM_STD),
])


def uncertainty_band(uncertainty: float) -> Tuple[str, str]:
    if uncertainty < 0.05:
        return "Very High Reliability", "#10b981"
    if uncertainty < 0.12:
        return "High Reliability", "#22d3ee"
    if uncertainty < 0.22:
        return "Moderate Reliability", "#f59e0b"
    return "Low Reliability", "#ef4444"


# ─────────────────────────────────────────────────────────────────────────────
# FONT IMPORT
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(
    '<link rel="preconnect" href="https://fonts.googleapis.com">'
    '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
    '<link href="https://fonts.googleapis.com/css2?'
    'family=Inter:wght@400;500;600;700;800&'
    'family=Sora:wght@600;700;800&'
    'family=JetBrains+Mono:wght@400;500;600;700&display=swap" '
    'rel="stylesheet">',
    unsafe_allow_html=True,
)

UI_CSS = r"""
/* ═══════════  01 · DESIGN TOKENS  ═══════════ */
:root {
  --ink-0:#050e22;
  --ink-1:#071530;
  --ink-2:#0a1d42;
  --ink-3:#0e2858;
  --ink-4:#143272;
  --ink-5:#1c4292;
  --ink-6:#2456b3;

  --aurora-cyan:#22d3ee;
  --aurora-cyan-deep:#0891b2;
  --aurora-emerald:#34d399;
  --aurora-emerald-deep:#059669;
  --aurora-blue:#60a5fa;
  --aurora-blue-deep:#2563eb;
  --aurora-violet:#a78bfa;
  --aurora-violet-deep:#7c3aed;
  --aurora-amber:#fbbf24;
  --aurora-rose:#fb7185;

  --accent:#22d3ee;
  --accent-2:#34d399;
  --accent-3:#a78bfa;
  --success:#10b981;
  --success-hi:#34d399;
  --warning:#f59e0b;
  --warning-hi:#fbbf24;
  --danger:#ef4444;
  --danger-hi:#f87171;

  --text:#f1f5fd;
  --text-2:#b8c8e3;
  --text-3:#7f9bbd;
  --text-4:#557099;
  --text-5:#3d5579;

  --line-1:rgba(96,165,250,.14);
  --line-2:rgba(96,165,250,.26);
  --line-3:rgba(96,165,250,.42);
  --line-glow:rgba(34,211,238,.52);
  --line-soft:rgba(148,163,184,.10);

  --el-1: 0 1px 2px rgba(0,0,0,.22), 0 1px 0 rgba(255,255,255,.03) inset;
  --el-2: 0 4px 16px rgba(0,0,0,.30), 0 1px 0 rgba(255,255,255,.05) inset;
  --el-3: 0 12px 36px rgba(0,0,0,.40), 0 1px 0 rgba(255,255,255,.06) inset;
  --el-4: 0 24px 64px rgba(0,0,0,.50), 0 1px 0 rgba(255,255,255,.08) inset;
  --glow-cyan: 0 0 28px rgba(34,211,238,.32), 0 0 56px rgba(34,211,238,.14);
  --glow-emerald: 0 0 28px rgba(52,211,153,.30), 0 0 56px rgba(52,211,153,.12);
  --glow-violet: 0 0 28px rgba(167,139,250,.30), 0 0 56px rgba(167,139,250,.12);

  --r-xs:6px;
  --r-sm:9px;
  --r-md:12px;
  --r-lg:15px;
  --r-xl:19px;
  --r-pill:999px;

  --sp-1:.25rem;
  --sp-2:.5rem;
  --sp-3:.75rem;
  --sp-4:1rem;
  --sp-5:1.25rem;
  --sp-6:1.5rem;

  --font-display:'Sora','Inter',system-ui,sans-serif;
  --font-body:'Inter',system-ui,-apple-system,sans-serif;
  --font-mono:'JetBrains Mono',ui-monospace,SFMono-Regular,Menlo,monospace;

  --ease:cubic-bezier(.22,.85,.28,1);
  --ease-spring:cubic-bezier(.34,1.56,.64,1);
  --ease-in-out:cubic-bezier(.65,0,.35,1);

  --sb-w:288px;
  --max-w:1420px;
}

/* ═══════════  02 · RESET & BASE  ═══════════ */
html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
  background:var(--ink-0) !important;
  font-family:var(--font-body);
  -webkit-font-smoothing:antialiased;
  -moz-osx-font-smoothing:grayscale;
  text-rendering:optimizeLegibility;
  font-feature-settings:"cv02","cv03","cv04","cv11";
}

[data-testid="stAppViewContainer"] > .main {
  background:
    radial-gradient(ellipse 100% 60% at 50% -15%, rgba(34,211,238,.18), transparent 65%),
    radial-gradient(circle at -5% 105%, rgba(52,211,153,.14), transparent 44rem),
    radial-gradient(circle at 105% 105%, rgba(96,165,250,.20), transparent 44rem),
    radial-gradient(circle at 105% -5%, rgba(167,139,250,.12), transparent 40rem),
    radial-gradient(circle at 50% 50%, rgba(20,50,114,.30), transparent 60rem),
    linear-gradient(180deg, #0d2a5a 0%, #071530 48%, #030a18 100%) !important;
  background-attachment:fixed;
}

[data-testid="stMainBlockContainer"] {
  width:100% !important;
  max-width:var(--max-w) !important;
  margin:0 auto !important;
  padding:1.7rem 1.4rem 3rem !important;
}

@keyframes nl-page-enter {
  from { opacity:0; transform:translateY(12px); filter:blur(3px); }
  60%  { filter:blur(0); }
  to   { opacity:1; transform:translateY(0); filter:blur(0); }
}
body.nl-page-transition [data-testid="stMainBlockContainer"] {
  animation: nl-page-enter .44s var(--ease) both;
}

/* ═══════════  03 · SCROLLBAR & SELECTION  ═══════════ */
::-webkit-scrollbar { width:10px; height:10px; }
::-webkit-scrollbar-track { background:rgba(5,14,34,.7); }
::-webkit-scrollbar-thumb {
  background:linear-gradient(180deg, rgba(34,211,238,.40), rgba(96,165,250,.34));
  border-radius:8px;
  border:2px solid rgba(5,14,34,.7);
  transition:background .2s var(--ease);
}
::-webkit-scrollbar-thumb:hover {
  background:linear-gradient(180deg, rgba(34,211,238,.65), rgba(52,211,153,.55));
}
::-webkit-scrollbar-corner { background:transparent; }

::selection {
  background:rgba(34,211,238,.35);
  color:#fff;
}

/* ═══════════  04 · FOCUS & A11Y  ═══════════ */
*:focus-visible {
  outline:2px solid var(--aurora-cyan) !important;
  outline-offset:2px !important;
  border-radius:8px;
  box-shadow:0 0 0 4px rgba(34,211,238,.18) !important;
}
button:focus-visible {
  outline-offset:3px !important;
}
@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after {
    animation-duration:.01ms !important;
    animation-iteration-count:1 !important;
    transition-duration:.01ms !important;
    scroll-behavior:auto !important;
  }
}

/* ═══════════  05 · SIDEBAR SHELL  ═══════════ */
[data-testid="stSidebar"] {
  background:
    linear-gradient(180deg, rgba(14,40,88,.99) 0%, rgba(5,14,34,.99) 100%),
    radial-gradient(circle at 0% 0%, rgba(34,211,238,.14), transparent 30rem) !important;
  border-right:1px solid var(--line-2) !important;
  box-shadow:
    inset -1px 0 0 rgba(34,211,238,.12),
    10px 0 40px rgba(0,0,0,.32);
  transition:
    width .30s var(--ease),
    min-width .30s var(--ease),
    transform .30s var(--ease) !important;
}
[data-testid="stSidebar"][aria-expanded="false"] {
  border-right:none !important;
  box-shadow:none !important;
}
[data-testid="stSidebar"] > div:first-child {
  width:100% !important;
  padding:0 .95rem 1.2rem !important;
}

/* ═══════════  06 · SIDEBAR BRAND  ═══════════ */
.sb-brand {
  display:flex; align-items:center; gap:.9rem;
  padding:1.05rem .5rem .85rem;
  border-bottom:1px solid var(--line-1);
  margin-bottom:.75rem;
  position:relative;
}
.sb-brand::after {
  content:'';
  position:absolute;
  bottom:-1px; left:.5rem; right:.5rem; height:1px;
  background:linear-gradient(90deg,transparent,rgba(34,211,238,.55),transparent);
}
.sb-brand-icon {
  position:relative;
  width:46px; height:46px;
  display:grid; place-items:center;
  border:1px solid rgba(34,211,238,.50);
  border-radius:14px;
  background:
    linear-gradient(145deg, rgba(37,99,235,.40), rgba(16,185,129,.24)),
    rgba(14,40,88,.96);
  box-shadow:
    0 0 32px rgba(34,211,238,.34),
    inset 0 1px 0 rgba(255,255,255,.12);
  flex:0 0 auto;
  transition:transform .3s var(--ease-spring), box-shadow .3s var(--ease);
}
.sb-brand-icon:hover {
  transform:scale(1.05) rotate(-3deg);
  box-shadow:0 0 40px rgba(34,211,238,.48), inset 0 1px 0 rgba(255,255,255,.15);
}
.sb-brand-icon::after {
  content:'';
  position:absolute; inset:-1px;
  border-radius:14px;
  padding:1px;
  background:linear-gradient(135deg, #22d3ee, #34d399, #22d3ee);
  background-size:200% 200%;
  -webkit-mask:linear-gradient(#000 0 0) content-box,linear-gradient(#000 0 0);
  -webkit-mask-composite:xor;
  mask-composite:exclude;
  opacity:.85;
  pointer-events:none;
  animation:sb-ring-flow 6s linear infinite;
}
@keyframes sb-ring-flow {
  0%   { background-position:0% 50%; }
  100% { background-position:200% 50%; }
}
.sb-brand-text { min-width:0; }
.sb-brand-name {
  color:var(--text);
  font:800 1.05rem/1.1 var(--font-display);
  white-space:nowrap;
  letter-spacing:-.014em;
}
.sb-brand-ai {
  background:linear-gradient(135deg, var(--aurora-cyan), var(--aurora-emerald));
  -webkit-background-clip:text;
  -webkit-text-fill-color:transparent;
  background-clip:text;
}
.sb-brand-ver {
  margin-left:.42rem;
  color:#8fb0d9;
  font:700 .55rem/1 var(--font-mono);
  padding:.18rem .4rem;
  border:1px solid var(--line-2);
  border-radius:6px;
  background:rgba(34,211,238,.08);
  letter-spacing:.04em;
}
.sb-brand-sub {
  margin-top:.34rem;
  color:#8fb0d9;
  font:500 .62rem/1.2 var(--font-body);
  letter-spacing:.025em;
}
.sb-brand-pulse {
  position:absolute; right:-3px; top:-3px;
  width:10px; height:10px; border-radius:50%;
  background:var(--aurora-emerald);
  box-shadow:
    0 0 12px rgba(52,211,153,1),
    0 0 24px rgba(52,211,153,.6),
    0 0 0 0 rgba(52,211,153,.4);
  animation:pulse-dot 2.4s ease-in-out infinite;
}
@keyframes pulse-dot {
  0%,100% {
    opacity:1; transform:scale(1);
    box-shadow:0 0 12px rgba(52,211,153,1), 0 0 24px rgba(52,211,153,.6), 0 0 0 0 rgba(52,211,153,.4);
  }
  50% {
    opacity:.75; transform:scale(1.15);
    box-shadow:0 0 14px rgba(52,211,153,1), 0 0 28px rgba(52,211,153,.75), 0 0 0 5px rgba(52,211,153,0);
  }
}

/* ═══════════  07 · SIDEBAR SESSION PILL  ═══════════ */
.sb-session {
  display:flex; align-items:center; justify-content:center; gap:.48rem;
  min-height:38px;
  padding:.5rem .6rem;
  margin:.05rem .05rem .65rem;
  border:1px solid rgba(34,211,238,.32);
  border-radius:var(--r-pill);
  background:
    linear-gradient(135deg,
      rgba(37,99,235,.22),
      rgba(34,211,238,.12),
      rgba(16,185,129,.10));
  box-shadow:0 0 18px rgba(34,211,238,.14), inset 0 1px 0 rgba(255,255,255,.06);
  color:var(--text-2);
  position:relative;
  overflow:hidden;
  width:fit-content;
  min-width:155px;
  margin-left:auto;
  margin-right:auto;
  backdrop-filter:blur(12px) saturate(140%);
  -webkit-backdrop-filter:blur(12px) saturate(140%);
}
.sb-session::before {
  content:'';
  position:absolute; inset:0; border-radius:var(--r-pill);
  padding:1px;
  background:linear-gradient(135deg,
    rgba(34,211,238,.55),
    rgba(52,211,153,.38),
    rgba(37,99,235,.55));
  background-size:200% 100%;
  -webkit-mask:linear-gradient(#000 0 0) content-box,linear-gradient(#000 0 0);
  -webkit-mask-composite:xor;
  mask-composite:exclude;
  pointer-events:none;
  opacity:.78;
  animation:sb-ring-flow 8s linear infinite;
}
.sb-session-txt {
  overflow:hidden; text-overflow:ellipsis; white-space:nowrap;
  font:700 .63rem var(--font-body); color:#d5e2f5; letter-spacing:.02em;
}
.sb-session-time {
  color:var(--aurora-cyan);
  font:700 .58rem var(--font-mono);
  background:rgba(34,211,238,.15);
  border:1px solid rgba(34,211,238,.32);
  border-radius:var(--r-pill);
  padding:.1rem .42rem;
  letter-spacing:.05em;
  font-variant-numeric:tabular-nums;
}
.sb-session-dot {
  width:7px; height:7px; border-radius:50%;
  flex:0 0 auto;
  animation:pulse-dot 2s ease-in-out infinite;
}

/* ═══════════  08 · SIDEBAR ENGINE PANEL  ═══════════ */
.sb-engine {
  margin:1rem .05rem 1.05rem;
  padding:1rem;
  border:1px solid var(--line-2);
  border-radius:13px;
  position:relative;
  background:linear-gradient(145deg, rgba(20,50,114,.96), rgba(5,14,34,.98));
  overflow:hidden;
  box-shadow:var(--el-2);
}
.sb-engine::before {
  content:'';
  position:absolute; top:-60px; right:-60px;
  width:160px; height:160px;
  border-radius:50%;
  background:radial-gradient(circle, rgba(34,211,238,.26), transparent 70%);
  pointer-events:none;
}
.sb-engine::after {
  content:'';
  position:absolute; bottom:-60px; left:-60px;
  width:140px; height:140px;
  border-radius:50%;
  background:radial-gradient(circle, rgba(52,211,153,.16), transparent 70%);
  pointer-events:none;
}
.sb-engine-head,
.sb-engine-status,
.sb-engine-confbar { display:flex; align-items:center; gap:.48rem; }
.sb-engine-head { margin-bottom:.7rem; position:relative; }
.sb-engine-title {
  color:#d5e2f5;
  font:700 .73rem var(--font-body);
  letter-spacing:.01em;
}
.sb-engine-live {
  margin-left:auto;
  color:var(--aurora-emerald);
  font:800 .56rem var(--font-mono);
  letter-spacing:.12em;
  padding:.20rem .5rem;
  border-radius:var(--r-pill);
  background:rgba(16,185,129,.15);
  border:1px solid rgba(16,185,129,.42);
  display:inline-flex; align-items:center; gap:.35rem;
}
.sb-engine-live-dot {
  display:inline-block; width:5px; height:5px;
  border-radius:50%; background:var(--aurora-emerald);
  box-shadow:0 0 8px var(--aurora-emerald);
  animation:pulse-dot 1.8s ease-in-out infinite;
}
.sb-engine-status {
  color:var(--text-2);
  font:600 .63rem var(--font-body);
  position:relative;
}
.sb-engine-confbar { margin-top:.75rem; position:relative; }
.sb-engine-confbar-track {
  height:6px; flex:1; overflow:hidden;
  border-radius:var(--r-pill);
  background:var(--ink-1);
  border:1px solid rgba(34,211,238,.18);
  box-shadow:inset 0 1px 2px rgba(0,0,0,.4);
}
.sb-engine-confbar-fill {
  height:100%;
  border-radius:var(--r-pill);
  background:linear-gradient(90deg, var(--aurora-cyan), var(--aurora-emerald));
  box-shadow:0 0 14px rgba(34,211,238,.65);
  transition:width .8s var(--ease);
}
.sb-engine-confbar-lbl {
  min-width:38px; text-align:right;
  color:#d5e2f5;
  font:700 .60rem var(--font-mono);
  font-variant-numeric:tabular-nums;
}
.sb-engine-grid {
  display:grid; grid-template-columns:1fr 1fr;
  gap:.5rem; margin-top:.8rem; position:relative;
}
.sb-mini {
  padding:.58rem .65rem;
  border:1px solid var(--line-1);
  border-radius:var(--r-sm);
  background:linear-gradient(135deg, rgba(20,50,114,.65), rgba(10,29,66,.40));
  transition:border-color .22s var(--ease), transform .22s var(--ease);
}
.sb-mini:hover {
  border-color:var(--line-3);
  transform:translateY(-1px);
}
.sb-mini-lbl {
  color:#8fb0d9;
  font:700 .50rem var(--font-body);
  letter-spacing:.12em;
}
.sb-mini-val {
  margin-top:.20rem;
  color:#e2ecf9;
  font:700 .71rem var(--font-mono);
  overflow:hidden; text-overflow:ellipsis; white-space:nowrap;
  font-variant-numeric:tabular-nums;
}
.sb-mini-white  { border-color:rgba(148,163,184,.28); }
.sb-mini-cyan   { border-color:rgba(34,211,238,.40); }
.sb-mini-violet { border-color:rgba(167,139,250,.40); }

/* ═══════════  09 · SIDEBAR CONFIRM DIALOGS  ═══════════ */
.sb-confirm {
  margin:.6rem 0;
  padding:.8rem;
  border:1px solid rgba(245,158,11,.35);
  border-radius:11px;
  background:linear-gradient(135deg, rgba(245,158,11,.12), rgba(245,158,11,.03));
  position:relative;
  overflow:hidden;
}
.sb-confirm::before {
  content:'';
  position:absolute; top:0; left:0; right:0; height:2px;
  background:linear-gradient(90deg,transparent,rgba(245,158,11,.85),transparent);
}
.sb-confirm-danger {
  border-color:rgba(239,68,68,.38);
  background:linear-gradient(135deg, rgba(239,68,68,.12), rgba(239,68,68,.03));
}
.sb-confirm-danger::before {
  background:linear-gradient(90deg,transparent,rgba(239,68,68,.85),transparent);
}
.sb-confirm-title {
  color:#e2ecf9;
  font:700 .71rem var(--font-body);
  display:flex; align-items:center; gap:.4rem;
}
.sb-confirm-text {
  margin-top:.4rem;
  color:var(--text-2);
  font:.62rem/1.5 var(--font-body);
}
.sb-warning {
  display:flex; gap:.6rem; align-items:flex-start;
  margin:1rem .1rem 0;
  padding:.8rem;
  border:1px solid rgba(245,158,11,.26);
  border-radius:11px;
  background:linear-gradient(135deg, rgba(245,158,11,.10), rgba(245,158,11,.02));
  color:var(--text-2);
  font:.62rem/1.5 var(--font-body);
  position:relative;
}
.sb-warning::before {
  content:'';
  position:absolute;
  left:0; top:8px; bottom:8px;
  width:2px;
  border-radius:2px;
  background:linear-gradient(180deg, var(--aurora-amber), rgba(245,158,11,.2));
}

/* ═══════════  10 · SIDEBAR NAV BUTTONS  ═══════════ */
[data-testid="stSidebar"] .stButton { width:100% !important; margin:.24rem 0 !important; }
[data-testid="stSidebar"] .stButton > button {
  position:relative !important;
  width:100% !important;
  min-height:46px !important;
  padding:.65rem .85rem .65rem 3rem !important;
  display:flex !important;
  align-items:center !important;
  justify-content:flex-start !important;
  text-align:left !important;
  border:1px solid transparent !important;
  border-radius:12px !important;
  background:transparent !important;
  color:var(--text-2) !important;
  font:600 .84rem/1.2 var(--font-body) !important;
  letter-spacing:.005em !important;
  box-shadow:none !important;
  overflow:hidden !important;
  transition:
    background .22s var(--ease),
    border-color .22s var(--ease),
    color .22s var(--ease),
    transform .22s var(--ease),
    box-shadow .22s var(--ease) !important;
}
[data-testid="stSidebar"] .stButton > button::after {
  content:'';
  position:absolute;
  left:0; top:0; bottom:0;
  width:3px;
  border-radius:2px;
  background:linear-gradient(180deg, var(--aurora-cyan), var(--aurora-emerald));
  opacity:0;
  transform:scaleY(.4);
  transition:opacity .22s var(--ease), transform .22s var(--ease);
}
[data-testid="stSidebar"] .stButton > button:hover {
  background:linear-gradient(90deg, rgba(34,211,238,.13), rgba(52,211,153,.07)) !important;
  border-color:var(--line-3) !important;
  color:var(--text) !important;
  transform:translateX(4px) !important;
  box-shadow:0 0 24px rgba(34,211,238,.16), var(--el-1) !important;
}
[data-testid="stSidebar"] .stButton > button:active {
  transform:translateX(2px) scale(.99) !important;
}
[data-testid="stSidebar"] .stButton > button p {
  margin:0 !important; padding:0 !important; width:100% !important;
  color:inherit !important; font:inherit !important;
}

.sb-nav-group {
  margin:1.3rem .5rem .5rem !important;
  font:800 .60rem/1.2 var(--font-body) !important;
  text-transform:uppercase !important;
  letter-spacing:.20em !important;
  background:linear-gradient(90deg, var(--aurora-cyan), var(--aurora-emerald));
  -webkit-background-clip:text;
  -webkit-text-fill-color:transparent;
  background-clip:text;
  opacity:.95;
  position:relative;
}
.sb-nav-group::after {
  content:'';
  display:block;
  height:1px;
  margin-top:.45rem;
  background:linear-gradient(90deg,
    rgba(34,211,238,.35),
    rgba(34,211,238,.05) 60%,
    transparent);
}

/* ═══════════  11 · SIDEBAR NAV ICONS  ═══════════ */
[data-testid="stSidebar"] .stButton > button::before {
  content:'' !important;
  position:absolute !important;
  left:1.1rem !important;
  top:50% !important;
  transform:translateY(-50%) !important;
  width:18px !important;
  height:18px !important;
  background-repeat:no-repeat !important;
  background-position:center !important;
  background-size:18px 18px !important;
  opacity:.85 !important;
  transition:
    opacity .22s var(--ease),
    transform .22s var(--ease),
    filter .22s var(--ease),
    background-image .22s var(--ease) !important;
  pointer-events:none !important;
}
[data-testid="stSidebar"] .stButton > button:hover::before {
  opacity:1 !important;
  transform:translateY(-50%) scale(1.10) !important;
}

[data-testid="stSidebar"] .st-key-nav_home button::before {
  background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%23b8c8e3' stroke-width='1.8' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='m3 9 9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z'/%3E%3Cpolyline points='9 22 9 12 15 12 15 22'/%3E%3C/svg%3E") !important;
}
[data-testid="stSidebar"] .st-key-nav_home button:hover::before {
  background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%23f1f5fd' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='m3 9 9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z'/%3E%3Cpolyline points='9 22 9 12 15 12 15 22'/%3E%3C/svg%3E") !important;
}
[data-testid="stSidebar"] .st-key-nav_mri_analysis button::before {
  background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%23b8c8e3' stroke-width='1.8' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M6 18h8'/%3E%3Cpath d='M3 22h18'/%3E%3Cpath d='M14 22a7 7 0 1 0 0-14h-1'/%3E%3Cpath d='M9 14h2'/%3E%3Cpath d='M9 12a2 2 0 0 1-2-2V6h6v4a2 2 0 0 1-2 2Z'/%3E%3Cpath d='M12 6V3a1 1 0 0 0-1-1H9a1 1 0 0 0-1 1v3'/%3E%3C/svg%3E") !important;
}
[data-testid="stSidebar"] .st-key-nav_mri_analysis button:hover::before {
  background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%23f1f5fd' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M6 18h8'/%3E%3Cpath d='M3 22h18'/%3E%3Cpath d='M14 22a7 7 0 1 0 0-14h-1'/%3E%3Cpath d='M9 14h2'/%3E%3Cpath d='M9 12a2 2 0 0 1-2-2V6h6v4a2 2 0 0 1-2 2Z'/%3E%3Cpath d='M12 6V3a1 1 0 0 0-1-1H9a1 1 0 0 0-1 1v3'/%3E%3C/svg%3E") !important;
}
[data-testid="stSidebar"] .st-key-nav_dashboard button::before {
  background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%23b8c8e3' stroke-width='1.8' stroke-linecap='round' stroke-linejoin='round'%3E%3Cline x1='18' y1='20' x2='18' y2='10'/%3E%3Cline x1='12' y1='20' x2='12' y2='4'/%3E%3Cline x1='6' y1='20' x2='6' y2='14'/%3E%3Cline x1='2' y1='20' x2='22' y2='20'/%3E%3C/svg%3E") !important;
}
[data-testid="stSidebar"] .st-key-nav_dashboard button:hover::before {
  background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%23f1f5fd' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cline x1='18' y1='20' x2='18' y2='10'/%3E%3Cline x1='12' y1='20' x2='12' y2='4'/%3E%3Cline x1='6' y1='20' x2='6' y2='14'/%3E%3Cline x1='2' y1='20' x2='22' y2='20'/%3E%3C/svg%3E") !important;
}
[data-testid="stSidebar"] .st-key-nav_history button::before {
  background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%23b8c8e3' stroke-width='1.8' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8'/%3E%3Cpath d='M3 3v5h5'/%3E%3Cpath d='M12 7v5l4 2'/%3E%3C/svg%3E") !important;
}
[data-testid="stSidebar"] .st-key-nav_history button:hover::before {
  background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%23f1f5fd' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8'/%3E%3Cpath d='M3 3v5h5'/%3E%3Cpath d='M12 7v5l4 2'/%3E%3C/svg%3E") !important;
}
[data-testid="stSidebar"] .st-key-nav_gradcam button::before {
  background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%23b8c8e3' stroke-width='1.8' stroke-linecap='round' stroke-linejoin='round'%3E%3Ccircle cx='12' cy='12' r='3'/%3E%3Cpath d='M12 2v3'/%3E%3Cpath d='M12 19v3'/%3E%3Cpath d='m4.22 4.22 2.12 2.12'/%3E%3Cpath d='m17.66 17.66 2.12 2.12'/%3E%3Cpath d='M2 12h3'/%3E%3Cpath d='M19 12h3'/%3E%3Cpath d='m4.22 19.78 2.12-2.12'/%3E%3Cpath d='m17.66 6.34 2.12-2.12'/%3E%3C/svg%3E") !important;
}
[data-testid="stSidebar"] .st-key-nav_gradcam button:hover::before {
  background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%23f1f5fd' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Ccircle cx='12' cy='12' r='3'/%3E%3Cpath d='M12 2v3'/%3E%3Cpath d='M12 19v3'/%3E%3Cpath d='m4.22 4.22 2.12 2.12'/%3E%3Cpath d='m17.66 17.66 2.12 2.12'/%3E%3Cpath d='M2 12h3'/%3E%3Cpath d='M19 12h3'/%3E%3Cpath d='m4.22 19.78 2.12-2.12'/%3E%3Cpath d='m17.66 6.34 2.12-2.12'/%3E%3C/svg%3E") !important;
}
[data-testid="stSidebar"] .st-key-nav_xai_lab button::before {
  background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%23b8c8e3' stroke-width='1.8' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M14.5 2v17.5c0 1.4-1.1 2.5-2.5 2.5h0c-1.4 0-2.5-1.1-2.5-2.5V2'/%3E%3Cpath d='M8.5 2h7'/%3E%3Cpath d='M14.5 16h-5'/%3E%3Cpath d='m8.5 13 5.5 3'/%3E%3C/svg%3E") !important;
}
[data-testid="stSidebar"] .st-key-nav_xai_lab button:hover::before {
  background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%23f1f5fd' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M14.5 2v17.5c0 1.4-1.1 2.5-2.5 2.5h0c-1.4 0-2.5-1.1-2.5-2.5V2'/%3E%3Cpath d='M8.5 2h7'/%3E%3Cpath d='M14.5 16h-5'/%3E%3Cpath d='m8.5 13 5.5 3'/%3E%3C/svg%3E") !important;
}
[data-testid="stSidebar"] .st-key-nav_settings button::before {
  background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%23b8c8e3' stroke-width='1.8' stroke-linecap='round' stroke-linejoin='round'%3E%3Ccircle cx='12' cy='12' r='3'/%3E%3Cpath d='M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 1 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 1 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 1 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 1 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z'/%3E%3C/svg%3E") !important;
}
[data-testid="stSidebar"] .st-key-nav_settings button:hover::before {
  background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%23f1f5fd' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'%3E%3Ccircle cx='12' cy='12' r='3'/%3E%3Cpath d='M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 1 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 1 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 1 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 1 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z'/%3E%3C/svg%3E") !important;
}

[data-testid="stSidebar"] .st-key-sb_clear button::before {
  background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%23fbbf24' stroke-width='1.8' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpolyline points='3 6 5 6 21 6'/%3E%3Cpath d='M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2'/%3E%3C/svg%3E") !important;
}
[data-testid="stSidebar"] .st-key-sb_clear button:hover::before {
  background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%23fcd34d' stroke-width='2.1' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpolyline points='3 6 5 6 21 6'/%3E%3Cpath d='M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2'/%3E%3C/svg%3E") !important;
  filter:drop-shadow(0 0 8px rgba(251,191,36,.9)) !important;
}
[data-testid="stSidebar"] .st-key-sb_clear button:hover {
  background:linear-gradient(90deg, rgba(245,158,11,.16), rgba(245,158,11,.05)) !important;
  border-color:rgba(245,158,11,.55) !important;
  color:#fbbf24 !important;
  box-shadow:0 0 26px rgba(245,158,11,.22) !important;
}
[data-testid="stSidebar"] .st-key-sb_clear button:hover p { color:#fbbf24 !important; }

[data-testid="stSidebar"] .st-key-sb_reset button::before {
  background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%23f87171' stroke-width='1.8' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M3 12a9 9 0 0 1 9-9 9.75 9.75 0 0 1 6.74 2.74L21 8'/%3E%3Cpath d='M21 3v5h-5'/%3E%3Cpath d='M21 12a9 9 0 0 1-9 9 9.75 9.75 0 0 1-6.74-2.74L3 16'/%3E%3Cpath d='M8 16H3v5'/%3E%3C/svg%3E") !important;
}
[data-testid="stSidebar"] .st-key-sb_reset button:hover::before {
  background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%23fca5a5' stroke-width='2.1' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M3 12a9 9 0 0 1 9-9 9.75 9.75 0 0 1 6.74 2.74L21 8'/%3E%3Cpath d='M21 3v5h-5'/%3E%3Cpath d='M21 12a9 9 0 0 1-9 9 9.75 9.75 0 0 1-6.74-2.74L3 16'/%3E%3Cpath d='M8 16H3v5'/%3E%3C/svg%3E") !important;
  filter:drop-shadow(0 0 8px rgba(248,113,113,.9)) !important;
  transform:translateY(-50%) rotate(45deg) !important;
}
[data-testid="stSidebar"] .st-key-sb_reset button:hover {
  background:linear-gradient(90deg, rgba(239,68,68,.16), rgba(239,68,68,.05)) !important;
  border-color:rgba(239,68,68,.55) !important;
  color:#f87171 !important;
  box-shadow:0 0 26px rgba(239,68,68,.22) !important;
}
[data-testid="stSidebar"] .st-key-sb_reset button:hover p { color:#f87171 !important; }

[data-testid="stSidebar"] .st-key-sb_clear_yes button,
[data-testid="stSidebar"] .st-key-sb_clear_no button,
[data-testid="stSidebar"] .st-key-sb_reset_yes button,
[data-testid="stSidebar"] .st-key-sb_reset_no button {
  padding:.58rem .65rem !important;
  justify-content:center !important;
  text-align:center !important;
  font-size:.76rem !important;
  font-weight:700 !important;
  letter-spacing:.06em !important;
  min-height:38px !important;
  border-radius:10px !important;
  transform:none !important;
}
[data-testid="stSidebar"] .st-key-sb_clear_yes button::before,
[data-testid="stSidebar"] .st-key-sb_clear_no button::before,
[data-testid="stSidebar"] .st-key-sb_reset_yes button::before,
[data-testid="stSidebar"] .st-key-sb_reset_no button::before {
  display:none !important;
  content:none !important;
  background-image:none !important;
}
[data-testid="stSidebar"] .st-key-sb_clear_yes button {
  background:linear-gradient(135deg, rgba(245,158,11,.26), rgba(245,158,11,.12)) !important;
  border-color:rgba(245,158,11,.60) !important;
  color:#fbbf24 !important;
}
[data-testid="stSidebar"] .st-key-sb_clear_yes button:hover {
  background:linear-gradient(135deg, rgba(245,158,11,.45), rgba(245,158,11,.22)) !important;
  border-color:rgba(245,158,11,.90) !important;
  color:#fcd34d !important;
  box-shadow:0 0 26px rgba(245,158,11,.42) !important;
  transform:translateY(-1px) !important;
}
[data-testid="stSidebar"] .st-key-sb_reset_yes button {
  background:linear-gradient(135deg, rgba(239,68,68,.26), rgba(239,68,68,.12)) !important;
  border-color:rgba(239,68,68,.60) !important;
  color:#f87171 !important;
}
[data-testid="stSidebar"] .st-key-sb_reset_yes button:hover {
  background:linear-gradient(135deg, rgba(239,68,68,.45), rgba(239,68,68,.22)) !important;
  border-color:rgba(239,68,68,.90) !important;
  color:#fca5a5 !important;
  box-shadow:0 0 26px rgba(239,68,68,.42) !important;
  transform:translateY(-1px) !important;
}
[data-testid="stSidebar"] .st-key-sb_clear_no button,
[data-testid="stSidebar"] .st-key-sb_reset_no button {
  background:linear-gradient(135deg, rgba(148,163,184,.15), rgba(148,163,184,.05)) !important;
  border-color:rgba(148,163,184,.35) !important;
  color:#b8c8e3 !important;
}
[data-testid="stSidebar"] .st-key-sb_clear_no button:hover,
[data-testid="stSidebar"] .st-key-sb_reset_no button:hover {
  background:linear-gradient(135deg, rgba(148,163,184,.24), rgba(148,163,184,.10)) !important;
  border-color:rgba(148,163,184,.58) !important;
  color:#d5e2f5 !important;
  box-shadow:0 0 18px rgba(148,163,184,.24) !important;
  transform:translateY(-1px) !important;
}

/* ═══════════  12 · GLOBAL BUTTONS  ═══════════ */
.stButton > button,
.stDownloadButton > button {
  min-height:44px !important;
  border-radius:12px !important;
  font:700 .82rem var(--font-body) !important;
  letter-spacing:.01em !important;
  transition:all .22s var(--ease) !important;
  position:relative;
  overflow:hidden;
}
.stButton > button::after,
.stDownloadButton > button::after {
  content:'';
  position:absolute;
  inset:0;
  background:linear-gradient(90deg,transparent,rgba(255,255,255,.08),transparent);
  transform:translateX(-100%);
  transition:transform .6s var(--ease);
  pointer-events:none;
}
.stButton > button:hover::after,
.stDownloadButton > button:hover::after {
  transform:translateX(100%);
}
.stButton > button {
  border:1px solid var(--line-3) !important;
  background:linear-gradient(135deg, rgba(20,50,114,.94), rgba(10,29,66,.94)) !important;
  color:var(--text) !important;
  box-shadow:var(--el-1) !important;
}
.stButton > button:hover,
.stDownloadButton > button:hover {
  border-color:var(--line-glow) !important;
  box-shadow:0 10px 30px rgba(34,211,238,.26), var(--el-2) !important;
  transform:translateY(-2px) !important;
}
.stButton > button:active,
.stDownloadButton > button:active {
  transform:translateY(0) scale(.99) !important;
}
.stButton > button[kind="primary"],
.stDownloadButton > button[kind="primary"] {
  background:linear-gradient(135deg, #0ea5e9 0%, var(--aurora-cyan) 45%, var(--aurora-emerald) 100%) !important;
  border:1px solid rgba(34,211,238,.60) !important;
  color:#fff !important;
  font-weight:700 !important;
  box-shadow:
    0 8px 28px rgba(34,211,238,.44),
    0 0 0 1px rgba(34,211,238,.20),
    inset 0 1px 0 rgba(255,255,255,.24) !important;
}
.stButton > button[kind="primary"]:hover,
.stDownloadButton > button[kind="primary"]:hover {
  background:linear-gradient(135deg, #06b6d4 0%, var(--aurora-emerald) 100%) !important;
  box-shadow:
    0 14px 40px rgba(34,211,238,.58),
    0 0 0 1px rgba(34,211,238,.32),
    inset 0 1px 0 rgba(255,255,255,.30) !important;
  transform:translateY(-3px) scale(1.008) !important;
}
.stDownloadButton > button { width:100% !important; }

@media (pointer: coarse) {
  .stButton > button,
  .stDownloadButton > button {
    min-height: 48px !important;
  }
  [data-testid="stSidebar"] .stButton > button {
    min-height: 50px !important;
  }
}

/* ═══════════  13 · STREAMLIT OVERRIDES  ═══════════ */
[data-testid="stHorizontalBlock"] { align-items:stretch !important; }
[data-testid="stHorizontalBlock"] > div { min-width:0 !important; }

.stTabs [data-baseweb="tab-list"] {
  gap:.35rem !important;
  border-bottom:1px solid var(--line-1) !important;
  padding-bottom:0;
}
.stTabs [data-baseweb="tab"] {
  min-height:44px !important;
  padding:0 1.1rem !important;
  color:#8fb0d9 !important;
  font:700 .75rem var(--font-body) !important;
  transition:color .20s var(--ease);
  letter-spacing:.01em;
  border-radius:10px 10px 0 0 !important;
}
.stTabs [data-baseweb="tab"]:hover { color:var(--text) !important; }
.stTabs [aria-selected="true"] { color:var(--aurora-cyan) !important; }
.stTabs [data-baseweb="tab-highlight"] {
  background:linear-gradient(90deg, var(--aurora-cyan), var(--aurora-emerald)) !important;
  height:2px !important;
  border-radius:2px;
}

.stFileUploader { border-radius:14px !important; }
.stFileUploader section {
  border:1.5px dashed rgba(34,211,238,.48) !important;
  background:linear-gradient(135deg, rgba(37,99,235,.10), rgba(16,185,129,.06)) !important;
  border-radius:14px !important;
  transition:all .24s var(--ease) !important;
}
.stFileUploader section:hover {
  border-color:rgba(34,211,238,.78) !important;
  background:linear-gradient(135deg, rgba(37,99,235,.18), rgba(16,185,129,.12)) !important;
  box-shadow:0 0 32px rgba(34,211,238,.22) !important;
}

.stAlert { border-radius:12px !important; }

[data-testid="stMetric"] {
  background:linear-gradient(135deg, rgba(20,50,114,.82), rgba(10,29,66,.82));
  border:1px solid var(--line-1);
  border-radius:12px;
  padding:.9rem 1rem !important;
  box-shadow:var(--el-1);
  transition:border-color .22s var(--ease), transform .22s var(--ease), box-shadow .22s var(--ease);
}
[data-testid="stMetric"]:hover {
  border-color:var(--line-2);
  transform:translateY(-2px);
  box-shadow:var(--el-2);
}
[data-testid="stMetricLabel"] {
  color:#8fb0d9 !important;
  font-weight:600 !important;
  letter-spacing:.02em;
}
[data-testid="stMetricValue"] {
  color:#e2ecf9 !important;
  font-weight:800 !important;
  font-variant-numeric:tabular-nums;
}

/* ═══════════  14 · STICKY HEADER  ═══════════ */
.sticky-header {
  position:relative;
  width:100%;
  margin:0 0 1.5rem 0;
  display:flex;
  align-items:center;
  flex-wrap:wrap;
  gap:.85rem 1.05rem;
  min-height:66px;
  padding:.75rem 1.2rem;
  border:1px solid rgba(96,165,250,.44);
  border-radius:16px;
  background:
    linear-gradient(90deg, rgba(20,50,114,.96), rgba(7,21,48,.96)),
    rgba(7,21,48,.92);
  backdrop-filter:blur(28px) saturate(180%);
  -webkit-backdrop-filter:blur(28px) saturate(180%);
  box-shadow:
    0 8px 36px rgba(0,0,0,.34),
    0 0 0 1px rgba(34,211,238,.14),
    inset 0 1px 0 rgba(255,255,255,.08);
  font-family:var(--font-body);
  z-index:5;
}
.sticky-header::before {
  content:'';
  position:absolute; inset:0;
  border-radius:16px;
  padding:1px;
  background:linear-gradient(90deg,
    rgba(96,165,250,.72),
    rgba(34,211,238,.74),
    rgba(52,211,153,.62),
    rgba(167,139,250,.62),
    rgba(96,165,250,.72));
  background-size:250% 100%;
  -webkit-mask:linear-gradient(#000 0 0) content-box,linear-gradient(#000 0 0);
  -webkit-mask-composite:xor;
  mask-composite:exclude;
  pointer-events:none;
  opacity:.92;
  animation:hd-border-flow 10s linear infinite;
}
.sticky-header::after {
  content:'';
  position:absolute;
  top:0; left:0; right:0; height:1px;
  background:linear-gradient(90deg,
    transparent,
    rgba(34,211,238,1),
    rgba(52,211,153,.85),
    transparent);
  background-size:200% 100%;
  animation:hd-shine 6s linear infinite;
  pointer-events:none;
}
@keyframes hd-border-flow {
  0%   { background-position:0% 50%; }
  100% { background-position:250% 50%; }
}
@keyframes hd-shine {
  0%   { background-position:200% 50%; opacity:.35; }
  50%  { background-position:0% 50%;   opacity:1; }
  100% { background-position:-200% 50%; opacity:.35; }
}

.hd-brand {
  display:flex; align-items:center; gap:.78rem;
  flex:0 0 auto;
  min-width:0;
  position:relative;
  z-index:1;
}
.hd-brand-icon {
  position:relative;
  width:42px; height:42px;
  display:grid; place-items:center;
  border:1px solid rgba(34,211,238,.52);
  border-radius:12px;
  background:
    linear-gradient(135deg, rgba(37,99,235,.40), rgba(16,185,129,.24)),
    rgba(14,40,88,.96);
  box-shadow:
    0 0 26px rgba(34,211,238,.36),
    inset 0 1px 0 rgba(255,255,255,.14);
  flex:0 0 auto;
  transition:transform .3s var(--ease-spring);
}
.hd-brand-icon:hover {
  transform:scale(1.06) rotate(-3deg);
}
.hd-brand-dot {
  position:absolute; width:8px; height:8px; right:-2px; top:-2px;
  border-radius:50%;
  background:var(--aurora-emerald);
  box-shadow:0 0 12px rgba(52,211,153,1), 0 0 26px rgba(52,211,153,.65);
  animation:pulse-dot 2.4s ease-in-out infinite;
}
.hd-brand-name {
  background:linear-gradient(135deg, #f1f5fd 22%, var(--aurora-cyan) 62%, var(--aurora-emerald) 100%);
  -webkit-background-clip:text;
  -webkit-text-fill-color:transparent;
  background-clip:text;
  font:800 .98rem var(--font-display);
  white-space:nowrap;
  letter-spacing:-.020em;
}
.hd-brand-tag {
  display:block;
  margin-top:.08rem;
  color:#8fb0d9;
  font:700 .55rem var(--font-mono);
  letter-spacing:.18em;
  text-transform:uppercase;
  white-space:nowrap;
}
.hd-divider {
  width:1px; height:34px;
  background:linear-gradient(180deg, transparent, rgba(34,211,238,.55), transparent);
  flex:0 0 auto;
}

/* ═══════════  15 · HEADER TICKER  ═══════════ */
.hd-ticker {
  flex:1 1 auto;
  min-width:180px;
  overflow-x:auto;
  overflow-y:hidden;
  position:relative;
  z-index:1;
  scrollbar-width:thin;
  scrollbar-color: rgba(34,211,238,.40) transparent;
}
.hd-ticker::-webkit-scrollbar { height:3px; }
.hd-ticker::-webkit-scrollbar-track { background:transparent; }
.hd-ticker::-webkit-scrollbar-thumb {
  background:rgba(34,211,238,.40);
  border-radius:2px;
}
.hd-ticker::-webkit-scrollbar-thumb:hover { background:rgba(34,211,238,.62); }
.hd-ticker-inner {
  display:inline-flex;
  align-items:center;
  gap:.9rem;
  white-space:nowrap;
  color:var(--text-2);
  font:.63rem var(--font-mono);
  padding:2px .5rem 2px 0;
}
.hd-tick { flex:0 0 auto; }
.hd-tick b {
  color:#e2ecf9;
  font-weight:700;
  font-variant-numeric:tabular-nums;
}
.hd-tick-idle { color:#8fb0d9; font-style:italic; }
.hd-live-badge {
  color:var(--aurora-emerald);
  font-weight:800;
  font-size:.56rem;
  letter-spacing:.14em;
  padding:.22rem .6rem;
  border:1px solid rgba(52,211,153,.46);
  border-radius:var(--r-pill);
  background:rgba(16,185,129,.14);
  box-shadow:0 0 16px rgba(52,211,153,.34);
  display:inline-flex;
  align-items:center;
  gap:.35rem;
  flex:0 0 auto;
}
.hd-live-badge::before {
  content:'';
  width:5px; height:5px;
  border-radius:50%;
  background:var(--aurora-emerald);
  box-shadow:0 0 8px var(--aurora-emerald);
  animation:pulse-dot 1.6s ease-in-out infinite;
}
.hd-status {
  flex:0 0 auto;
  display:flex; align-items:center; gap:.45rem;
  font:700 .64rem var(--font-body);
  padding:.40rem .85rem;
  border-radius:var(--r-pill);
  white-space:nowrap;
}
.hd-status-ok {
  color:var(--aurora-emerald);
  background:rgba(16,185,129,.15);
  border:1px solid rgba(16,185,129,.42);
  box-shadow:0 0 20px rgba(16,185,129,.24);
}
.hd-status-err {
  color:var(--danger-hi);
  background:rgba(239,68,68,.15);
  border:1px solid rgba(239,68,68,.42);
  box-shadow:0 0 20px rgba(239,68,68,.24);
}
.hd-status-dot {
  width:6px; height:6px; border-radius:50%;
  background:currentColor;
  box-shadow:0 0 8px currentColor;
  animation:pulse-dot 2s ease-in-out infinite;
}
.hd-page {
  flex:0 0 auto;
  color:#e2ecf9;
  font:700 .72rem var(--font-body);
  padding:.40rem .9rem;
  border-radius:var(--r-pill);
  background:linear-gradient(135deg, rgba(37,99,235,.28), rgba(34,211,238,.20));
  border:1px solid rgba(34,211,238,.46);
  box-shadow:0 0 22px rgba(34,211,238,.24);
  white-space:nowrap;
  letter-spacing:.01em;
}
.hd-page-dot {
  display:inline-block;
  width:5px; height:5px;
  margin-right:6px;
  border-radius:50%;
  background:var(--aurora-cyan);
  box-shadow:0 0 8px rgba(34,211,238,1);
  vertical-align:middle;
}

/* ═══════════  16 · HERO  ═══════════ */
.hero { padding:1.7rem 0 1.2rem; text-align:center; position:relative; }
.hero-eyebrow {
  display:inline-flex;
  align-items:center;
  gap:.5rem;
  padding:.38rem .9rem;
  border-radius:var(--r-pill);
  background:linear-gradient(135deg, rgba(37,99,235,.18), rgba(16,185,129,.14));
  border:1px solid rgba(34,211,238,.42);
  color:var(--aurora-cyan);
  font:600 .72rem var(--font-body);
  letter-spacing:.02em;
  margin-bottom:1.05rem;
  box-shadow:0 0 22px rgba(34,211,238,.16);
}
.hero h1 {
  font:800 2.75rem/1.02 var(--font-display);
  letter-spacing:-.038em;
  margin:0 0 .7rem;
  background:linear-gradient(135deg, #f1f5fd 18%, var(--aurora-cyan) 58%, var(--aurora-emerald) 100%);
  -webkit-background-clip:text;
  -webkit-text-fill-color:transparent;
  background-clip:text;
}
.hero-sub {
  color:var(--text-2);
  font:400 1rem/1.65 var(--font-body);
  max-width:72ch;
  margin:0 auto;
  text-align:center;
}
.hero-divider {
  height:1px;
  max-width:560px;
  margin:1.45rem auto 0;
  background:linear-gradient(90deg,
    transparent,
    rgba(34,211,238,.65),
    rgba(52,211,153,.45),
    transparent);
}

/* ═══════════  17 · METRIC CARDS  ═══════════ */
.metric-card {
  background:linear-gradient(135deg, rgba(20,50,114,.80), rgba(10,29,66,.80));
  border:1px solid var(--line-1);
  border-radius:14px;
  padding:1.15rem;
  height:100%;
  transition:
    border-color .24s var(--ease),
    transform .24s var(--ease),
    box-shadow .24s var(--ease);
  position:relative;
  overflow:hidden;
  box-shadow:var(--el-1);
}
.metric-card::before {
  content:'';
  position:absolute; top:-45px; right:-45px;
  width:130px; height:130px;
  border-radius:50%;
  background:radial-gradient(circle, rgba(34,211,238,.18), transparent 70%);
  pointer-events:none;
  transition:opacity .3s var(--ease);
}
.metric-card::after {
  content:'';
  position:absolute;
  top:0; left:0; right:0; height:2px;
  background:linear-gradient(90deg, transparent, rgba(34,211,238,.6), transparent);
  opacity:0;
  transition:opacity .24s var(--ease);
}
.metric-card:hover {
  border-color:rgba(34,211,238,.55);
  transform:translateY(-4px);
  box-shadow:0 18px 42px rgba(34,211,238,.20), var(--el-2);
}
.metric-card:hover::after { opacity:1; }
.metric-icon-wrap {
  width:38px; height:38px;
  display:grid; place-items:center;
  border-radius:11px;
  background:linear-gradient(135deg, rgba(37,99,235,.28), rgba(16,185,129,.18));
  border:1px solid rgba(34,211,238,.38);
  margin-bottom:.8rem;
  transition:transform .3s var(--ease-spring);
}
.metric-card:hover .metric-icon-wrap {
  transform:scale(1.08) rotate(-4deg);
}
.metric-value {
  font:800 1.65rem/1 var(--font-display);
  color:#e2ecf9;
  letter-spacing:-.024em;
  font-variant-numeric:tabular-nums;
}
.metric-label {
  margin-top:.4rem;
  color:#8fb0d9;
  font:600 .68rem var(--font-body);
  text-transform:uppercase;
  letter-spacing:.11em;
}

/* ═══════════  18 · INFO CARDS  ═══════════ */
.info-card {
  background:linear-gradient(135deg, rgba(20,50,114,.80), rgba(10,29,66,.80));
  border:1px solid var(--line-1);
  border-radius:14px;
  padding:1.2rem;
  height:100%;
  transition:
    border-color .24s var(--ease),
    transform .24s var(--ease),
    box-shadow .24s var(--ease);
  box-shadow:var(--el-1);
  position:relative;
  overflow:hidden;
}
.info-card::before {
  content:'';
  position:absolute;
  top:0; left:0; right:0; height:2px;
  background:linear-gradient(90deg, transparent, rgba(34,211,238,.55), transparent);
  opacity:0;
  transition:opacity .24s var(--ease);
}
.info-card:hover {
  border-color:var(--line-2);
  transform:translateY(-3px);
  box-shadow:var(--el-2);
}
.info-card:hover::before { opacity:1; }
.info-card-icon {
  width:46px; height:46px;
  display:grid; place-items:center;
  border-radius:13px;
  background:linear-gradient(135deg, rgba(37,99,235,.28), rgba(16,185,129,.18));
  border:1px solid rgba(34,211,238,.38);
  margin-bottom:.95rem;
  transition:transform .3s var(--ease-spring);
}
.info-card:hover .info-card-icon {
  transform:scale(1.06) rotate(-3deg);
}
.info-card h3 {
  color:#e2ecf9;
  font:700 1rem var(--font-display);
  margin:0 0 .55rem;
  letter-spacing:-.006em;
}
.info-card p {
  color:var(--text-2);
  font:400 .80rem/1.65 var(--font-body);
  margin:.3rem 0;
}
.info-card code {
  font-family:var(--font-mono);
  font-size:.74rem;
  color:var(--aurora-cyan);
  background:rgba(37,99,235,.20);
  padding:.12rem .4rem;
  border-radius:6px;
  border:1px solid rgba(34,211,238,.20);
}

/* ═══════════  19 · STEP CARDS  ═══════════ */
.step-card {
  text-align:center;
  padding:1.05rem .6rem;
  border-radius:12px;
  background:linear-gradient(180deg, rgba(20,50,114,.62), rgba(10,29,66,.62));
  border:1px solid rgba(96,165,250,.20);
  height:100%;
  transition:
    border-color .22s var(--ease),
    transform .22s var(--ease),
    box-shadow .22s var(--ease);
  position:relative;
  overflow:hidden;
}
.step-card:hover {
  border-color:rgba(34,211,238,.50);
  transform:translateY(-4px);
  box-shadow:0 14px 34px rgba(34,211,238,.18);
}
.step-num {
  width:34px; height:34px;
  margin:0 auto .65rem;
  display:grid; place-items:center;
  border-radius:50%;
  background:linear-gradient(135deg, #0ea5e9, var(--aurora-cyan), var(--aurora-emerald));
  color:#fff;
  font:800 .84rem var(--font-display);
  box-shadow:
    0 0 20px rgba(34,211,238,.55),
    inset 0 1px 0 rgba(255,255,255,.30);
}
.step-title {
  color:#e2ecf9;
  font:700 .82rem var(--font-body);
  margin-bottom:.28rem;
  letter-spacing:.005em;
}
.step-desc {
  color:#8fb0d9;
  font:500 .66rem var(--font-body);
}

/* ═══════════  20 · UPLOAD HERO  ═══════════ */
.upload-hero {
  padding:2rem 1.2rem 1.35rem;
  text-align:center;
  border-radius:16px;
  background:linear-gradient(180deg, rgba(37,99,235,.14), rgba(16,185,129,.05), transparent);
  border:1px solid rgba(34,211,238,.34);
  margin-bottom:1.2rem;
  position:relative;
  overflow:hidden;
  box-shadow:var(--el-2);
}
.upload-hero::before {
  content:'';
  position:absolute; top:-80px; left:50%;
  transform:translateX(-50%);
  width:320px; height:320px;
  border-radius:50%;
  background:radial-gradient(circle, rgba(34,211,238,.24), transparent 70%);
  pointer-events:none;
}
.upload-icon-wrap {
  width:68px; height:68px;
  margin:0 auto 1rem;
  display:grid; place-items:center;
  border-radius:18px;
  background:linear-gradient(135deg, rgba(37,99,235,.34), rgba(16,185,129,.24));
  border:1px solid rgba(34,211,238,.50);
  box-shadow:
    0 0 38px rgba(34,211,238,.36),
    inset 0 1px 0 rgba(255,255,255,.14);
  position:relative;
  transition:transform .4s var(--ease-spring);
}
.upload-icon-wrap:hover {
  transform:scale(1.05) rotate(-4deg);
}
.upload-title {
  color:#e2ecf9;
  font:700 1.15rem var(--font-display);
  position:relative;
  letter-spacing:-.012em;
}
.upload-sub {
  color:var(--text-2);
  font:400 .82rem var(--font-body);
  margin:.4rem 0 1rem;
  position:relative;
}
.upload-formats {
  display:flex;
  gap:.5rem;
  justify-content:center;
  flex-wrap:wrap;
  position:relative;
}
.fmt-badge {
  padding:.24rem .65rem;
  border-radius:8px;
  background:linear-gradient(135deg, rgba(37,99,235,.20), rgba(16,185,129,.14));
  border:1px solid rgba(34,211,238,.40);
  color:var(--aurora-cyan);
  font:700 .62rem var(--font-mono);
  letter-spacing:.08em;
}
.fmt-badge-muted {
  background:rgba(148,163,184,.12);
  border-color:rgba(148,163,184,.28);
  color:#8fb0d9;
}
.upload-note {
  margin-top:1.05rem;
  display:inline-flex;
  align-items:center;
  gap:.5rem;
  color:#8fb0d9;
  font:500 .72rem var(--font-body);
  position:relative;
}
.upload-note-dot {
  width:6px; height:6px;
  border-radius:50%;
  background:var(--aurora-emerald);
  box-shadow:0 0 10px rgba(52,211,153,.9);
  animation:pulse-dot 2s ease-in-out infinite;
}

/* ═══════════  21 · DIAGNOSTIC PANEL  ═══════════ */
.diagnostic-panel {
  padding:1.4rem 1.5rem;
  border-radius:15px;
  margin-bottom:1rem;
  background:linear-gradient(135deg, rgba(37,99,235,.18), rgba(16,185,129,.10), rgba(20,50,114,.90));
  border:1px solid rgba(34,211,238,.46);
  position:relative;
  overflow:hidden;
  box-shadow:var(--el-2);
}
.diagnostic-panel::before {
  content:'';
  position:absolute; top:-60px; right:-60px;
  width:190px; height:190px;
  border-radius:50%;
  background:radial-gradient(circle, rgba(34,211,238,.24), transparent 70%);
  pointer-events:none;
}
.diagnostic-panel::after {
  content:'';
  position:absolute;
  top:0; left:0; right:0; height:2px;
  background:linear-gradient(90deg, transparent, rgba(34,211,238,.75), rgba(52,211,153,.55), transparent);
}
.diag-header {
  display:flex;
  justify-content:space-between;
  align-items:center;
  margin-bottom:.85rem;
  gap:.6rem;
  flex-wrap:wrap;
  position:relative;
}
.diag-label {
  font:700 .64rem var(--font-body);
  background:linear-gradient(90deg, var(--aurora-cyan), var(--aurora-emerald));
  -webkit-background-clip:text;
  -webkit-text-fill-color:transparent;
  background-clip:text;
  text-transform:uppercase;
  letter-spacing:.18em;
}
.badge {
  display:inline-flex;
  align-items:center;
  gap:.35rem;
  padding:.28rem .7rem;
  border-radius:var(--r-pill);
  font:700 .62rem var(--font-body);
  letter-spacing:.05em;
}
.badge-research { background:rgba(167,139,250,.18); border:1px solid rgba(167,139,250,.44); color:var(--aurora-violet); }
.badge-high     { background:rgba(16,185,129,.20);  border:1px solid rgba(16,185,129,.48); color:var(--aurora-emerald); }
.badge-moderate { background:rgba(245,158,11,.20);  border:1px solid rgba(245,158,11,.48); color:var(--aurora-amber); }
.badge-low      { background:rgba(239,68,68,.20);   border:1px solid rgba(239,68,68,.48);  color:var(--danger-hi); }
.diag-prediction {
  font:800 2.2rem/1.04 var(--font-display);
  letter-spacing:-.034em;
  background:linear-gradient(135deg, #f1f5fd, var(--aurora-cyan));
  -webkit-background-clip:text;
  -webkit-text-fill-color:transparent;
  background-clip:text;
  position:relative;
}
.diag-confidence {
  margin-top:.35rem;
  color:var(--text-2);
  font:500 .86rem var(--font-body);
  position:relative;
}

/* ═══════════  22 · UNCERTAINTY CARD  ═══════════ */
.uncertainty-card {
  margin:1rem 0;
  padding:1.2rem 1.35rem;
  border-radius:14px;
  background:linear-gradient(135deg, rgba(167,139,250,.16), rgba(34,211,238,.08));
  border:1px solid rgba(167,139,250,.40);
  box-shadow:var(--el-1);
  position:relative;
  overflow:hidden;
}
.uncertainty-card::after {
  content:'';
  position:absolute;
  top:0; left:0; right:0; height:2px;
  background:linear-gradient(90deg, transparent, rgba(167,139,250,.7), transparent);
}
.unc-title {
  font:700 .62rem var(--font-body);
  background:linear-gradient(90deg, var(--aurora-violet), var(--aurora-cyan));
  -webkit-background-clip:text;
  -webkit-text-fill-color:transparent;
  background-clip:text;
  text-transform:uppercase;
  letter-spacing:.18em;
  margin-bottom:.8rem;
}
.unc-value {
  font:800 1.5rem/1 var(--font-mono);
  letter-spacing:-.018em;
  font-variant-numeric:tabular-nums;
}
.unc-band {
  margin-top:.3rem;
  font:600 .78rem var(--font-body);
}

/* ═══════════  23 · XAI CARD  ═══════════ */
.xai-card {
  margin:.9rem 0;
  padding:1.1rem 1.35rem;
  border-radius:14px;
  background:linear-gradient(135deg, rgba(37,99,235,.14), rgba(16,185,129,.08));
  border:1px solid rgba(34,211,238,.36);
  box-shadow:var(--el-1);
  position:relative;
  overflow:hidden;
}
.xai-card::after {
  content:'';
  position:absolute;
  top:0; left:0; right:0; height:2px;
  background:linear-gradient(90deg, transparent, rgba(34,211,238,.65), transparent);
}
.xai-title {
  display:flex;
  align-items:center;
  gap:.5rem;
  font:700 .62rem var(--font-body);
  background:linear-gradient(90deg, var(--aurora-cyan), var(--aurora-emerald));
  -webkit-background-clip:text;
  -webkit-text-fill-color:transparent;
  background-clip:text;
  text-transform:uppercase;
  letter-spacing:.16em;
  margin-bottom:.75rem;
}
.xai-text {
  color:var(--text-2);
  font:400 .82rem/1.7 var(--font-body);
}
.xai-text b  { color:#e2ecf9; font-weight:700; }
.xai-text em { color:#e2ecf9; font-style:italic; }

/* ═══════════  24 · DISCLAIMER  ═══════════ */
.disclaimer {
  display:flex;
  align-items:flex-start;
  gap:.7rem;
  margin:1rem 0;
  padding:.95rem 1.15rem;
  border-radius:12px;
  background:linear-gradient(135deg, rgba(245,158,11,.14), rgba(245,158,11,.03));
  border:1px solid rgba(245,158,11,.36);
  color:var(--text-2);
  font:400 .80rem/1.6 var(--font-body);
  box-shadow:var(--el-1);
  position:relative;
}
.disclaimer::before {
  content:'';
  position:absolute;
  left:0; top:10px; bottom:10px;
  width:3px;
  border-radius:3px;
  background:linear-gradient(180deg, var(--aurora-amber), rgba(245,158,11,.25));
}
.disclaimer b { color:var(--aurora-amber); font-weight:700; }

/* ═══════════  25 · PROBABILITY GRID  ═══════════ */
.prob-grid {
  display:grid;
  grid-template-columns:repeat(auto-fit, minmax(150px, 1fr));
  gap:.75rem;
  margin:.55rem 0 .3rem;
}
.prob-card {
  position:relative;
  padding:1.05rem 1.15rem;
  border-radius:13px;
  background:linear-gradient(135deg, rgba(20,50,114,.82), rgba(10,29,66,.82));
  border:1px solid var(--line-1);
  transition:
    border-color .24s var(--ease),
    transform .24s var(--ease),
    box-shadow .24s var(--ease);
  box-shadow:var(--el-1);
}
.prob-card:hover {
  border-color:rgba(34,211,238,.50);
  transform:translateY(-4px);
  box-shadow:0 16px 38px rgba(34,211,238,.18), var(--el-2);
}
.prob-card.is-top {
  border-color:rgba(34,211,238,.68);
  background:linear-gradient(135deg, rgba(37,99,235,.26), rgba(16,185,129,.18), rgba(20,50,114,.90));
  box-shadow:0 0 34px rgba(34,211,238,.30), var(--el-2);
}
.prob-card.is-top::after {
  content:'';
  position:absolute;
  top:0; left:0; right:0; height:2px;
  border-radius:13px 13px 0 0;
  background:linear-gradient(90deg, var(--aurora-cyan), var(--aurora-emerald));
}
.prob-value {
  font:800 1.5rem/1 var(--font-mono);
  color:#e2ecf9;
  letter-spacing:-.024em;
  font-variant-numeric:tabular-nums;
}
.prob-card.is-top .prob-value {
  background:linear-gradient(135deg, var(--aurora-cyan), var(--aurora-emerald));
  -webkit-background-clip:text;
  -webkit-text-fill-color:transparent;
  background-clip:text;
}
.prob-label {
  margin-top:.4rem;
  color:var(--text-2);
  font:600 .75rem var(--font-body);
}
.prob-top-tag {
  position:absolute;
  top:.6rem; right:.6rem;
  padding:.18rem .55rem;
  border-radius:var(--r-pill);
  background:linear-gradient(135deg, rgba(34,211,238,.32), rgba(16,185,129,.24));
  border:1px solid rgba(34,211,238,.58);
  color:var(--aurora-cyan);
  font:700 .53rem var(--font-body);
  letter-spacing:.08em;
  text-transform:uppercase;
}

/* ═══════════  26 · EMPTY STATE  ═══════════ */
.empty-state {
  padding:3.5rem 1.3rem;
  text-align:center;
  border-radius:16px;
  border:1.5px dashed rgba(96,165,250,.34);
  background:linear-gradient(135deg, rgba(20,50,114,.42), rgba(10,29,66,.42));
  position:relative;
  overflow:hidden;
}
.empty-state::before {
  content:'';
  position:absolute;
  top:50%; left:50%;
  transform:translate(-50%, -50%);
  width:340px; height:340px;
  border-radius:50%;
  background:radial-gradient(circle, rgba(34,211,238,.10), transparent 70%);
  pointer-events:none;
}
.empty-icon {
  width:78px; height:78px;
  margin:0 auto 1.1rem;
  display:grid; place-items:center;
  border-radius:19px;
  background:linear-gradient(135deg, rgba(37,99,235,.22), rgba(16,185,129,.14));
  border:1px solid rgba(96,165,250,.38);
  box-shadow:0 0 32px rgba(37,99,235,.18);
  position:relative;
}
.empty-title {
  color:#d5e2f5;
  font:700 1.05rem var(--font-display);
  letter-spacing:-.006em;
  position:relative;
}
.empty-text {
  margin-top:.5rem;
  color:#8fb0d9;
  font:400 .84rem var(--font-body);
  position:relative;
}

/* ═══════════  27 · DISTRIBUTION BARS  ═══════════ */
.dist-card {
  margin:.55rem 0;
  padding:.8rem 1rem;
  border-radius:12px;
  background:linear-gradient(135deg, rgba(20,50,114,.68), rgba(10,29,66,.68));
  border:1px solid rgba(96,165,250,.20);
  transition:border-color .22s var(--ease), transform .22s var(--ease);
}
.dist-card:hover {
  border-color:rgba(34,211,238,.38);
  transform:translateX(2px);
}
.dist-row {
  display:flex;
  justify-content:space-between;
  align-items:center;
  margin-bottom:.55rem;
}
.dist-name  { color:#d5e2f5; font:600 .84rem var(--font-body); }
.dist-count {
  color:var(--text-2);
  font:600 .74rem var(--font-mono);
  font-variant-numeric:tabular-nums;
}
.dist-track {
  height:7px;
  border-radius:var(--r-pill);
  background:rgba(96,165,250,.16);
  overflow:hidden;
  box-shadow:inset 0 1px 2px rgba(0,0,0,.35);
}
.dist-fill {
  height:100%;
  border-radius:var(--r-pill);
  background:linear-gradient(90deg, var(--aurora-cyan), var(--aurora-emerald));
  box-shadow:0 0 14px rgba(34,211,238,.60);
  transition:width 1.2s var(--ease);
  position:relative;
}
.dist-fill::after {
  content:'';
  position:absolute;
  inset:0;
  background:linear-gradient(90deg, transparent, rgba(255,255,255,.25), transparent);
  animation:dist-shine 2.8s linear infinite;
}
@keyframes dist-shine {
  0%   { transform:translateX(-100%); }
  100% { transform:translateX(100%); }
}

/* ═══════════  28 · LATEST CARD  ═══════════ */
.latest-card {
  padding:.9rem 1rem;
  border-radius:12px;
  background:linear-gradient(135deg, rgba(20,50,114,.65), rgba(10,29,66,.65));
  border:1px solid rgba(96,165,250,.22);
}
.latest-row {
  display:flex;
  justify-content:space-between;
  gap:.75rem;
  padding:.55rem 0;
  border-bottom:1px solid rgba(96,165,250,.14);
}
.latest-row:last-child { border-bottom:0; }
.latest-key {
  color:#8fb0d9;
  font:600 .73rem var(--font-body);
  letter-spacing:.01em;
}
.latest-val {
  color:#d5e2f5;
  font:600 .74rem var(--font-mono);
  text-align:right;
  word-break:break-word;
  font-variant-numeric:tabular-nums;
}

/* ═══════════  29 · THUMB CARD (HISTORY)  ═══════════ */
.thumb-card {
  display:flex;
  align-items:center;
  gap:1.05rem;
  padding:.9rem 1.15rem;
  margin:.55rem 0;
  border-radius:13px;
  background:linear-gradient(135deg, rgba(20,50,114,.68), rgba(10,29,66,.68));
  border:1px solid rgba(96,165,250,.20);
  transition:
    border-color .24s var(--ease),
    transform .24s var(--ease),
    box-shadow .24s var(--ease);
}
.thumb-card:hover {
  border-color:rgba(34,211,238,.48);
  transform:translateX(5px);
  box-shadow:0 8px 28px rgba(34,211,238,.18);
}
.thumb-icon {
  width:42px; height:42px;
  display:grid; place-items:center;
  border-radius:12px;
  background:linear-gradient(135deg, rgba(37,99,235,.24), rgba(16,185,129,.16));
  border:1px solid rgba(34,211,238,.38);
  flex:0 0 auto;
}
.thumb-info { flex:1; min-width:0; }
.thumb-title {
  color:#e2ecf9;
  font:700 .90rem var(--font-body);
  letter-spacing:-.005em;
}
.thumb-meta {
  margin-top:.25rem;
  color:#8fb0d9;
  font:500 .72rem var(--font-mono);
  overflow:hidden;
  text-overflow:ellipsis;
  white-space:nowrap;
}
.conf-badge {
  display:inline-block;
  padding:.34rem .8rem;
  border-radius:var(--r-pill);
  font:700 .76rem var(--font-mono);
  letter-spacing:.01em;
  font-variant-numeric:tabular-nums;
}
.conf-hi  { background:rgba(16,185,129,.22); border:1px solid rgba(16,185,129,.48); color:var(--aurora-emerald); }
.conf-mid { background:rgba(245,158,11,.22); border:1px solid rgba(245,158,11,.48); color:var(--aurora-amber); }
.conf-lo  { background:rgba(239,68,68,.22);  border:1px solid rgba(239,68,68,.48);  color:var(--danger-hi); }

/* ═══════════  30 · ACTIVITY FEED  ═══════════ */
.act-feed {
  padding:.75rem .85rem;
  border-radius:12px;
  max-height:390px;
  overflow-y:auto;
  background:linear-gradient(135deg, rgba(20,50,114,.65), rgba(10,29,66,.65));
  border:1px solid rgba(96,165,250,.20);
}
.act-row {
  display:flex;
  gap:.65rem;
  padding:.5rem .3rem;
  border-bottom:1px solid rgba(96,165,250,.12);
  font:500 .74rem var(--font-body);
}
.act-row:last-child { border-bottom:0; }
.act-time {
  color:#8fb0d9;
  font:600 .68rem var(--font-mono);
  flex:0 0 auto;
  font-variant-numeric:tabular-nums;
}
.act-msg { color:var(--text-2); word-break:break-word; }
.act-success .act-msg { color:var(--aurora-emerald); }
.act-warn    .act-msg { color:var(--aurora-amber); }
.act-error   .act-msg { color:var(--danger-hi); }

/* ═══════════  31 · EXPORT BLOCK  ═══════════ */
.export-head {
  display:flex;
  align-items:center;
  gap:.9rem;
  margin:1.35rem 0 .9rem;
  padding:.9rem 1.15rem;
  border-radius:13px;
  background:linear-gradient(135deg, rgba(20,50,114,.80), rgba(10,29,66,.80));
  border:1px solid rgba(96,165,250,.26);
  box-shadow:var(--el-1);
  position:relative;
  overflow:hidden;
}
.export-head::before {
  content:'';
  position:absolute;
  top:0; left:0; right:0; height:2px;
  background:linear-gradient(90deg, transparent, rgba(34,211,238,.55), transparent);
}
.export-head-icon {
  width:40px; height:40px;
  display:grid; place-items:center;
  border-radius:11px;
  background:linear-gradient(135deg, rgba(37,99,235,.24), rgba(16,185,129,.16));
  border:1px solid rgba(34,211,238,.38);
}
.export-head-title {
  color:#e2ecf9;
  font:700 .90rem var(--font-display);
  letter-spacing:-.005em;
}
.export-head-sub {
  color:#8fb0d9;
  font:500 .73rem var(--font-body);
}
.export-head-badge {
  margin-left:auto;
  padding:.24rem .7rem;
  border-radius:var(--r-pill);
  background:rgba(16,185,129,.20);
  border:1px solid rgba(16,185,129,.46);
  color:var(--aurora-emerald);
  font:700 .62rem var(--font-body);
  letter-spacing:.08em;
}
.export-item-label {
  color:#8fb0d9;
  font:700 .64rem var(--font-body);
  text-transform:uppercase;
  letter-spacing:.12em;
  margin:.35rem 0 .5rem;
}

/* ═══════════  32 · FOOTER  ═══════════ */
.app-footer {
  display:grid;
  grid-template-columns:1.6fr 1.1fr 0.95fr;
  gap:2.15rem;
  margin-top:3.25rem;
  padding:2.3rem;
  border-radius:20px;
  position:relative;
  background:
    linear-gradient(180deg, rgba(20,50,114,.90), rgba(3,10,24,.98)),
    radial-gradient(circle at 0% 0%, rgba(34,211,238,.14), transparent 30rem);
  border:1px solid var(--line-3);
  overflow:hidden;
  box-shadow:0 28px 72px rgba(0,0,0,.42);
}
.app-footer::before {
  content:'';
  position:absolute; inset:0;
  border-radius:20px;
  padding:1px;
  background:linear-gradient(135deg,
    rgba(37,99,235,.72),
    rgba(34,211,238,.62),
    rgba(52,211,153,.72));
  background-size:200% 200%;
  -webkit-mask:linear-gradient(#000 0 0) content-box,linear-gradient(#000 0 0);
  -webkit-mask-composite:xor;
  mask-composite:exclude;
  pointer-events:none;
  opacity:.92;
  animation:hd-border-flow 12s linear infinite;
}
.app-footer::after {
  content:'';
  position:absolute;
  top:-160px; right:-160px;
  width:360px; height:360px;
  border-radius:50%;
  background:radial-gradient(circle, rgba(34,211,238,.24), transparent 70%);
  pointer-events:none;
}
.footer-brand-lockup {
  display:flex;
  align-items:center;
  gap:.9rem;
  margin-bottom:1.1rem;
  position:relative;
  z-index:1;
}
.footer-brand-icon {
  width:60px; height:60px;
  display:grid; place-items:center;
  border-radius:16px;
  background:linear-gradient(135deg, rgba(37,99,235,.42), rgba(16,185,129,.28));
  border:1px solid rgba(34,211,238,.50);
  box-shadow:0 0 34px rgba(34,211,238,.36);
  position:relative;
}
.footer-brand-icon::after {
  content:'';
  position:absolute; inset:-1px;
  border-radius:16px;
  padding:1px;
  background:linear-gradient(135deg, var(--aurora-cyan), var(--aurora-emerald));
  -webkit-mask:linear-gradient(#000 0 0) content-box,linear-gradient(#000 0 0);
  -webkit-mask-composite:xor;
  mask-composite:exclude;
  opacity:.72;
  pointer-events:none;
}
.footer-brand-name {
  background:linear-gradient(135deg, #f1f5fd 18%, var(--aurora-cyan) 68%, var(--aurora-emerald) 100%);
  -webkit-background-clip:text;
  -webkit-text-fill-color:transparent;
  background-clip:text;
  font:800 1.25rem var(--font-display);
  letter-spacing:-.024em;
}
.footer-brand-tag {
  color:#8fb0d9;
  font:600 .72rem var(--font-body);
  letter-spacing:.08em;
  text-transform:uppercase;
  margin-top:.15rem;
}
.footer-desc {
  color:var(--text-2);
  font:400 .82rem/1.72 var(--font-body);
  max-width:50ch;
  margin-bottom:1.15rem;
  position:relative;
  z-index:1;
}
.footer-copy {
  color:#8fb0d9;
  font:600 .72rem var(--font-body);
  position:relative;
  z-index:1;
}
.footer-col-title {
  background:linear-gradient(90deg, var(--aurora-cyan), var(--aurora-emerald));
  -webkit-background-clip:text;
  -webkit-text-fill-color:transparent;
  background-clip:text;
  font:800 .64rem var(--font-body);
  text-transform:uppercase;
  letter-spacing:.18em;
  margin-bottom:.85rem;
  position:relative;
  z-index:1;
}
.footer-tech-badges {
  display:flex;
  flex-wrap:wrap;
  gap:.5rem;
  position:relative;
  z-index:1;
}
.ft-badge {
  display:inline-flex;
  align-items:center;
  gap:.42rem;
  padding:.4rem .85rem;
  border-radius:10px;
  background:linear-gradient(135deg, rgba(37,99,235,.22), rgba(16,185,129,.14));
  border:1px solid rgba(34,211,238,.36);
  color:#e2ecf9;
  font:600 .71rem var(--font-mono);
  transition:all .22s var(--ease);
}
.ft-badge:hover {
  border-color:rgba(34,211,238,.65);
  box-shadow:0 0 22px rgba(34,211,238,.36);
  transform:translateY(-2px);
}
.ft-badge-dot {
  width:5px; height:5px;
  border-radius:50%;
  background:linear-gradient(135deg, var(--aurora-cyan), var(--aurora-emerald));
  box-shadow:0 0 8px rgba(34,211,238,.95);
}
.footer-stats {
  display:flex;
  flex-direction:column;
  gap:.7rem;
  position:relative;
  z-index:1;
}
.f-stat {
  padding:.85rem 1rem;
  border-radius:12px;
  background:linear-gradient(135deg, rgba(20,50,114,.80), rgba(10,29,66,.65));
  border:1px solid var(--line-1);
  transition:all .22s var(--ease);
}
.f-stat:hover {
  border-color:var(--line-2);
  box-shadow:0 0 24px rgba(34,211,238,.20);
  transform:translateY(-2px);
}
.f-stat-wide { padding:.95rem 1rem; }
.f-stat-val {
  display:flex;
  align-items:center;
  gap:.6rem;
  background:linear-gradient(135deg, #f1f5fd, var(--aurora-cyan));
  -webkit-background-clip:text;
  -webkit-text-fill-color:transparent;
  background-clip:text;
  font:800 1.35rem/1 var(--font-display);
  letter-spacing:-.024em;
  font-variant-numeric:tabular-nums;
}
.f-stat-val .f-dot,
.f-stat-val .f-dot-off {
  -webkit-text-fill-color:currentColor;
  background:currentColor;
  -webkit-background-clip:initial;
  background-clip:initial;
}
.f-stat-lbl {
  margin-top:.45rem;
  color:#8fb0d9;
  font:700 .64rem var(--font-body);
  text-transform:uppercase;
  letter-spacing:.12em;
}
.f-dot {
  width:9px; height:9px;
  border-radius:50%;
  background:var(--aurora-emerald);
  box-shadow:0 0 12px rgba(52,211,153,1);
  animation:pulse-dot 2s ease-in-out infinite;
}
.f-dot-off {
  background:var(--danger-hi);
  box-shadow:0 0 12px rgba(248,113,113,1);
}

/* ═══════════  33 · COPYRIGHT  ═══════════ */
.copyright-line {
  margin-top:1.35rem;
  padding:1.35rem;
  text-align:center;
  font:500 .78rem var(--font-body);
  border-top:1px solid var(--line-1);
  background:linear-gradient(180deg, transparent, rgba(34,211,238,.05));
}

/* ═══════════  34 · UTILITY & DECORATION  ═══════════ */
.nl-divider {
  height:1px;
  margin:1.5rem 0;
  background:linear-gradient(90deg,
    transparent,
    rgba(34,211,238,.42),
    rgba(52,211,153,.28),
    transparent);
}

.nl-pill {
  display:inline-flex;
  align-items:center;
  gap:.35rem;
  padding:.28rem .7rem;
  border-radius:var(--r-pill);
  font:700 .62rem var(--font-body);
  letter-spacing:.06em;
  text-transform:uppercase;
}
.nl-pill-info    { background:rgba(34,211,238,.16); border:1px solid rgba(34,211,238,.42); color:var(--aurora-cyan); }
.nl-pill-success { background:rgba(16,185,129,.18); border:1px solid rgba(16,185,129,.46); color:var(--aurora-emerald); }
.nl-pill-warn    { background:rgba(245,158,11,.18); border:1px solid rgba(245,158,11,.46); color:var(--aurora-amber); }
.nl-pill-danger  { background:rgba(239,68,68,.18);  border:1px solid rgba(239,68,68,.46);  color:var(--danger-hi); }

.nl-chip {
  display:inline-block;
  padding:.22rem .6rem;
  border-radius:8px;
  background:rgba(34,211,238,.12);
  border:1px solid rgba(34,211,238,.32);
  color:var(--aurora-cyan);
  font:700 .62rem var(--font-mono);
  letter-spacing:.05em;
}

/* ═══════════  35 · LOADING / SKELETON  ═══════════ */
@keyframes nl-skeleton-shimmer {
  0%   { background-position:-200% 0; }
  100% { background-position:200% 0; }
}
.nl-skeleton {
  background:linear-gradient(
    90deg,
    rgba(96,165,250,.06) 25%,
    rgba(34,211,238,.14) 50%,
    rgba(96,165,250,.06) 75%
  );
  background-size:200% 100%;
  animation:nl-skeleton-shimmer 1.6s linear infinite;
  border-radius:8px;
}

@keyframes nl-spin {
  to { transform:rotate(360deg); }
}
.nl-spinner {
  width:22px; height:22px;
  border-radius:50%;
  border:2.5px solid rgba(34,211,238,.20);
  border-top-color:var(--aurora-cyan);
  animation:nl-spin .9s linear infinite;
}

.nl-progress-ring {
  transform:rotate(-90deg);
}
.nl-progress-ring__circle {
  transition:stroke-dashoffset .6s var(--ease);
  stroke-linecap:round;
}

/* ═══════════  36 · RESPONSIVE  ═══════════ */

@media (max-width: 1200px) {
  :root { --max-w: 1280px; }
}

@media (max-width: 1100px) {
  :root { --sb-w: 250px; }
  .app-footer { grid-template-columns: 1fr 1fr; }
  .hd-ticker { flex-basis: 100%; order: 10; }
  .hero h1 { font-size: 2.5rem; }
}

@media (max-width: 900px) {
  .app-footer { grid-template-columns: 1fr 1fr; gap: 1.6rem; }
  .hero h1 { font-size: 2.35rem; }
  .diag-prediction { font-size: 1.9rem; }
  .metric-value { font-size: 1.45rem; }
  .prob-value { font-size: 1.35rem; }
  [data-testid="stMainBlockContainer"] {
    padding: 1.5rem 1.15rem 2.8rem !important;
  }
}

@media (max-width: 760px) {
  .sb-engine-grid { grid-template-columns: 1fr 1fr; }
  .hd-brand-name { font-size: .88rem; }
  .hd-brand-tag { font-size: .52rem; }
  .hd-page { font-size: .64rem; padding: .34rem .75rem; }
  .hd-status { font-size: .60rem; padding: .34rem .7rem; }
  .hd-divider { height: 26px; }
  .hero h1 { font-size: 2.15rem; }
  .sticky-header {
    padding: .7rem 1rem;
    gap: .7rem .85rem;
    min-height: auto;
  }
  .metric-card, .info-card { padding: 1rem; }
  .prob-grid { grid-template-columns: repeat(2, 1fr); }
}

@media (max-width: 640px) {
  .app-footer {
    grid-template-columns: 1fr;
    padding: 1.6rem;
    gap: 1.4rem;
  }
  .hero h1 { font-size: 1.95rem; letter-spacing: -.03em; }
  .hero-sub { font-size: .92rem; line-height: 1.6; }
  .diag-prediction { font-size: 1.65rem; }
  .prob-grid { grid-template-columns: repeat(2, 1fr); gap: .6rem; }
  .sticky-header {
    padding: .65rem .9rem;
    gap: .6rem .75rem;
    border-radius: 14px;
  }
  [data-testid="stMainBlockContainer"] {
    padding: 1.3rem .95rem 2.4rem !important;
  }
  .upload-hero { padding: 1.6rem 1rem 1.2rem; }
  .diagnostic-panel { padding: 1.2rem 1.15rem; }
  .metric-card, .info-card, .step-card {
    padding: .95rem;
  }
  .sb-brand-name { font-size: .98rem; }
  .sb-engine { padding: .9rem; }
}

@media (max-width: 560px) {
  .stButton > button,
  .stDownloadButton > button {
    min-height: 48px !important;
    font-size: .80rem !important;
  }
  [data-testid="stMetric"] { padding: .75rem !important; }
  .stTabs [data-baseweb="tab"] {
    padding: 0 .65rem !important;
    font-size: .68rem !important;
    min-height: 42px !important;
  }
  .hero-sub { font-size: .90rem; }
  .prob-grid { grid-template-columns: 1fr 1fr; }
  .diag-prediction { font-size: 1.5rem; }
  .metric-value { font-size: 1.35rem; }
  .unc-value { font-size: 1.3rem; }
  .hd-brand-icon { width: 38px; height: 38px; }
  .hd-brand-name { font-size: .84rem; }
  .upload-icon-wrap { width: 60px; height: 60px; }
  .empty-icon { width: 68px; height: 68px; }
}

@media (max-width: 480px) {
  :root {
    --sp-4: .85rem;
    --sp-5: 1.05rem;
    --sp-6: 1.25rem;
  }
  .hero h1 { font-size: 1.75rem; }
  .hero-eyebrow { font-size: .68rem; padding: .32rem .75rem; }
  .diag-prediction { font-size: 1.4rem; }
  .prob-grid { grid-template-columns: 1fr; }
  .metric-card, .info-card {
    padding: .85rem;
  }
  .sticky-header {
    padding: .55rem .75rem;
    gap: .5rem;
    border-radius: 12px;
  }
  .hd-page, .hd-status {
    font-size: .58rem;
    padding: .28rem .6rem;
  }
  .hd-ticker-inner { font-size: .58rem; gap: .65rem; }
  [data-testid="stMainBlockContainer"] {
    padding: 1.1rem .8rem 2.2rem !important;
  }
  .app-footer { padding: 1.35rem; gap: 1.2rem; }
  .footer-brand-icon { width: 52px; height: 52px; }
  .footer-brand-name { font-size: 1.1rem; }
  .sb-brand-icon { width: 42px; height: 42px; }
  .sb-session { min-width: 140px; padding: .42rem .55rem; }
  [data-testid="stSidebar"] .stButton > button {
    min-height: 50px !important;
    padding: .7rem .8rem .7rem 3.1rem !important;
    font-size: .86rem !important;
  }
  [data-testid="stSidebar"] .st-key-sb_clear_yes button,
  [data-testid="stSidebar"] .st-key-sb_clear_no button,
  [data-testid="stSidebar"] .st-key-sb_reset_yes button,
  [data-testid="stSidebar"] .st-key-sb_reset_no button {
    min-height: 44px !important;
    font-size: .78rem !important;
  }
}

@media (max-width: 380px) {
  .hero h1 { font-size: 1.55rem; }
  .diag-prediction { font-size: 1.25rem; }
  .prob-value { font-size: 1.25rem; }
  .metric-value { font-size: 1.25rem; }
  .hd-brand-name { font-size: .78rem; }
  .hd-brand-tag { font-size: .48rem; }
  .upload-title { font-size: 1.05rem; }
  [data-testid="stMainBlockContainer"] {
    padding: 1rem .7rem 2rem !important;
  }
}

@media (max-height: 500px) and (orientation: landscape) {
  .sticky-header { min-height: 52px; padding: .5rem .9rem; }
  .hero { padding: 1rem 0 .8rem; }
  .hero h1 { font-size: 1.7rem; margin-bottom: .4rem; }
  .upload-hero { padding: 1.2rem 1rem 1rem; }
}

@media (hover: none) {
  .metric-card:hover,
  .info-card:hover,
  .prob-card:hover,
  .step-card:hover,
  .thumb-card:hover,
  .dist-card:hover,
  .f-stat:hover,
  .ft-badge:hover {
    transform: none;
  }
}

@media (-webkit-min-device-pixel-ratio: 2), (min-resolution: 192dpi) {
  .sb-brand-icon,
  .hd-brand-icon,
  .footer-brand-icon {
    box-shadow:
      0 0 28px rgba(34,211,238,.38),
      inset 0 1px 0 rgba(255,255,255,.14);
  }
}

/* ═══════════  37 · TEXT JUSTIFICATION  ═══════════ */
.info-card p,
.hero-sub,
.upload-sub,
.step-card .step-desc,
.xai-text,
.xai-text b,
.xai-text em,
.disclaimer,
.disclaimer span,
.unc-band,
.empty-text,
.footer-desc,
.act-msg,
.sb-warning span,
.sb-confirm-text,
.export-head-sub,
.thumb-meta,
.prob-label,
.latest-val {
  text-align:justify !important;
  text-justify:inter-word !important;
  -webkit-hyphens:auto !important;
  -ms-hyphens:auto !important;
  hyphens:auto !important;
  word-break:normal !important;
  overflow-wrap:anywhere !important;
}

.info-card p,
.hero-sub,
.upload-sub,
.step-card .step-desc,
.xai-text,
.disclaimer span,
.empty-text,
.footer-desc,
.act-msg,
.sb-warning span,
.sb-confirm-text,
.export-head-sub,
.latest-val {
  text-align-last:left !important;
}

.info-card h3,
.unc-title,
.xai-title,
.prob-label {
  text-align-last:left !important;
}

/* ═══════════  38 · PRINT  ═══════════ */
@media print {
  [data-testid="stSidebar"],
  .sticky-header,
  .app-footer,
  .copyright-line,
  .export-head,
  .stDownloadButton,
  .stButton { display:none !important; }

  html, body,
  [data-testid="stAppViewContainer"],
  [data-testid="stApp"],
  [data-testid="stMainBlockContainer"] {
    background:#fff !important;
    color:#000 !important;
  }
  .metric-card,
  .info-card,
  .prob-card,
  .diagnostic-panel,
  .xai-card,
  .uncertainty-card {
    background:#fff !important;
    border:1px solid #ccc !important;
    box-shadow:none !important;
    color:#000 !important;
  }
  .diag-prediction,
  .hero h1,
  .metric-value,
  .prob-value {
    background:none !important;
    -webkit-text-fill-color:#000 !important;
    color:#000 !important;
  }
}
"""

# ─────────────────────────────────────────────────────────────────────────────
# CSS INJECTION
# ─────────────────────────────────────────────────────────────────────────────

def inject_css_once(css: str) -> None:
    """Inject the full UI CSS into the parent document exactly once."""
    components.html(
        f"""
        <script>
        (function() {{
            try {{
                const d = window.parent.document;
                if (!d.getElementById('nl-ui-css')) {{
                    const s = d.createElement('style');
                    s.id = 'nl-ui-css';
                    s.textContent = {json.dumps(css)};
                    d.head.appendChild(s);
                }}
            }} catch (e) {{ /* ignore */ }}
        }})();
        </script>
        """,
        height=0,
        scrolling=False,
    )


st.markdown(
    "<style>"
    "html,body,[data-testid='stAppViewContainer'],[data-testid='stApp']"
    "{background:#0a1e3f!important;font-family:Inter,system-ui,sans-serif;}"
    "</style>",
    unsafe_allow_html=True,
)

inject_css_once(UI_CSS)


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
            nn.Linear(256, 256), nn.BatchNorm1d(256),
            nn.ReLU(inplace=True), nn.Dropout(0.35),
            nn.Linear(256, num_classes),
        )
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)


def build_resnet50(num_classes: int) -> nn.Module:
    model = resnet50(weights=None)
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 256),
        nn.BatchNorm1d(256),
        nn.ReLU(inplace=True),
        nn.Dropout(0.35),
        nn.Linear(256, num_classes),
    )
    return model


def build_efficientnet_b0(num_classes: int) -> nn.Module:
    model = efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.35),
        nn.Linear(in_features, num_classes),
    )
    return model


# ─────────────────────────────────────────────────────────────────────────────
# MODEL LOADING
# ─────────────────────────────────────────────────────────────────────────────

def _read_checkpoint(model_path: Path) -> dict:
    try:
        ckpt = torch.load(model_path, map_location="cpu", weights_only=True)
    except TypeError:
        try:
            ckpt = torch.load(model_path, map_location="cpu")
        except Exception as e:
            raise RuntimeError(f"torch.load failed: {e}") from e
    except Exception as e_safe:
        try:
            ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
        except Exception as e_full:
            raise RuntimeError(
                f"Could not load checkpoint (safe load: {e_safe}; "
                f"full load: {e_full})"
            ) from e_full

    if isinstance(ckpt, dict):
        return ckpt
    return {"model_state_dict": ckpt}


@st.cache_resource(show_spinner=False)
def load_model(model_path_str: str, force_cpu: bool):
    model_path = Path(model_path_str)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    device = torch.device(
        "cpu" if (force_cpu or not torch.cuda.is_available()) else "cuda"
    )
    if device.type == "cuda":
        try:
            torch.backends.cudnn.benchmark = True
        except Exception:
            pass

    checkpoint = _read_checkpoint(model_path)

    checkpoint_classes = checkpoint.get("class_names", CLASS_NAMES)
    normalized_classes = [str(n).strip().casefold().replace(" ", "") for n in checkpoint_classes]
    expected_classes   = [n.casefold().replace(" ", "") for n in CLASS_NAMES]
    if normalized_classes != expected_classes:
        raise ValueError(
            f"Unexpected class order. Expected {CLASS_NAMES}; found {checkpoint_classes}."
        )

    num_classes     = checkpoint.get("num_classes", len(CLASS_NAMES))
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
    clean_sd   = {
        (k[len("module."):] if k.startswith("module.") else k): v
        for k, v in state_dict.items()
    }

    try:
        model.load_state_dict(clean_sd, strict=True)
    except RuntimeError:
        model_sd = model.state_dict()
        filtered = {
            k: v for k, v in clean_sd.items()
            if k in model_sd and tuple(v.shape) == tuple(model_sd[k].shape)
        }
        model.load_state_dict(filtered, strict=False)

    model.to(device)
    model.eval()
    return model, list(CLASS_NAMES), best_model_name, device


# ─────────────────────────────────────────────────────────────────────────────
# INFERENCE
# ─────────────────────────────────────────────────────────────────────────────

def predict_image(
    image: Image.Image,
    model: nn.Module,
    class_names: List[str],
    device: torch.device,
) -> Tuple[str, float, Dict[str, float], float, float]:
    if model is None:
        raise RuntimeError("Neural engine unavailable.")
    model.eval()
    t0 = time.perf_counter()
    tensor = test_transforms(image).unsqueeze(0).to(device)
    preprocess_ms = (time.perf_counter() - t0) * 1000

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    t1 = time.perf_counter()
    with torch.inference_mode():
        probs = F.softmax(model(tensor), dim=1)[0].detach().cpu().numpy()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    inference_ms = (time.perf_counter() - t1) * 1000

    idx = int(np.argmax(probs))
    return (
        class_names[idx],
        float(probs[idx] * 100),
        {class_names[i]: float(probs[i] * 100) for i in range(len(class_names))},
        preprocess_ms,
        inference_ms,
    )


def mc_dropout_predict(
    image: Image.Image,
    model: nn.Module,
    class_names: List[str],
    device: torch.device,
    n_samples: int = MC_SAMPLES_DEF,
) -> Optional[Dict[str, Any]]:
    if model is None:
        return None

    n_samples = max(1, min(int(n_samples), 500))

    def _enable_dropout(m):
        if isinstance(m, (nn.Dropout, nn.Dropout2d)):
            m.train()

    model.eval()
    mc_preds: List[np.ndarray] = []
    try:
        model.apply(_enable_dropout)
        tensor = test_transforms(image).unsqueeze(0).to(device)
        with torch.no_grad():
            for _ in range(n_samples):
                mc_preds.append(F.softmax(model(tensor), dim=1)[0].detach().cpu().numpy())
    finally:
        model.eval()

    if not mc_preds:
        return None

    mc_preds   = np.stack(mc_preds)
    mean_probs = mc_preds.mean(axis=0)
    std_probs  = mc_preds.std(axis=0)
    pred_idx   = int(np.argmax(mean_probs))
    uncertainty = float(std_probs[pred_idx])
    band, color = uncertainty_band(uncertainty)

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

def _fig_to_png_bytes(fig, dpi: int = 160) -> bytes:
    buf = io.BytesIO()
    try:
        fig.savefig(
            buf, format="png", bbox_inches="tight", dpi=dpi,
            facecolor=fig.get_facecolor(), edgecolor="none",
        )
    finally:
        plt.close(fig)
    return buf.getvalue()


def _get_target_layer(model: nn.Module, model_name: Optional[str]) -> nn.Module:
    candidates = []
    if model_name == "ResNet50":
        candidates = [
            lambda m: m.layer4[-1].conv3,
            lambda m: m.layer4[-1].conv2,
            lambda m: m.layer4[-1],
            lambda m: m.layer4,
        ]
    elif model_name == "EfficientNet-B0":
        candidates = [
            lambda m: m.features[-1],
            lambda m: m.features[-2],
        ]
    else:
        candidates = [
            lambda m: m.features[4].block[0],
            lambda m: m.features[4],
            lambda m: m.features[-1],
        ]

    for fn in candidates:
        try:
            layer = fn(model)
            if isinstance(layer, nn.Module):
                return layer
        except (AttributeError, IndexError, TypeError):
            continue

    last_conv: Optional[nn.Module] = None
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            last_conv = m
    if last_conv is None:
        raise RuntimeError("Unable to locate a suitable Grad-CAM target layer.")
    return last_conv


def _cam_to_heatmap(cam_raw: np.ndarray) -> np.ndarray:
    cam = np.maximum(cam_raw, 0)
    cam -= cam.min()
    if cam.max() > 0:
        cam /= cam.max()
    return cam


def _hook_output_to_tensor(output):
    if isinstance(output, tuple):
        return output[0].detach()
    return output.detach()


def _render_cam_overlay(
    image: Image.Image,
    cam: np.ndarray,
    cmap: str,
    alpha: float,
) -> bytes:
    orig = np.asarray(image).astype(np.float32) / 255.0
    fig, ax = plt.subplots(figsize=(5, 5), dpi=120)
    fig.patch.set_alpha(0)
    try:
        ax.imshow(orig)
        ax.imshow(cam, cmap=cmap, alpha=alpha,
                  extent=(0, orig.shape[1], orig.shape[0], 0))
        ax.axis("off")
        fig.tight_layout(pad=0)
        return _fig_to_png_bytes(fig)
    except Exception:
        plt.close(fig)
        raise


def generate_gradcam(
    image: Image.Image,
    model: nn.Module,
    model_name: Optional[str],
    device: torch.device,
) -> bytes:
    if model is None:
        raise RuntimeError("Neural engine unavailable.")
    model.eval()
    activations: List[torch.Tensor] = []
    gradients: List[torch.Tensor] = []
    tl = _get_target_layer(model, model_name)

    fwd = tl.register_forward_hook(
        lambda m, i, o: activations.append(_hook_output_to_tensor(o))
    )
    bwd = tl.register_full_backward_hook(
        lambda m, gi, go: gradients.append(go[0].detach())
    )
    try:
        tensor = test_transforms(image).unsqueeze(0).to(device)
        model.zero_grad()
        out = model(tensor)
        out[0, int(out.argmax(dim=1).item())].backward()
        if not activations or not gradients:
            raise RuntimeError("Grad-CAM hooks did not produce activations.")

        act = activations[0][0]
        grd = gradients[0][0]
        w   = grd.mean(dim=(1, 2), keepdim=True)
        cam = _cam_to_heatmap((w * act).sum(dim=0).detach().cpu().numpy())
        return _render_cam_overlay(image, cam, cmap="jet", alpha=0.44)
    finally:
        try: fwd.remove()
        except Exception: pass
        try: bwd.remove()
        except Exception: pass


def generate_gradcam_pp(
    image: Image.Image,
    model: nn.Module,
    model_name: Optional[str],
    device: torch.device,
) -> bytes:
    if model is None:
        raise RuntimeError("Neural engine unavailable.")
    model.eval()
    activations: List[torch.Tensor] = []
    gradients: List[torch.Tensor] = []
    tl = _get_target_layer(model, model_name)

    fwd = tl.register_forward_hook(
        lambda m, i, o: activations.append(_hook_output_to_tensor(o))
    )
    bwd = tl.register_full_backward_hook(
        lambda m, gi, go: gradients.append(go[0].detach())
    )
    try:
        tensor = test_transforms(image).unsqueeze(0).to(device)
        model.zero_grad()
        out = model(tensor)
        out[0, int(out.argmax(dim=1).item())].backward()
        if not activations or not gradients:
            raise RuntimeError("Grad-CAM++ hooks did not produce activations.")

        act = activations[0][0].cpu().numpy()
        grd = gradients[0][0].cpu().numpy()
        an  = grd ** 2
        ad  = 2 * grd**2 + (act * grd**3).sum(axis=(1, 2), keepdims=True) + 1e-8
        alpha = an / ad
        w = (alpha * np.maximum(grd, 0)).sum(axis=(1, 2))

        cam_raw = np.zeros(act.shape[1:], dtype=np.float32)
        for ww, aa in zip(w, act):
            cam_raw += ww * aa
        cam = _cam_to_heatmap(cam_raw)
        return _render_cam_overlay(image, cam, cmap="inferno", alpha=0.46)
    finally:
        try: fwd.remove()
        except Exception: pass
        try: bwd.remove()
        except Exception: pass


def explanation_agreement(
    image: Image.Image,
    model: nn.Module,
    model_name: Optional[str],
    device: torch.device,
) -> Optional[float]:
    if model is None:
        return None
    model.eval()

    tl = _get_target_layer(model, model_name)
    acts: List[torch.Tensor] = []
    grds: List[torch.Tensor] = []

    fwd = tl.register_forward_hook(
        lambda m, i, o: acts.append(_hook_output_to_tensor(o))
    )
    bwd = tl.register_full_backward_hook(
        lambda m, gi, go: grds.append(go[0].detach())
    )
    try:
        tensor = test_transforms(image).unsqueeze(0).to(device)
        model.zero_grad()
        out = model(tensor)
        idx = int(out.argmax(dim=1).item())
        out[0, idx].backward()
    finally:
        try: fwd.remove()
        except Exception: pass
        try: bwd.remove()
        except Exception: pass

    if not acts or not grds:
        return None

    try:
        act = acts[0][0]
        grd = grds[0][0]

        w_gc = grd.mean(dim=(1, 2), keepdim=True)
        cam_gc = F.relu((w_gc * act).sum(dim=0)).detach().cpu().numpy()
        cam_gc = cam_gc / (cam_gc.max() + 1e-8)

        act_np = act.detach().cpu().numpy()
        grd_np = grd.cpu().numpy()
        an = grd_np ** 2
        ad = 2 * grd_np**2 + (act_np * grd_np**3).sum(axis=(1, 2), keepdims=True) + 1e-8
        w_pp = (an / ad * np.maximum(grd_np, 0)).sum(axis=(1, 2))
        cam_pp = np.zeros(act_np.shape[1:], dtype=np.float32)
        for ww, aa in zip(w_pp, act_np):
            cam_pp += ww * aa
        cam_pp = np.maximum(cam_pp, 0)
        cam_pp = cam_pp / (cam_pp.max() + 1e-8)

        f1, f2 = cam_gc.flatten(), cam_pp.flatten()
        if f1.std() < 1e-8 or f2.std() < 1e-8:
            return None
        corr = float(np.corrcoef(f1, f2)[0, 1])
        if not np.isfinite(corr):
            return None
        return float(max(0.0, min(1.0, corr)))
    except Exception:
        return None


# ─────────────────────────────────────────────────────────────────────────────
# CHARTING
# ─────────────────────────────────────────────────────────────────────────────

def _dark_fig(w: float = 6, h: float = 2.8):
    fig, ax = plt.subplots(figsize=(w, h))
    bg = "#0f2b57"
    fig.patch.set_facecolor(bg)
    ax.set_facecolor(bg)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#26497b")
    ax.tick_params(colors="#a8bcd8", labelsize=8)
    return fig, ax


def plot_confidence_trend(history):
    if len(history) < 2:
        return None
    fig, ax = _dark_fig()
    confs = [h["confidence"] for h in history]
    xs = list(range(1, len(history) + 1))
    ax.plot(xs, confs, marker="o", lw=2, ms=4, color="#22d3ee")
    ax.fill_between(xs, confs, alpha=0.10, color="#22d3ee")
    ax.set_ylim(0, 105)
    ax.set_xlabel("Analysis #", color="#a8bcd8", fontsize=9)
    ax.set_ylabel("Confidence %", color="#a8bcd8", fontsize=9)
    ax.grid(True, alpha=0.09, ls="--", color="#7ba3d6")
    fig.tight_layout(pad=0.5)
    return fig


def plot_latency_trend(history):
    if len(history) < 2:
        return None
    fig, ax = _dark_fig()
    lats = [h.get("latency_ms", 0) for h in history]
    ax.plot(range(1, len(history) + 1), lats, marker="o", lw=2, color="#34d399")
    ax.set_xlabel("Analysis #", color="#a8bcd8", fontsize=9)
    ax.set_ylabel("Latency ms", color="#a8bcd8", fontsize=9)
    ax.grid(True, alpha=0.09, ls="--", color="#7ba3d6")
    fig.tight_layout(pad=0.5)
    return fig


def plot_uncertainty_history(history):
    data = [
        (i + 1, h["uncertainty"])
        for i, h in enumerate(history)
        if h.get("uncertainty") is not None
    ]
    if len(data) < 2:
        return None
    fig, ax = _dark_fig()
    xs, ys = zip(*data)
    ax.plot(xs, ys, marker="s", lw=2, ms=4, color="#a78bfa")
    ax.fill_between(xs, ys, alpha=0.10, color="#a78bfa")
    ax.axhline(0.12, color="#f59e0b", lw=1, ls="--", alpha=0.65, zorder=1)
    ax.axhline(0.22, color="#ef4444", lw=1, ls="--", alpha=0.65, zorder=1)
    ax.set_xlabel("Analysis #", color="#a8bcd8", fontsize=9)
    ax.set_ylabel("MC Uncertainty σ", color="#a8bcd8", fontsize=9)
    ax.grid(True, alpha=0.09, ls="--", color="#7ba3d6")
    fig.tight_layout(pad=0.5)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE + NAVIGATION
# ─────────────────────────────────────────────────────────────────────────────

DEFAULTS: Dict[str, Any] = {
    "nav": "Home",
    "last_result": None,
    "last_image": None,
    "gradcam_png": None,
    "gradcam_pp_png": None,
    "mc_result": None,
    "agreement_score": None,
    "prediction_history": [],
    "activity_log": [],
    "live_session_start": None,
    "live_predictions_count": 0,
    "live_avg_confidence": 0.0,
    "live_class_counts": {},
    "live_throughput": 0.0,
    "live_last_confidence": 0.0,
    "live_latency_ms": 0.0,
    "confirm_clear": False,
    "confirm_reset": False,
    "confirm_clear_hist": False,
    "settings_confirm_reset": False,
    "last_nav_snapshot": "Home",
    "_page_changed": False,
    "force_cpu": False,
    "mc_samples": MC_SAMPLES_DEF,
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = copy.deepcopy(v)


def navigate_to(page: str) -> None:
    st.session_state.nav = page
    try:
        st.query_params["page"] = page
    except Exception:
        pass
    st.rerun()


def clear_prediction_history() -> None:
    for k in [
        "prediction_history", "last_result", "last_image",
        "gradcam_png", "gradcam_pp_png", "mc_result", "agreement_score",
        "live_predictions_count", "live_avg_confidence", "live_last_confidence",
        "live_class_counts", "live_throughput", "live_latency_ms",
    ]:
        st.session_state[k] = copy.deepcopy(DEFAULTS[k])


def log_activity(message: str, level: str = "info") -> None:
    ts = datetime.now().strftime("%H:%M:%S")
    log = list(st.session_state.activity_log)
    log.append({"timestamp": ts, "message": str(message), "level": level})
    st.session_state.activity_log = log[-60:]


def update_live_stats(result: Dict[str, Any], latency_ms: float) -> None:
    history = st.session_state.prediction_history
    st.session_state.live_predictions_count = len(history)
    if history:
        confs = [h["confidence"] for h in history]
        st.session_state.live_avg_confidence  = float(np.mean(confs))
        st.session_state.live_last_confidence = float(confs[-1])
    st.session_state.live_latency_ms = float(latency_ms)

    counts: Dict[str, int] = {}
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

_force_cpu = bool(st.session_state.get("force_cpu", False))
try:
    model, class_names, model_name, DEVICE = load_model(str(MODEL_PATH), _force_cpu)
    model_error: Optional[str] = None
except Exception as exc:
    model = None
    class_names = CLASS_NAMES
    model_name = None
    DEVICE = torch.device("cpu")
    model_error = str(exc)


# ─────────────────────────────────────────────────────────────────────────────
# NAV CONFIG
# ─────────────────────────────────────────────────────────────────────────────

NAV_GROUPS = [
    ("Main", [
        ("Home",         Icons.home(16, "#a8bcd8"),       "Home"),
        ("MRI Analysis", Icons.microscope(16, "#a8bcd8"), "MRI Analysis"),
        ("Dashboard",    Icons.chart(16, "#a8bcd8"),      "Dashboard"),
    ]),
    ("Analytics", [
        ("History",  Icons.history(16, "#a8bcd8"),  "History"),
        ("Grad-CAM", Icons.heatmap(16, "#a8bcd8"),  "Grad-CAM"),
        ("XAI Lab",  Icons.lab(16, "#a8bcd8"),      "XAI Lab"),
    ]),
    ("System", [
        ("Settings", Icons.settings(16, "#a8bcd8"), "Settings"),
    ]),
]

PAGE_LABELS = {
    "Home": "Home",
    "MRI Analysis": "MRI Analysis",
    "Dashboard": "Dashboard",
    "History": "History",
    "Grad-CAM": "Grad-CAM",
    "XAI Lab": "XAI Lab",
    "Settings": "Settings",
}

NAV_ICON_ACTIVE_URLS = {
    "home": "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%2322d3ee' stroke-width='2.0' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='m3 9 9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z'/%3E%3Cpolyline points='9 22 9 12 15 12 15 22'/%3E%3C/svg%3E",
    "mri_analysis": "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%2322d3ee' stroke-width='2.0' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M6 18h8'/%3E%3Cpath d='M3 22h18'/%3E%3Cpath d='M14 22a7 7 0 1 0 0-14h-1'/%3E%3Cpath d='M9 14h2'/%3E%3Cpath d='M9 12a2 2 0 0 1-2-2V6h6v4a2 2 0 0 1-2 2Z'/%3E%3Cpath d='M12 6V3a1 1 0 0 0-1-1H9a1 1 0 0 0-1 1v3'/%3E%3C/svg%3E",
    "dashboard": "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%2322d3ee' stroke-width='2.0' stroke-linecap='round' stroke-linejoin='round'%3E%3Cline x1='18' y1='20' x2='18' y2='10'/%3E%3Cline x1='12' y1='20' x2='12' y2='4'/%3E%3Cline x1='6' y1='20' x2='6' y2='14'/%3E%3Cline x1='2' y1='20' x2='22' y2='20'/%3E%3C/svg%3E",
    "history": "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%2322d3ee' stroke-width='2.0' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8'/%3E%3Cpath d='M3 3v5h5'/%3E%3Cpath d='M12 7v5l4 2'/%3E%3C/svg%3E",
    "gradcam": "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%2322d3ee' stroke-width='2.0' stroke-linecap='round' stroke-linejoin='round'%3E%3Ccircle cx='12' cy='12' r='3'/%3E%3Cpath d='M12 2v3'/%3E%3Cpath d='M12 19v3'/%3E%3Cpath d='m4.22 4.22 2.12 2.12'/%3E%3Cpath d='m17.66 17.66 2.12 2.12'/%3E%3Cpath d='M2 12h3'/%3E%3Cpath d='M19 12h3'/%3E%3Cpath d='m4.22 19.78 2.12-2.12'/%3E%3Cpath d='m17.66 6.34 2.12-2.12'/%3E%3C/svg%3E",
    "xai_lab": "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%2322d3ee' stroke-width='2.0' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M14.5 2v17.5c0 1.4-1.1 2.5-2.5 2.5h0c-1.4 0-2.5-1.1-2.5-2.5V2'/%3E%3Cpath d='M8.5 2h7'/%3E%3Cpath d='M14.5 16h-5'/%3E%3Cpath d='m8.5 13 5.5 3'/%3E%3C/svg%3E",
    "settings": "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%2322d3ee' stroke-width='2.0' stroke-linecap='round' stroke-linejoin='round'%3E%3Ccircle cx='12' cy='12' r='3'/%3E%3Cpath d='M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 1 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 1 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 1 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 1 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z'/%3E%3C/svg%3E",
}


# ─────────────────────────────────────────────────────────────────────────────
# URL ⇄ SESSION SYNC
# ─────────────────────────────────────────────────────────────────────────────

try:
    _qp_page = st.query_params.get("page")
except Exception:
    _qp_page = None

if _qp_page and _qp_page in PAGE_LABELS and _qp_page != st.session_state.nav:
    st.session_state.nav = _qp_page

st.session_state._page_changed = (
    st.session_state.nav != st.session_state.last_nav_snapshot
)
st.session_state.last_nav_snapshot = st.session_state.nav

try:
    if _qp_page != st.session_state.nav:
        st.query_params["page"] = st.session_state.nav
except Exception:
    pass


# ─────────────────────────────────────────────────────────────────────────────
# LIVE TICKER
# ─────────────────────────────────────────────────────────────────────────────

def render_live_ticker() -> None:
    counts = st.session_state.live_class_counts or {}
    items  = list(counts.items())
    text   = " · ".join(
        f"<b>{_escape_html(n)}</b>: {_escape_html(v)}" for n, v in items
    ) if items else "<b>—</b>"
    mc  = st.session_state.mc_result
    unc = f" · σ={mc['uncertainty']:.3f} ({_escape_html(mc['band'])})" if mc else ""

    _render_html(f"""
    <style>
    .tw{{overflow:hidden;border-radius:11px;border:1px solid rgba(34,211,238,.28);
    background:linear-gradient(90deg,rgba(24,75,138,.9),rgba(13,37,81,.9));padding:.75rem 0;width:100%;}}
    .tt{{display:inline-block;white-space:nowrap;padding-left:100%;
    animation:mq 28s linear infinite;color:#a8bcd8;font:500 .8rem 'JetBrains Mono',monospace}}
    .lp{{display:inline-block;width:7px;height:7px;border-radius:50%;background:#22d3ee;
    box-shadow:0 0 8px #22d3ee;margin-right:8px;vertical-align:middle}}
    @keyframes mq{{0%{{transform:translateX(0);}}100%{{transform:translateX(-100%);}}}}
    </style>
    <div class="tw"><div class="tt"><span class="lp"></span>LIVE ·
    {text} · Throughput: {st.session_state.live_throughput:.2f}/min ·
    Last Conf: {st.session_state.live_last_confidence:.1f}% ·
    Avg Conf: {st.session_state.live_avg_confidence:.1f}%{unc}
    </div></div>""", height=52)


# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────

def render_sticky_header() -> None:
    counts = st.session_state.live_class_counts or {}
    if counts:
        items_html = " ".join(
            f"<span class='hd-tick'><b>{_escape_html(k)}</b>: {_escape_html(v)}</span>"
            for k, v in counts.items()
        )
    else:
        items_html = "<span class='hd-tick hd-tick-idle'>—</span>"

    mc  = st.session_state.mc_result
    nav = PAGE_LABELS.get(st.session_state.nav, st.session_state.nav)
    eng = model_name or "—"
    eng_ok  = model_error is None and MODEL_PATH.exists()
    st_cls  = "hd-status-ok" if eng_ok else "hd-status-err"
    st_txt  = "Online" if eng_ok else "Offline"
    dev_tag = "GPU" if DEVICE.type == "cuda" else "CPU"
    mc_str  = f"<span class='hd-tick'>σ=<b>{mc['uncertainty']:.3f}</b></span>" if mc else ""

    brain_svg = Icons.brain(20, "#22d3ee")

    header_html = safe_html(f"""
    <div class="sticky-header">
        <div class="hd-brand">
            <div class="hd-brand-icon">
                {brain_svg}
                <span class="hd-brand-dot"></span>
            </div>
            <div>
                <div class="hd-brand-name">NeuroLens AI</div>
                <span class="hd-brand-tag">Neurodiagnostic Intelligence</span>
            </div>
        </div>
        <div class="hd-divider"></div>
        <div class="hd-ticker">
            <div class="hd-ticker-inner">
                <span class="hd-live-badge">LIVE</span>
                {items_html}
                <span class="hd-tick">Engine: <b>{_escape_html(eng)}</b></span>
                <span class="hd-tick">Device: <b>{dev_tag}</b></span>
                {mc_str}
            </div>
        </div>
        <div class="hd-divider"></div>
        <div class="hd-status {st_cls}">
            <span class="hd-status-dot"></span>{st_txt}
        </div>
        <div class="hd-page">
            <span class="hd-page-dot"></span>{_escape_html(nav)}
        </div>
    </div>""")

    st.markdown(header_html, unsafe_allow_html=True)

    page_changed = bool(st.session_state.get("_page_changed", False))

    if page_changed:
        scroll_js = (
            'const sc = d.querySelector(\'[data-testid="stAppViewContainer"]\')'
            ' || d.querySelector("section.main")'
            ' || d.scrollingElement'
            ' || d.documentElement;'
            'if (sc) sc.scrollTo({top:0, behavior:"auto"});'
        )
        transition_js = (
            'd.body.classList.add("nl-page-transition");'
            'setTimeout(()=>d.body.classList.remove("nl-page-transition"), 400);'
        )
    else:
        scroll_js = ""
        transition_js = ""

    components.html(f"""
    <script>
    (function() {{
        try {{
            const w = window.parent;
            const d = w.document;
            d.title = {json.dumps(nav + " · NeuroLens AI")};
            {scroll_js}
            {transition_js}
        }} catch (e) {{ console.error('[NeuroLens]', e); }}
    }})();
    </script>
    """, height=0, scrolling=False)


# ─────────────────────────────────────────────────────────────────────────────
# LIVE PROBABILITY BARS
# ─────────────────────────────────────────────────────────────────────────────

def render_live_probability_animation(
    result: Dict[str, Any],
    mc_result: Optional[Dict[str, Any]] = None,
) -> None:
    items = list(result["probabilities"].items())
    top   = result["prediction"]
    conf  = result["confidence"]

    mc_rows = ""
    if mc_result:
        mc_std = mc_result["std_probs"]
        mc_rows = "".join(
            f'<div class="lp-row"><div class="lp-label">{_escape_html(n)}'
            f'<span style="color:#a78bfa;font-family:\'JetBrains Mono\',monospace;font-size:.7rem"> ±{mc_std.get(n,0):.1f}%</span></div>'
            f'<div class="lp-track"><div class="lp-fill lp-mc" data-target="{p:.2f}" style="width:0%"></div></div>'
            f'<div class="lp-val">{p:.1f}%</div></div>'
            for n, p in mc_result["mean_probs"].items()
        )

    bars = "".join(
        f'<div class="lp-row"><div class="lp-label">{_escape_html(n)}</div>'
        f'<div class="lp-track"><div class="lp-fill" data-target="{p:.2f}" style="width:0%"></div></div>'
        f'<div class="lp-val">{p:.1f}%</div></div>'
        for n, p in items
    )

    unc_html = ""
    if mc_result:
        band  = mc_result["band"]
        uval  = mc_result["uncertainty"]
        color = mc_result["color"]
        unc_html = (
            f'<div style="margin-top:.7rem;padding:.55rem .8rem;border-radius:9px;'
            f'background:linear-gradient(135deg,rgba(139,92,246,.10),rgba(34,211,238,.05));border:1px solid rgba(139,92,246,.28)">'
            f'<span style="font-size:.6rem;color:#a78bfa;text-transform:uppercase;letter-spacing:.12em;font-weight:700;font-family:\'Inter\',sans-serif">MC Uncertainty</span>'
            f'<span style="float:right;font-weight:700;color:{color};font-family:\'JetBrains Mono\',monospace;font-size:.76rem">σ={uval:.4f} · {_escape_html(band)}</span></div>'
        )

    h = 130 + 34 * len(items) + (160 + 34 * len(items) if mc_result else 0)

    _render_html(f"""
    <style>
    .lp-wrap{{padding:1.05rem 1.3rem;border-radius:14px;border:1px solid rgba(34,211,238,.22);
    background:linear-gradient(135deg,rgba(24,75,138,.9),rgba(13,37,81,.9));font-family:Inter,sans-serif;color:#e2e8f0}}
    .lp-head{{display:flex;justify-content:space-between;align-items:center;margin-bottom:.75rem;padding-bottom:.55rem;
    border-bottom:1px solid rgba(34,211,238,.16);gap:.5rem;flex-wrap:wrap}}
    .lp-head-title{{font-size:.6rem;color:#22d3ee;text-transform:uppercase;letter-spacing:.14em;font-weight:800}}
    .lp-head-pred{{font-family:Sora,sans-serif;font-size:.88rem;font-weight:700;color:#e2e8f0;letter-spacing:-.01em}}
    .lp-row{{display:flex;align-items:center;gap:.65rem;margin:.38rem 0}}
    .lp-label{{min-width:132px;font-size:.76rem;color:#a8bcd8;font-weight:500}}
    .lp-track{{flex:1;height:6px;border-radius:999px;background:rgba(59,130,246,.16);overflow:hidden}}
    .lp-fill{{height:100%;width:0%;background:linear-gradient(90deg,#22d3ee,#34d399);border-radius:999px;
    transition:width 1.1s cubic-bezier(.2,.8,.2,1);box-shadow:0 0 10px rgba(34,211,238,.5)}}
    .lp-mc{{background:linear-gradient(90deg,#a78bfa,#22d3ee)!important}}
    .lp-val{{min-width:54px;text-align:right;font-family:'JetBrains Mono',monospace;font-weight:700;color:#e2e8f0;font-size:.74rem}}
    .lp-sec{{font-size:.58rem;color:#7ba3d6;text-transform:uppercase;letter-spacing:.14em;font-weight:800;margin:.7rem 0 .3rem}}
    </style>
    <div class="lp-wrap">
        <div class="lp-head">
            <span class="lp-head-title">Live Probability Stream</span>
            <span class="lp-head-pred">{_escape_html(top)} · {conf:.1f}%</span>
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


def render_activity_feed() -> None:
    log = st.session_state.activity_log[-14:][::-1]
    if not log:
        st.markdown(
            "<div style='color:#7ba3d6;font-size:.8rem;padding:.5rem'>No activity yet.</div>",
            unsafe_allow_html=True,
        )
        return
    rows = "".join(
        f"<div class='act-row act-{_escape_html(e['level'])}'>"
        f"<span class='act-time'>{_escape_html(e['timestamp'])}</span>"
        f"<span class='act-msg'>{_escape_html(e['message'])}</span></div>"
        for e in log
    )
    st.markdown(f"<div class='act-feed'>{rows}</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    _history_sidebar = st.session_state.prediction_history
    total_scans = len(_history_sidebar)
    avg_conf    = float(np.mean([x["confidence"] for x in _history_sidebar])) if _history_sidebar else 0.0
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

    brain_large = Icons.brain(22, "#22d3ee")
    st.markdown(safe_html(f"""
    <div class="sb-brand">
        <div class="sb-brand-icon">
            {brain_large}
            <span class="sb-brand-pulse"></span>
        </div>
        <div class="sb-brand-text">
            <div class="sb-brand-name">NeuroLens <span class="sb-brand-ai">AI</span><span class="sb-brand-ver">v3.7.3</span></div>
            <div class="sb-brand-sub">Neurodiagnostic Intelligence</div>
        </div>
    </div>"""), unsafe_allow_html=True)

    dot_color = "var(--success-hi)" if engine_ok else "var(--danger-hi)"
    st.markdown(safe_html(f"""
    <div class="sb-session">
        <span class="sb-session-dot" style="background:{dot_color};box-shadow:0 0 8px {dot_color}"></span>
        <span class="sb-session-txt">{_escape_html(active_nav)}</span>
        <span class="sb-session-time">{_escape_html(uptime)}</span>
    </div>"""), unsafe_allow_html=True)

    badge_css_parts = []
    if total_scans > 0:
        badge_css_parts.append(f"""
        [data-testid="stSidebar"] .st-key-nav_mri_analysis button::after,
        [data-testid="stSidebar"] .st-key-nav_history button::after {{
            content: "{total_scans}";
            position: absolute;
            right: 0.7rem;
            top: 50%;
            transform: translateY(-50%);
            padding: 0.08rem 0.45rem;
            border-radius: 999px;
            background: linear-gradient(135deg, rgba(37,99,235,0.28), rgba(34,211,238,0.20));
            color: #22d3ee;
            border: 1px solid rgba(34,211,238,0.42);
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.58rem;
            font-weight: 700;
            line-height: 1.4;
            letter-spacing: 0.02em;
        }}
        """)

    agree_vals = [h.get("agreement_score") for h in _history_sidebar if h.get("agreement_score") is not None]
    if agree_vals:
        avg_agree = float(np.mean(agree_vals))
        badge_css_parts.append(f"""
        [data-testid="stSidebar"] .st-key-nav_xai_lab button::after {{
            content: "{avg_agree:.2f}";
            position: absolute;
            right: 0.7rem;
            top: 50%;
            transform: translateY(-50%);
            padding: 0.08rem 0.45rem;
            border-radius: 999px;
            background: linear-gradient(135deg, rgba(139,92,246,0.28), rgba(34,211,238,0.16));
            color: #a78bfa;
            border: 1px solid rgba(139,92,246,0.42);
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.58rem;
            font-weight: 700;
            line-height: 1.4;
        }}
        """)

    if badge_css_parts:
        st.markdown(f"<style>{''.join(badge_css_parts)}</style>", unsafe_allow_html=True)

    active_css_parts = []
    for group_name, items in NAV_GROUPS:
        st.markdown(f'<div class="sb-nav-group">{_escape_html(group_name)}</div>', unsafe_allow_html=True)
        for label, icon_svg, key in items:
            is_active = active_nav == key
            slug = key.lower().replace("-", "").replace(" ", "_")

            if st.button(label, key=f"nav_{slug}", **_stretch()):
                navigate_to(key)

            if is_active:
                cyan_url = NAV_ICON_ACTIVE_URLS.get(slug, "")
                active_css_parts.append(f"""
                [data-testid="stSidebar"] .st-key-nav_{slug} button {{
                    background: linear-gradient(90deg, rgba(37,99,235,0.24), rgba(34,211,238,0.12), rgba(16,185,129,0.06)) !important;
                    border-color: rgba(34,211,238,0.55) !important;
                    color: #22d3ee !important;
                    font-weight: 700 !important;
                    box-shadow: inset 3px 0 0 0 #22d3ee, 0 0 22px rgba(34,211,238,0.20) !important;
                }}
                [data-testid="stSidebar"] .st-key-nav_{slug} button p {{
                    color: #22d3ee !important;
                    font-weight: 700 !important;
                }}
                [data-testid="stSidebar"] .st-key-nav_{slug} button::before {{
                    background-image: url("{cyan_url}") !important;
                    opacity: 1 !important;
                    filter: drop-shadow(0 0 8px rgba(34,211,238,.9)) !important;
                }}
                """)

    if active_css_parts:
        st.markdown(f"<style>{''.join(active_css_parts)}</style>", unsafe_allow_html=True)

    st.markdown(
        '<div style="height:1px;background:linear-gradient(90deg,transparent,rgba(34,211,238,.35),transparent);margin:0.7rem 0.85rem"></div>',
        unsafe_allow_html=True,
    )

    eng_color = "var(--success-hi)" if engine_ok else "var(--danger-hi)"
    eng_label = "Engine Ready" if engine_ok else "Engine Offline"
    dev_tag   = "CUDA" if DEVICE.type == "cuda" else "CPU"
    dev_color = "#22d3ee" if DEVICE.type == "cuda" else "#a8bcd8"

    live_chip = (
        '<div class="sb-engine-live"><span class="sb-engine-live-dot"></span>LIVE</div>'
        if engine_ok else ''
    )

    mc_row = ""
    if st.session_state.mc_result:
        mc = st.session_state.mc_result
        mc_row = f"""
        <div class="sb-engine-confbar" style="margin-top:0.4rem;margin-bottom:0">
            <span style="font-size:0.56rem;color:#a78bfa;font-weight:700;letter-spacing:0.1em;min-width:42px">σ MC</span>
            <span style="font-family:'JetBrains Mono',monospace;font-size:0.64rem;color:#a78bfa;font-weight:600">{mc['uncertainty']:.4f}</span>
            <span style="margin-left:auto;font-size:0.56rem;color:#a78bfa;font-weight:600">{_escape_html(mc['band'].replace(' Reliability',''))}</span>
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
            <span style="width:7px;height:7px;border-radius:50%;background:{eng_color};box-shadow:0 0 10px {eng_color};flex-shrink:0;display:inline-block"></span>
            <span style="color:{eng_color}">{eng_label}</span>
            <span style="margin-left:auto;font-size:0.56rem;color:{dev_color};font-weight:700;font-family:'JetBrains Mono',monospace;padding:0.08rem 0.45rem;border-radius:5px;border:1px solid {dev_color}55;background:{dev_color}1A;letter-spacing:0.08em">{dev_tag}</span>
        </div>
        <div class="sb-engine-confbar">
            <span style="font-size:0.56rem;color:var(--text-3);font-weight:700;letter-spacing:0.1em;min-width:42px">CONF</span>
            <div class="sb-engine-confbar-track">
                <div class="sb-engine-confbar-fill" style="width:{conf_pct:.1f}%"></div>
            </div>
            <span class="sb-engine-confbar-lbl">{conf_pct:.0f}%</span>
        </div>
        {mc_row}
        <div class="sb-engine-grid">
            <div class="sb-mini sb-mini-white"><div class="sb-mini-lbl">SCANS</div><div class="sb-mini-val">{total_scans}</div></div>
            <div class="sb-mini sb-mini-cyan"><div class="sb-mini-lbl">DEVICE</div><div class="sb-mini-val">{dev_tag}</div></div>
            <div class="sb-mini sb-mini-violet" style="grid-column:1 / -1"><div class="sb-mini-lbl">MODEL</div><div class="sb-mini-val">{_escape_html(model_name or '—')}</div></div>
        </div>
    </div>"""), unsafe_allow_html=True)

    st.markdown('<div class="sb-nav-group">Quick Actions</div>', unsafe_allow_html=True)

    if st.button("Clear History", key="sb_clear", **_stretch()):
        st.session_state.confirm_clear = True
    if st.session_state.get("confirm_clear", False):
        st.markdown(safe_html(f"""
        <div class="sb-confirm">
            <div class="sb-confirm-title">{Icons.alert_triangle(12, "#fbbf24")} Clear all records?</div>
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
        </div>"""), unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        if c1.button("Confirm", key="sb_reset_yes", **_stretch()):
            for k, v in DEFAULTS.items():
                st.session_state[k] = copy.deepcopy(v)
            navigate_to("Home")
        if c2.button("Cancel", key="sb_reset_no", **_stretch()):
            st.session_state.confirm_reset = False

    alert_svg = Icons.alert_triangle(13, "#f59e0b")
    st.markdown(safe_html(f"""
    <div class="sb-warning">
        {alert_svg}
        <span>Research prototype only.</span>
    </div>"""), unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# ROUTER
# ─────────────────────────────────────────────────────────────────────────────

render_sticky_header()
nav = st.session_state.nav

if nav not in PAGE_LABELS:
    nav = "Home"
    st.session_state.nav = "Home"


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
            AI-Powered Brain MRI Analysis
        </div>
        <h1>NeuroLens AI</h1>
        <div class="hero-divider"></div>
    </div>"""), unsafe_allow_html=True)

    _l, _c, _r = st.columns([1.8, 1, 1.8])
    with _c:
        if st.button("Run MRI Analysis", type="primary", key="home_cta", **_stretch()):
            navigate_to("MRI Analysis")

    st.write("")
    _home_hist = st.session_state.prediction_history
    mc1, mc2, mc3, mc4 = st.columns(4)
    pulse_svg = Icons.pulse(18, "#22d3ee")
    tgt_svg   = Icons.target(18, "#22d3ee")
    zap_svg   = Icons.zap(18, "#22d3ee")
    cpu_svg   = Icons.cpu(18, "#22d3ee")

    metrics = [
        (pulse_svg, st.session_state.live_predictions_count, "Live Scans"),
        (tgt_svg,   f"{st.session_state.live_avg_confidence:.1f}%", "Avg. Confidence"),
        (zap_svg,   f"{st.session_state.live_throughput:.2f}/m", "Throughput"),
        (cpu_svg,
         f"{np.mean([x.get('latency_ms',0) for x in _home_hist]):.0f} ms" if _home_hist else "—",
         "Avg. Inference"),
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
        (micro_svg, "Brain MRI Classification", ""),
        (shld_svg,  "MC Dropout Uncertainty", ""),
        (eye_svg,   "Dual XAI (Grad-CAM++)", ""),
        (zap2_svg,  "Real-Time Inference", ""),
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
    st.markdown(
        _icon_header(Icons.zap(20, "#22d3ee"), "Pipeline"),
        unsafe_allow_html=True,
    )
    steps = [
        ("1", "Upload", ""),
        ("2", "Preprocess", ""),
        ("3", "Inference", ""),
        ("4", "MC Dropout", ""),
        ("5", "Dual XAI", ""),
        ("6", "Agreement", ""),
        ("7", "Report", ""),
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
        <div class="info-card" style="border-color:rgba(239,68,68,.35);background:rgba(239,68,68,.06)">
            <div style="display:flex;align-items:center;gap:.5rem;margin-bottom:.5rem">{err_svg}
                <h3 style="color:#f87171;margin:0">Neural Engine Unavailable</h3></div>
            <p>Expected: <code>{_escape_html(MODEL_PATH)}</code></p>
        </div>"""), unsafe_allow_html=True)
        with st.expander("Technical Details"):
            st.code(model_error)
    else:
        chk_svg = Icons.check_circle(18, "#34d399")
        st.markdown(safe_html(f"""
        <div class="info-card" style="border-color:rgba(16,185,129,.42);background:linear-gradient(135deg,rgba(16,185,129,.08),rgba(37,99,235,.06))">
            <div style="display:flex;align-items:center;gap:.5rem;margin-bottom:.9rem">{chk_svg}
                <h3 style="color:#34d399;margin:0">Neural Engine Ready</h3></div>
            <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:.7rem">
                <div style="background:linear-gradient(135deg,rgba(37,99,235,.16),rgba(16,185,129,.10));padding:.65rem .85rem;border-radius:10px;border:1px solid rgba(34,211,238,.28)">
                    <div style="font-size:.6rem;color:#7ba3d6;letter-spacing:.08em;font-weight:700;margin-bottom:.22rem">ARCHITECTURE</div>
                    <div style="font-size:.88rem;font-weight:700;color:#e2e8f0;font-family:var(--font-mono)">{_escape_html(model_name)}</div>
                </div>
                <div style="background:linear-gradient(135deg,rgba(37,99,235,.16),rgba(16,185,129,.10));padding:.65rem .85rem;border-radius:10px;border:1px solid rgba(34,211,238,.28)">
                    <div style="font-size:.6rem;color:#7ba3d6;letter-spacing:.08em;font-weight:700;margin-bottom:.22rem">DEVICE</div>
                    <div style="font-size:.88rem;font-weight:700;color:#e2e8f0;font-family:var(--font-mono)">{_escape_html(str(DEVICE))}</div>
                </div>
                <div style="background:linear-gradient(135deg,rgba(37,99,235,.16),rgba(16,185,129,.10));padding:.65rem .85rem;border-radius:10px;border:1px solid rgba(34,211,238,.28)">
                    <div style="font-size:.6rem;color:#7ba3d6;letter-spacing:.08em;font-weight:700;margin-bottom:.22rem">CLASSES</div>
                    <div style="font-size:.82rem;font-weight:700;color:#22d3ee">{', '.join(class_names)}</div>
                </div>
                <div style="background:linear-gradient(135deg,rgba(37,99,235,.16),rgba(16,185,129,.10));padding:.65rem .85rem;border-radius:10px;border:1px solid rgba(34,211,238,.28)">
                    <div style="font-size:.6rem;color:#7ba3d6;letter-spacing:.08em;font-weight:700;margin-bottom:.22rem">MC DROPOUT</div>
                    <div style="font-size:.82rem;font-weight:700;color:#e2e8f0">{st.session_state.get('mc_samples', MC_SAMPLES_DEF)} passes</div>
                </div>
            </div>
        </div>"""), unsafe_allow_html=True)

    if st.session_state.last_result:
        st.write("")
        st.markdown(
            _icon_header(Icons.file_text(20, "#22d3ee"), "Last Analysis"),
            unsafe_allow_html=True,
        )
        render_live_probability_animation(
            st.session_state.last_result,
            mc_result=st.session_state.mc_result,
        )


# ══════════════════════════════════════════════════════════════════════════════
# MRI ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

elif nav == "MRI Analysis":
    micro_h = Icons.microscope(22, "#22d3ee")
    st.markdown(
        f"<h1 style='display:flex;align-items:center;gap:.5rem;flex-wrap:wrap'>"
        f"{micro_h} MRI Analysis</h1>",
        unsafe_allow_html=True,
    )
    render_live_ticker()

    if model_error:
        st.error("Neural engine unavailable.")
        st.info(f"Expected model: `{MODEL_PATH}`")
        with st.expander("Technical Details"):
            st.code(model_error)
    else:
        up_svg = Icons.upload(28, "#22d3ee")
        st.markdown(safe_html(f"""
        <div class="upload-hero">
            <div class="upload-icon-wrap">{up_svg}</div>
            <div class="upload-title">Upload MRI Scan</div>
            <div class="upload-formats">
                <span class="fmt-badge">JPG</span><span class="fmt-badge">JPEG</span>
                <span class="fmt-badge">PNG</span><span class="fmt-badge">WEBP</span>
                <span class="fmt-badge fmt-badge-muted">≤ 200 MB</span>
            </div>
            <div class="upload-note">
                <span class="upload-note-dot"></span>
                Processed locally
            </div>
        </div>"""), unsafe_allow_html=True)

        _ul, _uc, _ur = st.columns([1, 2, 1])
        with _uc:
            uploaded_file = st.file_uploader(
                "Upload MRI Scan",
                type=["jpg", "jpeg", "png", "webp"],
                key="mri_uploader",
                label_visibility="collapsed",
            )

        image_id: Optional[str] = None
        image: Optional[Image.Image] = None
        if uploaded_file is not None:
            upload_bytes = uploaded_file.getvalue()

            if len(upload_bytes) > MAX_UPLOAD_BYTES:
                st.error(
                    f"File too large: {len(upload_bytes) / (1024 * 1024):.1f} MB. "
                    f"Maximum allowed is {MAX_UPLOAD_BYTES // (1024 * 1024)} MB."
                )
                image = None
            else:
                image_id = hashlib.sha256(upload_bytes).hexdigest()
                prev = st.session_state.last_result
                if prev is not None and prev.get("image_id") != image_id:
                    for k in [
                        "last_result", "last_image", "gradcam_png",
                        "gradcam_pp_png", "mc_result", "agreement_score",
                    ]:
                        st.session_state[k] = None

                try:
                    image = Image.open(io.BytesIO(upload_bytes)).convert("RGB")
                except Image.DecompressionBombError:
                    st.error("Image rejected: exceeds safe pixel limit.")
                    image = None
                except (UnidentifiedImageError, OSError, ValueError) as img_err:
                    st.error(f"Could not read image. ({img_err})")
                    image = None

            if image is not None:
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.markdown(
                        _icon_header(Icons.image(20, "#22d3ee"), "Preview", level=4),
                        unsafe_allow_html=True,
                    )
                    st.image(image, **_stretch())
                    st.caption(
                        f"{image.width}×{image.height}px · "
                        f"Processed: {IMG_SIZE}×{IMG_SIZE}px"
                    )

                with col2:
                    st.markdown(
                        _icon_header(Icons.sliders(20, "#22d3ee"), "Configuration", level=4),
                        unsafe_allow_html=True,
                    )
                    cf1, cf2, cf3 = st.columns(3)
                    cf1.metric("Architecture", model_name or "—")
                    cf2.metric("Device", str(DEVICE).upper())
                    cf3.metric("Classes", len(class_names))

                    mc_n = int(st.session_state.get("mc_samples", MC_SAMPLES_DEF))
                    run_mc    = st.checkbox(f"MC Dropout ({mc_n} passes)", value=True, key="run_mc_cb")
                    run_xai   = st.checkbox("Dual XAI", value=True, key="run_xai_cb")
                    run_agree = st.checkbox("Agreement Score", value=True, key="run_agree_cb")

                    if st.button("Run Analysis", type="primary", key="analyze_btn", **_content()):
                        if st.session_state.live_session_start is None:
                            st.session_state.live_session_start = datetime.now()
                        status = st.empty()
                        prog   = st.empty()

                        stages = ["Image Loaded", "Preprocessing", "Normalization", "Tensor Prep"]
                        if run_mc:
                            stages.append("MC Dropout")
                        if run_xai:
                            stages += ["Grad-CAM", "Grad-CAM++"]
                        if run_agree and run_xai:
                            stages.append("Agreement Score")
                        stages.append("Report")
                        total_stages = len(stages)

                        class _Tracker:
                            def __init__(self, total, status_el, prog_el):
                                self.i = 0
                                self.total = total
                                self.status = status_el
                                self.prog = prog_el
                            def step(self, label):
                                self.i += 1
                                self.status.info(f"Processing: {label}…")
                                self.prog.progress(min(self.i / self.total, 1.0))

                        tracker = _Tracker(total_stages, status, prog)
                        for stage in ["Image Loaded", "Preprocessing", "Normalization", "Tensor Prep"]:
                            tracker.step(stage)

                        try:
                            total_t0 = time.perf_counter()

                            log_activity("Preprocessing started", "info")
                            (predicted_class, confidence, probability_dict,
                             preprocessing_ms, inference_ms) = predict_image(
                                image, model, class_names, DEVICE
                            )
                            log_activity(
                                f"Inference: {predicted_class} ({confidence:.1f}%)",
                                "success",
                            )

                            mc_result = None
                            if run_mc:
                                tracker.step("MC Dropout")
                                try:
                                    mc_result = mc_dropout_predict(
                                        image, model, class_names, DEVICE,
                                        n_samples=mc_n,
                                    )
                                    if mc_result:
                                        log_activity(
                                            f"MC Dropout: σ={mc_result['uncertainty']:.4f}",
                                            "info",
                                        )
                                except Exception as xai_err:
                                    log_activity(f"MC Dropout failed: {xai_err}", "warn")
                                    st.warning(f"MC Dropout failed: {xai_err}")

                            gradcam_png = gradcam_ms = None
                            gradcam_pp_png = gradcam_pp_ms = None
                            if run_xai:
                                tracker.step("Grad-CAM")
                                try:
                                    t_gc = time.perf_counter()
                                    gradcam_png = generate_gradcam(image, model, model_name, DEVICE)
                                    gradcam_ms = (time.perf_counter() - t_gc) * 1000
                                    log_activity("Grad-CAM generated", "info")
                                except Exception as xai_err:
                                    log_activity(f"Grad-CAM failed: {xai_err}", "warn")
                                    st.warning(f"Grad-CAM failed: {xai_err}")

                                tracker.step("Grad-CAM++")
                                try:
                                    t_pp = time.perf_counter()
                                    gradcam_pp_png = generate_gradcam_pp(image, model, model_name, DEVICE)
                                    gradcam_pp_ms = (time.perf_counter() - t_pp) * 1000
                                    log_activity("Grad-CAM++ generated", "info")
                                except Exception as xai_err:
                                    log_activity(f"Grad-CAM++ failed: {xai_err}", "warn")
                                    st.warning(f"Grad-CAM++ failed: {xai_err}")

                            agree_score = None
                            if run_agree and run_xai:
                                tracker.step("Agreement Score")
                                try:
                                    agree_score = explanation_agreement(
                                        image, model, model_name, DEVICE
                                    )
                                    if agree_score is not None:
                                        log_activity(f"Agreement: {agree_score:.3f}", "info")
                                    else:
                                        log_activity("Agreement unavailable.", "warn")
                                        st.warning("Agreement could not be computed.")
                                except Exception as xai_err:
                                    log_activity(f"Agreement failed: {xai_err}", "warn")
                                    st.warning(f"Agreement failed: {xai_err}")

                            total_ms = (time.perf_counter() - total_t0) * 1000
                            tracker.step("Report")

                            result = {
                                "prediction": predicted_class,
                                "confidence": confidence,
                                "probabilities": probability_dict,
                                "model": model_name,
                                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                "latency_ms": inference_ms,
                                "preprocessing_ms": preprocessing_ms,
                                "gradcam_ms": gradcam_ms if gradcam_png else None,
                                "gradcam_pp_ms": gradcam_pp_ms if gradcam_pp_png else None,
                                "total_ms": total_ms,
                                "gradcam_available": gradcam_png is not None,
                                "gradcam_pp_available": gradcam_pp_png is not None,
                                "image_id": image_id,
                                "uncertainty": mc_result["uncertainty"] if mc_result else None,
                                "mc_band": mc_result["band"] if mc_result else None,
                                "agreement_score": agree_score,
                            }

                            st.session_state.gradcam_png    = gradcam_png
                            st.session_state.gradcam_pp_png = gradcam_pp_png
                            st.session_state.last_result    = result
                            st.session_state.last_image     = image.copy()
                            st.session_state.mc_result      = mc_result
                            st.session_state.agreement_score = agree_score
                            st.session_state.prediction_history.append(result)
                            st.session_state.prediction_history = (
                                st.session_state.prediction_history[-MAX_HISTORY:]
                            )
                            update_live_stats(result, inference_ms)
                            log_activity("Report generated", "success")
                            status.success(
                                f"{predicted_class} · {confidence:.2f}% · {inference_ms:.0f} ms"
                            )
                            prog.empty()

                        except Exception as exc:
                            err_l = str(exc).lower()
                            if "out of memory" in err_l:
                                msg = "CUDA out of memory. Enable Force CPU in Settings and retry."
                            elif "cuda" in err_l:
                                msg = f"CUDA error: {exc}. Try Force CPU in Settings."
                            else:
                                msg = "Analysis failed."
                            status.error(msg)
                            with st.expander("Technical Details"):
                                st.code(str(exc))
                            log_activity(f"Analysis failed: {exc}", "error")

        if (image_id is not None
                and st.session_state.last_result is not None
                and st.session_state.last_result.get("image_id") == image_id):

            result = st.session_state.last_result
            mc_res = st.session_state.mc_result
            conf_v = result["confidence"]
            conf_lbl = "High" if conf_v >= 80 else "Moderate" if conf_v >= 60 else "Low"
            conf_cls = "high" if conf_v >= 80 else "moderate" if conf_v >= 60 else "low"

            st.markdown(safe_html(f"""
            <div class="diagnostic-panel">
                <div class="diag-header">
                    <span class="diag-label">AI Classification Result</span>
                </div>
                <div class="diag-prediction">{_escape_html(result['prediction'])}</div>
                <div class="diag-confidence">{conf_v:.2f}% model confidence</div>
                <div style="margin-top:.8rem">
                    <span class="badge badge-{conf_cls}">{conf_lbl} Confidence</span>
                </div>
            </div>"""), unsafe_allow_html=True)

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Confidence",    f"{conf_v:.2f}%")
            m2.metric("Inference",     f"{result['latency_ms']:.1f} ms")
            m3.metric("Preprocessing", f"{result.get('preprocessing_ms',0):.1f} ms")
            m4.metric("Total Time",    f"{result.get('total_ms',0):.1f} ms")

            if mc_res:
                unc   = mc_res["uncertainty"]
                band  = mc_res["band"]
                color = mc_res["color"]
                st.markdown(safe_html(f"""
                <div class="uncertainty-card">
                    <div class="unc-title">MC Dropout Uncertainty ({st.session_state.get('mc_samples', MC_SAMPLES_DEF)} passes)</div>
                    <div style="display:flex;justify-content:space-between;align-items:center;gap:1rem;flex-wrap:wrap">
                        <div>
                            <div class="unc-value" style="color:{color}">σ = {unc:.4f}</div>
                            <div class="unc-band" style="color:{color}">{_escape_html(band)}</div>
                        </div>
                        <div style="text-align:right;font-size:.76rem;color:#a8bcd8">
                            MC Confidence: <b style="color:#22d3ee;font-family:var(--font-mono)">{mc_res['confidence']:.2f}%</b><br>
                            Prediction: <b>{_escape_html(mc_res['prediction'])}</b>
                        </div>
                    </div>
                </div>"""), unsafe_allow_html=True)

            agree = result.get("agreement_score")
            if agree is not None:
                a_color = "#34d399" if agree >= 0.7 else "#f59e0b" if agree >= 0.5 else "#ef4444"
                a_label = ("High Agreement" if agree >= 0.7
                           else "Moderate Agreement" if agree >= 0.5
                           else "Low Agreement")
                st.markdown(safe_html(f"""
                <div class="xai-card">
                    <div class="xai-title">{Icons.eye(12, "#22d3ee")} Explanation Agreement Score</div>
                    <div style="display:flex;justify-content:space-between;align-items:center;gap:1rem;flex-wrap:wrap">
                        <div class="xai-text">
                            Pearson correlation between Grad-CAM and Grad-CAM++ heatmaps.
                        </div>
                        <div style="text-align:right;min-width:90px;margin-left:1rem">
                            <div style="font-family:var(--font-display);font-size:1.45rem;font-weight:800;color:{a_color};letter-spacing:-.02em">{agree:.3f}</div>
                            <div style="font-size:.66rem;color:{a_color};font-weight:700">{a_label}</div>
                        </div>
                    </div>
                </div>"""), unsafe_allow_html=True)

            st.write("")
            st.markdown(
                _icon_header(Icons.pie_chart(20, "#22d3ee"), "Probability Distribution", level=4),
                unsafe_allow_html=True,
            )
            prob_html = '<div class="prob-grid">' + "".join(
                f'<div class="prob-card {"is-top" if label == result["prediction"] else ""}">'
                f'<div class="prob-value">{prob:.1f}%</div>'
                f'<div class="prob-label">{_escape_html(label)}</div>'
                f'{"<div class=\"prob-top-tag\">Top Prediction</div>" if label == result["prediction"] else ""}'
                f'</div>'
                for label, prob in result["probabilities"].items()
            ) + '</div>'
            st.markdown(prob_html, unsafe_allow_html=True)

            st.write("")
            st.markdown(
                _icon_header(Icons.activity(20, "#22d3ee"), "Live Probability Stream", level=4),
                unsafe_allow_html=True,
            )
            render_live_probability_animation(result, mc_result=mc_res)

            gc_png = st.session_state.get("gradcam_png")
            pp_png = st.session_state.get("gradcam_pp_png")

            if gc_png or pp_png:
                st.write("")
                st.markdown(
                    _icon_header(Icons.eye(20, "#22d3ee"), "Dual XAI Visualization", level=4),
                    unsafe_allow_html=True,
                )
                gc_col, pp_col = st.columns(2)
                with gc_col:
                    st.markdown(
                        _icon_header(Icons.heatmap(18, "#22d3ee"), "Grad-CAM · Jet", level=5),
                        unsafe_allow_html=True,
                    )
                    if gc_png:
                        st.image(gc_png, **_stretch())
                        st.caption("α=0.44")
                with pp_col:
                    st.markdown(
                        _icon_header(Icons.heatmap(18, "#22d3ee"), "Grad-CAM++ · Inferno", level=5),
                        unsafe_allow_html=True,
                    )
                    if pp_png:
                        st.image(pp_png, **_stretch())
                        st.caption("α=0.46")

            dl_svg = Icons.download(16, "#22d3ee")
            st.markdown(safe_html(f"""
            <div class="export-head">
                <div class="export-head-icon">{dl_svg}</div>
                <div>
                    <div class="export-head-title">Export</div>
                </div>
                <span class="export-head-badge">Ready</span>
            </div>"""), unsafe_allow_html=True)

            if gc_png and pp_png:
                dc1, dc2 = st.columns(2)
                with dc1:
                    st.markdown('<div class="export-item-label">Grad-CAM · PNG</div>', unsafe_allow_html=True)
                    st.download_button("Download Grad-CAM", gc_png, "gradcam.png",
                                       "image/png", key="dl_gc", **_stretch())
                with dc2:
                    st.markdown('<div class="export-item-label">Grad-CAM++ · PNG</div>', unsafe_allow_html=True)
                    st.download_button("Download Grad-CAM++", pp_png, "gradcam_pp.png",
                                       "image/png", key="dl_pp", **_stretch())
            elif gc_png:
                st.markdown('<div class="export-item-label">Grad-CAM · PNG</div>', unsafe_allow_html=True)
                st.download_button("Download Grad-CAM", gc_png, "gradcam.png",
                                   "image/png", key="dl_gc", **_stretch())
            elif pp_png:
                st.markdown('<div class="export-item-label">Grad-CAM++ · PNG</div>', unsafe_allow_html=True)
                st.download_button("Download Grad-CAM++", pp_png, "gradcam_pp.png",
                                   "image/png", key="dl_pp", **_stretch())

            st.markdown(
                '<div class="export-item-label" style="margin-top:.9rem">Report · TXT</div>',
                unsafe_allow_html=True,
            )
            st.download_button(
                "Download Report",
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
                    "Agreement Score     : " + (
                        f"{result.get('agreement_score'):.4f}"
                        if result.get('agreement_score') is not None else 'N/A'
                    ),
                    "─────────────────────────────────────────────────────",
                    "  TIMING",
                    "─────────────────────────────────────────────────────",
                    f"Preprocessing       : {result.get('preprocessing_ms',0):.2f} ms",
                    f"Inference           : {result['latency_ms']:.2f} ms",
                    f"Grad-CAM            : {result.get('gradcam_ms') or 0:.2f} ms",
                    f"Grad-CAM++          : {result.get('gradcam_pp_ms') or 0:.2f} ms",
                    f"Total               : {result.get('total_ms',0):.2f} ms",
                    "═══════════════════════════════════════════════════════",
                ]),
                "neurolens_report.txt",
                "text/plain",
                key="dl_report",
                type="primary",
                **_stretch(),
            )


# ══════════════════════════════════════════════════════════════════════════════
# DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════

elif nav == "Dashboard":
    chart_h = Icons.chart(22, "#22d3ee")
    st.markdown(
        f"<h1 style='display:flex;align-items:center;gap:.5rem;flex-wrap:wrap'>"
        f"{chart_h} Dashboard</h1>",
        unsafe_allow_html=True,
    )
    render_live_ticker()

    if st.session_state.live_session_start is None:
        st.session_state.live_session_start = datetime.now()
    dash_hist = st.session_state.prediction_history

    if not dash_hist:
        chart_svg = Icons.chart(32, "#7ba3d6")
        st.markdown(safe_html(f"""
        <div class="empty-state">
            <div class="empty-icon">{chart_svg}</div>
            <div class="empty-title">No data</div>
        </div>"""), unsafe_allow_html=True)
    else:
        total   = st.session_state.live_predictions_count
        avg_c   = st.session_state.live_avg_confidence
        unique  = len(st.session_state.live_class_counts)
        avg_lat = float(np.mean([h.get("latency_ms", 0) for h in dash_hist]))
        unc_data = [h["uncertainty"] for h in dash_hist if h.get("uncertainty") is not None]
        avg_unc = float(np.mean(unc_data)) if unc_data else None
        agree_data = [h["agreement_score"] for h in dash_hist if h.get("agreement_score") is not None]
        avg_agree  = float(np.mean(agree_data)) if agree_data else None

        p_svg = Icons.pulse(18, "#22d3ee")
        t_svg = Icons.target(18, "#22d3ee")
        f_svg = Icons.flask(18, "#22d3ee")
        z_svg = Icons.zap(18, "#22d3ee")
        sh_svg = Icons.shield(18, "#22d3ee")

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
            <div style="margin:.7rem 0;padding:.75rem 1.15rem;border-radius:11px;background:linear-gradient(135deg,rgba(37,99,235,.14),rgba(16,185,129,.08));border:1px solid rgba(34,211,238,.32);display:flex;justify-content:space-between;align-items:center;gap:1rem;flex-wrap:wrap">
                <span style="font-size:.66rem;color:#22d3ee;font-weight:700;letter-spacing:.1em">Avg. Explanation Agreement</span>
                <span style="font-family:var(--font-display);font-size:1.25rem;font-weight:800;color:#e2e8f0;letter-spacing:-.02em">{avg_agree:.3f}</span>
            </div>"""), unsafe_allow_html=True)

        st.write("")
        col_l, col_r = st.columns([2, 1])
        with col_l:
            st.markdown(
                _icon_header(Icons.pie_chart(20, "#22d3ee"), "Prediction Distribution"),
                unsafe_allow_html=True,
            )
            counts = st.session_state.live_class_counts
            mx = max(counts.values()) if counts else 1
            for label, count in counts.items():
                pct = (count / mx) * 100
                live_pct = (count / total) * 100 if total else 0
                st.markdown(safe_html(f"""
                <div class="dist-card">
                    <div class="dist-row"><div class="dist-name">{_escape_html(label)}</div><div class="dist-count">{count} · {live_pct:.1f}%</div></div>
                    <div class="dist-track"><div class="dist-fill" style="width:{pct:.0f}%"></div></div>
                </div>"""), unsafe_allow_html=True)

            st.write("")
            st.markdown(
                _icon_header(Icons.trending_up(20, "#22d3ee"), "Confidence Trend"),
                unsafe_allow_html=True,
            )
            cf = plot_confidence_trend(dash_hist)
            if cf:
                st.pyplot(cf, **_stretch_pyplot())
                plt.close(cf)
            else:
                st.caption("Need ≥2 analyses.")

            st.markdown(
                _icon_header(Icons.clock(20, "#22d3ee"), "Inference Latency"),
                unsafe_allow_html=True,
            )
            lf = plot_latency_trend(dash_hist)
            if lf:
                st.pyplot(lf, **_stretch_pyplot())
                plt.close(lf)
            else:
                st.caption("Need ≥2 analyses.")

            if unc_data:
                st.markdown(
                    _icon_header(Icons.activity(20, "#22d3ee"), "Uncertainty Trend"),
                    unsafe_allow_html=True,
                )
                uf = plot_uncertainty_history(dash_hist)
                if uf:
                    st.pyplot(uf, **_stretch_pyplot())
                    plt.close(uf)

        with col_r:
            st.markdown(
                _icon_header(Icons.activity(18, "#22d3ee"), "Activity Feed", level=4),
                unsafe_allow_html=True,
            )
            render_activity_feed()

            st.write("")
            st.markdown(
                _icon_header(Icons.file_text(18, "#22d3ee"), "Latest Result", level=4),
                unsafe_allow_html=True,
            )
            latest = dash_hist[-1]
            el = "—"
            if st.session_state.live_session_start:
                e = datetime.now() - st.session_state.live_session_start
                el = f"{int(e.total_seconds())}s"
            rows = [
                ("Prediction",    latest["prediction"]),
                ("Confidence",    f"{latest['confidence']:.2f}%"),
                ("Architecture",  latest["model"]),
                ("Inference",     f"{latest.get('latency_ms',0):.0f} ms"),
                ("Uncertainty σ",
                 f"{latest['uncertainty']:.4f}"
                 if latest.get("uncertainty") is not None else "—"),
                ("Agreement",
                 f"{latest['agreement_score']:.3f}"
                 if latest.get("agreement_score") is not None else "—"),
                ("Session Time",  el),
                ("Analyzed At",   latest["timestamp"]),
            ]
            st.markdown('<div class="latest-card">' + "".join(
                f'<div class="latest-row"><div class="latest-key">{_escape_html(k)}</div><div class="latest-val">{_escape_html(v)}</div></div>'
                for k, v in rows
            ) + '</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# HISTORY
# ══════════════════════════════════════════════════════════════════════════════

elif nav == "History":
    hist_h = Icons.history(22, "#22d3ee")
    st.markdown(
        f"<h1 style='display:flex;align-items:center;gap:.5rem;flex-wrap:wrap'>"
        f"{hist_h} History</h1>",
        unsafe_allow_html=True,
    )
    render_live_ticker()
    hist_list = st.session_state.prediction_history

    if not hist_list:
        hist_svg = Icons.history(32, "#7ba3d6")
        st.markdown(safe_html(f"""
        <div class="empty-state">
            <div class="empty-icon">{hist_svg}</div>
            <div class="empty-title">No history</div>
        </div>"""), unsafe_allow_html=True)
    else:
        st.markdown(
            _icon_header(Icons.filter_icon(18, "#22d3ee"), "Filter & Search", level=4),
            unsafe_allow_html=True,
        )
        fc1, fc2, fc3, fc4 = st.columns([1, 1, 1, 2])
        pred_filter = fc1.selectbox("Prediction", ["All"] + CLASS_NAMES)
        conf_filter = fc2.selectbox(
            "Confidence", ["All", "High (≥80%)", "Moderate (60–79%)", "Low (<60%)"]
        )
        sort_order  = fc3.selectbox("Sort", ["Newest first", "Oldest first"])
        search_q    = fc4.text_input("Search")

        filtered = []
        for rec in hist_list:
            c = rec["confidence"]
            if pred_filter != "All" and rec["prediction"] != pred_filter:
                continue
            if conf_filter == "High (≥80%)" and c < 80:
                continue
            if conf_filter == "Moderate (60–79%)" and not (60 <= c < 80):
                continue
            if conf_filter == "Low (<60%)" and c >= 60:
                continue
            q = search_q.strip().lower()
            if q and q not in rec["model"].lower() and q not in rec["prediction"].lower():
                continue
            filtered.append(rec)

        if sort_order == "Newest first":
            filtered = list(reversed(filtered))

        if not filtered:
            st.info("No records match these filters.")
        else:
            st.caption(f"Showing {len(filtered)} of {len(hist_list)} records")
            scan_sm = Icons.scan(18, "#22d3ee")
            for item in filtered:
                conf = item["confidence"]
                cc = "conf-hi" if conf >= 80 else "conf-mid" if conf >= 60 else "conf-lo"
                unc = f" · σ={item['uncertainty']:.4f}" if item.get("uncertainty") is not None else ""
                ag  = (f" · Agr={item['agreement_score']:.3f}"
                       if item.get("agreement_score") is not None else "")
                st.markdown(safe_html(f"""
                <div class="thumb-card">
                    <div class="thumb-icon">{scan_sm}</div>
                    <div class="thumb-info">
                        <div class="thumb-title">{_escape_html(item['prediction'])}</div>
                        <div class="thumb-meta">{_escape_html(item['timestamp'])} · {_escape_html(item['model'])}{unc}{ag}</div>
                    </div>
                    <div><span class="conf-badge {cc}">{conf:.1f}%</span></div>
                </div>"""), unsafe_allow_html=True)

                with st.expander(f"Details — {item['timestamp']}"):
                    d1, d2 = st.columns(2)
                    with d1:
                        st.markdown(
                            _icon_header(Icons.list_icon(16, "#22d3ee"), "Report Summary", level=5),
                            unsafe_allow_html=True,
                        )
                        st.write(f"**Prediction:** {item['prediction']}")
                        st.write(f"**Confidence:** {item['confidence']:.4f}%")
                        st.write(f"**Architecture:** {item['model']}")
                        st.write(f"**Inference:** {item.get('latency_ms',0):.1f} ms")
                    with d2:
                        st.markdown(
                            _icon_header(Icons.database(16, "#22d3ee"), "Extended Metrics", level=5),
                            unsafe_allow_html=True,
                        )
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
    st.markdown(
        f"<h1 style='display:flex;align-items:center;gap:.5rem;flex-wrap:wrap'>"
        f"{hm_h} Grad-CAM</h1>",
        unsafe_allow_html=True,
    )
    render_live_ticker()

    gc_png = st.session_state.get("gradcam_png")
    pp_png = st.session_state.get("gradcam_pp_png")

    if not gc_png and not pp_png:
        hm_svg = Icons.heatmap(32, "#7ba3d6")
        st.markdown(safe_html(f"""
        <div class="empty-state">
            <div class="empty-icon">{hm_svg}</div>
            <div class="empty-title">No Grad-CAM yet</div>
        </div>"""), unsafe_allow_html=True)
    else:
        oc, gcol, pcol = st.columns(3)
        with oc:
            st.markdown(
                _icon_header(Icons.image(18, "#22d3ee"), "Original", level=4),
                unsafe_allow_html=True,
            )
            if st.session_state.last_image:
                st.image(st.session_state.last_image, **_stretch())
        with gcol:
            st.markdown(
                _icon_header(Icons.heatmap(18, "#22d3ee"), "Grad-CAM · Jet", level=4),
                unsafe_allow_html=True,
            )
            if gc_png:
                st.image(gc_png, **_stretch())
        with pcol:
            st.markdown(
                _icon_header(Icons.heatmap(18, "#22d3ee"), "Grad-CAM++ · Inferno", level=4),
                unsafe_allow_html=True,
            )
            if pp_png:
                st.image(pp_png, **_stretch())

        agree = st.session_state.agreement_score
        if agree is not None:
            a_color = "#34d399" if agree >= 0.7 else "#f59e0b" if agree >= 0.5 else "#ef4444"
            a_label = "High Agreement" if agree >= 0.7 else "Moderate" if agree >= 0.5 else "Low Agreement"
            st.markdown(safe_html(f"""
            <div class="xai-card">
                <div class="xai-title">{Icons.eye(12, "#22d3ee")} Explanation Agreement Score</div>
                <div style="display:flex;justify-content:space-between;align-items:center;gap:1rem;flex-wrap:wrap">
                    <div style="text-align:right;min-width:85px;margin-left:1rem">
                        <div style="font-family:var(--font-display);font-size:1.45rem;font-weight:800;color:{a_color};letter-spacing:-.02em">{agree:.3f}</div>
                        <div style="font-size:.66rem;color:{a_color};font-weight:700">{a_label}</div>
                    </div>
                </div>
            </div>"""), unsafe_allow_html=True)

        if st.session_state.last_result:
            res = st.session_state.last_result
            c1, c2 = st.columns(2)
            c1.metric("Prediction", res["prediction"])
            c2.metric("Confidence", f"{res['confidence']:.2f}%")

        dl1, dl2 = st.columns(2)
        if gc_png:
            dl1.download_button("Download Grad-CAM", gc_png,
                                "gradcam.png", "image/png",
                                key="gradcam_dl", **_stretch())
        if pp_png:
            dl2.download_button("Download Grad-CAM++", pp_png,
                                "gradcam_pp.png", "image/png",
                                key="gradcam_pp_dl", **_stretch())


# ══════════════════════════════════════════════════════════════════════════════
# XAI LAB
# ══════════════════════════════════════════════════════════════════════════════

elif nav == "XAI Lab":
    lab_h = Icons.lab(22, "#22d3ee")
    st.markdown(
        f"<h1 style='display:flex;align-items:center;gap:.5rem;flex-wrap:wrap'>"
        f"{lab_h} XAI Lab</h1>",
        unsafe_allow_html=True,
    )
    render_live_ticker()
    xai_hist = st.session_state.prediction_history

    if not xai_hist:
        lab_svg = Icons.lab(32, "#7ba3d6")
        st.markdown(safe_html(f"""
        <div class="empty-state">
            <div class="empty-icon">{lab_svg}</div>
            <div class="empty-title">No XAI data yet</div>
        </div>"""), unsafe_allow_html=True)
    else:
        st.markdown(
            _icon_header(Icons.activity(20, "#22d3ee"), "Uncertainty Distribution"),
            unsafe_allow_html=True,
        )
        unc_data = [h["uncertainty"] for h in xai_hist if h.get("uncertainty") is not None]
        if len(unc_data) >= 2:
            uf = plot_uncertainty_history(xai_hist)
            if uf:
                st.pyplot(uf, **_stretch_pyplot())
                plt.close(uf)
        else:
            st.caption("Need ≥2 analyses with MC Dropout enabled.")

        bands: Dict[str, int] = {}
        for h in xai_hist:
            b = h.get("mc_band")
            if b:
                bands[b] = bands.get(b, 0) + 1

        if bands:
            st.write("")
            st.markdown(
                _icon_header(Icons.bar_chart(20, "#22d3ee"), "Reliability Band Distribution"),
                unsafe_allow_html=True,
            )
            bc1, bc2, bc3, bc4 = st.columns(4)
            band_map = {
                "Very High Reliability": (bc1, "#34d399"),
                "High Reliability":      (bc2, "#22d3ee"),
                "Moderate Reliability":  (bc3, "#f59e0b"),
                "Low Reliability":       (bc4, "#ef4444"),
            }
            for band, (col, color) in band_map.items():
                cnt = bands.get(band, 0)
                with col:
                    st.markdown(safe_html(f"""
                    <div class="metric-card" style="border-color:{color}55">
                        <div class="metric-value" style="color:{color};font-size:1.6rem">{cnt}</div>
                        <div class="metric-label">{_escape_html(band.replace(' Reliability',''))}</div>
                    </div>"""), unsafe_allow_html=True)

        agree_hist = [
            (i + 1, h["agreement_score"])
            for i, h in enumerate(xai_hist)
            if h.get("agreement_score") is not None
        ]
        if len(agree_hist) >= 2:
            st.write("")
            st.markdown(
                _icon_header(Icons.trending_up(20, "#22d3ee"), "Agreement Score Trend"),
                unsafe_allow_html=True,
            )
            fig2, ax2 = _dark_fig()
            try:
                xs, ys = zip(*agree_hist)
                ax2.plot(xs, ys, marker="D", lw=2, ms=4, color="#22d3ee")
                ax2.fill_between(xs, ys, alpha=0.08, color="#22d3ee")
                ax2.axhline(0.7, color="#34d399", lw=1, ls="--", alpha=0.65)
                ax2.axhline(0.5, color="#f59e0b", lw=1, ls="--", alpha=0.65)
                ax2.set_ylim(0, 1.05)
                ax2.set_xlabel("Analysis #", color="#a8bcd8", fontsize=9)
                ax2.set_ylabel("Agreement Score", color="#a8bcd8", fontsize=9)
                ax2.grid(True, alpha=0.09, ls="--", color="#7ba3d6")
                fig2.tight_layout(pad=0.5)
                st.pyplot(fig2, **_stretch_pyplot())
            finally:
                plt.close(fig2)

        st.write("")
        st.markdown(
            _icon_header(Icons.grid(20, "#22d3ee"), "Per-Class Uncertainty Analysis"),
            unsafe_allow_html=True,
        )
        class_unc: Dict[str, List[float]] = {c: [] for c in CLASS_NAMES}
        for h in xai_hist:
            if h.get("uncertainty") is not None and h.get("prediction") in class_unc:
                class_unc[h["prediction"]].append(h["uncertainty"])
        rows = []
        for cls, vals in class_unc.items():
            if vals:
                rows.append({
                    "Class":  cls,
                    "Count":  len(vals),
                    "Mean σ": f"{np.mean(vals):.6f}",
                    "Min σ":  f"{np.min(vals):.6f}",
                    "Max σ":  f"{np.max(vals):.6f}",
                })
        if rows:
            st.dataframe(pd.DataFrame(rows), hide_index=True, **_stretch())
        else:
            st.caption("No per-class uncertainty data yet.")


# ══════════════════════════════════════════════════════════════════════════════
# SETTINGS
# ══════════════════════════════════════════════════════════════════════════════

elif nav == "Settings":
    set_h = Icons.settings(22, "#22d3ee")
    st.markdown(
        f"<h1 style='display:flex;align-items:center;gap:.5rem;flex-wrap:wrap'>"
        f"{set_h} Settings</h1>",
        unsafe_allow_html=True,
    )

    c1, c2 = st.columns(2)
    with c1:
        lay_svg = Icons.layers(18, "#22d3ee")
        st.markdown(safe_html(f"""
        <div class="info-card">
            <div class="info-card-icon">{lay_svg}</div>
            <h3>Neural Engine</h3>
            <p><b>Architecture:</b> {_escape_html(model_name or 'Unavailable')}</p>
            <p><b>Status:</b> {"Ready" if not model_error else "Failed"}</p>
            <p><b>Device:</b> {_escape_html(str(DEVICE))}</p>
            <p><b>Input Resolution:</b> {IMG_SIZE} × {IMG_SIZE}</p>
            <p><b>Classes:</b> {_escape_html(', '.join(CLASS_NAMES))}</p>
            <p><b>Checkpoint:</b> {_escape_html(MODEL_PATH.name if MODEL_PATH.exists() else 'Not found')}</p>
            <p><b>MC Dropout Passes:</b> {st.session_state.get('mc_samples', MC_SAMPLES_DEF)}</p>
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
            <h3>System</h3>
            <p><b>PyTorch:</b> {_escape_html(torch.__version__)}</p>
            <p><b>Torchvision:</b> {_escape_html(tv)}</p>
            <p><b>Streamlit:</b> {_escape_html(st.__version__)}</p>
            <p><b>Python:</b> {_escape_html(platform.python_version())}</p>
            <p><b>CUDA Available:</b> {torch.cuda.is_available()}</p>
            <p><b>CUDA Device:</b> {_escape_html(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')}</p>
            <p><b>XAI Methods:</b> Grad-CAM · Grad-CAM++</p>
        </div>"""), unsafe_allow_html=True)

    st.write("")
    st.markdown(
        _icon_header(Icons.sliders(18, "#22d3ee"), "Runtime Controls", level=4),
        unsafe_allow_html=True,
    )

    rc1, rc2 = st.columns(2)
    with rc1:
        prev_force_cpu = bool(st.session_state.get("force_cpu", False))

        def _on_force_cpu_change():
            try:
                load_model.clear()
            except Exception:
                pass

        st.toggle(
            "Force CPU Inference",
            key="force_cpu",
            on_change=_on_force_cpu_change,
            help="Reloads the neural engine on CPU.",
        )

        if st.session_state.force_cpu != prev_force_cpu:
            st.info("Device preference changed — reloading engine…")
            st.rerun()

        if st.session_state.force_cpu and DEVICE.type == "cpu":
            st.success("Engine is running on CPU (forced).")
        elif st.session_state.force_cpu and DEVICE.type == "cuda":
            st.warning("Force CPU requested but engine still on CUDA — reloading…")
        elif not st.session_state.force_cpu and DEVICE.type == "cuda":
            st.success("Engine is running on CUDA (GPU).")
        else:
            st.warning("CUDA unavailable; running on CPU.")

    with rc2:
        mc_val = st.number_input(
            "MC Dropout samples",
            min_value=5, max_value=200, step=5,
            value=int(st.session_state.get("mc_samples", MC_SAMPLES_DEF)),
            key="mc_samples_widget",
            help="More samples → better uncertainty estimate, slower runtime.",
        )
        if mc_val != st.session_state.get("mc_samples"):
            st.session_state.mc_samples = int(mc_val)

    st.markdown(
        _icon_header(Icons.terminal(18, "#22d3ee"), "Runtime Details", level=4),
        unsafe_allow_html=True,
    )
    show_tech = st.toggle("Show technical details", key="show_tech")
    if show_tech:
        with st.expander("Checkpoint & Runtime", expanded=True):
            st.write(f"Checkpoint: `{MODEL_PATH}`")
            st.write(f"Device: `{DEVICE}` · Input: `{IMG_SIZE}×{IMG_SIZE}`")
            st.write(f"Norm mean: `{NORM_MEAN}` · std: `{NORM_STD}`")
            st.write(f"Force CPU: `{st.session_state.force_cpu}` · MC samples: `{st.session_state.mc_samples}`")
            if model_error:
                st.code(model_error)

    st.write("")
    if st.button("Reset Session", type="primary", key="settings_reset"):
        st.session_state.settings_confirm_reset = True
    if st.session_state.get("settings_confirm_reset", False):
        st.warning("Reset all session results?")
        rc1b, rc2b, _ = st.columns([1, 1, 4])
        if rc1b.button("Confirm", key="settings_confirm_yes", **_stretch()):
            for k, v in DEFAULTS.items():
                st.session_state[k] = copy.deepcopy(v)
            st.session_state.settings_confirm_reset = False
            st.success("Session cleared.")
            navigate_to("Home")
        if rc2b.button("Cancel", key="settings_confirm_no", **_stretch()):
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
brain_ft   = Icons.brain(26, "#22d3ee")

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
    </div>

    <div>
        <div class="footer-col-title">Tech Stack</div>
        <div class="footer-tech-badges">
            <span class="ft-badge"><span class="ft-badge-dot"></span>PyTorch {_escape_html(torch_ver)}</span>
            <span class="ft-badge"><span class="ft-badge-dot"></span>Streamlit {_escape_html(st_ver)}</span>
            <span class="ft-badge"><span class="ft-badge-dot"></span>Grad-CAM</span>
            <span class="ft-badge"><span class="ft-badge-dot"></span>Grad-CAM++</span>
            <span class="ft-badge"><span class="ft-badge-dot"></span>MC Dropout</span>
            <span class="ft-badge"><span class="ft-badge-dot"></span>{_escape_html(model_name or 'CustomCNN')}</span>
        </div>
        <div class="footer-col-title" style="margin-top:.9rem">Build</div>
        <div style="font-family:var(--font-mono);font-size:.66rem;color:#7ba3d6">{_escape_html(build_time)}</div>
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