# =============================================================================
# NeuroLens AI — Neurodiagnostic Intelligence Platform
# Production SaaS Edition v3.6.2 — Vibrant Blue Edition + Premium UI v6.2
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
from urllib.parse import quote

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
    """Escape user/class-name text for safe HTML injection."""
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


def _render_html(content: str, height: int = 200, scrolling: bool = False):
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
        if (major, minor) >= (1, 37):
            return {"width": "stretch" if container else "content"}
    except Exception:
        pass
    return {"use_container_width": bool(container)}


def _stretch() -> dict:
    return _st_width_arg(True)


def _content() -> dict:
    return _st_width_arg(False)


def _stretch_pyplot() -> dict:
    return {"use_container_width": True}


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


# ─────────────────────────────────────────────────────────────────────────────
# RESPONSIVE UI / DESIGN SYSTEM — VIBRANT BLUE EDITION (UPGRADED)
# ─────────────────────────────────────────────────────────────────────────────
UI_CSS = r"""
:root {
  /* ═══ NEW: Vibrant Blue Background System ═══ */
  --bg:#0a1e3f;
  --bg-2:#0d2551;
  --bg-3:#123067;
  --surface:#123a6f;
  --surface-2:#184b8a;
  --surface-3:#215ba8;
  --surface-4:#2d72c9;

  --blue:#2563eb;
  --blue-hi:#3b82f6;
  --blue-deep:#1e40af;
  --cyan:#06b6d4;
  --cyan-hi:#22d3ee;

  --green:#10b981;
  --green-hi:#22c55e;
  --emerald:#34d399;

  --accent:#06b6d4;
  --accent-hi:#22d3ee;
  --success:#10b981;
  --success-hi:#34d399;
  --warning:#f59e0b;
  --danger:#ef4444;
  --danger-hi:#f87171;
  --violet:#8b5cf6;

  --text:#f1f5f9;
  --text-2:#a8bcd8;
  --text-3:#64748b;

  --line:rgba(59,130,246,.22);
  --line-strong:rgba(59,130,246,.42);
  --line-glow:rgba(34,211,238,.45);

  --font-display:'Sora','Inter',system-ui,sans-serif;
  --font-mono:'JetBrains Mono',ui-monospace,SFMono-Regular,Menlo,monospace;

  --sb-w:280px;
  --hd-h:64px;
}

html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
  background:var(--bg) !important;
  font-family:'Inter',system-ui,sans-serif;
}

/* ═══ NEW: Rich Blue Radial Gradient Background ═══ */
[data-testid="stAppViewContainer"] > .main {
  background:
    radial-gradient(ellipse 80% 55% at 50% 0%, rgba(34,211,238,.14), transparent 60%),
    radial-gradient(circle at 0% 100%, rgba(16,185,129,.10), transparent 40rem),
    radial-gradient(circle at 100% 100%, rgba(59,130,246,.18), transparent 40rem),
    radial-gradient(circle at 100% 0%, rgba(139,92,246,.08), transparent 35rem),
    linear-gradient(180deg, #0d2551 0%, #0a1e3f 45%, #061733 100%) !important;
}

[data-testid="stMainBlockContainer"] {
  width:100% !important;
  max-width:1380px !important;
  margin:0 auto !important;
  padding:calc(var(--hd-h) + 1.5rem) 1.25rem 2.5rem !important;
}

/* ═══════════════════════════════════════════════════════════════════════
   SIDEBAR
   ═══════════════════════════════════════════════════════════════════════ */
[data-testid="stSidebar"] {
  background:
    linear-gradient(180deg, rgba(13,37,81,.98) 0%, rgba(6,23,51,.98) 100%),
    radial-gradient(circle at 0% 0%, rgba(34,211,238,.10), transparent 25rem) !important;
  border-right:1px solid var(--line-strong) !important;
  min-width:var(--sb-w) !important;
  max-width:var(--sb-w) !important;
  width:var(--sb-w) !important;
  box-shadow:inset -1px 0 0 rgba(34,211,238,.08), 4px 0 24px rgba(0,0,0,.20);
}
[data-testid="stSidebar"] > div:first-child {
  width:100% !important;
  padding:0 .85rem 1rem !important;
}

[data-testid="stSidebar"] .stButton { width:100% !important; margin:.22rem 0 !important; }
[data-testid="stSidebar"] .stButton > button {
  position:relative !important;
  width:100% !important;
  min-height:44px !important;
  padding:.6rem .75rem .6rem 2.85rem !important;
  display:flex !important;
  align-items:center !important;
  justify-content:flex-start !important;
  text-align:left !important;
  border:1px solid transparent !important;
  border-radius:10px !important;
  background:transparent !important;
  color:var(--text-2) !important;
  font:600 .82rem/1.2 'Inter',system-ui,sans-serif !important;
  letter-spacing:.01em !important;
  box-shadow:none !important;
  overflow:hidden !important;
  transition:background .18s ease,border-color .18s ease,color .18s ease,transform .18s ease,box-shadow .18s ease !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
  background:linear-gradient(90deg,rgba(34,211,238,.10),rgba(16,185,129,.06)) !important;
  border-color:var(--line-strong) !important;
  color:var(--text) !important;
  transform:translateX(2px) !important;
  box-shadow:0 0 20px rgba(34,211,238,.10) !important;
}
[data-testid="stSidebar"] .stButton > button p {
  margin:0 !important; padding:0 !important; width:100% !important;
  color:inherit !important; font:inherit !important;
}

.sb-nav-group {
  margin:1.1rem .4rem .4rem !important;
  font:800 .62rem/1.2 'Inter',system-ui,sans-serif !important;
  text-transform:uppercase !important;
  letter-spacing:.16em !important;
  background:linear-gradient(90deg,#22d3ee,#34d399);
  -webkit-background-clip:text;
  -webkit-text-fill-color:transparent;
  background-clip:text;
}

.sb-brand {
  display:flex; align-items:center; gap:.8rem;
  padding:.9rem .4rem .75rem;
  border-bottom:1px solid var(--line);
  margin-bottom:.6rem;
}
.sb-brand-icon {
  position:relative;
  width:42px; height:42px;
  display:grid; place-items:center;
  border:1px solid rgba(34,211,238,.42);
  border-radius:12px;
  background:
    linear-gradient(145deg,rgba(37,99,235,.30),rgba(16,185,129,.18)),
    rgba(18,58,111,.9);
  box-shadow:
    0 0 26px rgba(34,211,238,.28),
    inset 0 1px 0 rgba(255,255,255,.08);
  flex:0 0 auto;
}
.sb-brand-icon::after {
  content:'';
  position:absolute;
  inset:-1px;
  border-radius:12px;
  padding:1px;
  background:linear-gradient(135deg,#22d3ee,#10b981);
  -webkit-mask:linear-gradient(#000 0 0) content-box,linear-gradient(#000 0 0);
  -webkit-mask-composite:xor;
  mask-composite:exclude;
  opacity:.75;
  pointer-events:none;
}
.sb-brand-text { min-width:0; }
.sb-brand-name {
  color:#f1f5f9;
  font:800 1rem/1.1 var(--font-display);
  white-space:nowrap;
  letter-spacing:-.01em;
}
.sb-brand-ai {
  background:linear-gradient(135deg,#22d3ee,#34d399);
  -webkit-background-clip:text;
  -webkit-text-fill-color:transparent;
  background-clip:text;
}
.sb-brand-ver {
  margin-left:.4rem;
  color:#7ba3d6;
  font:700 .55rem/1 var(--font-mono);
  padding:.15rem .35rem;
  border:1px solid var(--line);
  border-radius:4px;
  background:rgba(34,211,238,.06);
}
.sb-brand-sub {
  margin-top:.28rem;
  color:#7ba3d6;
  font:500 .6rem/1.2 'Inter',sans-serif;
  letter-spacing:.02em;
}
.sb-brand-pulse {
  position:absolute; right:-2px; top:-2px;
  width:8px; height:8px; border-radius:50%;
  background:#34d399;
  box-shadow:0 0 10px rgba(52,211,153,.9),0 0 20px rgba(52,211,153,.5);
  animation:pulse-dot 2s ease-in-out infinite;
}
@keyframes pulse-dot {
  0%,100% { opacity:1; transform:scale(1); }
  50%     { opacity:.6; transform:scale(1.15); }
}

.sb-session {
  display:flex; align-items:center; gap:.5rem;
  min-height:34px;
  padding:.4rem .65rem;
  margin:.05rem .05rem .6rem;
  border:1px solid var(--line);
  border-radius:9px;
  background:linear-gradient(135deg,rgba(24,75,138,.75),rgba(13,37,81,.65));
  color:#a8bcd8;
}
.sb-session-txt { overflow:hidden; text-overflow:ellipsis; white-space:nowrap; font:600 .68rem Inter,sans-serif; }
.sb-session-time { margin-left:auto; color:#7ba3d6; font:600 .6rem var(--font-mono); }
.sb-session-dot { width:7px; height:7px; border-radius:50%; flex:0 0 auto; }

.sb-engine {
  margin:.85rem .05rem .95rem;
  padding:.85rem;
  border:1px solid var(--line);
  border-radius:11px;
  position:relative;
  background:linear-gradient(145deg,rgba(24,75,138,.92),rgba(6,23,51,.96));
  overflow:hidden;
}
.sb-engine::before {
  content:'';
  position:absolute; top:-40px; right:-40px;
  width:120px; height:120px;
  border-radius:50%;
  background:radial-gradient(circle,rgba(34,211,238,.20),transparent 70%);
  pointer-events:none;
}
.sb-engine-head,.sb-engine-status,.sb-engine-confbar { display:flex; align-items:center; gap:.45rem; }
.sb-engine-head { margin-bottom:.6rem; position:relative; }
.sb-engine-title { color:#cbd5e1; font:700 .7rem Inter,sans-serif; }
.sb-engine-live {
  margin-left:auto;
  color:#34d399;
  font:800 .56rem var(--font-mono);
  letter-spacing:.1em;
  padding:.15rem .4rem;
  border-radius:99px;
  background:rgba(16,185,129,.12);
  border:1px solid rgba(16,185,129,.35);
}
.sb-engine-live-dot {
  display:inline-block; width:5px; height:5px; margin-right:4px;
  border-radius:50%; background:#34d399;
  box-shadow:0 0 8px #34d399;
}
.sb-engine-status { color:#a8bcd8; font:600 .62rem Inter,sans-serif; position:relative; }
.sb-engine-confbar { margin-top:.65rem; position:relative; }
.sb-engine-confbar-track {
  height:5px; flex:1; overflow:hidden;
  border-radius:99px; background:#0d2551;
  border:1px solid rgba(34,211,238,.14);
}
.sb-engine-confbar-fill {
  height:100%; border-radius:99px;
  background:linear-gradient(90deg,#22d3ee,#34d399);
  box-shadow:0 0 12px rgba(34,211,238,.5);
  transition:width .6s cubic-bezier(.2,.8,.2,1);
}
.sb-engine-confbar-lbl {
  min-width:34px; text-align:right;
  color:#cbd5e1; font:700 .6rem var(--font-mono);
}
.sb-engine-grid { display:grid; grid-template-columns:1fr 1fr; gap:.4rem; margin-top:.7rem; position:relative; }
.sb-mini {
  padding:.5rem .55rem;
  border:1px solid var(--line);
  border-radius:7px;
  background:linear-gradient(135deg,rgba(24,75,138,.5),rgba(13,37,81,.3));
}
.sb-mini-lbl { color:#7ba3d6; font:700 .5rem Inter,sans-serif; letter-spacing:.1em; }
.sb-mini-val {
  margin-top:.16rem; color:#e2e8f0;
  font:700 .68rem var(--font-mono);
  overflow:hidden; text-overflow:ellipsis; white-space:nowrap;
}
/* FIX: Added missing mini variant classes */
.sb-mini-white  { border-color:rgba(148,163,184,.25); }
.sb-mini-cyan   { border-color:rgba(34,211,238,.35); }
.sb-mini-violet { border-color:rgba(139,92,246,.35); }

.sb-confirm {
  margin:.5rem 0; padding:.7rem;
  border:1px solid rgba(245,158,11,.30);
  border-radius:9px;
  background:rgba(245,158,11,.08);
}
.sb-confirm-danger { border-color:rgba(239,68,68,.32); background:rgba(239,68,68,.08); }
.sb-confirm-title { color:#e2e8f0; font:700 .68rem Inter,sans-serif; }
.sb-confirm-text { margin-top:.3rem; color:#a8bcd8; font:.6rem/1.4 Inter,sans-serif; }
.sb-warning {
  display:flex; gap:.5rem; align-items:flex-start;
  margin:.85rem .1rem 0; padding:.7rem;
  border:1px solid rgba(245,158,11,.22);
  border-radius:9px;
  background:linear-gradient(135deg,rgba(245,158,11,.08),rgba(245,158,11,.02));
  color:#a8bcd8;
  font:.6rem/1.4 Inter,sans-serif;
}

.stButton > button, .stDownloadButton > button {
  min-height:42px !important;
  border-radius:10px !important;
  font:700 .78rem Inter,system-ui,sans-serif !important;
  transition:all .18s ease !important;
}
.stButton > button {
  border:1px solid var(--line-strong) !important;
  background:linear-gradient(135deg,rgba(24,75,138,.9),rgba(13,37,81,.9)) !important;
  color:var(--text) !important;
}
.stButton > button:hover, .stDownloadButton > button:hover {
  border-color:var(--line-glow) !important;
  box-shadow:0 6px 24px rgba(34,211,238,.22) !important;
  transform:translateY(-1px) !important;
}
.stButton > button[kind="primary"], .stDownloadButton > button[kind="primary"] {
  background:linear-gradient(135deg,#0ea5e9 0%,#22d3ee 50%,#10b981 100%) !important;
  border:1px solid rgba(34,211,238,.50) !important;
  color:#fff !important;
  font-weight:700 !important;
  box-shadow:
    0 4px 20px rgba(34,211,238,.35),
    0 0 0 1px rgba(34,211,238,.15),
    inset 0 1px 0 rgba(255,255,255,.18) !important;
}
.stButton > button[kind="primary"]:hover, .stDownloadButton > button[kind="primary"]:hover {
  background:linear-gradient(135deg,#06b6d4 0%,#22c55e 100%) !important;
  box-shadow:
    0 6px 28px rgba(34,211,238,.50),
    0 0 0 1px rgba(34,211,238,.25),
    inset 0 1px 0 rgba(255,255,255,.22) !important;
  transform:translateY(-2px) !important;
}
.stDownloadButton > button { width:100% !important; }
[data-testid="stHorizontalBlock"] { align-items:stretch !important; }
[data-testid="stHorizontalBlock"] > div { min-width:0 !important; }
.stTabs [data-baseweb="tab-list"] { gap:.25rem !important; border-bottom:1px solid var(--line) !important; }
.stTabs [data-baseweb="tab"] {
  min-height:40px !important; padding:0 .9rem !important;
  color:#7ba3d6 !important; font:700 .72rem Inter,sans-serif !important;
}
.stTabs [aria-selected="true"] { color:#22d3ee !important; }

.stFileUploader { border-radius:12px !important; }
.stFileUploader section {
  border:1px dashed rgba(34,211,238,.42) !important;
  background:linear-gradient(135deg,rgba(37,99,235,.08),rgba(16,185,129,.05)) !important;
  border-radius:12px !important;
}
.stAlert { border-radius:10px !important; }
[data-testid="stMetric"] {
  background:linear-gradient(135deg,rgba(24,75,138,.72),rgba(13,37,81,.72));
  border:1px solid var(--line);
  border-radius:10px; padding:.8rem !important;
}
[data-testid="stMetricLabel"] { color:#7ba3d6 !important; }
[data-testid="stMetricValue"] { color:#e2e8f0 !important; }

/* ═══════════════════════════════════════════════════════════════════════
   STICKY HEADER — PREMIUM BLUE UPGRADE
   ═══════════════════════════════════════════════════════════════════════ */
.sticky-header {
  position:fixed; z-index:9999;
  top:.7rem; left:calc(var(--sb-w) + .75rem); right:.75rem;
  min-height:var(--hd-h);
  display:flex; align-items:center; gap:.9rem;
  padding:.55rem 1.05rem;
  border:1px solid rgba(59,130,246,.42);
  border-radius:14px;
  background:
    linear-gradient(90deg,rgba(18,58,111,.94),rgba(10,30,63,.94)),
    rgba(10,30,63,.90);
  backdrop-filter:blur(22px) saturate(150%);
  -webkit-backdrop-filter:blur(22px) saturate(150%);
  box-shadow:
    0 12px 44px rgba(0,0,0,.35),
    0 0 0 1px rgba(34,211,238,.10),
    inset 0 1px 0 rgba(255,255,255,.06);
  overflow:hidden;
}
.sticky-header::before {
  content:'';
  position:absolute;
  inset:0;
  border-radius:14px;
  padding:1px;
  background:linear-gradient(90deg,
    rgba(59,130,246,.65),
    rgba(34,211,238,.65),
    rgba(16,185,129,.55),
    rgba(59,130,246,.65));
  -webkit-mask:linear-gradient(#000 0 0) content-box,linear-gradient(#000 0 0);
  -webkit-mask-composite:xor;
  mask-composite:exclude;
  pointer-events:none;
  opacity:.85;
  background-size:200% 100%;
  animation:hd-border-flow 8s linear infinite;
}
.sticky-header::after {
  content:'';
  position:absolute;
  top:0; left:0; right:0; height:1px;
  background:linear-gradient(90deg,
    transparent,
    rgba(34,211,238,.85),
    rgba(16,185,129,.65),
    transparent);
  background-size:200% 100%;
  animation:hd-shine 5s linear infinite;
  pointer-events:none;
}
@keyframes hd-border-flow {
  0%   { background-position:0% 50%; }
  100% { background-position:200% 50%; }
}
@keyframes hd-shine {
  0%   { background-position:200% 50%; opacity:.3; }
  50%  { background-position:0% 50%;   opacity:.95; }
  100% { background-position:-200% 50%; opacity:.3; }
}

.hd-brand { display:flex; align-items:center; gap:.65rem; flex:0 0 auto; position:relative; z-index:1; }
.hd-brand-icon {
  position:relative;
  width:38px; height:38px;
  display:grid; place-items:center;
  border:1px solid rgba(34,211,238,.45);
  border-radius:10px;
  background:
    linear-gradient(135deg,rgba(37,99,235,.32),rgba(16,185,129,.20)),
    rgba(18,58,111,.9);
  box-shadow:
    0 0 20px rgba(34,211,238,.25),
    inset 0 1px 0 rgba(255,255,255,.10);
}
.hd-brand-dot {
  position:absolute; width:7px; height:7px; right:-2px; top:-2px;
  border-radius:50%; background:#34d399;
  box-shadow:0 0 10px rgba(52,211,153,.95), 0 0 20px rgba(52,211,153,.55);
  animation:pulse-dot 2s ease-in-out infinite;
}
.hd-brand-name {
  background:linear-gradient(135deg,#f1f5f9 25%,#22d3ee 65%,#34d399 100%);
  -webkit-background-clip:text;
  -webkit-text-fill-color:transparent;
  background-clip:text;
  font:800 .92rem var(--font-display);
  white-space:nowrap;
  letter-spacing:-.015em;
}
.hd-brand-tag {
  display:block;
  margin-top:.06rem;
  color:#7ba3d6;
  font:700 .54rem var(--font-mono);
  letter-spacing:.14em;
  text-transform:uppercase;
}
.hd-divider {
  width:1px; height:30px;
  background:linear-gradient(180deg,transparent,rgba(34,211,238,.45),transparent);
  flex:0 0 auto;
}
.hd-ticker { min-width:0; flex:1 1 auto; overflow:hidden; position:relative; z-index:1; }
.hd-ticker-inner {
  display:flex; align-items:center; gap:.75rem; overflow:hidden;
  white-space:nowrap; color:#a8bcd8;
  font:.62rem var(--font-mono);
}
.hd-tick { flex:0 0 auto; }
.hd-tick b { color:#e2e8f0; }
/* FIX: Added missing idle style */
.hd-tick-idle { color:#7ba3d6; font-style:italic; }
.hd-live-badge {
  color:#34d399; font-weight:800; font-size:.56rem;
  letter-spacing:.12em;
  padding:.18rem .5rem;
  border:1px solid rgba(52,211,153,.42);
  border-radius:99px;
  background:rgba(16,185,129,.10);
  box-shadow:0 0 12px rgba(52,211,153,.28);
  display:inline-flex; align-items:center; gap:.3rem;
}
.hd-live-badge::before {
  content:''; width:5px; height:5px; border-radius:50%;
  background:#34d399;
  box-shadow:0 0 8px #34d399;
  animation:pulse-dot 1.6s ease-in-out infinite;
}
.hd-status {
  flex:0 0 auto; display:flex; align-items:center; gap:.4rem;
  font:700 .62rem 'Inter',sans-serif;
  padding:.35rem .75rem;
  border-radius:99px;
}
.hd-status-ok {
  color:#34d399;
  background:rgba(16,185,129,.12);
  border:1px solid rgba(16,185,129,.38);
  box-shadow:0 0 16px rgba(16,185,129,.18);
}
.hd-status-err {
  color:#f87171;
  background:rgba(239,68,68,.12);
  border:1px solid rgba(239,68,68,.38);
  box-shadow:0 0 16px rgba(239,68,68,.18);
}
.hd-status-dot {
  width:6px; height:6px; border-radius:50%;
  background:currentColor;
  box-shadow:0 0 8px currentColor;
  animation:pulse-dot 2s ease-in-out infinite;
}
.hd-page {
  flex:0 0 auto;
  color:#e2e8f0;
  font:700 .68rem 'Inter',sans-serif;
  padding:.35rem .8rem;
  border-radius:99px;
  background:linear-gradient(135deg,rgba(37,99,235,.24),rgba(34,211,238,.16));
  border:1px solid rgba(34,211,238,.38);
  box-shadow:0 0 18px rgba(34,211,238,.18);
}
.hd-page-dot {
  display:inline-block; width:5px; height:5px; margin-right:6px;
  border-radius:50%;
  background:#22d3ee;
  box-shadow:0 0 8px rgba(34,211,238,.95);
}

/* ── HERO ── */
.hero { padding:1.5rem 0 1rem; }
.hero-eyebrow {
  display:inline-flex; align-items:center; gap:.45rem;
  padding:.32rem .75rem; border-radius:999px;
  background:linear-gradient(135deg,rgba(37,99,235,.14),rgba(16,185,129,.10));
  border:1px solid rgba(34,211,238,.35);
  color:#22d3ee; font:600 .68rem Inter,sans-serif;
  letter-spacing:.02em; margin-bottom:.9rem;
}
.hero h1 {
  font:800 2.6rem/1.05 var(--font-display);
  letter-spacing:-.03em; margin:0 0 .6rem;
  background:linear-gradient(135deg,#f1f5f9 20%,#22d3ee 60%,#34d399 100%);
  -webkit-background-clip:text; -webkit-text-fill-color:transparent;
  background-clip:text;
}
.hero-sub { color:#a8bcd8; font:400 .95rem/1.55 Inter,sans-serif; max-width:68ch; margin:0; }
.hero-divider {
  height:1px; max-width:480px; margin:1.2rem 0 0;
  background:linear-gradient(90deg,rgba(34,211,238,.55),rgba(16,185,129,.35),transparent);
}

/* ── METRIC CARDS ── */
.metric-card {
  background:linear-gradient(135deg,rgba(24,75,138,.72),rgba(13,37,81,.72));
  border:1px solid var(--line);
  border-radius:12px; padding:1rem; height:100%;
  transition:border-color .18s,transform .18s,box-shadow .18s;
  position:relative;
  overflow:hidden;
}
.metric-card::before {
  content:'';
  position:absolute; top:-30px; right:-30px;
  width:100px; height:100px;
  border-radius:50%;
  background:radial-gradient(circle,rgba(34,211,238,.14),transparent 70%);
  pointer-events:none;
}
.metric-card:hover {
  border-color:rgba(34,211,238,.48);
  transform:translateY(-2px);
  box-shadow:0 8px 24px rgba(34,211,238,.16);
}
.metric-icon-wrap {
  width:34px; height:34px; display:grid; place-items:center;
  border-radius:9px;
  background:linear-gradient(135deg,rgba(37,99,235,.22),rgba(16,185,129,.14));
  border:1px solid rgba(34,211,238,.30);
  margin-bottom:.7rem;
}
.metric-value {
  font:800 1.55rem/1 var(--font-display);
  color:#e2e8f0; letter-spacing:-.02em;
}
.metric-label {
  margin-top:.3rem; color:#7ba3d6;
  font:600 .68rem Inter,sans-serif;
  text-transform:uppercase; letter-spacing:.08em;
}

/* ── INFO CARDS ── */
.info-card {
  background:linear-gradient(135deg,rgba(24,75,138,.72),rgba(13,37,81,.72));
  border:1px solid var(--line);
  border-radius:12px; padding:1.1rem; height:100%;
}
.info-card-icon {
  width:42px; height:42px; display:grid; place-items:center;
  border-radius:11px;
  background:linear-gradient(135deg,rgba(37,99,235,.22),rgba(16,185,129,.14));
  border:1px solid rgba(34,211,238,.30);
  margin-bottom:.85rem;
}
.info-card h3 { color:#e2e8f0; font:700 .95rem var(--font-display); margin:0 0 .5rem; }
.info-card p  { color:#a8bcd8; font:400 .78rem/1.55 Inter,sans-serif; margin:.25rem 0; }
.info-card code {
  font-family:var(--font-mono); font-size:.72rem; color:#22d3ee;
  background:rgba(37,99,235,.16); padding:.08rem .3rem; border-radius:4px;
}

/* ── STEP CARDS ── */
.step-card {
  text-align:center; padding:.9rem .5rem; border-radius:10px;
  background:linear-gradient(180deg,rgba(24,75,138,.55),rgba(13,37,81,.55));
  border:1px solid rgba(59,130,246,.16);
  height:100%;
  transition:border-color .18s,transform .18s;
}
.step-card:hover {
  border-color:rgba(34,211,238,.38);
  transform:translateY(-2px);
}
.step-num {
  width:30px; height:30px; margin:0 auto .55rem; display:grid; place-items:center;
  border-radius:50%;
  background:linear-gradient(135deg,#0ea5e9,#22d3ee,#10b981);
  color:#fff; font:800 .8rem var(--font-display);
  box-shadow:0 0 14px rgba(34,211,238,.45);
}
.step-title { color:#e2e8f0; font:700 .78rem Inter,sans-serif; margin-bottom:.22rem; }
.step-desc  { color:#7ba3d6; font:500 .64rem Inter,sans-serif; }

/* ── UPLOAD HERO ── */
.upload-hero {
  padding:1.6rem 1rem 1.1rem; text-align:center; border-radius:14px;
  background:linear-gradient(180deg,rgba(37,99,235,.10),rgba(16,185,129,.04),transparent);
  border:1px solid rgba(34,211,238,.28);
  margin-bottom:1rem;
  position:relative;
  overflow:hidden;
}
.upload-hero::before {
  content:'';
  position:absolute; top:-60px; left:50%;
  transform:translateX(-50%);
  width:240px; height:240px;
  border-radius:50%;
  background:radial-gradient(circle,rgba(34,211,238,.16),transparent 70%);
  pointer-events:none;
}
.upload-icon-wrap {
  width:60px; height:60px; margin:0 auto .85rem; display:grid; place-items:center;
  border-radius:15px;
  background:linear-gradient(135deg,rgba(37,99,235,.26),rgba(16,185,129,.18));
  border:1px solid rgba(34,211,238,.42);
  box-shadow:0 0 30px rgba(34,211,238,.28);
  position:relative;
}
.upload-title { color:#e2e8f0; font:700 1.05rem var(--font-display); position:relative; }
.upload-sub   { color:#a8bcd8; font:400 .78rem Inter,sans-serif; margin:.3rem 0 .85rem; position:relative; }
.upload-formats { display:flex; gap:.4rem; justify-content:center; flex-wrap:wrap; position:relative; }
.fmt-badge {
  padding:.2rem .55rem; border-radius:6px;
  background:linear-gradient(135deg,rgba(37,99,235,.16),rgba(16,185,129,.10));
  border:1px solid rgba(34,211,238,.32);
  color:#22d3ee; font:700 .6rem var(--font-mono); letter-spacing:.06em;
}
.fmt-badge-muted {
  background:rgba(148,163,184,.10); border-color:rgba(148,163,184,.24); color:#7ba3d6;
}
.upload-note {
  margin-top:.95rem; display:inline-flex; align-items:center; gap:.4rem;
  color:#7ba3d6; font:500 .68rem Inter,sans-serif;
  position:relative;
}
.upload-note-dot {
  width:6px; height:6px; border-radius:50%; background:#34d399;
  box-shadow:0 0 10px rgba(52,211,153,.8);
}

/* ── DIAGNOSTIC PANEL ── */
.diagnostic-panel {
  padding:1.2rem 1.3rem; border-radius:13px; margin-bottom:.85rem;
  background:linear-gradient(135deg,rgba(37,99,235,.14),rgba(16,185,129,.08),rgba(24,75,138,.85));
  border:1px solid rgba(34,211,238,.38);
  position:relative;
  overflow:hidden;
}
.diagnostic-panel::before {
  content:'';
  position:absolute; top:-40px; right:-40px;
  width:150px; height:150px;
  border-radius:50%;
  background:radial-gradient(circle,rgba(34,211,238,.20),transparent 70%);
  pointer-events:none;
}
.diag-header {
  display:flex; justify-content:space-between; align-items:center;
  margin-bottom:.7rem; gap:.5rem; flex-wrap:wrap;
  position:relative;
}
.diag-label {
  font:700 .62rem Inter,sans-serif;
  background:linear-gradient(90deg,#22d3ee,#34d399);
  -webkit-background-clip:text;
  -webkit-text-fill-color:transparent;
  background-clip:text;
  text-transform:uppercase; letter-spacing:.14em;
}
.badge {
  display:inline-flex; align-items:center; gap:.3rem;
  padding:.24rem .6rem; border-radius:999px;
  font:700 .6rem Inter,sans-serif; letter-spacing:.04em;
}
.badge-research { background:rgba(139,92,246,.14); border:1px solid rgba(139,92,246,.35); color:#a78bfa; }
.badge-high     { background:rgba(16,185,129,.16);  border:1px solid rgba(16,185,129,.40); color:#34d399; }
.badge-moderate { background:rgba(245,158,11,.16);  border:1px solid rgba(245,158,11,.40); color:#fbbf24; }
.badge-low      { background:rgba(239,68,68,.16);   border:1px solid rgba(239,68,68,.40);  color:#f87171; }
.diag-prediction {
  font:800 2rem/1.05 var(--font-display);
  letter-spacing:-.03em;
  background:linear-gradient(135deg,#f1f5f9,#22d3ee);
  -webkit-background-clip:text;
  -webkit-text-fill-color:transparent;
  background-clip:text;
  position:relative;
}
.diag-confidence { margin-top:.25rem; color:#a8bcd8; font:500 .82rem Inter,sans-serif; position:relative; }

/* ── UNCERTAINTY / XAI CARDS ── */
.uncertainty-card {
  margin:.85rem 0; padding:1.05rem 1.2rem; border-radius:12px;
  background:linear-gradient(135deg,rgba(139,92,246,.12),rgba(34,211,238,.06));
  border:1px solid rgba(139,92,246,.32);
}
.unc-title {
  font:700 .6rem Inter,sans-serif;
  background:linear-gradient(90deg,#a78bfa,#22d3ee);
  -webkit-background-clip:text;
  -webkit-text-fill-color:transparent;
  background-clip:text;
  text-transform:uppercase; letter-spacing:.14em; margin-bottom:.7rem;
}
.unc-value { font:800 1.4rem/1 var(--font-mono); letter-spacing:-.01em; }
.unc-band  { margin-top:.22rem; font:600 .74rem Inter,sans-serif; }

.xai-card {
  margin:.75rem 0; padding:.95rem 1.2rem; border-radius:12px;
  background:linear-gradient(135deg,rgba(37,99,235,.10),rgba(16,185,129,.06));
  border:1px solid rgba(34,211,238,.28);
}
.xai-title {
  display:flex; align-items:center; gap:.4rem;
  font:700 .6rem Inter,sans-serif;
  background:linear-gradient(90deg,#22d3ee,#34d399);
  -webkit-background-clip:text;
  -webkit-text-fill-color:transparent;
  background-clip:text;
  text-transform:uppercase; letter-spacing:.12em; margin-bottom:.65rem;
}
.xai-text { color:#a8bcd8; font:400 .78rem/1.6 Inter,sans-serif; }
.xai-text b  { color:#e2e8f0; font-weight:700; }
.xai-text em { color:#e2e8f0; font-style:italic; }

.disclaimer {
  display:flex; align-items:flex-start; gap:.6rem;
  margin:.85rem 0; padding:.8rem 1rem; border-radius:10px;
  background:linear-gradient(135deg,rgba(245,158,11,.10),rgba(245,158,11,.03));
  border:1px solid rgba(245,158,11,.28);
  color:#a8bcd8; font:400 .76rem/1.5 Inter,sans-serif;
}
.disclaimer b { color:#fbbf24; }

.prob-grid {
  display:grid; grid-template-columns:repeat(auto-fit,minmax(140px,1fr));
  gap:.65rem; margin:.4rem 0 .2rem;
}
.prob-card {
  position:relative; padding:.9rem 1rem; border-radius:11px;
  background:linear-gradient(135deg,rgba(24,75,138,.72),rgba(13,37,81,.72));
  border:1px solid var(--line);
  transition:border-color .18s,transform .18s,box-shadow .18s;
}
.prob-card:hover {
  border-color:rgba(34,211,238,.42);
  transform:translateY(-2px);
}
.prob-card.is-top {
  border-color:rgba(34,211,238,.58);
  background:linear-gradient(135deg,rgba(37,99,235,.22),rgba(16,185,129,.14),rgba(24,75,138,.85));
  box-shadow:0 0 28px rgba(34,211,238,.22);
}
.prob-value {
  font:800 1.4rem/1 var(--font-mono); color:#e2e8f0; letter-spacing:-.02em;
}
.prob-card.is-top .prob-value {
  background:linear-gradient(135deg,#22d3ee,#34d399);
  -webkit-background-clip:text;
  -webkit-text-fill-color:transparent;
  background-clip:text;
}
.prob-label { margin-top:.32rem; color:#a8bcd8; font:600 .72rem Inter,sans-serif; }
.prob-top-tag {
  position:absolute; top:.5rem; right:.5rem;
  padding:.14rem .45rem; border-radius:999px;
  background:linear-gradient(135deg,rgba(34,211,238,.28),rgba(16,185,129,.20));
  border:1px solid rgba(34,211,238,.50);
  color:#22d3ee; font:700 .52rem Inter,sans-serif;
  letter-spacing:.06em; text-transform:uppercase;
}

.empty-state {
  padding:3rem 1rem; text-align:center; border-radius:14px;
  border:1px dashed rgba(59,130,246,.30);
  background:linear-gradient(135deg,rgba(24,75,138,.35),rgba(13,37,81,.35));
}
.empty-icon {
  width:68px; height:68px; margin:0 auto .95rem; display:grid; place-items:center;
  border-radius:16px;
  background:linear-gradient(135deg,rgba(37,99,235,.18),rgba(16,185,129,.10));
  border:1px solid rgba(59,130,246,.30);
}
.empty-title { color:#cbd5e1; font:700 .98rem var(--font-display); }
.empty-text  { margin-top:.4rem; color:#7ba3d6; font:400 .8rem Inter,sans-serif; }

.dist-card {
  margin:.45rem 0; padding:.7rem .9rem; border-radius:10px;
  background:linear-gradient(135deg,rgba(24,75,138,.6),rgba(13,37,81,.6));
  border:1px solid rgba(59,130,246,.16);
}
.dist-row {
  display:flex; justify-content:space-between; align-items:center; margin-bottom:.48rem;
}
.dist-name  { color:#cbd5e1; font:600 .8rem Inter,sans-serif; }
.dist-count { color:#a8bcd8; font:600 .72rem var(--font-mono); }
.dist-track { height:6px; border-radius:999px; background:rgba(59,130,246,.16); overflow:hidden; }
.dist-fill {
  height:100%; border-radius:999px;
  background:linear-gradient(90deg,#22d3ee,#34d399);
  box-shadow:0 0 12px rgba(34,211,238,.5);
  transition:width 1s cubic-bezier(.2,.8,.2,1);
}

.latest-card {
  padding:.75rem .9rem; border-radius:10px;
  background:linear-gradient(135deg,rgba(24,75,138,.55),rgba(13,37,81,.55));
  border:1px solid rgba(59,130,246,.18);
}
.latest-row {
  display:flex; justify-content:space-between; gap:.6rem;
  padding:.45rem 0; border-bottom:1px solid rgba(59,130,246,.14);
}
.latest-row:last-child { border-bottom:0; }
.latest-key { color:#7ba3d6; font:600 .7rem Inter,sans-serif; }
.latest-val {
  color:#cbd5e1; font:600 .72rem var(--font-mono);
  text-align:right; word-break:break-word;
}

.thumb-card {
  display:flex; align-items:center; gap:.9rem;
  padding:.8rem 1rem; margin:.45rem 0; border-radius:11px;
  background:linear-gradient(135deg,rgba(24,75,138,.6),rgba(13,37,81,.6));
  border:1px solid rgba(59,130,246,.16);
  transition:border-color .18s,transform .18s,box-shadow .18s;
}
.thumb-card:hover {
  border-color:rgba(34,211,238,.40);
  transform:translateX(3px);
  box-shadow:0 4px 20px rgba(34,211,238,.14);
}
.thumb-icon {
  width:38px; height:38px; display:grid; place-items:center;
  border-radius:10px;
  background:linear-gradient(135deg,rgba(37,99,235,.18),rgba(16,185,129,.12));
  border:1px solid rgba(34,211,238,.30);
  flex:0 0 auto;
}
.thumb-info  { flex:1; min-width:0; }
.thumb-title { color:#e2e8f0; font:700 .86rem Inter,sans-serif; }
.thumb-meta {
  margin-top:.18rem; color:#7ba3d6; font:500 .7rem var(--font-mono);
  overflow:hidden; text-overflow:ellipsis; white-space:nowrap;
}
.conf-badge {
  display:inline-block; padding:.3rem .7rem; border-radius:999px;
  font:700 .74rem var(--font-mono); letter-spacing:.01em;
}
.conf-hi  { background:rgba(16,185,129,.18); border:1px solid rgba(16,185,129,.42); color:#34d399; }
.conf-mid { background:rgba(245,158,11,.18); border:1px solid rgba(245,158,11,.42); color:#fbbf24; }
.conf-lo  { background:rgba(239,68,68,.18);  border:1px solid rgba(239,68,68,.42);  color:#f87171; }

.act-feed {
  padding:.6rem .7rem; border-radius:10px; max-height:360px; overflow-y:auto;
  background:linear-gradient(135deg,rgba(24,75,138,.55),rgba(13,37,81,.55));
  border:1px solid rgba(59,130,246,.16);
}
.act-row {
  display:flex; gap:.55rem; padding:.4rem .25rem;
  border-bottom:1px solid rgba(59,130,246,.11);
  font:500 .72rem Inter,sans-serif;
}
.act-row:last-child { border-bottom:0; }
.act-time { color:#7ba3d6; font:600 .66rem var(--font-mono); flex:0 0 auto; }
.act-msg  { color:#a8bcd8; word-break:break-word; }
.act-success .act-msg { color:#34d399; }
.act-warn    .act-msg { color:#fbbf24; }
.act-error   .act-msg { color:#f87171; }

.export-head {
  display:flex; align-items:center; gap:.75rem;
  margin:1.1rem 0 .7rem; padding:.75rem 1rem; border-radius:11px;
  background:linear-gradient(135deg,rgba(24,75,138,.72),rgba(13,37,81,.72));
  border:1px solid rgba(59,130,246,.22);
}
.export-head-icon {
  width:36px; height:36px; display:grid; place-items:center;
  border-radius:9px;
  background:linear-gradient(135deg,rgba(37,99,235,.18),rgba(16,185,129,.12));
  border:1px solid rgba(34,211,238,.30);
}
.export-head-title { color:#e2e8f0; font:700 .86rem var(--font-display); }
.export-head-sub   { color:#7ba3d6; font:500 .7rem Inter,sans-serif; }
.export-head-badge {
  margin-left:auto; padding:.2rem .6rem; border-radius:999px;
  background:rgba(16,185,129,.16); border:1px solid rgba(16,185,129,.38);
  color:#34d399; font:700 .6rem Inter,sans-serif; letter-spacing:.06em;
}
.export-item-label {
  color:#7ba3d6; font:700 .62rem Inter,sans-serif;
  text-transform:uppercase; letter-spacing:.1em; margin:.25rem 0 .4rem;
}

/* ═══════════════════════════════════════════════════════════════════════
   FOOTER — PREMIUM BLUE UPGRADE
   ═══════════════════════════════════════════════════════════════════════ */
.app-footer {
  display:grid;
  grid-template-columns:1.6fr 1.1fr 0.95fr;
  gap:1.85rem;
  margin-top:2.75rem;
  padding:2rem;
  border-radius:18px;
  position:relative;
  background:
    linear-gradient(180deg,rgba(18,58,111,.85),rgba(6,23,51,.96)),
    radial-gradient(circle at 0% 0%, rgba(34,211,238,.10), transparent 25rem);
  border:1px solid var(--line-strong);
  overflow:hidden;
  box-shadow:0 20px 50px rgba(0,0,0,.30);
}
.app-footer::before {
  content:'';
  position:absolute; inset:0;
  border-radius:18px;
  padding:1px;
  background:linear-gradient(135deg,
    rgba(37,99,235,.65),
    rgba(34,211,238,.55),
    rgba(16,185,129,.65));
  -webkit-mask:linear-gradient(#000 0 0) content-box,linear-gradient(#000 0 0);
  -webkit-mask-composite:xor;
  mask-composite:exclude;
  pointer-events:none;
  opacity:.85;
}
.app-footer::after {
  content:'';
  position:absolute;
  top:-120px; right:-120px;
  width:280px; height:280px;
  border-radius:50%;
  background:radial-gradient(circle,rgba(34,211,238,.20),transparent 70%);
  pointer-events:none;
}
.footer-brand-lockup { display:flex; align-items:center; gap:.75rem; margin-bottom:.9rem; position:relative; z-index:1; }
.footer-brand-icon {
  width:52px; height:52px;
  display:grid; place-items:center;
  border-radius:14px;
  background:linear-gradient(135deg,rgba(37,99,235,.35),rgba(16,185,129,.24));
  border:1px solid rgba(34,211,238,.42);
  box-shadow:0 0 26px rgba(34,211,238,.28);
  position:relative;
}
.footer-brand-icon::after {
  content:'';
  position:absolute; inset:-1px;
  border-radius:14px;
  padding:1px;
  background:linear-gradient(135deg,#22d3ee,#34d399);
  -webkit-mask:linear-gradient(#000 0 0) content-box,linear-gradient(#000 0 0);
  -webkit-mask-composite:xor;
  mask-composite:exclude;
  opacity:.65;
  pointer-events:none;
}
.footer-brand-name {
  background:linear-gradient(135deg,#f1f5f9 20%,#22d3ee 70%,#34d399 100%);
  -webkit-background-clip:text;
  -webkit-text-fill-color:transparent;
  background-clip:text;
  font:800 1.15rem var(--font-display);
  letter-spacing:-.02em;
}
.footer-brand-tag {
  color:#7ba3d6;
  font:600 .68rem 'Inter',sans-serif;
  letter-spacing:.06em;
  text-transform:uppercase;
  margin-top:.1rem;
}
.footer-desc {
  color:#a8bcd8; font:400 .78rem/1.65 Inter,sans-serif;
  max-width:46ch; margin-bottom:1rem;
  position:relative; z-index:1;
}
.footer-copy { color:#7ba3d6; font:600 .68rem Inter,sans-serif; position:relative; z-index:1; }
.footer-col-title {
  background:linear-gradient(90deg,#22d3ee,#34d399);
  -webkit-background-clip:text;
  -webkit-text-fill-color:transparent;
  background-clip:text;
  font:800 .62rem 'Inter',sans-serif;
  text-transform:uppercase;
  letter-spacing:.15em;
  margin-bottom:.7rem;
  position:relative; z-index:1;
}
.footer-tech-badges { display:flex; flex-wrap:wrap; gap:.4rem; position:relative; z-index:1; }
.ft-badge {
  display:inline-flex; align-items:center; gap:.35rem;
  padding:.35rem .72rem;
  border-radius:8px;
  background:linear-gradient(135deg,rgba(37,99,235,.18),rgba(16,185,129,.10));
  border:1px solid rgba(34,211,238,.30);
  color:#e2e8f0;
  font:600 .68rem 'JetBrains Mono',monospace;
  transition:all .18s ease;
}
.ft-badge:hover {
  border-color:rgba(34,211,238,.55);
  box-shadow:0 0 16px rgba(34,211,238,.28);
  transform:translateY(-1px);
}
.ft-badge-dot {
  width:5px; height:5px; border-radius:50%;
  background:linear-gradient(135deg,#22d3ee,#34d399);
  box-shadow:0 0 8px rgba(34,211,238,.9);
}
.footer-stats { display:flex; flex-direction:column; gap:.6rem; position:relative; z-index:1; }
.f-stat {
  padding:.75rem .9rem;
  border-radius:10px;
  background:linear-gradient(135deg,rgba(24,75,138,.75),rgba(13,37,81,.55));
  border:1px solid var(--line);
  transition:all .18s ease;
}
.f-stat:hover {
  border-color:var(--line-strong);
  box-shadow:0 0 18px rgba(34,211,238,.14);
}
/* FIX: Added missing f-stat-wide modifier */
.f-stat-wide { padding:.85rem .9rem; }
.f-stat-val {
  display:flex; align-items:center; gap:.5rem;
  background:linear-gradient(135deg,#f1f5f9,#22d3ee);
  -webkit-background-clip:text;
  -webkit-text-fill-color:transparent;
  background-clip:text;
  font:800 1.25rem/1 var(--font-display);
  letter-spacing:-.02em;
}
/* FIX: Ensure .f-dot inside f-stat-val is NOT gradient-filled */
.f-stat-val .f-dot,
.f-stat-val .f-dot-off {
  -webkit-text-fill-color:currentColor;
  background:currentColor;
  -webkit-background-clip:initial;
  background-clip:initial;
}
.f-stat-lbl {
  margin-top:.35rem;
  color:#7ba3d6;
  font:700 .62rem 'Inter',sans-serif;
  text-transform:uppercase;
  letter-spacing:.1em;
}
.f-dot {
  width:9px; height:9px; border-radius:50%;
  background:#34d399;
  box-shadow:0 0 10px rgba(52,211,153,.9);
  animation:pulse-dot 2s ease-in-out infinite;
}
.f-dot-off {
  background:#f87171;
  box-shadow:0 0 10px rgba(248,113,113,.9);
}

.copyright-line {
  margin-top:1.1rem;
  padding:1.1rem;
  text-align:center;
  font:500 .74rem 'Inter',sans-serif;
  border-top:1px solid var(--line);
  background:linear-gradient(180deg,transparent,rgba(34,211,238,.03));
}

/* ═══════════════════════════════════════════════════════════════════════
   SIDEBAR NAV — SVG ICON INJECTION
   ═══════════════════════════════════════════════════════════════════════ */
[data-testid="stSidebar"] .stButton > button::before {
  content: '' !important;
  position: absolute !important;
  left: 1rem !important;
  top: 50% !important;
  transform: translateY(-50%) !important;
  width: 18px !important;
  height: 18px !important;
  background-repeat: no-repeat !important;
  background-position: center !important;
  background-size: 18px 18px !important;
  opacity: .85 !important;
  transition: opacity .18s ease, transform .18s ease, filter .18s ease !important;
  pointer-events: none !important;
}
[data-testid="stSidebar"] .stButton > button:hover::before {
  opacity: 1 !important;
  transform: translateY(-50%) scale(1.06) !important;
}

/* ── HOME ── */
[data-testid="stSidebar"] .st-key-nav_home button::before {
  background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%23a8bcd8' stroke-width='1.75' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='m3 9 9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z'/%3E%3Cpolyline points='9 22 9 12 15 12 15 22'/%3E%3C/svg%3E") !important;
}
[data-testid="stSidebar"] .st-key-nav_home button:hover::before {
  background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%23f1f5f9' stroke-width='1.9' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='m3 9 9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z'/%3E%3Cpolyline points='9 22 9 12 15 12 15 22'/%3E%3C/svg%3E") !important;
}

/* ── MRI ANALYSIS ── */
[data-testid="stSidebar"] .st-key-nav_mri_analysis button::before {
  background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%23a8bcd8' stroke-width='1.75' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M6 18h8'/%3E%3Cpath d='M3 22h18'/%3E%3Cpath d='M14 22a7 7 0 1 0 0-14h-1'/%3E%3Cpath d='M9 14h2'/%3E%3Cpath d='M9 12a2 2 0 0 1-2-2V6h6v4a2 2 0 0 1-2 2Z'/%3E%3Cpath d='M12 6V3a1 1 0 0 0-1-1H9a1 1 0 0 0-1 1v3'/%3E%3C/svg%3E") !important;
}
[data-testid="stSidebar"] .st-key-nav_mri_analysis button:hover::before {
  background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%23f1f5f9' stroke-width='1.9' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M6 18h8'/%3E%3Cpath d='M3 22h18'/%3E%3Cpath d='M14 22a7 7 0 1 0 0-14h-1'/%3E%3Cpath d='M9 14h2'/%3E%3Cpath d='M9 12a2 2 0 0 1-2-2V6h6v4a2 2 0 0 1-2 2Z'/%3E%3Cpath d='M12 6V3a1 1 0 0 0-1-1H9a1 1 0 0 0-1 1v3'/%3E%3C/svg%3E") !important;
}

/* ── DASHBOARD ── */
[data-testid="stSidebar"] .st-key-nav_dashboard button::before {
  background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%23a8bcd8' stroke-width='1.75' stroke-linecap='round' stroke-linejoin='round'%3E%3Cline x1='18' y1='20' x2='18' y2='10'/%3E%3Cline x1='12' y1='20' x2='12' y2='4'/%3E%3Cline x1='6' y1='20' x2='6' y2='14'/%3E%3Cline x1='2' y1='20' x2='22' y2='20'/%3E%3C/svg%3E") !important;
}
[data-testid="stSidebar"] .st-key-nav_dashboard button:hover::before {
  background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%23f1f5f9' stroke-width='1.9' stroke-linecap='round' stroke-linejoin='round'%3E%3Cline x1='18' y1='20' x2='18' y2='10'/%3E%3Cline x1='12' y1='20' x2='12' y2='4'/%3E%3Cline x1='6' y1='20' x2='6' y2='14'/%3E%3Cline x1='2' y1='20' x2='22' y2='20'/%3E%3C/svg%3E") !important;
}

/* ── HISTORY ── */
[data-testid="stSidebar"] .st-key-nav_history button::before {
  background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%23a8bcd8' stroke-width='1.75' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8'/%3E%3Cpath d='M3 3v5h5'/%3E%3Cpath d='M12 7v5l4 2'/%3E%3C/svg%3E") !important;
}
[data-testid="stSidebar"] .st-key-nav_history button:hover::before {
  background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%23f1f5f9' stroke-width='1.9' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8'/%3E%3Cpath d='M3 3v5h5'/%3E%3Cpath d='M12 7v5l4 2'/%3E%3C/svg%3E") !important;
}

/* ── GRAD-CAM ── */
[data-testid="stSidebar"] .st-key-nav_gradcam button::before {
  background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%23a8bcd8' stroke-width='1.75' stroke-linecap='round' stroke-linejoin='round'%3E%3Ccircle cx='12' cy='12' r='3'/%3E%3Cpath d='M12 2v3'/%3E%3Cpath d='M12 19v3'/%3E%3Cpath d='m4.22 4.22 2.12 2.12'/%3E%3Cpath d='m17.66 17.66 2.12 2.12'/%3E%3Cpath d='M2 12h3'/%3E%3Cpath d='M19 12h3'/%3E%3Cpath d='m4.22 19.78 2.12-2.12'/%3E%3Cpath d='m17.66 6.34 2.12-2.12'/%3E%3C/svg%3E") !important;
}
[data-testid="stSidebar"] .st-key-nav_gradcam button:hover::before {
  background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%23f1f5f9' stroke-width='1.9' stroke-linecap='round' stroke-linejoin='round'%3E%3Ccircle cx='12' cy='12' r='3'/%3E%3Cpath d='M12 2v3'/%3E%3Cpath d='M12 19v3'/%3E%3Cpath d='m4.22 4.22 2.12 2.12'/%3E%3Cpath d='m17.66 17.66 2.12 2.12'/%3E%3Cpath d='M2 12h3'/%3E%3Cpath d='M19 12h3'/%3E%3Cpath d='m4.22 19.78 2.12-2.12'/%3E%3Cpath d='m17.66 6.34 2.12-2.12'/%3E%3C/svg%3E") !important;
}

/* ── XAI LAB ── */
[data-testid="stSidebar"] .st-key-nav_xai_lab button::before {
  background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%23a8bcd8' stroke-width='1.75' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M14.5 2v17.5c0 1.4-1.1 2.5-2.5 2.5h0c-1.4 0-2.5-1.1-2.5-2.5V2'/%3E%3Cpath d='M8.5 2h7'/%3E%3Cpath d='M14.5 16h-5'/%3E%3Cpath d='m8.5 13 5.5 3'/%3E%3C/svg%3E") !important;
}
[data-testid="stSidebar"] .st-key-nav_xai_lab button:hover::before {
  background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%23f1f5f9' stroke-width='1.9' stroke-linecap='round' stroke-linejoin='round'%3E%3Cpath d='M14.5 2v17.5c0 1.4-1.1 2.5-2.5 2.5h0c-1.4 0-2.5-1.1-2.5-2.5V2'/%3E%3Cpath d='M8.5 2h7'/%3E%3Cpath d='M14.5 16h-5'/%3E%3Cpath d='m8.5 13 5.5 3'/%3E%3C/svg%3E") !important;
}

/* ── SETTINGS ── */
[data-testid="stSidebar"] .st-key-nav_settings button::before {
  background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%23a8bcd8' stroke-width='1.75' stroke-linecap='round' stroke-linejoin='round'%3E%3Ccircle cx='12' cy='12' r='3'/%3E%3Cpath d='M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 1 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 1 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 1 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 1 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z'/%3E%3C/svg%3E") !important;
}
[data-testid="stSidebar"] .st-key-nav_settings button:hover::before {
  background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 24 24' fill='none' stroke='%23f1f5f9' stroke-width='1.9' stroke-linecap='round' stroke-linejoin='round'%3E%3Ccircle cx='12' cy='12' r='3'/%3E%3Cpath d='M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 1 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 1 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 1 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 1 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z'/%3E%3C/svg%3E") !important;
}

/* Active nav — glow icon in cyan */
[data-testid="stSidebar"] [class*="st-key-nav_"] button[data-active="true"]::before {
  filter: drop-shadow(0 0 6px rgba(34,211,238,.65));
}

@media (max-width: 1100px) {
  :root { --sb-w:240px; }
  [data-testid="stSidebar"] { min-width:240px !important; max-width:240px !important; width:240px !important; }
  .sticky-header { left:calc(var(--sb-w) + .6rem); }
  .hd-ticker { display:none; }
  .app-footer { grid-template-columns:1fr 1fr; }
}
@media (max-width: 900px) {
  .app-footer { grid-template-columns:1fr 1fr; }
}
@media (max-width: 760px) {
  :root { --sb-w:0px; --hd-h:56px; }
  [data-testid="stSidebar"] { min-width:0 !important; max-width:0 !important; width:0 !important; }
  [data-testid="stSidebar"] > div:first-child { padding:0 !important; }
  [data-testid="stMainBlockContainer"] { padding:calc(var(--hd-h) + .75rem) .75rem 2rem !important; }
  .sticky-header { top:.35rem; left:.35rem; right:.35rem; min-height:var(--hd-h); }
  .hd-brand-name { display:none; }
  .hd-divider, .hd-status { display:none; }
  .hd-page { margin-left:auto; }
  .sb-engine-grid { grid-template-columns:1fr 1fr; }
}
@media (max-width: 600px) {
  .app-footer { grid-template-columns:1fr; padding:1.4rem; }
  .hero h1 { font-size:2rem; }
  .diag-prediction { font-size:1.55rem; }
  .prob-grid { grid-template-columns:repeat(2,1fr); }
}
@media (max-width: 560px) {
  .stButton > button, .stDownloadButton > button { min-height:44px !important; }
  [data-testid="stMetric"] { padding:.65rem !important; }
  .stTabs [data-baseweb="tab"] { padding:0 .55rem !important; font-size:.67rem !important; }
}
"""
st.markdown(f"<style>{UI_CSS}</style>", unsafe_allow_html=True)


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


# ─────────────────────────────────────────────────────────────────────────────
# MODEL LOADING
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource
def load_model(model_path):
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    try:
        checkpoint = torch.load(model_path, map_location=DEVICE, weights_only=True)
    except Exception:
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
        k[len("module."):] if k.startswith("module.") else k: v
        for k, v in state_dict.items()
    }
    model.load_state_dict(clean_sd, strict=True)
    model.to(DEVICE)
    model.eval()
    return model, CLASS_NAMES, best_model_name


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
        probs = F.softmax(model(tensor), dim=1)[0].detach().cpu().numpy()
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


def mc_dropout_predict(image, model, class_names, n_samples=MC_SAMPLES):
    if model is None:
        return None

    def _enable_dropout(m):
        if isinstance(m, (nn.Dropout, nn.Dropout2d)):
            m.train()

    model.eval()
    mc_preds = []
    try:
        model.apply(_enable_dropout)
        tensor = test_transforms(image).unsqueeze(0).to(DEVICE)
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


def _cam_to_heatmap(cam_raw):
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
    tl  = _get_target_layer(model, model_name)
    fwd = tl.register_forward_hook(lambda m, i, o: activations.append(o.detach()))
    bwd = tl.register_full_backward_hook(lambda m, gi, go: gradients.append(go[0].detach()))
    fig = None
    try:
        tensor = test_transforms(image).unsqueeze(0).to(DEVICE)
        model.zero_grad()
        out = model(tensor)
        out[0, int(out.argmax(dim=1).item())].backward()
        if not activations or not gradients:
            raise RuntimeError("Hooks failed.")

        act = activations[0][0]
        grd = gradients[0][0]
        w   = grd.mean(dim=(1, 2), keepdim=True)
        cam = _cam_to_heatmap((w * act).sum(dim=0).detach().cpu().numpy())

        orig = np.asarray(image).astype(np.float32) / 255.0
        fig, ax = plt.subplots(figsize=(5, 5), dpi=120)
        fig.patch.set_alpha(0)
        ax.imshow(orig)
        ax.imshow(cam, cmap="jet", alpha=0.44,
                  extent=(0, orig.shape[1], orig.shape[0], 0))
        ax.axis("off")
        fig.tight_layout(pad=0)
        return fig
    except Exception:
        if fig:
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
    tl  = _get_target_layer(model, model_name)
    fwd = tl.register_forward_hook(lambda m, i, o: activations.append(o.detach()))
    bwd = tl.register_full_backward_hook(lambda m, gi, go: gradients.append(go[0].detach()))
    fig = None
    try:
        tensor = test_transforms(image).unsqueeze(0).to(DEVICE)
        model.zero_grad()
        out = model(tensor)
        out[0, int(out.argmax(dim=1).item())].backward()
        if not activations or not gradients:
            raise RuntimeError("Hooks failed.")

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

        orig = np.asarray(image).astype(np.float32) / 255.0
        fig, ax = plt.subplots(figsize=(5, 5), dpi=120)
        fig.patch.set_alpha(0)
        ax.imshow(orig)
        ax.imshow(cam, cmap="inferno", alpha=0.46,
                  extent=(0, orig.shape[1], orig.shape[0], 0))
        ax.axis("off")
        fig.tight_layout(pad=0)
        return fig
    except Exception:
        if fig:
            plt.close(fig)
        raise
    finally:
        try: fwd.remove()
        except Exception: pass
        try: bwd.remove()
        except Exception: pass


def explanation_agreement(image, model, model_name):
    if model is None:
        return None
    model.eval()
    handles = []
    try:
        tl = _get_target_layer(model, model_name)
        acts_gc, grds_gc, acts_pp, grds_pp = [], [], [], []

        fwd1 = tl.register_forward_hook(lambda m, i, o: acts_gc.append(o.detach()))
        bwd1 = tl.register_full_backward_hook(lambda m, gi, go: grds_gc.append(go[0].detach()))
        handles.extend((fwd1, bwd1))

        tensor = test_transforms(image).unsqueeze(0).to(DEVICE)
        model.zero_grad()
        out = model(tensor)
        idx = int(out.argmax(dim=1).item())
        out[0, idx].backward()

        try: fwd1.remove()
        except Exception: pass
        try: bwd1.remove()
        except Exception: pass

        if not acts_gc or not grds_gc:
            return None

        act_gc = acts_gc[0][0]
        grd_gc = grds_gc[0][0]
        w_gc   = grd_gc.mean(dim=(1, 2), keepdim=True)
        cam_gc = F.relu((w_gc * act_gc).sum(dim=0)).detach().cpu().numpy()
        cam_gc = cam_gc / (cam_gc.max() + 1e-8)

        fwd2 = tl.register_forward_hook(lambda m, i, o: acts_pp.append(o.detach()))
        bwd2 = tl.register_full_backward_hook(lambda m, gi, go: grds_pp.append(go[0].detach()))
        handles.extend((fwd2, bwd2))

        model.zero_grad()
        out2 = model(tensor)
        out2[0, idx].backward()
        try: fwd2.remove()
        except Exception: pass
        try: bwd2.remove()
        except Exception: pass

        if not acts_pp or not grds_pp:
            return None

        act_pp = acts_pp[0][0].cpu().numpy()
        grd_pp = grds_pp[0][0].cpu().numpy()
        an  = grd_pp ** 2
        ad  = 2 * grd_pp**2 + (act_pp * grd_pp**3).sum(axis=(1, 2), keepdims=True) + 1e-8
        w_pp = (an / ad * np.maximum(grd_pp, 0)).sum(axis=(1, 2))
        cam_pp = np.zeros(act_pp.shape[1:], dtype=np.float32)
        for ww, aa in zip(w_pp, act_pp):
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
    finally:
        for h in handles:
            try: h.remove()
            except Exception: pass


# ─────────────────────────────────────────────────────────────────────────────
# CHARTING
# ─────────────────────────────────────────────────────────────────────────────

def _dark_fig(w=6, h=2.8):
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
    ax.axhline(0.12, color="#f59e0b", lw=1, ls="--", alpha=0.65, label="Moderate threshold", zorder=1)
    ax.axhline(0.22, color="#ef4444", lw=1, ls="--", alpha=0.65, label="Low reliability", zorder=1)
    ax.set_xlabel("Analysis #", color="#a8bcd8", fontsize=9)
    ax.set_ylabel("MC Uncertainty σ", color="#a8bcd8", fontsize=9)
    ax.grid(True, alpha=0.09, ls="--", color="#7ba3d6")
    ax.legend(fontsize=7, labelcolor="#a8bcd8", facecolor="#0f2b57", edgecolor="#26497b")
    fig.tight_layout(pad=0.5)
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────

DEFAULTS = {
    "nav": "Home",
    "last_result": None,
    "last_image": None,
    "gradcam_image": None,
    "gradcam_pp_image": None,
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
    "live_inference_running": False,
}
for k, v in DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = copy.deepcopy(v)


def clear_prediction_history():
    for fk in ("gradcam_image", "gradcam_pp_image"):
        old = st.session_state.get(fk)
        if old:
            plt.close(old)
    for k in [
        "prediction_history", "last_result", "last_image",
        "gradcam_image", "gradcam_pp_image", "mc_result", "agreement_score",
        "live_predictions_count", "live_avg_confidence", "live_last_confidence",
        "live_class_counts", "live_throughput", "live_latency_ms",
    ]:
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


# ─────────────────────────────────────────────────────────────────────────────
# LIVE TICKER
# ─────────────────────────────────────────────────────────────────────────────

def render_live_ticker():
    counts = st.session_state.live_class_counts or {}
    items  = list(counts.items())
    text   = " · ".join(
        f"<b>{_escape_html(n)}</b>: {_escape_html(v)}" for n, v in items
    ) if items else "<b>Awaiting first scan</b>"
    mc  = st.session_state.mc_result
    unc = f" · σ={mc['uncertainty']:.3f} ({_escape_html(mc['band'])})" if mc else ""

    _render_html(f"""
    <style>
    .tw{{overflow:hidden;border-radius:9px;border:1px solid rgba(34,211,238,.28);
    background:linear-gradient(90deg,rgba(24,75,138,.9),rgba(13,37,81,.9));padding:.45rem 0;}}
    .tt{{display:inline-block;white-space:nowrap;padding-left:100%;
    animation:mq 28s linear infinite;color:#a8bcd8;font:500 .72rem 'JetBrains Mono',monospace}}
    .lp{{display:inline-block;width:6px;height:6px;border-radius:50%;background:#22d3ee;
    box-shadow:0 0 8px #22d3ee;margin-right:8px;vertical-align:middle}}
    @keyframes mq{{0%{{transform:translateX(0);}}100%{{transform:translateX(-100%);}}}}
    </style>
    <div class="tw"><div class="tt"><span class="lp"></span>LIVE ·
    {text} · Throughput: {st.session_state.live_throughput:.2f}/min ·
    Last Conf: {st.session_state.live_last_confidence:.1f}% ·
    Avg Conf: {st.session_state.live_avg_confidence:.1f}%{unc}
    </div></div>""", height=36)


# ─────────────────────────────────────────────────────────────────────────────
# STICKY HEADER — UPGRADED PREMIUM
# ─────────────────────────────────────────────────────────────────────────────

def render_sticky_header():
    counts = st.session_state.live_class_counts or {}
    if counts:
        items_html = " ".join(
            f"<span class='hd-tick'><b>{_escape_html(k)}</b>: {_escape_html(v)}</span>"
            for k, v in counts.items()
        )
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

    brain_svg = Icons.brain(20, "#22d3ee")

    st.markdown(safe_html(f"""
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
        band = mc_result["band"]
        uval = mc_result["uncertainty"]
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


# ─────────────────────────────────────────────────────────────────────────────
# ACTIVITY FEED
# ─────────────────────────────────────────────────────────────────────────────

def render_activity_feed():
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

    brain_large = Icons.brain(22, "#22d3ee")
    st.markdown(safe_html(f"""
    <div class="sb-brand">
        <div class="sb-brand-icon">
            {brain_large}
            <span class="sb-brand-pulse"></span>
        </div>
        <div class="sb-brand-text">
            <div class="sb-brand-name">NeuroLens <span class="sb-brand-ai">AI</span><span class="sb-brand-ver">v3.6.2</span></div>
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

    # ── Nav badges ──
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

    agree_vals = [h.get("agreement_score") for h in history if h.get("agreement_score") is not None]
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

    # ── Nav groups ──
    active_css_parts = []
    for group_name, items in NAV_GROUPS:
        st.markdown(f'<div class="sb-nav-group">{_escape_html(group_name)}</div>', unsafe_allow_html=True)
        for label, icon_svg, key in items:
            is_active = active_nav == key
            slug = key.lower().replace("-", "").replace(" ", "_")

            if st.button(label, key=f"nav_{slug}", **_stretch()):
                st.session_state.nav = key
                st.rerun()

            if is_active:
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
                    opacity: 1 !important;
                    filter: drop-shadow(0 0 6px rgba(34,211,238,.7)) !important;
                }}
                """)

    if active_css_parts:
        st.markdown(f"<style>{''.join(active_css_parts)}</style>", unsafe_allow_html=True)

    st.markdown(
        '<div style="height:1px;background:linear-gradient(90deg,transparent,rgba(34,211,238,.35),transparent);margin:0.7rem 0.85rem"></div>',
        unsafe_allow_html=True,
    )

    # ── Neural Engine Panel ──
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
                if old:
                    plt.close(old)
            for k, v in DEFAULTS.items():
                st.session_state[k] = copy.deepcopy(v)
            st.rerun()
        if c2.button("Cancel", key="sb_reset_no", **_stretch()):
            st.session_state.confirm_reset = False

    alert_svg = Icons.alert_triangle(13, "#f59e0b")
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
        "<div style='text-align:center;color:#7ba3d6;font-size:.76rem;margin-top:.5rem;line-height:1.5'>"
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
        (cpu_svg,
         f"{np.mean([x.get('latency_ms',0) for x in h2]):.0f} ms" if h2 else "—",
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
        (micro_svg, "Brain MRI Classification",
         "Four-class classification: Glioma, Meningioma, No Tumor, Pituitary using state-of-the-art CNN architectures."),
        (shld_svg,  "MC Dropout Uncertainty",
         "Bayesian uncertainty estimation via Monte Carlo Dropout. Reliability bands quantify how confident the model really is."),
        (eye_svg,   "Dual XAI (Grad-CAM++)",
         "Side-by-side Grad-CAM and Grad-CAM++ visualizations with an agreement score showing heatmap consistency."),
        (zap2_svg,  "Real-Time Inference",
         "Detailed timing breakdown: preprocessing, inference, and XAI generation with CUDA-accurate latency measurement."),
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
        _icon_header(Icons.zap(20, "#22d3ee"), "How NeuroLens Works"),
        unsafe_allow_html=True,
    )
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
                    <div style="font-size:.6rem;color:#7ba3d6;letter-spacing:.08em;font-weight:700;margin-bottom:.22rem">MC DROPOUT &amp; XAI</div>
                    <div style="font-size:.82rem;font-weight:700;color:#e2e8f0">{MC_SAMPLES} passes · Grad-CAM + Grad-CAM++</div>
                </div>
            </div>
        </div>"""), unsafe_allow_html=True)

    if st.session_state.last_result:
        st.write("")
        st.markdown(
            _icon_header(Icons.file_text(20, "#22d3ee"), "Last Analysis — Live Stream"),
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
        f"{micro_h} MRI Diagnostic Analysis</h1>",
        unsafe_allow_html=True,
    )
    st.caption("Upload a brain MRI scan for AI-powered classification with uncertainty estimation and dual explainability.")
    render_live_ticker()

    if model_error:
        st.error("Neural engine unavailable. Cannot run inference.")
        st.info(f"Expected model: `{MODEL_PATH}`")
        with st.expander("Technical Details"):
            st.code(model_error)
    else:
        up_svg = Icons.upload(28, "#22d3ee")
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
                "Upload Brain MRI Scan",
                type=["jpg", "jpeg", "png", "webp"],
                key="mri_uploader",
                label_visibility="collapsed",
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
                    if old:
                        plt.close(old)
                for k in [
                    "last_result", "last_image", "gradcam_image",
                    "gradcam_pp_image", "mc_result", "agreement_score",
                ]:
                    st.session_state[k] = None

            try:
                image = Image.open(io.BytesIO(upload_bytes)).convert("RGB")
            except (UnidentifiedImageError, OSError, ValueError):
                st.error("Invalid image. Please upload a valid JPG, PNG, or WEBP file.")
                image = None

            if image is not None:
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.markdown(
                        _icon_header(Icons.image(20, "#22d3ee"), "MRI Preview", level=4),
                        unsafe_allow_html=True,
                    )
                    st.image(image, **_stretch())
                    st.caption(
                        f"Original: {image.width}×{image.height}px · "
                        f"Processed: {IMG_SIZE}×{IMG_SIZE}px"
                    )

                with col2:
                    st.markdown(
                        _icon_header(Icons.sliders(20, "#22d3ee"), "Analysis Configuration", level=4),
                        unsafe_allow_html=True,
                    )
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
                        status = st.empty()
                        prog = st.empty()
                        try:
                            total_t0 = time.perf_counter()
                            stages = [
                                "Image Loaded", "Preprocessing", "Normalization",
                                "Tensor Prep", "Neural Inference", "Probability Calc",
                            ]
                            if run_mc:
                                stages.append("MC Dropout")
                            if run_xai:
                                stages += ["Grad-CAM", "Grad-CAM++"]
                            if run_agree:
                                stages.append("Agreement Score")
                            stages.append("Report")
                            total_stages = len(stages)

                            for i, stage in enumerate(stages[:-4]):
                                status.info(f"Processing: {stage}…")
                                prog.progress((i + 1) / total_stages)

                            log_activity("MRI uploaded; preprocessing started", "info")
                            (predicted_class, confidence, probability_dict,
                             preprocessing_ms, inference_ms) = predict_image(
                                image, model, class_names
                            )
                            log_activity(
                                f"Inference complete: {predicted_class} ({confidence:.1f}%)",
                                "success",
                            )

                            step_off = 4
                            mc_result = None
                            if run_mc:
                                status.info("Processing: MC Dropout…")
                                prog.progress((step_off + 1) / total_stages)
                                step_off += 1
                                try:
                                    mc_result = mc_dropout_predict(image, model, class_names, MC_SAMPLES)
                                    if mc_result:
                                        log_activity(
                                            f"MC Dropout: σ={mc_result['uncertainty']:.4f} ({mc_result['band']})",
                                            "info",
                                        )
                                except Exception:
                                    log_activity("MC Dropout failed", "warn")

                            gradcam_fig = gradcam_ms = None
                            gradcam_pp_fig = gradcam_pp_ms = None
                            if run_xai:
                                status.info("Processing: Grad-CAM…")
                                prog.progress((step_off + 1) / total_stages)
                                step_off += 1
                                try:
                                    t_gc = time.perf_counter()
                                    gradcam_fig = generate_gradcam(image, model, model_name)
                                    gradcam_ms = (time.perf_counter() - t_gc) * 1000
                                    log_activity("Grad-CAM generated", "info")
                                except Exception:
                                    log_activity("Grad-CAM failed", "warn")

                                status.info("Processing: Grad-CAM++…")
                                prog.progress((step_off + 1) / total_stages)
                                step_off += 1
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
                                prog.progress((step_off + 1) / total_stages)
                                step_off += 1
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
                                "prediction": predicted_class,
                                "confidence": confidence,
                                "probabilities": probability_dict,
                                "model": model_name,
                                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                "latency_ms": inference_ms,
                                "preprocessing_ms": preprocessing_ms,
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

                            for fk, nf in [("gradcam_image", gradcam_fig),
                                           ("gradcam_pp_image", gradcam_pp_fig)]:
                                old = st.session_state.get(fk)
                                if old:
                                    plt.close(old)
                                st.session_state[fk] = nf

                            st.session_state.last_result = result
                            st.session_state.last_image = image.copy()
                            st.session_state.mc_result = mc_result
                            st.session_state.agreement_score = agree_score
                            st.session_state.prediction_history.append(result)
                            st.session_state.prediction_history = (
                                st.session_state.prediction_history[-MAX_HISTORY:]
                            )
                            update_live_stats(result, inference_ms)
                            log_activity("Analysis report generated", "success")
                            status.success(
                                f"{predicted_class} · {confidence:.2f}% · {inference_ms:.0f} ms"
                            )
                            prog.empty()

                        except Exception as exc:
                            msg = (
                                "CUDA out of memory. Try CPU inference."
                                if "out of memory" in str(exc).lower()
                                else "Analysis failed. Check image and model."
                            )
                            status.error(msg)
                            with st.expander("Technical Details"):
                                st.code(str(exc))
                            log_activity("Analysis failed", "error")
                        finally:
                            st.session_state.live_inference_running = False

        if (image_id is not None
                and st.session_state.last_result is not None
                and st.session_state.last_result.get("image_id") == image_id):

            result = st.session_state.last_result
            mc_res = st.session_state.mc_result
            conf_v = result["confidence"]
            conf_lbl = "High" if conf_v >= 80 else "Moderate" if conf_v >= 60 else "Low"
            conf_cls = "high" if conf_v >= 80 else "moderate" if conf_v >= 60 else "low"

            shld_r = Icons.shield(16, "#a8bcd8")
            st.markdown(safe_html(f"""
            <div class="diagnostic-panel">
                <div class="diag-header">
                    <span class="diag-label">AI Classification Result</span>
                    <span class="badge badge-research">{shld_r} Research Prototype</span>
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
                unc = mc_res["uncertainty"]
                band = mc_res["band"]
                color = mc_res["color"]
                st.markdown(safe_html(f"""
                <div class="uncertainty-card">
                    <div class="unc-title">MC Dropout Uncertainty Estimation ({MC_SAMPLES} passes)</div>
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
                            Pearson correlation between Grad-CAM and Grad-CAM++ heatmaps.<br>
                            High score (&gt;0.70) means both methods highlight similar regions.
                        </div>
                        <div style="text-align:right;min-width:90px;margin-left:1rem">
                            <div style="font-family:var(--font-display);font-size:1.45rem;font-weight:800;color:{a_color};letter-spacing:-.02em">{agree:.3f}</div>
                            <div style="font-size:.66rem;color:{a_color};font-weight:700">{a_label}</div>
                        </div>
                    </div>
                </div>"""), unsafe_allow_html=True)

            alert_svg = Icons.alert_triangle(16, "#f59e0b")
            st.markdown(safe_html(f"""
            <div class="disclaimer">
                {alert_svg}
                <span>NeuroLens AI is a research prototype for educational use. Model outputs are not medical diagnoses and must not replace evaluation by a qualified healthcare professional.</span>
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

            if st.session_state.gradcam_image or st.session_state.gradcam_pp_image:
                st.write("")
                st.markdown(
                    _icon_header(Icons.eye(20, "#22d3ee"), "Dual XAI Visualization", level=4),
                    unsafe_allow_html=True,
                )
                gc_col, pp_col = st.columns(2)
                with gc_col:
                    st.markdown(
                        _icon_header(Icons.heatmap(18, "#22d3ee"), "Grad-CAM · Jet colormap", level=5),
                        unsafe_allow_html=True,
                    )
                    if st.session_state.gradcam_image:
                        st.pyplot(st.session_state.gradcam_image, **_stretch_pyplot())
                        st.caption("Weighted class activations · α=0.44")
                with pp_col:
                    st.markdown(
                        _icon_header(Icons.heatmap(18, "#22d3ee"), "Grad-CAM++ · Inferno colormap", level=5),
                        unsafe_allow_html=True,
                    )
                    if st.session_state.gradcam_pp_image:
                        st.pyplot(st.session_state.gradcam_pp_image, **_stretch_pyplot())
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
                    st.download_button("Download Grad-CAM", gc_buf, "gradcam.png",
                                       "image/png", key="dl_gc", **_stretch())
                with dc2:
                    st.markdown('<div class="export-item-label">Grad-CAM++ · PNG</div>', unsafe_allow_html=True)
                    st.download_button("Download Grad-CAM++", pp_buf, "gradcam_pp.png",
                                       "image/png", key="dl_pp", **_stretch())
            elif gc_buf:
                st.markdown('<div class="export-item-label">Grad-CAM · PNG</div>', unsafe_allow_html=True)
                st.download_button("Download Grad-CAM", gc_buf, "gradcam.png",
                                   "image/png", key="dl_gc", **_stretch())
            elif pp_buf:
                st.markdown('<div class="export-item-label">Grad-CAM++ · PNG</div>', unsafe_allow_html=True)
                st.download_button("Download Grad-CAM++", pp_buf, "gradcam_pp.png",
                                   "image/png", key="dl_pp", **_stretch())

            st.markdown(
                '<div class="export-item-label" style="margin-top:.9rem">Analysis Report · TXT</div>',
                unsafe_allow_html=True,
            )
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
                    "  DISCLAIMER",
                    "─────────────────────────────────────────────────────",
                    "  NeuroLens AI is a research prototype for educational use.",
                    "  Outputs do NOT constitute medical diagnoses.",
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
        f"{chart_h} Neurodiagnostic Dashboard</h1>",
        unsafe_allow_html=True,
    )
    st.caption("Session analytics from completed MRI analyses")
    render_live_ticker()

    if st.session_state.live_session_start is None:
        st.session_state.live_session_start = datetime.now()
    history = st.session_state.prediction_history

    if not history:
        chart_svg = Icons.chart(32, "#7ba3d6")
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
                <span style="font-size:.66rem;color:#22d3ee;font-weight:700;letter-spacing:.1em">Average Explanation Agreement Score</span>
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
            cf = plot_confidence_trend(history)
            if cf:
                st.pyplot(cf, **_stretch_pyplot())
                plt.close(cf)
            else:
                st.caption("Need ≥2 analyses.")

            st.markdown(
                _icon_header(Icons.clock(20, "#22d3ee"), "Inference Latency"),
                unsafe_allow_html=True,
            )
            lf = plot_latency_trend(history)
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
                uf = plot_uncertainty_history(history)
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
        f"{hist_h} Analysis History</h1>",
        unsafe_allow_html=True,
    )
    st.caption("Review all session AI diagnostic reports")
    render_live_ticker()
    history = st.session_state.prediction_history

    if not history:
        hist_svg = Icons.history(32, "#7ba3d6")
        st.markdown(safe_html(f"""
        <div class="empty-state">
            <div class="empty-icon">{hist_svg}</div>
            <div class="empty-title">No diagnostic history</div>
            <div class="empty-text">Analysis reports appear here after running MRI scans.</div>
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
        for rec in history:
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
            st.caption(f"Showing {len(filtered)} of {len(history)} records")
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
        f"{hm_h} Grad-CAM Explainability</h1>",
        unsafe_allow_html=True,
    )
    st.caption("Visualize which MRI regions influenced the model's classification")
    render_live_ticker()

    if not st.session_state.gradcam_image and not st.session_state.gradcam_pp_image:
        hm_svg = Icons.heatmap(32, "#7ba3d6")
        st.markdown(safe_html(f"""
        <div class="empty-state">
            <div class="empty-icon">{hm_svg}</div>
            <div class="empty-title">No Grad-CAM visualization yet</div>
            <div class="empty-text">Run an MRI analysis with XAI enabled to generate heatmaps.</div>
        </div>"""), unsafe_allow_html=True)
    else:
        oc, gc, pc = st.columns(3)
        with oc:
            st.markdown(
                _icon_header(Icons.image(18, "#22d3ee"), "Original MRI", level=4),
                unsafe_allow_html=True,
            )
            if st.session_state.last_image:
                st.image(st.session_state.last_image, **_stretch())
        with gc:
            st.markdown(
                _icon_header(Icons.heatmap(18, "#22d3ee"), "Grad-CAM · Jet", level=4),
                unsafe_allow_html=True,
            )
            if st.session_state.gradcam_image:
                st.pyplot(st.session_state.gradcam_image, **_stretch_pyplot())
        with pc:
            st.markdown(
                _icon_header(Icons.heatmap(18, "#22d3ee"), "Grad-CAM++ · Inferno", level=4),
                unsafe_allow_html=True,
            )
            if st.session_state.gradcam_pp_image:
                st.pyplot(st.session_state.gradcam_pp_image, **_stretch_pyplot())

        st.markdown(
            "<div style='font-size:.74rem;color:#7ba3d6;margin:.55rem 0'>"
            "Heatmap influence: Low (dark) ░░░▒▒▒████ High (bright)</div>",
            unsafe_allow_html=True,
        )

        agree = st.session_state.agreement_score
        if agree is not None:
            a_color = "#34d399" if agree >= 0.7 else "#f59e0b" if agree >= 0.5 else "#ef4444"
            a_label = "High Agreement" if agree >= 0.7 else "Moderate" if agree >= 0.5 else "Low Agreement"
            st.markdown(safe_html(f"""
            <div class="xai-card">
                <div class="xai-title">{Icons.eye(12, "#22d3ee")} Explanation Agreement Score</div>
                <div style="display:flex;justify-content:space-between;align-items:center;gap:1rem;flex-wrap:wrap">
                    <div class="xai-text">Correlation between Grad-CAM and Grad-CAM++ attention regions.<br>High scores (&gt;0.70) indicate consistent heatmaps.</div>
                    <div style="text-align:right;min-width:85px;margin-left:1rem">
                        <div style="font-family:var(--font-display);font-size:1.45rem;font-weight:800;color:{a_color};letter-spacing:-.02em">{agree:.3f}</div>
                        <div style="font-size:.66rem;color:{a_color};font-weight:700">{a_label}</div>
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
            dl1.download_button("Download Grad-CAM", buf.getvalue(),
                                "gradcam.png", "image/png",
                                key="gradcam_dl", **_stretch())
        if st.session_state.gradcam_pp_image:
            buf2 = io.BytesIO()
            st.session_state.gradcam_pp_image.savefig(buf2, format="png", bbox_inches="tight", dpi=160)
            dl2.download_button("Download Grad-CAM++", buf2.getvalue(),
                                "gradcam_pp.png", "image/png",
                                key="gradcam_pp_dl", **_stretch())


# ══════════════════════════════════════════════════════════════════════════════
# XAI LAB
# ══════════════════════════════════════════════════════════════════════════════

elif nav == "XAI Lab":
    lab_h = Icons.lab(22, "#22d3ee")
    st.markdown(
        f"<h1 style='display:flex;align-items:center;gap:.5rem;flex-wrap:wrap'>"
        f"{lab_h} XAI Research Lab</h1>",
        unsafe_allow_html=True,
    )
    st.caption("Uncertainty, explainability, and model behavior analysis")
    render_live_ticker()
    history = st.session_state.prediction_history

    if not history:
        lab_svg = Icons.lab(32, "#7ba3d6")
        st.markdown(safe_html(f"""
        <div class="empty-state">
            <div class="empty-icon">{lab_svg}</div>
            <div class="empty-title">No XAI data yet</div>
            <div class="empty-text">Run analyses with MC Dropout and dual XAI enabled.</div>
        </div>"""), unsafe_allow_html=True)
    else:
        st.markdown(
            _icon_header(Icons.activity(20, "#22d3ee"), "Uncertainty Distribution"),
            unsafe_allow_html=True,
        )
        unc_data = [h["uncertainty"] for h in history if h.get("uncertainty") is not None]
        if len(unc_data) >= 2:
            uf = plot_uncertainty_history(history)
            if uf:
                st.pyplot(uf, **_stretch_pyplot())
                plt.close(uf)
        else:
            st.caption("Need ≥2 analyses with MC Dropout enabled.")

        bands = {}
        for h in history:
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
            for i, h in enumerate(history)
            if h.get("agreement_score") is not None
        ]
        if len(agree_hist) >= 2:
            st.write("")
            st.markdown(
                _icon_header(Icons.trending_up(20, "#22d3ee"), "Agreement Score Trend"),
                unsafe_allow_html=True,
            )
            fig2, ax2 = _dark_fig()
            xs, ys = zip(*agree_hist)
            ax2.plot(xs, ys, marker="D", lw=2, ms=4, color="#22d3ee")
            ax2.fill_between(xs, ys, alpha=0.08, color="#22d3ee")
            ax2.axhline(0.7, color="#34d399", lw=1, ls="--", alpha=0.65, label="High agreement")
            ax2.axhline(0.5, color="#f59e0b", lw=1, ls="--", alpha=0.65, label="Moderate")
            ax2.set_ylim(0, 1.05)
            ax2.set_xlabel("Analysis #", color="#a8bcd8", fontsize=9)
            ax2.set_ylabel("Agreement Score", color="#a8bcd8", fontsize=9)
            ax2.legend(fontsize=7, labelcolor="#a8bcd8",
                       facecolor="#0f2b57", edgecolor="#26497b")
            ax2.grid(True, alpha=0.09, ls="--", color="#7ba3d6")
            fig2.tight_layout(pad=0.5)
            st.pyplot(fig2, **_stretch_pyplot())
            plt.close(fig2)

        st.write("")
        st.markdown(
            _icon_header(Icons.grid(20, "#22d3ee"), "Per-Class Uncertainty Analysis"),
            unsafe_allow_html=True,
        )
        class_unc = {c: [] for c in CLASS_NAMES}
        for h in history:
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
        f"{set_h} System Settings</h1>",
        unsafe_allow_html=True,
    )
    st.caption("Neural engine configuration and session management")

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
            <p><b>PyTorch:</b> {_escape_html(torch.__version__)}</p>
            <p><b>Torchvision:</b> {_escape_html(tv)}</p>
            <p><b>Streamlit:</b> {_escape_html(st.__version__)}</p>
            <p><b>Python:</b> {_escape_html(platform.python_version())}</p>
            <p><b>CUDA Available:</b> {torch.cuda.is_available()}</p>
            <p><b>CUDA Device:</b> {_escape_html(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')}</p>
            <p><b>XAI Methods:</b> Grad-CAM · Grad-CAM++</p>
        </div>"""), unsafe_allow_html=True)

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
                if model_error:
                    st.code(model_error)

        st.write("")
        alert_svg = Icons.alert_triangle(16, "#f59e0b")
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
# FOOTER — PREMIUM BLUE UPGRADE
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
        <div class="footer-desc">
            An advanced research platform for brain MRI classification with dual XAI explainability and Bayesian uncertainty estimation — built for researchers, students, and clinical AI exploration.
        </div>
        <div class="footer-copy">© {year} NeuroLens AI · All Rights Reserved</div>
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

st.markdown(safe_html(f"""
<div class="copyright-line">
    <span style="color:#a8bcd8;font-weight:600">© {year} NeuroLens AI</span> · All Rights Reserved
    <br>
    <span style="font-size:.68rem;color:#7ba3d6">
        Developed by
        <span style="color:#22d3ee;font-weight:700">MD. Atique Shahriar</span>
        &amp;
        <span style="color:#22d3ee;font-weight:700">Aronna Das</span>
    </span>
</div>"""), unsafe_allow_html=True)