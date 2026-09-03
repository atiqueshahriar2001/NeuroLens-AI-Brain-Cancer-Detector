import warnings
import time
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

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import transforms
from torchvision.models import resnet50, efficientnet_b0

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="NeuroLens AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)


STYLES = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

:root {
    --bg-primary: #0f172a;
    --bg-secondary: #1e293b;
    --bg-card: rgba(15, 23, 42, 0.75);
    --bg-glass: rgba(255, 255, 255, 0.03);
    --border-subtle: rgba(255, 255, 255, 0.06);
    --border-hover: rgba(14, 165, 233, 0.4);
    --accent: #0ea5e9;
    --accent-light: #38bdf8;
    --accent-glow: rgba(14, 165, 233, 0.15);
    --text-primary: #f8fafc;
    --text-secondary: #94a3b8;
    --text-muted: #64748b;
    --success: #10b981;
    --warning: #f59e0b;
    --error: #ef4444;
    --gradient-1: linear-gradient(135deg, #0ea5e9 0%, #06b6d4 100%);
    --shadow-glow: 0 0 25px rgba(14, 165, 233, 0.2);
    --diagnostic: #06b6d4;
    --neural-core: rgba(14, 165, 233, 0.1);
}

* { font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; }

.stApp { background: var(--bg-primary); color: var(--text-primary); }

header[data-testid="stHeader"] { background: transparent; }
#MainMenu, footer { visibility: hidden; }

[data-testid="stSidebar"] {
    background: var(--bg-secondary);
    border-right: 1px solid var(--border-subtle);
}
[data-testid="stSidebar"] > div:first-child { padding-top: 1.5rem; }

.sidebar-brand { padding: 0 1rem 1.5rem; }
.sidebar-brand-title {
    font-size: 1.5rem; font-weight: 800;
    background: var(--gradient-1);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.25rem;
    letter-spacing: -0.02em;
}
.sidebar-brand-subtitle {
    font-size: 0.75rem; color: var(--text-muted);
    font-weight: 500; letter-spacing: 0.05em;
    text-transform: uppercase;
}

/* Premium Sidebar Navigation Cards */
[data-testid="stSidebar"] .stButton { margin: 0.42rem 0 !important; }
[data-testid="stSidebar"] .stButton > button {
    width: 100% !important; min-height: 54px !important; display: flex !important;
    justify-content: flex-start !important; align-items: center !important; text-align: left !important;
    padding: 0.85rem 1rem !important; border: 1px solid rgba(255,255,255,0.055) !important;
    border-radius: 16px !important;
    background: linear-gradient(135deg, rgba(255,255,255,0.035), rgba(255,255,255,0.012)) !important;
    color: var(--text-secondary) !important; font-weight: 600 !important; font-size: 0.92rem !important;
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.025), 0 8px 20px rgba(0,0,0,0.10) !important;
    transition: all 0.22s cubic-bezier(.2,.8,.2,1) !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: linear-gradient(135deg, rgba(14,165,233,0.16), rgba(6,182,212,0.06)) !important;
    border-color: rgba(56,189,248,0.42) !important; color: var(--text-primary) !important;
    transform: translateX(6px) scale(1.015) !important;
    box-shadow: 0 10px 26px rgba(14,165,233,0.13) !important;
}
[data-testid="stSidebar"] .stButton > button:active { transform: translateX(3px) scale(.99) !important; }
[data-testid="stSidebar"] .stButton > button p { width:100% !important; text-align:left !important; margin:0 !important; }
.nav-active {
    position: relative !important; border-radius: 16px !important; margin: 0.42rem 0 !important;
    padding-left: 4px !important; background: linear-gradient(135deg, rgba(14,165,233,0.20), rgba(6,182,212,0.08)) !important;
    border: 1px solid rgba(56,189,248,0.45) !important;
    box-shadow: 0 0 0 1px rgba(14,165,233,0.08), 0 12px 28px rgba(14,165,233,0.14) !important;
}
.nav-active::before {
    content: "" !important; position: absolute !important; left: -1px !important; top: 14px !important; bottom: 14px !important;
    width: 4px !important; border-radius: 0 8px 8px 0 !important;
    background: linear-gradient(180deg, var(--accent-light), #06b6d4) !important;
    box-shadow: 0 0 14px var(--accent) !important; z-index: 2 !important;
}
.nav-active [data-testid="stSidebar"] .stButton > button {
    color: var(--text-primary) !important; font-weight: 700 !important; background: transparent !important;
    border-color: transparent !important; box-shadow: none !important; transform: none !important;
}

.sidebar-status {
    margin-top: 1rem; padding: 0.875rem;
    border-radius: 12px; background: var(--bg-glass);
    border: 1px solid var(--border-subtle);
    text-align: center;
}
.status-title {
    font-size: 0.7rem; color: var(--text-muted);
    text-transform: uppercase; letter-spacing: 0.1em; font-weight: 600;
    text-align: center;
}
.status-value { margin-top: 0.35rem; font-size: 0.85rem; font-weight: 600; color: var(--text-primary); }

.topbar {
    display: flex; justify-content: space-between; align-items: center;
    padding: 1rem 1.5rem; margin-bottom: 2rem;
    border: 1px solid var(--border-subtle);
    border-radius: 16px; background: var(--bg-card);
}
.topbar-title { font-size: 1.25rem; font-weight: 700; color: var(--text-primary); }
.topbar-page { font-size: 0.85rem; color: var(--text-secondary); font-weight: 500; }

/* ====== STICKY COMPACT HEADER ====== */
.sticky-header {
    position: fixed;
    top: 0.6rem;
    left: calc(300px + 0.4rem);
    right: 0.4rem;
    z-index: 9999;
    display: flex;
    align-items: center;
    gap: 1.25rem;
    padding: 1rem 1.5rem;
    border-radius: 16px;
    background: linear-gradient(135deg, rgba(15,23,42,0.85), rgba(30,41,59,0.65));
    backdrop-filter: blur(18px) saturate(140%);
    -webkit-backdrop-filter: blur(18px) saturate(140%);
    border: 1px solid rgba(56,189,248,0.18);
    box-shadow: 0 8px 28px rgba(0,0,0,0.35), 0 0 22px rgba(14,165,233,0.08);
    transition: all 0.3s ease;
}
.sticky-header:hover {
    border-color: rgba(56,189,248,0.32);
    box-shadow: 0 10px 32px rgba(0,0,0,0.42), 0 0 26px rgba(14,165,233,0.14);
}
.sticky-brand {
    display: flex; align-items: center; gap: 0.6rem;
    padding-right: 0.85rem;
    border-right: 1px solid rgba(255,255,255,0.07);
}
.sticky-brand-logo {
    width: 44px; height: 44px;
    border-radius: 11px;
    background: var(--gradient-1);
    display: flex; align-items: center; justify-content: center;
    font-size: 1.4rem;
    box-shadow: 0 0 16px rgba(14,165,233,0.55);
}
.sticky-brand-name {
    font-size: 1.25rem; font-weight: 800;
    color: var(--text-primary);
    letter-spacing: -0.01em;
}
.sticky-brand-version {
    font-size: 0.72rem; color: var(--text-muted);
    font-weight: 600; letter-spacing: 0.08em;
    text-transform: uppercase;
}
.sticky-divider {
    width: 1px; height: 32px;
    background: rgba(255,255,255,0.07);
}
.sticky-ticker {
    flex: 1;
    overflow: hidden;
    border-radius: 10px;
    border: 1px solid rgba(56,189,248,0.18);
    background: linear-gradient(90deg, rgba(14,165,233,0.08), rgba(6,182,212,0.02));
    padding: 0.65rem 0;
    min-width: 0;
}
.sticky-ticker-inner {
    display: inline-block;
    white-space: nowrap;
    padding-left: 100%;
    color: #7dd3fc;
    font: 600 0.92rem Inter, Arial, sans-serif;
    letter-spacing: 0.04em;
    animation: ticker-scroll 24s linear infinite;
}
@keyframes ticker-scroll {
    0% { transform: translateX(0); }
    100% { transform: translateX(-100%); }
}
.sticky-page {
    display: flex; align-items: center; gap: 0.6rem;
    padding: 0.55rem 1rem;
    border-radius: 10px;
    background: rgba(14,165,233,0.10);
    border: 1px solid rgba(56,189,248,0.30);
    font-size: 1rem; font-weight: 700;
    color: var(--text-primary);
    white-space: nowrap;
}
.sticky-page-dot {
    width: 9px; height: 9px; border-radius: 50%;
    background: var(--success);
    box-shadow: 0 0 8px var(--success);
    animation: header-dot-pulse 1.6s infinite;
}
@keyframes header-dot-pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.4; }
}
.sticky-clock {
    display: flex; flex-direction: column; align-items: flex-end;
    font-family: 'JetBrains Mono', monospace;
    line-height: 1.1;
    white-space: nowrap;
}
.sticky-clock-time {
    font-size: 1.05rem; font-weight: 700;
    color: var(--text-primary);
    letter-spacing: 0.02em;
}
.sticky-clock-date {
    font-size: 0.74rem; color: var(--text-muted);
    letter-spacing: 0.06em; text-transform: uppercase;
}

/* Add top padding to main content so sticky header doesn't overlap */
.main .block-container {
    padding-top: 7rem !important;
}

/* =========================================
   RESPONSIVE DESIGN - MOBILE FIRST
   ========================================= */

/* Small phones */
@media (max-width: 480px) {
    .sticky-header {
        left: 0.2rem !important;
        right: 0.2rem !important;
        gap: 0.4rem !important;
        padding: 0.4rem 0.6rem !important;
        border-radius: 10px !important;
    }
    .sticky-brand-logo { width: 24px !important; height: 24px !important; font-size: 0.75rem !important; }
    .sticky-brand-name { font-size: 0.8rem !important; }
    .sticky-page { padding: 0.25rem 0.5rem !important; font-size: 0.7rem !important; }
    .sticky-clock-time { font-size: 0.7rem !important; }
    .sticky-clock-date { font-size: 0.55rem !important; }
    .sticky-divider { height: 16px !important; }
    
    .hero { padding: 1.5rem !important; }
    .hero h1 { font-size: 1.5rem !important; }
    .hero p { font-size: 0.85rem !important; }
    
    .info-card { padding: 1rem !important; min-height: auto !important; }
    .metric-card { padding: 1rem !important; }
    .metric-value { font-size: 1.5rem !important; }
    .metric-icon { font-size: 1.5rem !important; }
    
    .diagnostic-panel { padding: 1rem !important; }
    .diagnostic-prediction { font-size: 1.25rem !important; }
    
    .probability-grid { grid-template-columns: 1fr !important; }
    
    .main .block-container { padding-top: 4.5rem !important; padding-left: 0.5rem !important; padding-right: 0.5rem !important; }
    
    [data-testid="stSidebar"] .stButton > button { min-height: 44px !important; font-size: 0.8rem !important; }
    .sidebar-brand-title { font-size: 1.1rem !important; }
    .sidebar-brand-subtitle { font-size: 0.6rem !important; }
}

/* Tablets and small laptops */
@media (max-width: 768px) {
    .sticky-header {
        left: 0.3rem !important;
        right: 0.3rem !important;
        gap: 0.6rem !important;
        padding: 0.5rem 0.75rem !important;
    }
    .sticky-ticker { display: none; }
    .sticky-brand-version { display: none; }
    .sticky-page { padding: 0.3rem 0.6rem !important; }
    
    .hero { padding: 2rem !important; }
    .hero h1 { font-size: 2rem !important; }
    .hero p { font-size: 0.95rem !important; }
    .metric-value { font-size: 1.75rem !important; }
    
    .main .block-container { padding-top: 5rem !important; }
    
    [data-testid="stSidebar"],
    [data-testid="stSidebar"] > div:first-child {
        min-width: 100% !important;
        max-width: 100% !important;
        width: 100% !important;
    }
}

/* Medium screens */
@media (min-width: 769px) and (max-width: 1024px) {
    .sticky-header {
        left: calc(260px + 0.3rem) !important;
        right: 0.3rem !important;
        gap: 0.7rem !important;
    }
    [data-testid="stSidebar"] {
        min-width: 260px !important;
        max-width: 260px !important;
        width: 260px !important;
    }
    [data-testid="stSidebar"] > div:first-child {
        width: 260px !important;
    }
    .main .block-container { padding-top: 5rem !important; }
}

/* Large screens */
@media (min-width: 1025px) and (max-width: 1400px) {
    .sticky-header {
        left: calc(280px + 0.3rem) !important;
        right: 0.3rem !important;
    }
    [data-testid="stSidebar"] {
        min-width: 280px !important;
        max-width: 280px !important;
        width: 280px !important;
    }
    [data-testid="stSidebar"] > div:first-child {
        width: 280px !important;
    }
}

/* Extra large screens */
@media (min-width: 1401px) {
    .sticky-header {
        left: calc(320px + 0.3rem) !important;
        right: 0.3rem !important;
    }
    [data-testid="stSidebar"] {
        min-width: 320px !important;
        max-width: 320px !important;
        width: 320px !important;
    }
    [data-testid="stSidebar"] > div:first-child {
        width: 320px !important;
    }
    .main .block-container { padding-top: 6rem !important; }
}

/* Landscape mobile */
@media (max-width: 768px) and (orientation: landscape) {
    .sticky-header {
        padding: 0.35rem 0.6rem !important;
        gap: 0.4rem !important;
    }
    .sticky-brand-logo { width: 22px !important; height: 22px !important; }
    .sticky-brand-name { font-size: 0.75rem !important; }
    .sticky-page { font-size: 0.7rem !important; padding: 0.2rem 0.5rem !important; }
    .main .block-container { padding-top: 3.5rem !important; }
}

/* High DPI displays */
@media (-webkit-min-device-pixel-ratio: 2), (min-resolution: 192dpi) {
    .sticky-header {
        border-width: 0.5px;
    }
    .info-card, .metric-card, .result-card {
        border-width: 0.5px;
    }
}

/* Reduced motion preference */
@media (prefers-reduced-motion: reduce) {
    *, *::before, *::after {
        animation-duration: 0.01ms !important;
        animation-iteration-count: 1 !important;
        transition-duration: 0.01ms !important;
    }
    .sticky-ticker-inner { animation: none !important; }
}

/* Dark mode preference (already dark, but ensure consistency) */
@media (prefers-color-scheme: dark) {
    :root {
        --bg-primary: #0f172a;
        --bg-secondary: #1e293b;
    }
}

/* ====== FOOTER ====== */
.app-footer {
    margin-top: 3rem;
    margin-left: -0.4rem;
    margin-right: -0.4rem;
    padding: 1.85rem 2rem;
    border-top: 1px solid var(--border-subtle);
    background: linear-gradient(180deg, transparent, rgba(15,23,42,0.45));
    border-radius: 20px 20px 0 0;
    display: flex; flex-wrap: wrap;
    align-items: center; justify-content: space-between;
    gap: 1.25rem;
    color: var(--text-muted);
    font-size: 0.95rem;
}
.footer-left {
    display: flex; align-items: center; gap: 1.1rem;
    flex-wrap: wrap;
}
.footer-brand {
    display: flex; align-items: center; gap: 0.65rem;
    font-weight: 700; color: var(--text-secondary);
    font-size: 1.05rem;
}
.footer-dot {
    width: 10px; height: 10px; border-radius: 50%;
    background: var(--success);
    box-shadow: 0 0 10px var(--success);
}
.footer-divider { color: rgba(255,255,255,0.12); }
.footer-meta {
    display: flex; align-items: center; gap: 1.1rem;
    flex-wrap: wrap;
    font-size: 0.95rem;
}
.footer-disclaimer {
    flex-basis: 100%;
    padding: 1.1rem 1.25rem;
    border-radius: 12px;
    background: rgba(245,158,11,0.05);
    border: 1px solid rgba(245,158,11,0.18);
    color: var(--text-secondary);
    font-size: 0.9rem;
    line-height: 1.6;
}
.footer-disclaimer b { color: var(--warning); }

.hero {
    padding: 3rem; border-radius: 24px; margin-bottom: 2rem;
    background: var(--bg-card); border: 1px solid var(--border-subtle);
    position: relative; overflow: hidden;
}
.hero::before {
    content: ""; position: absolute;
    top: -50%; right: -10%;
    width: 500px; height: 500px;
    background: radial-gradient(circle, var(--accent-glow) 0%, transparent 70%);
    pointer-events: none;
}
.hero::after {
    content: ""; position: absolute;
    inset: 0;
    background-image: radial-gradient(circle at 50% 50%, var(--neural-core) 1px, transparent 1px);
    background-size: 40px 40px;
    opacity: 0.3;
    pointer-events: none;
}
.hero h1 { margin: 0; font-size: 3rem; line-height: 1.1; color: var(--text-primary); letter-spacing: -0.03em; }
.hero p { color: var(--text-secondary); font-size: 1.1rem; margin-top: 1rem; max-width: 600px; line-height: 1.6; }
.hero-badge {
    display: inline-flex; align-items: center; gap: 0.4rem;
    padding: 0.4rem 0.9rem; border-radius: 8px;
    background: var(--accent-glow); border: 1px solid var(--border-hover);
    font-size: 0.75rem; font-weight: 600; color: var(--accent-light);
    margin-top: 0.75rem;
}

.info-card {
    padding: 1.5rem; min-height: 160px;
    border-radius: 20px; background: var(--bg-card);
    border: 1px solid var(--border-subtle);
    transition: all 0.3s ease;
}
.info-card:hover {
    border-color: var(--border-hover);
    box-shadow: var(--shadow-glow);
    transform: translateY(-2px);
}
.info-card h3 { margin-top: 0; color: var(--text-primary); font-size: 1.1rem; font-weight: 600; }
.info-card p { color: var(--text-secondary); line-height: 1.6; font-size: 0.9rem; }

.result-card {
    padding: 1.25rem 1.5rem; margin-top: 1.5rem;
    border-radius: 16px; background: var(--bg-card);
    border: 1px solid var(--border-hover);
    box-shadow: var(--shadow-glow);
}
.result-label {
    color: var(--accent-light); font-size: 0.75rem;
    font-weight: 700; text-transform: uppercase; letter-spacing: 0.1em;
}

[data-testid="stFileUploader"] {
    background: var(--bg-card); border-radius: 16px;
    padding: 1.25rem; border: 2px dashed var(--border-subtle);
}
[data-testid="stFileUploader"]:hover {
    border-color: var(--accent); background: var(--accent-glow);
}

[data-testid="stExpander"] {
    border-radius: 14px !important;
    border: 1px solid var(--border-subtle) !important;
    background: var(--bg-card) !important;
}

.disclaimer {
    margin-top: 2rem; padding: 1rem 1.25rem;
    border-radius: 14px; background: rgba(245, 158, 11, 0.08);
    border: 1px solid rgba(245, 158, 11, 0.2);
    color: var(--text-secondary); font-size: 0.85rem; line-height: 1.6;
}

.metric-card {
    padding: 1.5rem; border-radius: 20px;
    background: var(--bg-card); border: 1px solid var(--border-subtle);
    text-align: center; transition: all 0.3s ease;
}
.metric-card:hover {
    border-color: var(--border-hover);
    box-shadow: var(--shadow-glow);
    transform: translateY(-2px);
}
.metric-value { font-size: 2.5rem; font-weight: 800; color: var(--text-primary); margin: 0.5rem 0; }
.metric-label {
    font-size: 0.75rem; color: var(--text-muted);
    text-transform: uppercase; letter-spacing: 0.1em; font-weight: 600;
}
.metric-icon { font-size: 2rem; margin-bottom: 0.5rem; }

.dist-card {
    padding: 1rem 1.25rem; border-radius: 14px;
    background: var(--bg-card); border: 1px solid var(--border-subtle);
    margin-bottom: 0.75rem; transition: all 0.2s ease;
}
.dist-card:hover { border-color: var(--border-hover); }
.dist-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem; }
.dist-name { font-size: 0.9rem; font-weight: 600; color: var(--text-primary); }
.dist-count { font-size: 0.8rem; color: var(--accent-light); font-weight: 600; }
.dist-bar-bg {
    height: 6px; border-radius: 3px;
    background: rgba(255, 255, 255, 0.06); overflow: hidden;
}
.dist-bar-fill {
    height: 100%; border-radius: 3px;
    background: var(--gradient-1);
    transition: width 0.5s cubic-bezier(0.4, 0, 0.2, 1);
}

.latest-card { padding: 1.25rem; border-radius: 16px; background: var(--bg-card); border: 1px solid var(--border-subtle); }
.latest-row {
    display: flex; justify-content: space-between; align-items: center;
    padding: 0.75rem 0; border-bottom: 1px solid var(--border-subtle);
}
.latest-row:last-child { border-bottom: none; }
.latest-key { font-size: 0.85rem; color: var(--text-secondary); font-weight: 500; }
.latest-val { font-size: 0.85rem; color: var(--text-primary); font-weight: 600; }

.thumbnail-card {
    display: flex; align-items: center; gap: 1rem;
    padding: 1rem; border-radius: 16px;
    background: var(--bg-card); border: 1px solid var(--border-subtle);
    transition: all 0.2s ease; margin-bottom: 0.75rem;
}
.thumbnail-card:hover { border-color: var(--border-hover); }
.thumbnail-img {
    width: 64px; height: 64px; border-radius: 12px;
    background: var(--bg-glass);
    display: flex; align-items: center; justify-content: center;
    font-size: 1.5rem; border: 1px solid var(--border-subtle);
}
.thumbnail-info { flex: 1; min-width: 0; }
.thumbnail-title { font-size: 0.95rem; font-weight: 600; color: var(--text-primary); margin-bottom: 0.25rem; }
.thumbnail-meta { font-size: 0.75rem; color: var(--text-muted); }
.thumbnail-confidence { text-align: right; }
.confidence-badge {
    display: inline-block; padding: 0.35rem 0.75rem;
    border-radius: 8px; font-size: 0.85rem; font-weight: 700;
}
.confidence-high { background: rgba(16, 185, 129, 0.15); color: #10b981; border: 1px solid rgba(16, 185, 129, 0.3); }
.confidence-medium { background: rgba(245, 158, 11, 0.15); color: #f59e0b; border: 1px solid rgba(245, 158, 11, 0.3); }
.confidence-low { background: rgba(239, 68, 68, 0.15); color: #ef4444; border: 1px solid rgba(239, 68, 68, 0.3); }

.empty-state {
    padding: 3rem; border-radius: 20px;
    background: var(--bg-card); border: 1px solid var(--border-subtle);
    text-align: center; color: var(--text-secondary);
}
.empty-state-icon { font-size: 3rem; margin-bottom: 1rem; opacity: 0.6; }
.empty-state-title { font-size: 1.1rem; font-weight: 600; color: var(--text-primary); margin-bottom: 0.5rem; }
.empty-state-text { font-size: 0.9rem; }

.image-preview-frame {
    padding: 0.75rem; border-radius: 16px;
    background: var(--bg-card); border: 1px solid var(--border-subtle);
    text-align: center;
}
.image-preview-frame.medical-scan {
    border: 2px solid var(--diagnostic);
    box-shadow: 0 0 0 4px var(--neural-core);
}
.image-preview-label {
    font-size: 0.7rem; color: var(--text-muted);
    text-transform: uppercase; letter-spacing: 0.12em;
    margin-bottom: 0.75rem; font-weight: 700;
}

.medical-badge {
    display: inline-flex; align-items: center; gap: 0.3rem;
    padding: 0.3rem 0.7rem; border-radius: 6px;
    font-size: 0.75rem; font-weight: 700;
}
.badge-benign { background: rgba(16, 185, 129, 0.15); color: var(--success); border: 1px solid rgba(16, 185, 129, 0.3); }
.badge-malignant { background: rgba(239, 68, 68, 0.15); color: var(--error); border: 1px solid rgba(239, 68, 68, 0.3); }
.badge-uncertain { background: rgba(245, 158, 11, 0.15); color: var(--warning); border: 1px solid rgba(245, 158, 11, 0.3); }

.diagnostic-panel {
    padding: 1.5rem; border-radius: 20px;
    background: var(--bg-card); border: 1px solid var(--border-hover);
    box-shadow: var(--shadow-glow); margin-top: 1.5rem;
}
.diagnostic-header {
    display: flex; justify-content: space-between; align-items: center;
    margin-bottom: 1rem; padding-bottom: 0.75rem;
    border-bottom: 1px solid var(--border-subtle);
}
.diagnostic-title { font-size: 0.8rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.1em; font-weight: 700; }
.diagnostic-prediction { font-size: 1.75rem; font-weight: 800; color: var(--text-primary); }

.probability-grid {
    display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
    gap: 0.75rem; margin-top: 1rem;
}
.probability-card {
    padding: 1rem; border-radius: 12px;
    background: var(--bg-glass); border: 1px solid var(--border-subtle);
    text-align: center; transition: all 0.2s ease;
}
.probability-card:hover { border-color: var(--border-hover); transform: translateY(-2px); }
.probability-value { font-size: 1.5rem; font-weight: 800; color: var(--accent-light); margin-bottom: 0.25rem; }
.probability-label { font-size: 0.75rem; color: var(--text-secondary); font-weight: 500; }

.chart-container { padding: 1.25rem; border-radius: 16px; background: var(--bg-card); border: 1px solid var(--border-subtle); margin-top: 1rem; }

.stButton > button { border-radius: 12px !important; font-weight: 600 !important; transition: all 0.2s ease !important; }
.stButton > button:hover { transform: translateY(-1px); }
[data-testid="stMetricValue"] { font-size: 1.75rem !important; font-weight: 700 !important; color: var(--text-primary) !important; }
[data-testid="stMetricLabel"] { font-size: 0.8rem !important; color: var(--text-muted) !important; text-transform: uppercase !important; letter-spacing: 0.05em !important; }
[data-testid="stSidebar"] .stMetric { text-align: center !important; }
[data-testid="stSidebar"] .stCaption { text-align: center !important; }
.stProgress > div > div > div > div { background: var(--gradient-1) !important; border-radius: 4px !important; }
.stAlert { border-radius: 12px !important; border: 1px solid var(--border-subtle) !important; }
hr { border-color: var(--border-subtle) !important; margin: 1.5rem 0 !important; }
.stImage { border-radius: 16px !important; overflow: hidden !important; border: 1px solid var(--border-subtle) !important; }


/* ===== PREMIUM ROBUST SIDEBAR UPGRADE ===== */
.sidebar-section-label {
    margin: 1.1rem 0 0.45rem 0;
    padding-left: 0.25rem;
    font-size: 0.68rem;
    font-weight: 800;
    letter-spacing: 0.14em;
    color: var(--text-muted);
    text-align: center;
}
.sidebar-active-indicator {
    display:flex; align-items:center; gap:.55rem;
    padding:.7rem .85rem; margin:.25rem 0 .45rem 0;
    border-radius:12px;
    background:var(--accent-glow);
    border:1px solid var(--border-hover);
    color:var(--accent-light);
    font-size:.78rem; font-weight:700;
    box-shadow:var(--shadow-glow);
}
.sidebar-active-dot { width:8px; height:8px; border-radius:50%; background:var(--success); box-shadow:0 0 10px var(--success); }
.sidebar-engine-grid { display:grid; grid-template-columns:1fr 1fr; gap:.5rem; margin-top:.8rem; }
.sidebar-mini-stat { padding:.65rem; border-radius:10px; background:rgba(255,255,255,.025); border:1px solid var(--border-subtle); }
.sidebar-mini-label { font-size:.62rem; color:var(--text-muted); text-transform:uppercase; letter-spacing:.08em; }
.sidebar-mini-value { margin-top:.25rem; font-size:.8rem; font-weight:700; color:var(--text-primary); overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
.sidebar-footer { margin-top:1rem; padding:.85rem; border-radius:14px; background:rgba(245,158,11,.06); border:1px solid rgba(245,158,11,.18); font-size:.72rem; color:var(--text-secondary); text-align:center; line-height:1.45; }


/* =========================================
   BALANCED PREMIUM SIDEBAR (250px)
========================================= */
[data-testid="stSidebar"] {
    min-width: 300px !important;
    max-width: 300px !important;
    width: 300px !important;
    background: var(--bg-secondary) !important;
    border-right: 1px solid var(--border-subtle) !important;
    transition: all 0.3s ease !important;
}

[data-testid="stSidebar"] > div:first-child {
    width: 300px !important;
    padding-top: 1.2rem !important;
    padding-left: 0.7rem !important;
    padding-right: 0.7rem !important;
}

.sidebar-brand { padding: 0.4rem 0.5rem 1.2rem !important; }
.sidebar-brand-title { font-size: 1.3rem !important; font-weight: 800 !important; }
.sidebar-brand-subtitle { font-size: 0.65rem !important; letter-spacing: 0.07em !important; }

[data-testid="stSidebar"] .stButton > button {
    width: 100% !important;
    min-height: 46px !important;
    display: flex !important;
    align-items: center !important;
    justify-content: flex-start !important;
    padding: 0.65rem 0.85rem !important;
    margin: 0.22rem 0 !important;
    border-radius: 12px !important;
    background: rgba(255, 255, 255, 0.025) !important;
    border: 1px solid rgba(255, 255, 255, 0.06) !important;
    color: var(--text-secondary) !important;
    font-size: 0.85rem !important;
    font-weight: 600 !important;
    transition: transform 0.25s ease, background 0.25s ease, border-color 0.25s ease, box-shadow 0.25s ease !important;
}

[data-testid="stSidebar"] .stButton > button:hover {
    background: rgba(14, 165, 233, 0.08) !important;
    border-color: rgba(14, 165, 233, 0.35) !important;
    color: var(--text-primary) !important;
    transform: translateX(4px) !important;
    box-shadow: 0 4px 15px rgba(14, 165, 233, 0.10) !important;
}

[data-testid="stSidebar"] .stButton > button:active {
    transform: translateX(2px) scale(0.98) !important;
}

.nav-active {
    position: relative !important;
    border-radius: 12px !important;
    margin: 0.22rem 0 !important;
    background: linear-gradient(135deg, rgba(14, 165, 233, 0.18), rgba(6, 182, 212, 0.06)) !important;
    border: 1px solid rgba(14, 165, 233, 0.40) !important;
    box-shadow: 0 0 20px rgba(14, 165, 233, 0.12) !important;
}

.nav-active::before {
    content: "" !important;
    position: absolute !important;
    left: 0 !important;
    top: 20% !important;
    width: 3px !important;
    height: 60% !important;
    border-radius: 0 4px 4px 0 !important;
    background: var(--accent-light) !important;
    box-shadow: 0 0 10px var(--accent) !important;
    z-index: 10 !important;
}

.nav-active .stButton > button {
    background: transparent !important;
    border-color: transparent !important;
    color: var(--text-primary) !important;
    font-weight: 700 !important;
    box-shadow: none !important;
    transform: none !important;
}

[data-testid="stSidebar"] hr {
    margin: 0.9rem 0 !important;
    border-color: rgba(255, 255, 255, 0.06) !important;
}

.sidebar-status {
    margin-top: 0.9rem !important;
    padding: 0.8rem !important;
    border-radius: 12px !important;
    background: rgba(255, 255, 255, 0.025) !important;
    border: 1px solid rgba(255, 255, 255, 0.07) !important;
    transition: all 0.25s ease !important;
    text-align: center !important;
}

.sidebar-status:hover {
    border-color: rgba(14, 165, 233, 0.3) !important;
    box-shadow: 0 0 18px rgba(14, 165, 233, 0.08) !important;
}

.status-title { font-size: 0.65rem !important; letter-spacing: 0.08em !important; text-align: center !important; }
.status-value { margin-top: 0.4rem !important; font-size: 0.78rem !important; font-weight: 600 !important; }

[data-testid="stSidebar"] ::-webkit-scrollbar { width: 5px; }
[data-testid="stSidebar"] ::-webkit-scrollbar-track { background: transparent; }
[data-testid="stSidebar"] ::-webkit-scrollbar-thumb { background: rgba(14, 165, 233, 0.25); border-radius: 10px; }
[data-testid="stSidebar"] ::-webkit-scrollbar-thumb:hover { background: rgba(14, 165, 233, 0.5); }

</style>
"""

st.markdown(STYLES, unsafe_allow_html=True)


def render_neural_animation():
    components.html("""
    <div id="neuro-wrap"><canvas id="neuro-canvas"></canvas><div class="neuro-overlay"><div class="pulse"></div><span>LIVE NEURAL NETWORK</span></div></div>
    <style>
    html,body{margin:0;padding:0;background:transparent;overflow:hidden}
    #neuro-wrap{position:relative;width:100%;height:170px;overflow:hidden;border-radius:20px;background:radial-gradient(circle at 50% 50%,rgba(14,165,233,.14),rgba(15,23,42,.15) 55%,rgba(15,23,42,.7));border:1px solid rgba(56,189,248,.22)}
    #neuro-canvas{width:100%;height:100%;display:block}
    .neuro-overlay{position:absolute;left:18px;bottom:14px;display:flex;align-items:center;gap:8px;font:600 11px Inter,Arial,sans-serif;letter-spacing:.13em;color:#7dd3fc;text-shadow:0 0 12px rgba(56,189,248,.8)}
    .pulse{width:8px;height:8px;border-radius:50%;background:#22d3ee;box-shadow:0 0 0 0 rgba(34,211,238,.65);animation:pulse 1.5s infinite}
    @keyframes pulse{70%{box-shadow:0 0 0 10px rgba(34,211,238,0)}100%{box-shadow:0 0 0 0 rgba(34,211,238,0)}}
    </style>
    <script>
    const canvas=document.getElementById('neuro-canvas'),ctx=canvas.getContext('2d');let particles=[];
    function resize(){const r=canvas.getBoundingClientRect(),d=window.devicePixelRatio||1;canvas.width=r.width*d;canvas.height=r.height*d;ctx.setTransform(d,0,0,d,0,0);particles=Array.from({length:Math.max(24,Math.floor(r.width/28))},()=>({x:Math.random()*r.width,y:Math.random()*r.height,vx:(Math.random()-.5)*.35,vy:(Math.random()-.5)*.35,r:1.5+Math.random()*2.2}))}
    function draw(){const r=canvas.getBoundingClientRect();ctx.clearRect(0,0,r.width,r.height);for(const p of particles){p.x+=p.vx;p.y+=p.vy;if(p.x<0||p.x>r.width)p.vx*=-1;if(p.y<0||p.y>r.height)p.vy*=-1}for(let i=0;i<particles.length;i++)for(let j=i+1;j<particles.length;j++){const a=particles[i],b=particles[j],d=Math.hypot(a.x-b.x,a.y-b.y);if(d<115){ctx.strokeStyle=`rgba(56,189,248,${(1-d/115)*.26})`;ctx.lineWidth=.7;ctx.beginPath();ctx.moveTo(a.x,a.y);ctx.lineTo(b.x,b.y);ctx.stroke()}}for(const p of particles){ctx.beginPath();ctx.arc(p.x,p.y,p.r,0,Math.PI*2);ctx.fillStyle='#38bdf8';ctx.shadowBlur=12;ctx.shadowColor='#0ea5e9';ctx.fill()}ctx.shadowBlur=0;requestAnimationFrame(draw)}
    window.addEventListener('resize',resize);resize();draw();
    </script>
    """,height=170,scrolling=False)


NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]
IMG_SIZE = 224

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "brain_tumor_detector.pth"
if not MODEL_PATH.exists():
    MODEL_PATH = BASE_DIR.parent / "brain_tumor_detector.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

test_transforms = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(NORM_MEAN, NORM_STD),
])


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

    def forward(self, x):
        return self.block(x)


class CustomCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            ConvBlock(3, 32),
            ConvBlock(32, 64),
            ConvBlock(64, 128),
            ConvBlock(128, 256),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.30),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.pool(self.features(x)))


def build_resnet50(num_classes):
    model = resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def build_efficientnet_b0(num_classes):
    model = efficientnet_b0(weights=None)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    return model


@st.cache_resource
def load_model(model_path):
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found:\n{model_path}")

    checkpoint = torch.load(model_path, map_location=DEVICE)
    class_names = checkpoint.get("class_names", ["Glioma", "Meningioma", "No Tumor", "Pituitary"])
    num_classes = checkpoint.get("num_classes", len(class_names))
    best_model_name = checkpoint.get("best_model_name", "CustomCNN")

    if best_model_name == "ResNet50":
        model = build_resnet50(num_classes)
    elif best_model_name == "EfficientNet-B0":
        model = build_efficientnet_b0(num_classes)
    else:
        model = CustomCNN(num_classes)

    state_dict = checkpoint.get("model_state_dict", checkpoint)
    clean_state_dict = {
        key[len("module."):] if key.startswith("module.") else key: value
        for key, value in state_dict.items()
    }
    model.load_state_dict(clean_state_dict, strict=False)
    model.to(DEVICE)
    model.eval()
    return model, class_names, best_model_name


defaults = {
    "nav": "🏠 Home",
    "last_result": None,
    "last_image": None,
    "gradcam_image": None,
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
    "live_inference_progress": 0.0,
    "live_inference_stage": "Idle",
}
for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

model = None
class_names = None
model_name = None
model_error = None

NAV_ITEMS = [
    "🏠 Home",
    "🔬 MRI Analysis",
    "📊 Dashboard",
    "🕘 History",
    "🔥 Grad-CAM",
    "⚙️ Settings",
]

# ===================== PREMIUM ROBUST SIDEBAR =====================
with st.sidebar:
    history = st.session_state.prediction_history
    total_scans = len(history)
    avg_conf = float(np.mean([x["confidence"] for x in history])) if history else 0.0
    last_pred = history[-1]["prediction"] if history else "No analysis yet"
    active_nav = st.session_state.nav

    st.markdown("""
    <div class="sidebar-brand">
        <div class="sidebar-brand-title">🧠 NeuroLens AI</div>
        <div class="sidebar-brand-subtitle">Neurodiagnostic Intelligence Platform · v2.1</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="sidebar-active-indicator">
        <span class="sidebar-active-dot"></span>
        ACTIVE MODULE: {active_nav}
    </div>
    """, unsafe_allow_html=True)

    groups = [
        ("MAIN", ["🏠 Home", "🔬 MRI Analysis", "📊 Dashboard"]),
        ("ANALYTICS", ["🕘 History", "🔥 Grad-CAM"]),
        ("SYSTEM", ["⚙️ Settings"]),
    ]

    for group_name, items in groups:
        st.markdown(f'<div class="sidebar-section-label">{group_name}</div>', unsafe_allow_html=True)
        for item in items:
            label = f"▶ {item}" if active_nav == item else item
            if st.button(label, key=f"premium_nav_{item}", use_container_width=True):
                st.session_state.nav = item
                st.rerun()

    st.markdown("---")
    engine_status = "🟢 ENGINE READY" if (MODEL_PATH.exists() and model_error is None) else "🔴 MODEL MISSING"
    st.markdown(f"""
    <div class="sidebar-status">
        <div class="status-title">NEURAL ENGINE</div>
        <div class="status-value">{engine_status}</div>
        <div class="sidebar-engine-grid">
            <div class="sidebar-mini-stat"><div class="sidebar-mini-label">Device</div><div class="sidebar-mini-value">{DEVICE}</div></div>
            <div class="sidebar-mini-stat"><div class="sidebar-mini-label">Scans</div><div class="sidebar-mini-value">{total_scans}</div></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section-label">SESSION OVERVIEW</div>', unsafe_allow_html=True)
    a, b = st.columns(2)
    with a:
        st.metric("Total", total_scans)
    with b:
        st.metric("Avg", f"{avg_conf:.0f}%")
    st.caption(f"🧠 Last: **{last_pred}**")

    st.markdown('<div class="sidebar-section-label">QUICK ACTIONS</div>', unsafe_allow_html=True)
    if st.button("🗑️ Clear History", key="sidebar_clear_history", use_container_width=True):
        old_fig = st.session_state.get("gradcam_image")
        if old_fig is not None:
            try:
                plt.close(old_fig)
            except Exception:
                pass
        st.session_state.prediction_history = []
        st.session_state.last_result = None
        st.session_state.last_image = None
        st.session_state.gradcam_image = None
        st.rerun()

    if st.button("🔄 Reset Session", key="sidebar_reset_session", use_container_width=True):
        old_fig = st.session_state.get("gradcam_image")
        if old_fig is not None:
            try:
                plt.close(old_fig)
            except Exception:
                pass
        for key, value in defaults.items():
            st.session_state[key] = value
        st.rerun()


# ==================================================================

nav = st.session_state.nav


def render_sticky_header():
    counts = st.session_state.live_class_counts or {}
    if counts:
        ticker_text = " · ".join(f"<b>{k}</b>: {v}" for k, v in counts.items())
    else:
        ticker_text = "<b>Awaiting first scan</b>"
    last_conf = st.session_state.live_last_confidence
    avg_conf = st.session_state.live_avg_confidence
    throughput = st.session_state.live_throughput
    engine_label = model_name or "—"

    st.markdown(
        f"""
        <div class="sticky-header">
            <div class="sticky-brand">
                <div class="sticky-brand-logo">🧠</div>
                <div>
                    <div class="sticky-brand-name">NeuroLens AI</div>
                    <div class="sticky-brand-version">v2.1 · Live</div>
                </div>
            </div>
            <div class="sticky-divider"></div>
            <div class="sticky-ticker">
                <div class="sticky-ticker-inner">
                    ⚡ LIVE · {ticker_text} · Throughput: {throughput:.2f}/min · Last: {last_conf:.1f}% · Avg: {avg_conf:.1f}% · Engine: {engine_label}
                </div>
            </div>
            <div class="sticky-divider"></div>
            <div class="sticky-page">
                <span class="sticky-page-dot"></span>
                {nav}
            </div>
            <div class="sticky-clock" id="sticky-clock">
                <div class="sticky-clock-time" id="clocktime"></div>
                <div class="sticky-clock-date" id="clockdate"></div>
            </div>
        </div>
        <script>
        (function() {{
            function tick() {{
                const d = new Date();
                const pad = n => String(n).padStart(2,'0');
                const t = pad(d.getHours()) + ":" + pad(d.getMinutes()) + ":" + pad(d.getSeconds());
                const months = ['JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC'];
                const dt = months[d.getMonth()] + " " + pad(d.getDate()) + ", " + d.getFullYear();
                const tEl = document.getElementById('clocktime');
                const dEl = document.getElementById('clockdate');
                if (tEl) tEl.textContent = t;
                if (dEl) dEl.textContent = dt;
            }}
            tick(); setInterval(tick, 1000);
        }})();
        </script>
        """,
        unsafe_allow_html=True,
    )


try:
    model, class_names, model_name = load_model(MODEL_PATH)
except Exception as e:
    model_error = str(e)

render_sticky_header()


def predict_image(image, model, class_names):
    if model is None or class_names is None:
        raise RuntimeError("Neural engine unavailable — cannot run inference.")
    tensor = test_transforms(image).unsqueeze(0).to(DEVICE)
    model.eval()
    with torch.no_grad():
        outputs = model(tensor)
        probs = F.softmax(outputs, dim=1)[0].detach().cpu().numpy()

    idx = int(np.argmax(probs))
    return class_names[idx], float(probs[idx] * 100), {
        class_names[i]: float(probs[i] * 100) for i in range(len(class_names))
    }


MAX_HISTORY = 200


def log_activity(message, level="info"):
    timestamp = datetime.now().strftime("%H:%M:%S")
    st.session_state.activity_log.append(
        {"timestamp": timestamp, "message": message, "level": level}
    )
    if len(st.session_state.activity_log) > 50:
        st.session_state.activity_log = st.session_state.activity_log[-50:]


def update_live_stats(result, latency_ms):
    history = st.session_state.prediction_history
    st.session_state.live_predictions_count = len(history)
    if history:
        confs = [h["confidence"] for h in history]
        st.session_state.live_avg_confidence = float(np.mean(confs))
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


def render_live_ticker():
    counts = st.session_state.live_class_counts or {}
    items = list(counts.items())
    if not items:
        items = [("Awaiting first scan", "—")]
    ticker_html = " · ".join(f"<b>{name}</b>: {n}" for name, n in items)
    components.html(
        f"""
        <style>
        @keyframes marquee {{
            0% {{ transform: translateX(100%); }}
            100% {{ transform: translateX(-100%); }}
        }}
        .ticker-wrap {{
            overflow: hidden;
            border-radius: 12px;
            border: 1px solid rgba(56,189,248,0.3);
            background: linear-gradient(90deg, rgba(14,165,233,0.12), rgba(6,182,212,0.04));
            padding: 0.65rem 0;
            box-shadow: 0 0 20px rgba(14,165,233,0.08);
        }}
        .ticker {{
            display: inline-block;
            white-space: nowrap;
            padding-left: 100%;
            animation: marquee 22s linear infinite;
            color: #7dd3fc;
            font: 600 0.85rem Inter, Arial, sans-serif;
            letter-spacing: 0.04em;
        }}
        .live-pulse {{
            display: inline-block;
            width: 8px; height: 8px;
            border-radius: 50%;
            background: #22d3ee;
            margin-right: 10px;
            box-shadow: 0 0 0 0 rgba(34,211,238,0.7);
            animation: tpulse 1.4s infinite;
            vertical-align: middle;
        }}
        @keyframes tpulse {{
            70% {{ box-shadow: 0 0 0 10px rgba(34,211,238,0); }}
            100% {{ box-shadow: 0 0 0 0 rgba(34,211,238,0); }}
        }}
        </style>
        <div class="ticker-wrap">
            <div class="ticker"><span class="live-pulse"></span>LIVE · {ticker_html} · Throughput: {st.session_state.live_throughput:.2f}/min · Last: {st.session_state.live_last_confidence:.1f}% · Avg: {st.session_state.live_avg_confidence:.1f}%</div>
        </div>
        """,
        height=46,
    )


def render_activity_feed():
    log = st.session_state.activity_log[-12:][::-1]
    if not log:
        st.markdown(
            "<div style='color: var(--text-muted); font-size: 0.85rem;'>No activity yet — run an MRI scan to populate the live feed.</div>",
            unsafe_allow_html=True,
        )
        return
    rows = "".join(
        f"<div class='activity-row activity-{e['level']}'>"
        f"<span class='activity-time'>{e['timestamp']}</span>"
        f"<span class='activity-msg'>{e['message']}</span></div>"
        for e in log
    )
    st.markdown(
        f"""
        <div class="activity-feed">{rows}</div>
        <style>
        .activity-feed {{
            border-radius: 14px;
            border: 1px solid rgba(255,255,255,0.06);
            background: rgba(15,23,42,0.6);
            max-height: 320px;
            overflow-y: auto;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.78rem;
        }}
        .activity-row {{
            display: flex; gap: 0.75rem;
            padding: 0.45rem 0.85rem;
            border-bottom: 1px solid rgba(255,255,255,0.04);
        }}
        .activity-row:last-child {{ border-bottom: none; }}
        .activity-time {{ color: #64748b; min-width: 70px; }}
        .activity-msg {{ color: #cbd5e1; }}
        .activity-success .activity-msg {{ color: #6ee7b7; }}
        .activity-warn .activity-msg {{ color: #fbbf24; }}
        .activity-error .activity-msg {{ color: #fca5a5; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_live_probability_animation(result):
    items = list(result["probabilities"].items())
    bars = "".join(
        f'<div class="lp-row"><div class="lp-label">{name}</div>'
        f'<div class="lp-track"><div class="lp-fill" style="width:0%" data-target="{pct:.2f}"></div></div>'
        f'<div class="lp-val">{pct:.1f}%</div></div>'
        for name, pct in items
    )
    top = result["prediction"]
    conf = result["confidence"]
    components.html(
        f"""
        <style>
        .live-prob {{
            padding: 1.25rem 1.5rem;
            border-radius: 18px;
            border: 1px solid rgba(56,189,248,0.35);
            background: linear-gradient(135deg, rgba(14,165,233,0.10), rgba(6,182,212,0.04));
            box-shadow: 0 0 25px rgba(14,165,233,0.12);
            font-family: Inter, sans-serif;
            color: #e2e8f0;
        }}
        .live-prob-head {{
            display:flex; justify-content:space-between; align-items:center;
            margin-bottom: 0.9rem;
            padding-bottom: 0.6rem;
            border-bottom: 1px solid rgba(255,255,255,0.06);
        }}
        .live-prob-title {{ font-size: 0.75rem; color: #7dd3fc; text-transform: uppercase; letter-spacing: 0.12em; font-weight: 700; }}
        .live-prob-pred {{ font-size: 1.05rem; font-weight: 700; color: #f8fafc; }}
        .lp-row {{ display:flex; align-items:center; gap: 0.75rem; margin: 0.45rem 0; }}
        .lp-label {{ min-width: 110px; font-size: 0.82rem; color: #cbd5e1; font-weight: 600; }}
        .lp-track {{
            flex:1; height: 10px; border-radius: 999px;
            background: rgba(255,255,255,0.06); overflow:hidden;
            position: relative;
        }}
        .lp-fill {{
            height: 100%; width: 0%;
            background: linear-gradient(90deg, #0ea5e9, #06b6d4);
            border-radius: 999px;
            box-shadow: 0 0 12px rgba(14,165,233,0.6);
            transition: width 1.4s cubic-bezier(.2,.8,.2,1);
        }}
        .lp-fill::after {{
            content:""; position:absolute; top:0; left:0; height:100%; width:30px;
            background: linear-gradient(90deg, transparent, rgba(255,255,255,0.55), transparent);
            animation: shimmer 2s infinite;
        }}
        @keyframes shimmer {{ 0% {{transform:translateX(-100%);}} 100% {{transform:translateX(400%);}} }}
        .lp-val {{ min-width: 60px; text-align:right; font-weight: 700; color: #38bdf8; font-size: 0.85rem; }}
        </style>
        <div class="live-prob">
            <div class="live-prob-head">
                <span class="live-prob-title">⚡ Live Probability Stream</span>
                <span class="live-prob-pred">{top} · {conf:.1f}%</span>
            </div>
            {bars}
        </div>
        <script>
        requestAnimationFrame(()=>{{
            document.querySelectorAll('.lp-fill').forEach(el=>{{
                const t = parseFloat(el.getAttribute('data-target'));
                requestAnimationFrame(()=>{{ el.style.width = t + '%'; }});
            }});
        }});
        </script>
        """,
        height=40 + 38 * len(items),
    )


def generate_gradcam(image, model, model_name):
    if model is None:
        raise RuntimeError("Neural engine unavailable — cannot compute Grad-CAM.")
    model.eval()
    activations, gradients = [], []

    if model_name == "ResNet50":
        target_layer = model.layer4[-1].conv3
    elif model_name == "EfficientNet-B0":
        target_layer = model.features[-1]
    else:
        target_layer = model.features[3].block[0]

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
            raise RuntimeError("Grad-CAM hooks failed to capture activations or gradients.")

        activation = activations[0][0]
        gradient = gradients[0][0]
        weights = gradient.mean(dim=(1, 2), keepdim=True)
        cam = F.relu((weights * activation).sum(dim=0)).detach().cpu().numpy()
        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()

        original = np.asarray(image).astype(np.float32) / 255.0
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.imshow(original)
        ax.imshow(cam, cmap="jet", alpha=0.42,
                  extent=(0, original.shape[1], original.shape[0], 0))
        ax.axis("off")
        fig.tight_layout(pad=0)
        return fig
    finally:
        try:
            fwd.remove()
        except Exception:
            pass
        try:
            bwd.remove()
        except Exception:
            pass
        if fig is None:
            plt.close("all")




def plot_confidence_trend(history):
    if len(history) < 2:
        return None
    fig, ax = plt.subplots(figsize=(6, 2.8))
    confidences = [item["confidence"] for item in history]
    indices = list(range(1, len(history) + 1))
    ax.plot(indices, confidences, marker="o", linewidth=2, markersize=5, color="#6366f1")
    ax.fill_between(indices, confidences, alpha=0.15, color="#6366f1")
    ax.set_facecolor("none")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#3f3f5f")
    ax.tick_params(colors="#94a3b8", labelsize=8)
    ax.set_xlabel("Analysis #", color="#94a3b8", fontsize=9)
    ax.set_ylabel("Confidence %", color="#94a3b8", fontsize=9)
    ax.grid(True, alpha=0.15, linestyle="--")
    ax.set_ylim(0, 105)
    fig.tight_layout(pad=0.5)
    return fig


if nav == "🏠 Home":
    render_neural_animation()
    render_live_ticker()
    st.markdown(
        dedent("""
        <div class="hero">
            <div class="hero-badge">🧠 AI-Powered Neurodiagnostic Engine v2.0 · Live Mode</div>
            <h1>🧠 NeuroLens AI</h1>
            <p>Advanced computational neuroimaging platform utilizing deep convolutional
            neural networks for brain MRI analysis, diagnostic classification, and
            explainable AI visualization of oncological findings.</p>
        </div>
        """),
        unsafe_allow_html=True,
    )

    hc1, hc2, hc3, hc4 = st.columns(4)
    live_metrics = [
        ("📡", st.session_state.live_predictions_count, "Live Scans"),
        ("🎯", f"{st.session_state.live_avg_confidence:.1f}%", "Avg. Confidence"),
        ("⚡", f"{st.session_state.live_throughput:.2f}/m", "Throughput"),
        ("⏱️", f"{st.session_state.live_latency_ms:.0f} ms", "Latency"),
    ]
    for col, (icon, value, label) in zip([hc1, hc2, hc3, hc4], live_metrics):
        with col:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-icon">{icon}</div>
                    <div class="metric-value">{value}</div>
                    <div class="metric-label">{label}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    st.write("")

    if st.session_state.live_inference_running:
        st.markdown(
            f"""
            <div class="result-card">
                <div class="result-label">⚡ Live Inference</div>
                <div style="font-size:1.05rem; margin-top:0.4rem;">
                    <b>{st.session_state.live_inference_stage}</b> — {int(st.session_state.live_inference_progress*100)}%
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.progress(st.session_state.live_inference_progress)

    if st.session_state.last_result:
        show_live_prob = True
    else:
        show_live_prob = False

    c1, c2, c3 = st.columns(3)
    cards = [
        ("🔬", "MRI Diagnostic Analysis", "Upload a brain MRI scan and receive AI-powered tumor classification using our trained deep neural network model."),
        ("📊", "Probability Diagnostics", "View confidence-weighted class probability distributions with clinical diagnostic certainty scores."),
        ("🔥", "Explainable AI (XAI)", "Grad-CAM heatmaps highlight the specific brain regions that influenced the model's diagnostic decision."),
    ]
    for col, (icon, title, desc) in zip([c1, c2, c3], cards):
        with col:
            st.markdown(
                f"""
                <div class="info-card">
                    <div style="font-size: 2rem; margin-bottom: 0.75rem;">{icon}</div>
                    <h3>{title}</h3>
                    <p>{desc}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.write("")

    if model_error:
        st.markdown(
            f"""
            <div class="info-card" style="border-color: rgba(239,68,68,0.3); background: rgba(239,68,68,0.08);">
                <h3 style="color: var(--error);">⚠️ Neural Engine Unavailable</h3>
                <p>The trained diagnostic model could not be loaded. Expected location:</p>
                <div style="padding: 0.75rem; border-radius: 8px; background: rgba(0,0,0,0.25); font-family: monospace; font-size: 0.8rem; color: var(--text-muted);">
                    {MODEL_PATH}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div class="info-card" style="border-color: rgba(16,185,129,0.3); background: rgba(16,185,129,0.08);">
                <h3 style="color: var(--success);">✅ Neural Engine Ready</h3>
                <p>Diagnostic Architecture: <b>{model_name}</b></p>
                <p>Trained Classes: <b style="color: var(--accent-light);">{', '.join(class_names)}</b></p>
                <p>Inference Device: <b>{DEVICE}</b></p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if show_live_prob and st.session_state.last_result is not None:
        st.write("")
        st.subheader("⚡ Live Probability Stream")
        render_live_probability_animation(st.session_state.last_result)



elif nav == "🔬 MRI Analysis":
    st.title("🔬 MRI Diagnostic Analysis · Live")
    st.caption("Upload a brain MRI scan to generate an AI-powered diagnostic report.")
    render_live_ticker()

    if model_error:
        st.error("🚨 Unable to load the neural diagnostic engine.")
        st.code(model_error)
        st.info(f"Expected model location:\n{MODEL_PATH}")
    else:
        uploaded_file = st.file_uploader(
            "Upload Brain MRI Scan",
            type=["jpg", "jpeg", "png", "webp"],
            key="mri_uploader",
        )

        if uploaded_file is not None:
            try:
                image = Image.open(uploaded_file).convert("RGB")
            except UnidentifiedImageError:
                st.error("⚠️ Invalid image file. Please upload a JPG, JPEG, PNG, or WEBP scan.")
                st.stop()
            except Exception as e:
                st.error(f"Unable to open the uploaded scan: {e}")
                st.stop()

            col1, col2 = st.columns([1, 2])

            with col1:
                st.markdown(
                    """
                    <div class="image-preview-frame medical-scan">
                        <div class="image-preview-label">MRI Scan</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.image(image, use_container_width=True)

            with col2:
                st.markdown("### 🧠 Neurodiagnostic Report")
                st.caption(f"Neural Engine: **{model_name}** · Compute: **{DEVICE}**")

                if st.button("🔍 Run Diagnostic Analysis", type="primary",
                             use_container_width=True, key="analyze_mri_button"):
                    if st.session_state.live_session_start is None:
                        st.session_state.live_session_start = datetime.now()

                    st.session_state.live_inference_running = True
                    st.session_state.live_inference_progress = 0.0
                    st.session_state.live_inference_stage = "Initializing"

                    progress_bar = st.progress(0.0, text="🧠 Stage: Initializing neural engine...")
                    status_box = st.empty()

                    try:
                        stages = [
                            ("Preprocessing MRI scan", 0.15, 0.18),
                            ("Normalizing pixel values", 0.30, 0.12),
                            ("Loading tensors onto " + str(DEVICE), 0.45, 0.10),
                            ("Forward pass through " + str(model_name), 0.70, 0.30),
                            ("Computing softmax probabilities", 0.85, 0.10),
                            ("Generating Grad-CAM heatmap", 0.97, 0.15),
                            ("Compiling diagnostic report", 1.00, 0.05),
                        ]

                        t_start = datetime.now()
                        for stage_label, pct, sleep_s in stages:
                            st.session_state.live_inference_stage = stage_label
                            status_box.markdown(
                                f"<div class='result-card'><div class='result-label'>⚡ Live Status</div>"
                                f"<div style='font-size:1rem; margin-top:0.4rem;'>"
                                f"<b>{stage_label}</b> — {int(pct*100)}% complete</div></div>",
                                unsafe_allow_html=True,
                            )
                            progress_bar.progress(pct, text=f"🧠 Stage: {stage_label}...")
                            log_activity(f"Pipeline → {stage_label}", level="info")
                            time.sleep(sleep_s)

                        predicted_class, confidence, probability_dict = predict_image(
                            image, model, class_names
                        )

                        gradcam_fig = None
                        try:
                            gradcam_fig = generate_gradcam(image, model, model_name)
                        except Exception as ge:
                            log_activity(f"Grad-CAM unavailable: {ge}", level="warn")

                        latency_ms = (datetime.now() - t_start).total_seconds() * 1000.0

                        result = {
                            "prediction": predicted_class,
                            "confidence": confidence,
                            "probabilities": probability_dict,
                            "model": model_name,
                            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "latency_ms": latency_ms,
                        }
                        st.session_state.last_result = result
                        st.session_state.last_image = image
                        # Close previous Grad-CAM figure to release memory before reassignment.
                        old_fig = st.session_state.get("gradcam_image")
                        if old_fig is not None:
                            try:
                                plt.close(old_fig)
                            except Exception:
                                pass
                        st.session_state.gradcam_image = gradcam_fig
                        st.session_state.prediction_history.append(result)
                        if len(st.session_state.prediction_history) > MAX_HISTORY:
                            st.session_state.prediction_history = (
                                st.session_state.prediction_history[-MAX_HISTORY:]
                            )

                        update_live_stats(result, latency_ms)
                        log_activity(
                            f"Prediction: {predicted_class} ({confidence:.1f}%) in {latency_ms:.0f}ms",
                            level="success",
                        )

                        progress_bar.progress(1.0, text="✅ Analysis complete")
                        status_box.success(
                            f"✅ Diagnostic analysis complete. Prediction: **{predicted_class}** "
                            f"(inference latency: {latency_ms:.0f} ms)"
                        )
                    except Exception as e:
                        progress_bar.empty()
                        status_box.error(f"⚠️ An error occurred during diagnostic analysis: {e}")
                        log_activity(f"Analysis failed: {e}", level="error")
                    finally:
                        st.session_state.live_inference_running = False
                        st.session_state.live_inference_progress = 0.0
                        st.session_state.live_inference_stage = "Idle"

        if st.session_state.last_result is not None and st.session_state.last_image is not None:
            result = st.session_state.last_result

            is_tumor = result["prediction"] != "No Tumor"
            badge_class = "badge-malignant" if result['confidence'] > 80 else "badge-uncertain"

            st.markdown(
                f"""
                <div class="diagnostic-panel">
                    <div class="diagnostic-header">
                        <span class="diagnostic-title">🩺 Diagnostic Report</span>
                        <span class="medical-badge {badge_class}">{'⚠️ TUMOR DETECTED' if is_tumor else '✅ NO TUMOR DETECTED'}</span>
                    </div>
                    <div class="diagnostic-prediction">{result['prediction']}</div>
                    <div class="result-label">AI Certainty Score</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            m1, m2 = st.columns(2)
            with m1:
                st.metric("Diagnostic Confidence", f"{result['confidence']:.2f}%",
                          delta="AI certainty" if result['confidence'] > 70 else "- Low certainty")
            with m2:
                st.metric("Deep Model", result["model"])

            st.write("")
            st.markdown("### 📊 Probability Distribution")

            st.markdown(
                '<div class="probability-grid">' +
                "".join(
                    f'<div class="probability-card"><div class="probability-value">{prob:.1f}%</div>'
                    f'<div class="probability-label">{label}</div></div>'
                    for label, prob in result["probabilities"].items()
                )
                + '</div>',
                unsafe_allow_html=True,
            )

            st.write("")
            st.markdown("### ⚡ Live Probability Stream")
            render_live_probability_animation(result)

            st.write("")
            st.markdown("### 📈 Class Probabilities")
            for label, prob in result["probabilities"].items():
                is_pred = label == result["prediction"]
                st.caption(f"**{label}** {'(top prediction)' if is_pred else ''}")
                st.progress(min(int(prob), 100))

            if st.session_state.gradcam_image is not None:
                st.write("")
                st.markdown("### 🔥 Grad-CAM Visualization")
                st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                st.pyplot(st.session_state.gradcam_image, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
                st.caption("Red/yellow regions indicate areas most influential to the model's diagnostic decision.")

    st.write("")


elif nav == "📊 Dashboard":
    st.title("📊 Neurodiagnostic Dashboard · Live")
    st.caption("🔴 Real-time clinical overview — auto-refreshing every 3 seconds")

    render_live_ticker()

    if st.session_state.live_session_start is None:
        st.session_state.live_session_start = datetime.now()

    history = st.session_state.prediction_history

    if not history:
        st.markdown(
            """
            <div class="empty-state">
                <div class="empty-state-icon">📊</div>
                <div class="empty-state-title">Awaiting live diagnostic data</div>
                <div class="empty-state-text">Run an MRI diagnostic analysis to populate the live dashboard.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        total = st.session_state.live_predictions_count
        avg_conf = st.session_state.live_avg_confidence
        unique = len(st.session_state.live_class_counts)
        throughput = st.session_state.live_throughput
        latency = st.session_state.live_latency_ms

        m1, m2, m3, m4, m5 = st.columns(5)
        metrics = [
            ("📈", total, "Total Scans"),
            ("🎯", f"{avg_conf:.1f}%", "Avg. Confidence"),
            ("🧬", unique, "Classes Seen"),
            ("⚡", f"{throughput:.2f}/m", "Throughput"),
            ("⏱️", f"{latency:.0f} ms", "Last Latency"),
        ]
        for col, (icon, value, label) in zip([m1, m2, m3, m4, m5], metrics):
            with col:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <div class="metric-icon">{icon}</div>
                        <div class="metric-value">{value}</div>
                        <div class="metric-label">{label}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        st.write("")
        col_left, col_right = st.columns([2, 1])

        with col_left:
            st.subheader("📊 Live Diagnostic Distribution")
            counts = st.session_state.live_class_counts
            max_count = max(counts.values()) if counts else 1
            for label, count in counts.items():
                pct = (count / max_count) * 100
                live_pct = (count / total) * 100 if total else 0
                st.markdown(
                    f"""
                    <div class="dist-card">
                        <div class="dist-header">
                            <div class="dist-name">{label}</div>
                            <div class="dist-count">{count} analyses · {live_pct:.1f}%</div>
                        </div>
                        <div class="dist-bar-bg">
                            <div class="dist-bar-fill" style="width: {pct:.0f}%;"></div>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.write("")
            st.subheader("📈 Real-Time Certainty Trend")
            trend_fig = plot_confidence_trend(history)
            if trend_fig is not None:
                st.markdown('<div class="chart-container">', unsafe_allow_html=True)
                st.pyplot(trend_fig, use_container_width=True)
                st.markdown('</div>', unsafe_allow_html=True)
                plt.close(trend_fig)
            else:
                st.caption("Need at least 2 analyses to show trend.")

        with col_right:
            st.subheader("📡 Live Activity Feed")
            render_activity_feed()

        st.write("")
        st.subheader("🧠 Latest Diagnostic Report")
        latest = history[-1]
        elapsed_str = "—"
        if st.session_state.live_session_start:
            elapsed_dt = datetime.now() - st.session_state.live_session_start
            elapsed_str = f"{int(elapsed_dt.total_seconds())}s"
        st.markdown(
            f"""
            <div class="latest-card">
                <div class="latest-row"><div class="latest-key">Diagnosis</div>
                    <div class="latest-val">{latest["prediction"]}</div></div>
                <div class="latest-row"><div class="latest-key">AI Certainty</div>
                    <div class="latest-val">{latest['confidence']:.2f}%</div></div>
                <div class="latest-row"><div class="latest-key">Neural Engine</div>
                    <div class="latest-val">{latest["model"]}</div></div>
                <div class="latest-row"><div class="latest-key">Analyzed At</div>
                    <div class="latest-val">{latest["timestamp"]}</div></div>
                <div class="latest-row"><div class="latest-key">Inference Latency</div>
                    <div class="latest-val">{latest.get('latency_ms', 0):.0f} ms</div></div>
                <div class="latest-row"><div class="latest-key">Session Uptime</div>
                    <div class="latest-val">{elapsed_str}</div></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    components.html(
        """
        <script>
        setTimeout(function() {
            const btn = window.parent.document.querySelector('button[aria-label="Rerun"]');
            if (btn) btn.click();
        }, 3000);
        </script>
        """,
        height=0,
    )



elif nav == "🕘 History":
    st.title("🕘 Diagnostic History · Live")
    st.caption("Review past AI diagnostic reports and MRI analyses")
    render_live_ticker()
    history = st.session_state.prediction_history

    if not history:
        st.markdown(
            """
            <div class="empty-state">
                <div class="empty-state-icon">🕘</div>
                <div class="empty-state-title">No diagnostic history</div>
                <div class="empty-state-text">Your AI diagnostic reports will appear here after analysis.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        for item in reversed(history):
            conf = item["confidence"]
            conf_class = (
                "confidence-high" if conf >= 80
                else "confidence-medium" if conf >= 60
                else "confidence-low"
            )
            st.markdown(
                f"""
                <div class="thumbnail-card">
                    <div class="thumbnail-img">🧠</div>
                    <div class="thumbnail-info">
                        <div class="thumbnail-title">{item['prediction']}</div>
                        <div class="thumbnail-meta">{item['timestamp']} · {item['model']}</div>
                    </div>
                    <div class="thumbnail-confidence">
                        <span class="confidence-badge {conf_class}">{conf:.1f}%</span>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            with st.expander(f"View diagnostic details — {item['model']}", expanded=False):
                st.write(f"**Diagnosis:** {item['prediction']}")
                st.write(f"**AI Certainty:** {item['confidence']:.2f}%")
                st.write(f"**Neural Engine:** {item['model']}")
                st.write(f"**Analyzed At:** {item['timestamp']}")
                st.write("**Class Probabilities:**")
                for label, prob in item["probabilities"].items():
                    st.write(f"- {label}: {prob:.2f}%")

        st.write("")
        if st.button("🗑️ Clear History", key="clear_history_button"):
            old_fig = st.session_state.get("gradcam_image")
            if old_fig is not None:
                try:
                    plt.close(old_fig)
                except Exception:
                    pass
            st.session_state.prediction_history = []
            st.session_state.last_result = None
            st.session_state.last_image = None
            st.session_state.gradcam_image = None
            st.rerun()



elif nav == "🔥 Grad-CAM":
    st.title("🔥 Grad-CAM Explainability (XAI) · Live")
    st.caption("Visualize which MRI regions influenced the AI's diagnostic decision")
    render_live_ticker()

    if st.session_state.gradcam_image is None:
        st.markdown(
            """
            <div class="empty-state">
                <div class="empty-state-icon">🔥</div>
                <div class="empty-state-title">No Grad-CAM visualization yet</div>
                <div class="empty-state-text">Run an MRI diagnostic analysis first to generate an explainable AI heatmap.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        tab1, tab2 = st.tabs(["🖼️ Original MRI", "🌡️ Heatmap Overlay"])

        with tab1:
            st.markdown(
                """
                <div class="image-preview-frame medical-scan">
                    <div class="image-preview-label">Original MRI Scan</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.image(st.session_state.last_image, width=400)

        with tab2:
            st.markdown(
                """
                <div class="image-preview-frame medical-scan">
                    <div class="image-preview-label">Grad-CAM Heatmap Overlay</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.pyplot(st.session_state.gradcam_image, use_container_width=True)

        st.write("")
        c1, c2 = st.columns(2)
        with c1:
            if st.session_state.last_result is not None:
                result = st.session_state.last_result
                st.markdown(
                    f"""
                    <div class="info-card">
                        <div style="font-size: 0.75rem; color: var(--text-muted);
                             text-transform: uppercase; letter-spacing: 0.1em;
                             margin-bottom: 0.5rem; font-weight: 600;">Prediction</div>
                        <div style="font-size: 1.25rem; font-weight: 700;
                             color: var(--text-primary);">{result['prediction']}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        with c2:
            if st.session_state.last_result is not None:
                result = st.session_state.last_result
                st.markdown(
                    f"""
                    <div class="info-card">
                        <div style="font-size: 0.75rem; color: var(--text-muted);
                             text-transform: uppercase; letter-spacing: 0.1em;
                             margin-bottom: 0.5rem; font-weight: 600;">Confidence</div>
                        <div style="font-size: 1.5rem; font-weight: 800;
                             color: var(--accent-light);">{result['confidence']:.1f}%</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )



elif nav == "⚙️ Settings":
    st.title("⚙️ System Settings")
    st.caption("Neural diagnostic engine configuration and session management")

    c1, c2 = st.columns(2)

    with c1:
        st.markdown(
            f"""
            <div class="info-card">
                <h3>📋 Neural Engine</h3>
                <p><b>Compute Device:</b> {DEVICE}</p>
                <p><b>Architecture:</b> {model_name if model_name else 'Unavailable'}</p>
                <p><b>Model Status:</b> {'Loaded' if MODEL_PATH.exists() else 'Not found'}</p>
                <p><b>Input Resolution:</b> {IMG_SIZE} × {IMG_SIZE}</p>
                <p><b>Diagnostic Classes:</b> {', '.join(class_names) if class_names else 'Unavailable'}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            """
            <div class="info-card">
                <h3>🖥️ System Information</h3>
                <p><b>Platform:</b> Streamlit</p>
                <p><b>Framework:</b> PyTorch (Neural Network)</p>
                <p><b>Backend:</b> CPU / CUDA</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.write("")
        st.markdown(
            """
            <div class="disclaimer">
                <b>⚠️ Reset Session</b><br>
                Clearing all diagnostic reports and AI results is permanent.
            </div>
            """,
            unsafe_allow_html=True,
        )

        if st.button("🧹 Reset Session", key="reset_session_button",
                     use_container_width=True, type="primary"):
            old_fig = st.session_state.get("gradcam_image")
            if old_fig is not None:
                try:
                    plt.close(old_fig)
                except Exception:
                    pass
            for key, value in defaults.items():
                st.session_state[key] = value
            st.success("Session cleared. All diagnostic reports reset.")
            st.rerun()


def render_footer():
    year = datetime.now().year
    build_time = datetime.now().strftime("%Y-%m-%d %H:%M UTC")
    total_scans = st.session_state.live_predictions_count
    avg_conf = st.session_state.live_avg_confidence
    engine_name = model_name or "—"
    engine_status = "Online" if (model_error is None and MODEL_PATH.exists()) else "Unavailable"

    st.markdown(
        f"""
        <div class="app-footer">
            <div class="footer-left">
                <div class="footer-brand">🧠 NeuroLens AI</div>
                <span class="footer-divider">·</span>
                <span>© {year} NeuroLens Diagnostics</span>
                <span class="footer-divider">·</span>
                <span>Build {build_time}</span>
            </div>
            <div class="footer-meta">
                <span><b>Engine:</b> {engine_name}</span>
                <span class="footer-divider">·</span>
                <span><b>Scans:</b> {total_scans}</span>
                <span class="footer-divider">·</span>
                <span><b>Avg Conf:</b> {avg_conf:.1f}%</span>
                <span class="footer-divider">·</span>
                <span style="display:inline-flex; align-items:center; gap:0.4rem;">
                    <span class="footer-dot"></span> {engine_status}
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


render_footer()

