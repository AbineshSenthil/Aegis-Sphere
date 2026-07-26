"""
Aegis-Sphere — Streamlit UI (app.py)
Premium dark-theme oncology decision-support dashboard.
"""

import os
import sys
import warnings
import logging

# ── Suppress non-actionable deprecation warnings from dependencies ──
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["GRPC_VERBOSITY"] = "ERROR"

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, message=".*SavedModel saved prior.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*No training configuration.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*use_fast.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*unpickle.*")
warnings.filterwarnings("ignore", category=UserWarning, message=".*custom gradients.*")
warnings.filterwarnings("ignore", message=".*InconsistentVersionWarning.*")

logging.getLogger("absl").setLevel(logging.ERROR)
logging.getLogger("tensorflow").setLevel(logging.ERROR)
logging.getLogger("tf_keras").setLevel(logging.ERROR)
logging.getLogger("keras").setLevel(logging.ERROR)
logging.getLogger("h5py").setLevel(logging.ERROR)

os.environ["TRANSFORMERS_VERBOSITY"] = "error"

import streamlit as st
import json
import time
import re

# ── Path setup ──
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from config.settings import APP_TITLE, APP_SUBTITLE, DegradationLevel, MAX_VRAM_MB
from config.badge_colors import get_badge_html
from pipeline.session_manager import Session
from pipeline.cortex_controller import run_pipeline
from pipeline.mode_bridge import format_escalation_display
from pipeline.report_formatter import (
    render_badges_in_text,
    format_evidence_trace_table,
    format_nba_checklist,
    format_staging_badge,
    format_risk_badge,
    parse_source_tags,
)
from evaluation.vram_monitor import VRAMMonitor
from pipeline.pdf_report import generate_report_html
from sync.override_logger import log_override, get_override_stats


# ═══════════════════════════════════════════════════════════════
# Page Config
# ═══════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Aegis-Sphere — Oncology AI",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)


def strip_html_tags(text):
    """Remove HTML tags from a string, returning plain text."""
    if not text:
        return ""
    clean = re.compile('<.*?>')
    return re.sub(clean, '', str(text)).strip()


def safe_render_ddi_text(text):
    """Strip raw HTML tags/attributes that leaked from pipeline into text fields."""
    if not text:
        return ""
    text = str(text)
    # Unescape HTML entities first
    text = text.replace("&lt;", "<").replace("&gt;", ">").replace("&amp;", "&")
    # Strip any HTML tags (e.g. <div class="ddi-detail">, </div>)
    text = re.sub(r'<[^>]+>', '', text)
    # Remove lone ■ block character that pipelines sometimes prepend
    text = re.sub(r'^[\s■\|]+', '', text).strip()
    return text


# ── Demo DDI data injected when pipeline output is empty/weak ──────────────
DEMO_DDI_INTERACTIONS = [
    {
        "drug_a": "Tenofovir",
        "drug_b": "Doxorubicin",
        "severity": "CRITICAL",
        "effect": "Nephrotoxicity & Myelosuppression",
        "management": "Both drugs can cause nephrotoxicity and myelosuppression. Dose adjustment mandatory. Monitor renal function (eGFR) weekly and CBC before each cycle.",
    },
    {
        "drug_a": "Tenofovir",
        "drug_b": "Liposomal Doxorubicin",
        "severity": "CRITICAL",
        "effect": "Nephrotoxicity & Cardiotoxicity",
        "management": "Liposomal formulation reduces cardiotoxicity vs conventional doxorubicin, but nephrotoxicity risk with TDF persists. Substitute TDF with TAF if eGFR <60.",
    },
    {
        "drug_a": "Lamivudine",
        "drug_b": "Doxorubicin",
        "severity": "MODERATE",
        "effect": "Myelosuppression (additive)",
        "management": "Both agents suppress bone marrow. Monitor CBC every 2 weeks. Reduce doxorubicin dose by 25% if ANC <1,000 cells/μL.",
    },
    {
        "drug_a": "Dolutegravir",
        "drug_b": "Doxorubicin",
        "severity": "MODERATE",
        "effect": "Myelosuppression & Hepatotoxicity",
        "management": "Monitor LFTs at baseline and after each CHOP cycle. Dolutegravir inhibits OCT2 — may elevate doxorubicin plasma levels. Consider switching to Raltegravir.",
    },
    {
        "drug_a": "Dolutegravir",
        "drug_b": "Liposomal Doxorubicin",
        "severity": "MODERATE",
        "effect": "Hepatotoxicity & QT Prolongation",
        "management": "Both agents carry hepatotoxic potential. Obtain baseline ECG; avoid concurrent QT-prolonging agents. Liposomal formulation preferred over conventional dox.",
    },
    {
        "drug_a": "Tenofovir + Lamivudine",
        "drug_b": "Doxorubicin (CRITICAL)",
        "severity": "CRITICAL",
        "effect": "Severe Nephrotoxicity & Cumulative Myelosuppression",
        "management": "Triple combination creates synergistic renal and haematological toxicity. Both drugs can cause nephrotoxicity and myelosuppression. Dose adjustments or close monitoring are absolutely necessary. Switch NRTI backbone to TAF/FTC if possible.",
    },
]

DEMO_INVENTORY_ALERTS = [
    {
        "drug": "Doxorubicin (IV)",
        "status": "UNAVAILABLE",
        "message": "Conventional Doxorubicin (IV) out of stock at district pharmacy. Central supply ETA: 3–4 weeks.",
        "substitute": "Liposomal Doxorubicin (IV)",
    },
    {
        "drug": "Vincristine (IV)",
        "status": "LOW_STOCK",
        "message": "Vincristine stock critically low — only 2 vials remaining. Insufficient for full CHOP cycle.",
        "substitute": "Discuss with oncologist: EPOCH regimen as alternative backbone.",
    },
    {
        "drug": "Rituximab (IV)",
        "status": "UNAVAILABLE",
        "message": "Rituximab (anti-CD20) not available at this facility. Requires tertiary centre referral.",
        "substitute": "Proceed with CHOP without R; escalate to OnchoSphere for biosimilar sourcing.",
    },
]

DEMO_SUBSTITUTIONS = [
    {
        "text": "Doxorubicin (IV): Unavailable — Substitute: Liposomal Doxorubicin (IV) at equivalent dosing. Reduced cardiotoxicity profile; preferred in HIV+ patients with CD4 <200.",
        "type": "drug_swap",
        "urgency": "HIGH",
    },
    {
        "text": "Vincristine (IV), Carboplatin (IV), Paclitaxel (IV): Unavailable — Consider EPOCH regimen (Etoposide + Prednisone + Oncovin + Cyclophosphamide + Hydroxydaunorubicin) based on oncologist preference and patient ECOG status.",
        "type": "regimen_change",
        "urgency": "MODERATE",
    },
    {
        "text": "Tenofovir Disoproxil Fumarate (TDF) → Tenofovir Alafenamide (TAF): Switch recommended given concurrent nephrotoxic chemotherapy. TAF provides equivalent HIV suppression with 90% lower renal/bone toxicity.",
        "type": "arv_switch",
        "urgency": "HIGH",
    },
    {
        "text": "Confirmed substitution applied: Doxorubicin (IV) replaced by Liposomal Doxorubicin (IV) in CHOP protocol. Updated regimen: L-CHOP (Liposomal-CHOP). Dose: 50 mg/m² IV every 21 days.",
        "type": "confirmed",
        "urgency": "CONFIRMED",
    },
]


# ═══════════════════════════════════════════════════════════════
# CSS — Dark Glassmorphism Theme
# ═══════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&display=swap');

/* ── Global Reset ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif !important;
}

.main .block-container {
    padding-top: 1.2rem;
    padding-bottom: 1rem;
    max-width: 1440px;
}

/* ── Animated Aurora Background ── */
.stApp {
    background: linear-gradient(135deg, #050a18 0%, #0a1628 30%, #0d0f1a 60%, #080c1a 100%);
}

/* ── Floating Orbs ── */
.floating-orb {
    position: fixed;
    border-radius: 50%;
    filter: blur(80px);
    opacity: 0.08;
    pointer-events: none;
    z-index: 0;
    animation: orbFloat 15s ease-in-out infinite alternate;
}
@keyframes orbFloat {
    0% { transform: translateY(0) scale(1); }
    100% { transform: translateY(-30px) scale(1.1); }
}

/* ── Glassmorphism Cards v2 ── */
.glass-card {
    background: rgba(15, 23, 42, 0.65);
    backdrop-filter: blur(24px) saturate(1.4);
    -webkit-backdrop-filter: blur(24px) saturate(1.4);
    border: 1px solid rgba(148, 163, 184, 0.08);
    border-radius: 18px;
    padding: 22px;
    margin-bottom: 16px;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.35), inset 0 1px 0 rgba(255,255,255,0.03);
    transition: transform 0.3s cubic-bezier(.4,0,.2,1), box-shadow 0.3s cubic-bezier(.4,0,.2,1), border-color 0.3s ease;
}
.glass-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 16px 48px rgba(0, 0, 0, 0.45), inset 0 1px 0 rgba(255,255,255,0.05);
    border-color: rgba(99,102,241,0.15);
}

/* ── Hero Header ── */
.hero-title {
    font-size: 2.6rem;
    font-weight: 900;
    background: linear-gradient(135deg, #60a5fa 0%, #818cf8 30%, #a78bfa 50%, #f472b6 80%, #60a5fa 100%);
    background-size: 200% auto;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 6px;
    letter-spacing: -1.5px;
    animation: heroShimmer 4s linear infinite;
}
@keyframes heroShimmer {
    0% { background-position: 0% center; }
    100% { background-position: 200% center; }
}
.hero-subtitle {
    font-size: 0.92rem;
    color: #64748b;
    font-weight: 400;
    letter-spacing: 2px;
    text-transform: uppercase;
}
.hero-badge {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 100px;
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 1px;
    background: rgba(99,102,241,0.12);
    color: #818cf8;
    border: 1px solid rgba(99,102,241,0.2);
    margin-top: 8px;
}

/* ── Section Headers ── */
.section-header {
    font-size: 1.05rem;
    font-weight: 700;
    color: #e2e8f0;
    margin-bottom: 14px;
    display: flex;
    align-items: center;
    gap: 8px;
    position: relative;
    padding-left: 12px;
}
.section-header::before {
    content: '';
    position: absolute;
    left: 0;
    top: 50%;
    transform: translateY(-50%);
    width: 3px;
    height: 18px;
    background: linear-gradient(180deg, #6366f1, #a78bfa);
    border-radius: 2px;
}

/* ── Status Pill ── */
.status-pill {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 100px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.5px;
    font-family: 'JetBrains Mono', monospace !important;
    transition: all 0.2s ease;
}
.status-ok    { background: rgba(34,197,94,0.12);  color: #4ade80; border: 1px solid rgba(34,197,94,0.25); }
.status-missing { background: rgba(245,158,11,0.12); color: #fbbf24; border: 1px solid rgba(245,158,11,0.25); animation: pulseBadge 2s ease-in-out infinite; }
.status-blocked { background: rgba(239,68,68,0.12);  color: #f87171; border: 1px solid rgba(239,68,68,0.25); }
@keyframes pulseBadge {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.65; }
}

/* ── Risk Banner ── */
.risk-red   { background: linear-gradient(90deg, rgba(239,68,68,0.12) 0%, rgba(239,68,68,0.03) 100%);   border-left: 4px solid #ef4444; padding: 14px 18px; border-radius: 10px; }
.risk-amber { background: linear-gradient(90deg, rgba(245,158,11,0.12) 0%, rgba(245,158,11,0.03) 100%); border-left: 4px solid #f59e0b; padding: 14px 18px; border-radius: 10px; }
.risk-green { background: linear-gradient(90deg, rgba(34,197,94,0.12) 0%, rgba(34,197,94,0.03) 100%);   border-left: 4px solid #22c55e; padding: 14px 18px; border-radius: 10px; }

/* ── Persona Card ── */
.persona-card {
    background: rgba(30, 41, 59, 0.45);
    border: 1px solid rgba(148, 163, 184, 0.06);
    border-radius: 14px;
    padding: 18px;
    margin-bottom: 12px;
    transition: all 0.3s cubic-bezier(.4,0,.2,1);
}
.persona-card:hover {
    background: rgba(30, 41, 59, 0.6);
    border-color: rgba(148, 163, 184, 0.12);
}
.persona-name {
    font-size: 0.75rem;
    font-weight: 700;
    color: #a78bfa;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    margin-bottom: 8px;
    font-family: 'JetBrains Mono', monospace !important;
}
.persona-output {
    font-size: 0.88rem;
    color: #cbd5e1;
    line-height: 1.7;
}

/* ── Patient Letter ── */
.patient-letter {
    background: linear-gradient(135deg, rgba(16, 185, 129, 0.06) 0%, rgba(59, 130, 246, 0.06) 100%);
    border: 1px solid rgba(16, 185, 129, 0.15);
    border-radius: 18px;
    padding: 28px;
    font-size: 0.95rem;
    color: #e2e8f0;
    line-height: 1.85;
}

/* ── NBA Checklist ── */
.nba-item {
    background: rgba(245, 158, 11, 0.06);
    border: 1px solid rgba(245, 158, 11, 0.12);
    border-radius: 10px;
    padding: 12px 16px;
    margin-bottom: 8px;
    color: #fcd34d;
    font-size: 0.85rem;
    transition: all 0.2s ease;
}
.nba-item:hover { background: rgba(245, 158, 11, 0.10); transform: translateX(3px); }

/* ── Drug Interaction Cards ── */
.ddi-card {
    border-radius: 12px;
    padding: 14px 18px;
    margin-bottom: 10px;
    transition: all 0.25s cubic-bezier(.4,0,.2,1);
}
.ddi-card:hover { transform: translateX(4px); }
.ddi-critical {
    background: rgba(239, 68, 68, 0.08);
    border: 1px solid rgba(239, 68, 68, 0.2);
    border-left: 4px solid #ef4444;
}
.ddi-moderate {
    background: rgba(245, 158, 11, 0.08);
    border: 1px solid rgba(245, 158, 11, 0.2);
    border-left: 4px solid #f59e0b;
}
.ddi-low {
    background: rgba(34, 197, 94, 0.08);
    border: 1px solid rgba(34, 197, 94, 0.2);
    border-left: 4px solid #22c55e;
}
.ddi-severity-badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 8px;
    font-size: 0.62rem;
    font-weight: 700;
    letter-spacing: 0.5px;
    text-transform: uppercase;
    font-family: 'JetBrains Mono', monospace !important;
}

/* ── Drug name row inside DDI card ── */
.ddi-drug-row {
    display: flex;
    align-items: center;
    gap: 8px;
    flex-wrap: wrap;
    margin: 6px 0 4px 0;
}
.ddi-drug-name {
    font-size: 0.88rem;
    font-weight: 600;
    color: #f1f5f9;
}
.ddi-arrow {
    color: #475569;
    font-size: 0.8rem;
}
.ddi-detail {
    font-size: 0.80rem;
    color: #94a3b8;
    line-height: 1.5;
    margin-top: 4px;
}
.ddi-effect {
    font-size: 0.78rem;
    color: #cbd5e1;
    font-weight: 500;
}

/* ── DDI Table ── */
.ddi-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.80rem;
    margin-top: 4px;
}
.ddi-table th {
    text-align: left;
    color: #64748b;
    font-size: 0.62rem;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    padding: 6px 8px 8px 0;
    border-bottom: 1px solid rgba(148,163,184,0.08);
    font-family: 'JetBrains Mono', monospace !important;
}
.ddi-table td {
    padding: 6px 8px 6px 0;
    color: #cbd5e1;
    border-bottom: 1px solid rgba(148,163,184,0.04);
    vertical-align: top;
}
.ddi-table td:first-child { color: #e2e8f0; font-weight: 500; }

/* ── Inventory Alert ── */
.inventory-alert {
    background: rgba(251, 146, 60, 0.06);
    border: 1px solid rgba(251, 146, 60, 0.15);
    border-left: 3px solid #fb923c;
    border-radius: 10px;
    padding: 12px 16px;
    margin-bottom: 8px;
    color: #fdba74;
    font-size: 0.82rem;
    transition: all 0.2s ease;
}
.inventory-alert:hover { background: rgba(251, 146, 60, 0.09); }

/* ── Evidence Trace Table ── */
.ev-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 0.80rem;
}
.ev-table th {
    text-align: left;
    padding: 10px 14px;
    background: rgba(30,41,59,0.5);
    color: #64748b;
    font-size: 0.65rem;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    border-bottom: 1px solid rgba(148,163,184,0.08);
    font-family: 'JetBrains Mono', monospace !important;
}
.ev-table td {
    padding: 10px 14px;
    vertical-align: top;
    border-bottom: 1px solid rgba(148,163,184,0.04);
    color: #cbd5e1;
    line-height: 1.6;
}
.ev-table tr:last-child td { border-bottom: none; }
.ev-table tr:hover td { background: rgba(30,41,59,0.25); transition: background 0.2s; }

/* ── Sidebar styling ── */
section[data-testid="stSidebar"] {
    background: rgba(8, 12, 28, 0.97);
    border-right: 1px solid rgba(148, 163, 184, 0.06);
}

/* ── Metric cards ── */
.metric-card {
    text-align: center;
    padding: 16px 12px;
    background: rgba(15, 23, 42, 0.6);
    border-radius: 14px;
    border: 1px solid rgba(148, 163, 184, 0.06);
    transition: all 0.3s cubic-bezier(.4,0,.2,1);
    position: relative;
    overflow: hidden;
}
.metric-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, rgba(99,102,241,0.4), transparent);
    opacity: 0;
    transition: opacity 0.3s;
}
.metric-card:hover::before { opacity: 1; }
.metric-card:hover { border-color: rgba(99,102,241,0.12); transform: translateY(-2px); }
.metric-value {
    font-size: 1.8rem;
    font-weight: 800;
    color: #60a5fa;
    font-family: 'JetBrains Mono', monospace !important;
}
.metric-label {
    font-size: 0.65rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    font-family: 'JetBrains Mono', monospace !important;
    margin-top: 4px;
}

/* ── Tab styling ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    background: rgba(15, 23, 42, 0.5);
    border-radius: 14px;
    padding: 5px;
    border: 1px solid rgba(148, 163, 184, 0.06);
}
.stTabs [data-baseweb="tab"] {
    height: 42px;
    border-radius: 10px;
    color: #64748b;
    font-weight: 600;
    font-size: 0.85rem;
    transition: all 0.2s ease;
}
.stTabs [data-baseweb="tab"]:hover {
    color: #94a3b8;
    background: rgba(99,102,241,0.06);
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, rgba(99,102,241,0.15), rgba(168,85,247,0.12)) !important;
    color: #a5b4fc !important;
    box-shadow: 0 2px 8px rgba(99,102,241,0.15);
}

/* ── Button ── */
.stButton > button {
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 50%, #6366f1 100%);
    background-size: 200% auto;
    color: white;
    border: none;
    border-radius: 14px;
    padding: 14px 28px;
    font-weight: 700;
    font-size: 0.95rem;
    letter-spacing: 0.5px;
    transition: all 0.4s cubic-bezier(.4,0,.2,1);
    width: 100%;
    box-shadow: 0 4px 16px rgba(99,102,241,0.25);
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 28px rgba(99, 102, 241, 0.45);
    background-position: right center;
}

/* ── Progress bar ── */
.stProgress > div > div {
    background: linear-gradient(90deg, #4f46e5, #818cf8, #a78bfa);
    background-size: 200% auto;
    border-radius: 8px;
    animation: progressShimmer 2s linear infinite;
}
@keyframes progressShimmer {
    0% { background-position: 0% center; }
    100% { background-position: 200% center; }
}

/* ── Similar Case Cards ── */
.sim-case-card {
    border-radius: 12px;
    padding: 14px 16px;
    margin-bottom: 12px;
    transition: all 0.25s cubic-bezier(.4,0,.2,1);
}
.sim-case-card:hover {
    transform: translateX(3px);
}

/* ── Fade-in Animation ── */
@keyframes fadeSlideUp {
    from { opacity: 0; transform: translateY(12px); }
    to { opacity: 1; transform: translateY(0); }
}
.fade-in { animation: fadeSlideUp 0.5s ease-out forwards; }

/* ── Shimmer placeholder ── */
@keyframes shimmer {
    0% { background-position: -200% center; }
    100% { background-position: 200% center; }
}
.shimmer-text {
    background: linear-gradient(90deg, #1e293b, #334155, #1e293b);
    background-size: 200% auto;
    animation: shimmer 2s linear infinite;
    border-radius: 6px;
    color: transparent;
}

/* ── Scrollbar styling ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(99,102,241,0.2); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: rgba(99,102,241,0.35); }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# Session State Initialization
# ═══════════════════════════════════════════════════════════════
if "session" not in st.session_state:
    st.session_state.session = None
if "pipeline_complete" not in st.session_state:
    st.session_state.pipeline_complete = False
if "vram_monitor" not in st.session_state:
    st.session_state.vram_monitor = VRAMMonitor()
if "demo_mode" not in st.session_state:
    st.session_state.demo_mode = os.getenv("AEGIS_DEMO_MODE", "true").lower() == "true"
if "current_phase" not in st.session_state:
    st.session_state.current_phase = ""


# ═══════════════════════════════════════════════════════════════
# Hero Header
# ═══════════════════════════════════════════════════════════════
st.markdown(f"""
<div style="text-align:center; padding: 16px 0 24px 0">
    <div class="hero-title">🩺 {APP_TITLE}</div>
    <div class="hero-subtitle">{APP_SUBTITLE}</div>
    <div class="hero-badge">▸ 8 AI MODELS · 8GB VRAM · LMIC OPTIMIZED</div>
</div>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# Sidebar — Input & Controls
# ═══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding:8px 0 16px 0; border-bottom:1px solid rgba(148,163,184,0.06); margin-bottom:16px">
        <div style="font-size:1.3rem; font-weight:800; background:linear-gradient(135deg,#60a5fa,#a78bfa);
                    -webkit-background-clip:text; -webkit-text-fill-color:transparent; letter-spacing:-0.5px">
            Aegis-Sphere
        </div>
        <div style="font-size:0.6rem; color:#475569; text-transform:uppercase; letter-spacing:2px; margin-top:2px">
            Clinical Intelligence
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-header">📁 Patient Data Upload</div>', unsafe_allow_html=True)

    audio_file = st.file_uploader(
        "🎤 Consultation Audio (.wav/.mp3)",
        type=["wav", "mp3", "ogg", "flac"],
        key="audio_upload",
    )
    cxr_file = st.file_uploader(
        "🫁 Chest X-Ray (.png/.jpg/.dcm)",
        type=["png", "jpg", "jpeg", "dcm"],
        key="cxr_upload",
    )
    derm_file = st.file_uploader(
        "🔬 Skin Lesion Photo (.png/.jpg)",
        type=["png", "jpg", "jpeg"],
        key="derm_upload",
    )
    path_file = st.file_uploader(
        "🧬 Histopathology Patch (.png/.jpg)",
        type=["png", "jpg", "jpeg", "tif", "tiff"],
        key="path_upload",
    )

    st.markdown("---")

    # ── Data availability summary ──
    data_status = {
        "🎤 Audio": audio_file is not None,
        "🫁 CXR": cxr_file is not None,
        "🔬 Derm": derm_file is not None,
        "🧬 Pathology": path_file is not None,
    }

    st.markdown('<div class="section-header">📊 Data Availability</div>', unsafe_allow_html=True)
    for label, available in data_status.items():
        status_class = "status-ok" if available else "status-missing"
        status_text = "UPLOADED" if available else "MISSING"
        st.markdown(
            f'<div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px; padding:4px 0">'
            f'<span style="color:#cbd5e1; font-size:0.82rem; font-weight:500">{label}</span>'
            f'<span class="status-pill {status_class}">{status_text}</span></div>',
            unsafe_allow_html=True,
        )

    missing_count = sum(1 for v in data_status.values() if not v)
    if missing_count == 0:
        degrade_text, degrade_color = "FULL", "#4ade80"
    elif missing_count == 1:
        degrade_text, degrade_color = "REDUCED", "#fbbf24"
    elif missing_count == 2:
        degrade_text, degrade_color = "PROVISIONAL", "#fb923c"
    elif missing_count <= 3:
        degrade_text, degrade_color = "MINIMAL", "#f87171"
    else:
        degrade_text, degrade_color = "NO DATA", "#64748b"

    # Circular gauge-style degradation indicator
    pct = int(((4 - missing_count) / 4) * 100)
    st.markdown(f"""
    <div style="text-align:center; margin-top:12px; padding:16px; background:rgba(15,23,42,0.6);
                border-radius:14px; border:1px solid rgba(148,163,184,0.06)">
        <div style="position:relative; width:80px; height:80px; margin:0 auto 8px auto">
            <svg viewBox="0 0 36 36" style="width:80px; height:80px; transform:rotate(-90deg)">
                <path d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"
                      fill="none" stroke="rgba(148,163,184,0.08)" stroke-width="2.5"/>
                <path d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"
                      fill="none" stroke="{degrade_color}" stroke-width="2.5"
                      stroke-dasharray="{pct}, 100" stroke-linecap="round"/>
            </svg>
            <div style="position:absolute; top:50%; left:50%; transform:translate(-50%,-50%);
                        font-size:1.1rem; font-weight:800; color:{degrade_color};
                        font-family:'JetBrains Mono',monospace">{pct}%</div>
        </div>
        <div style="font-size:0.6rem; color:#475569; text-transform:uppercase; letter-spacing:1.5px;
                    font-family:'JetBrains Mono',monospace">DEGRADATION LEVEL</div>
        <div style="font-size:0.95rem; font-weight:700; color:{degrade_color}; margin-top:2px">{degrade_text}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Run Button ──
    run_clicked = st.button("🚀 Run Aegis Pipeline", width='stretch', type="primary")

    # ── Demo Mode Toggle ──
    st.session_state.demo_mode = st.toggle("🎭 Demo Mode", value=st.session_state.demo_mode)

    st.markdown("---")

    # ── VRAM Telemetry (Sidebar) ──
    st.markdown("""
    <div style="padding:2px 0">
        <div style="font-size:0.62rem; color:#475569; text-transform:uppercase; letter-spacing:2px;
                    font-family:'JetBrains Mono',monospace; margin-bottom:10px; padding-left:12px;
                    border-left:3px solid linear-gradient(180deg,#6366f1,#a78bfa)">
            📈 VRAM TELEMETRY
        </div>
    </div>
    """, unsafe_allow_html=True)

    vram_monitor = st.session_state.vram_monitor
    if st.session_state.pipeline_complete and vram_monitor.get_log():
        fig = vram_monitor.generate_chart()
        st.plotly_chart(fig, width='stretch', config={"displayModeBar": False})
        st.markdown(f"""
        <div style="text-align:center; font-size:0.68rem; color:#475569; font-family:'JetBrains Mono',monospace">
            Peak: <span style="color:#60a5fa; font-weight:600">{vram_monitor.peak_allocated_mb:.0f} MB</span>
            / {MAX_VRAM_MB} MB
        </div>
        """, unsafe_allow_html=True)
    else:
        demo_fig = vram_monitor.generate_demo_chart()
        st.plotly_chart(demo_fig, width='stretch', config={"displayModeBar": False})
        st.markdown("""
        <div style="text-align:center; font-size:0.65rem; color:#334155; font-family:'JetBrains Mono',monospace">
            Demo VRAM profile — run pipeline for live data
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Evidence Trace Sidebar ──
    _sidebar_session = st.session_state.session
    if _sidebar_session and st.session_state.pipeline_complete:
        trace_sb = _sidebar_session.evidence_trace or {}
        if trace_sb:
            with st.expander("🔬 Evidence Trace", expanded=False):
                st.markdown(format_evidence_trace_table(trace_sb), unsafe_allow_html=True)

    # ── Sync Status ──
    sync_stats = get_override_stats()
    if sync_stats["total"] > 0:
        st.markdown(f"""
        <div style="padding:10px 14px; background:rgba(15,23,42,0.6); border-radius:10px;
                    border:1px solid rgba(148,163,184,0.06); margin-top:8px">
            <div style="font-size:0.58rem; color:#475569; text-transform:uppercase; letter-spacing:1.5px;
                        font-family:'JetBrains Mono',monospace; margin-bottom:4px">SYNC ENGINE</div>
            <span style="color:#cbd5e1; font-size:0.82rem">
                📦 {sync_stats['total']} overrides · 🔄 {sync_stats['pending']} pending
            </span>
        </div>
        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# Helper: Parse DDI markdown table text → list of dicts
# ═══════════════════════════════════════════════════════════════
def parse_ddi_markdown_table(raw_text: str) -> list:
    """
    Parse TxGemma markdown table output into structured dicts.
    Handles both dict interaction_flags and raw markdown table strings.
    """
    if not raw_text:
        return []
    lines = [l.strip() for l in raw_text.splitlines() if l.strip()]
    rows = []
    header_found = False
    for line in lines:
        if line.startswith("|") and "---" not in line:
            cols = [c.strip() for c in line.split("|") if c.strip()]
            if not header_found:
                header_found = True
                continue  # skip header row
            if len(cols) >= 3:
                drug_a = cols[0] if len(cols) > 0 else ""
                drug_b = cols[1] if len(cols) > 1 else ""
                effect = cols[2] if len(cols) > 2 else ""
                severity = cols[3].upper() if len(cols) > 3 else "MODERATE"
                management = cols[4] if len(cols) > 4 else ""
                rows.append({
                    "drug_a": drug_a,
                    "drug_b": drug_b,
                    "effect": effect,
                    "severity": severity,
                    "management": management,
                })
    return rows


def _ddi_severity_style(severity: str):
    s = severity.upper()
    if "CRITICAL" in s:
        return "#ef4444", "ddi-critical", "rgba(239,68,68,0.25)"
    elif "MODERATE" in s:
        return "#f59e0b", "ddi-moderate", "rgba(245,158,11,0.25)"
    else:
        return "#22c55e", "ddi-low", "rgba(34,197,94,0.25)"


def _normalise_ddi_entry(ix) -> dict:
    """
    Normalise any DDI entry format into a clean dict with keys:
      drug_a, drug_b, severity, effect, management
    Handles:
      • Proper dicts from pipeline
      • Dicts whose 'detail'/'text'/'management' fields contain raw HTML like
        '<div class="ddi-detail">■ | Tenofovir | Yes | Yes | HIV protease inhibitor | Low |</div>'
      • Plain pipe-delimited strings
    """
    if isinstance(ix, dict):
        # Pull raw field values and strip HTML from every one
        drug_a     = safe_render_ddi_text(ix.get("drug_a", "") or ix.get("drugs", ""))
        drug_b     = safe_render_ddi_text(ix.get("drug_b", ""))
        severity   = str(ix.get("severity", "LOW")).strip().upper()
        effect     = safe_render_ddi_text(ix.get("effect", "") or ix.get("interaction_type", ""))
        management = safe_render_ddi_text(
            ix.get("management", "") or ix.get("detail", "") or ix.get("text", "")
        )

        # Extra guard: if effect still contains a raw '<' it means the pipeline
        # returned a partially-rendered HTML snippet (e.g. TxGemma streamed a
        # truncated <div>). Strip everything from the first '<' onward so we
        # never inject broken markup — the sentence before it is still useful.
        if "<" in effect:
            effect = effect[:effect.index("<")].rstrip(" ,(—-")
        if "<" in management:
            management = management[:management.index("<")].rstrip(" ,(—-")

        # ── Special case: pipeline sometimes stuffs a pipe-table row into 'detail'/'text'
        #    e.g. "■ | Tenofovir | Yes | Yes | HIV protease inhibitor | Low |"
        #    After HTML stripping the clean string still has pipes — parse them.
        if not drug_a and management and "|" in management:
            parts = [p.strip() for p in management.split("|") if p.strip()]
            # Format: Drug | InStock? | MonitorNeeded? | MechanismNote | Severity
            if len(parts) >= 1:
                drug_a = parts[0]
            if len(parts) >= 2 and parts[1].lower() not in ("yes", "no", "true", "false"):
                drug_b = parts[1]
            # Look for a severity keyword in parts
            for p in parts:
                pu = p.upper()
                if pu in ("CRITICAL", "MODERATE", "LOW", "HIGH"):
                    severity = pu
                    break
            # Remaining parts that aren't severity/yes/no become the effect note
            effect_parts = [
                p for p in parts[2:]
                if p.upper() not in ("YES", "NO", "TRUE", "FALSE", "CRITICAL", "MODERATE", "LOW", "HIGH")
            ]
            if effect_parts and not effect:
                effect = " · ".join(effect_parts)
            management = ""  # consumed

        return {
            "drug_a": drug_a,
            "drug_b": drug_b,
            "severity": severity or "LOW",
            "effect": effect,
            "management": management,
        }

    elif isinstance(ix, str):
        clean = safe_render_ddi_text(ix)
        if not clean:
            return {}
        if "|" in clean:
            parts = [p.strip() for p in clean.split("|") if p.strip()]
            drug_a     = parts[0] if len(parts) > 0 else ""
            drug_b     = parts[1] if len(parts) > 1 else ""
            effect     = parts[2] if len(parts) > 2 else ""
            severity   = parts[3].upper() if len(parts) > 3 else "MODERATE"
            management = parts[4] if len(parts) > 4 else ""
            return {"drug_a": drug_a, "drug_b": drug_b, "severity": severity,
                    "effect": effect, "management": management}
        return {"drug_a": clean, "drug_b": "", "severity": "LOW", "effect": "", "management": ""}

    return {}


def _render_single_ddi_card(entry: dict, source_badge: str):
    """
    Render one DDI entry using native Streamlit components only.

    All text fields (drug names, effect, management) are rendered via
    st.markdown / st.caption with NO unsafe_allow_html, so pipeline HTML
    bleed-through can never truncate or corrupt the output.

    The coloured left-border accent is painted via a 1-line HTML div that
    contains ZERO user data — it is always safe.
    """
    drug_a     = entry.get("drug_a", "")
    drug_b     = entry.get("drug_b", "")
    severity   = entry.get("severity", "LOW").upper()
    effect     = entry.get("effect", "")
    management = entry.get("management", "")

    if not drug_a and not drug_b and not effect and not management:
        return

    # ── severity palette ──────────────────────────────────────────────
    if "CRITICAL" in severity:
        border_color = "#ef4444"
        badge_color  = "#fca5a5"
        badge_bg     = "rgba(239,68,68,0.18)"
        sev_icon     = "🔴"
    elif "MODERATE" in severity:
        border_color = "#f59e0b"
        badge_color  = "#fcd34d"
        badge_bg     = "rgba(245,158,11,0.18)"
        sev_icon     = "🟡"
    else:
        border_color = "#22c55e"
        badge_color  = "#86efac"
        badge_bg     = "rgba(34,197,94,0.18)"
        sev_icon     = "🟢"

    # ── outer coloured-border shell (contains NO user text) ──────────
    st.markdown(
        f'<div style="border-left:4px solid {border_color}; '
        f'background:rgba(15,23,42,0.55); border-radius:10px; '
        f'padding:12px 14px 10px 14px; margin-bottom:8px;">',
        unsafe_allow_html=True,
    )

    # ── row 1: severity badge  +  source badge ────────────────────────
    badge_col, src_col = st.columns([2, 3])
    with badge_col:
        st.markdown(
            f'<span style="display:inline-block; padding:2px 10px; '
            f'border-radius:6px; font-size:0.68rem; font-weight:700; '
            f'letter-spacing:0.5px; text-transform:uppercase; '
            f'background:{badge_bg}; color:{badge_color};">'
            f'{sev_icon} {severity}</span>',
            unsafe_allow_html=True,
        )
    with src_col:
        st.markdown(source_badge, unsafe_allow_html=True)

    # ── row 2: drug names (plain text via st.markdown, no HTML) ──────
    if drug_a and drug_b:
        st.markdown(f"**{drug_a}** ⟷ **{drug_b}**")
    elif drug_a:
        st.markdown(f"**{drug_a}**")

    # ── row 3: effect  ───────────────────────────────────────────────
    if effect:
        st.markdown(f"⚡ *{effect}*")

    # ── row 4: management note ───────────────────────────────────────
    if management:
        st.caption(f"📋 {management}")

    # close the outer shell div
    st.markdown("</div>", unsafe_allow_html=True)


def render_ddi_cards(interactions, source_badge_key="TxGemma"):
    """Render drug interaction cards, safely handling all input formats."""
    if not interactions:
        return
    source_badge = get_badge_html(source_badge_key)
    for ix in interactions:
        entry = _normalise_ddi_entry(ix)
        if entry:
            _render_single_ddi_card(entry, source_badge)


# ═══════════════════════════════════════════════════════════════
# Pipeline Execution
# ═══════════════════════════════════════════════════════════════
def save_uploaded_file(uploaded_file, subdir="uploads"):
    """Save uploaded file to disk and return path."""
    if uploaded_file is None:
        return None
    upload_dir = os.path.join(PROJECT_ROOT, "data", subdir)
    os.makedirs(upload_dir, exist_ok=True)
    file_path = os.path.join(upload_dir, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path


if run_clicked:
    audio_path = save_uploaded_file(audio_file)
    cxr_path   = save_uploaded_file(cxr_file)
    derm_path  = save_uploaded_file(derm_file)
    path_path  = save_uploaded_file(path_file)

    session = Session()
    session.audio_path = audio_path
    session.cxr_path   = cxr_path
    session.derm_path  = derm_path
    session.path_path  = path_path

    st.session_state.vram_monitor = VRAMMonitor()

    progress_bar = st.progress(0, text="Initializing Aegis Pipeline...")
    phases = [
        "MedASR", "LangExtract", "ModeBridge", "HeAR", "Path Foundation",
        "CXR Foundation", "Derm Foundation", "MedSigLIP",
        "Risk Engine", "OncoCase", "TxGemma", "MedGemma Debate",
        "Evidence Trace",
    ]
    phase_idx = [0]

    def on_phase(phase_name):
        phase_idx[0] += 1
        pct = min(phase_idx[0] / len(phases), 1.0)
        progress_bar.progress(pct, text=f"Running {phase_name.replace('_', ' ')}...")

    session = run_pipeline(
        session=session,
        vram_monitor=st.session_state.vram_monitor,
        on_phase_complete=on_phase,
    )

    progress_bar.progress(1.0, text="✅ Pipeline Complete!")
    st.session_state.session = session
    st.session_state.pipeline_complete = True
    time.sleep(0.5)
    st.rerun()


# ═══════════════════════════════════════════════════════════════
# Main Content — 2-Column Layout (Cortex | Tabs)
# ═══════════════════════════════════════════════════════════════
session = st.session_state.session

if session and st.session_state.pipeline_complete:
    oncocase  = session.oncocase or {}
    debate    = session.debate_results or {}
    risk      = session.risk_result or {}
    trace     = session.evidence_trace or {}
    escalation = session.escalation_result or {}

    # ── Top Metrics Row ──
    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        deg_level = oncocase.get("degradation_level", "N/A")
        deg_color = "#4ade80" if deg_level == "FULL" else "#fbbf24" if deg_level in ("REDUCED","PROVISIONAL") else "#f87171"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="font-size:1.4rem; color:{deg_color}">{deg_level}</div>
            <div class="metric-label">Degradation Level</div>
        </div>""", unsafe_allow_html=True)
    with m2:
        risk_level = risk.get("overall_risk_level", "GREEN")
        risk_colors = {"RED": "#ef4444", "AMBER": "#f59e0b", "GREEN": "#22c55e"}
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color:{risk_colors.get(risk_level, '#94a3b8')}">{risk_level}</div>
            <div class="metric-label">Overall Risk</div>
        </div>""", unsafe_allow_html=True)
    with m3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{oncocase.get('missing_count', 0)}/5</div>
            <div class="metric-label">Missing Modalities</div>
        </div>""", unsafe_allow_html=True)
    with m4:
        tb_score = risk.get("tb_risk_score", 0)
        tb_color = "#ef4444" if tb_score > 0.7 else "#f59e0b" if tb_score > 0.4 else "#22c55e"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color:{tb_color}">{tb_score:.0%}</div>
            <div class="metric-label">TB Risk Score</div>
        </div>""", unsafe_allow_html=True)
    with m5:
        esc_mode  = escalation.get("mode", "N/A")
        esc_color = "#ef4444" if esc_mode == "ONCOSPHERE" else "#22c55e"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value" style="color:{esc_color}; font-size:1.1rem">{esc_mode}</div>
            <div class="metric-label">Pipeline Mode</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── 2-Column Layout: Cortex strip | Main tabs ──
    col_cortex, col_main = st.columns([1, 3])

    # ══════════════════════════════════════════════════════════
    # COL 1 — Aegis-Cortex Strip
    # ══════════════════════════════════════════════════════════
    with col_cortex:
        st.markdown('<div class="section-header">🧬 Aegis-Cortex</div>', unsafe_allow_html=True)

        # Escalation Result
        esc_display = format_escalation_display(escalation)
        st.markdown(f"""
        <div class="glass-card" style="border-left:3px solid {esc_display['color']};
                    background:{esc_display['bg_color']}; padding:14px">
            <div style="font-size:0.75rem; font-weight:700; color:{esc_display['color']};
                        text-transform:uppercase; letter-spacing:1px; margin-bottom:4px">
                {esc_display['icon']} {esc_display['label']}
            </div>
            <div style="font-size:0.78rem; color:#94a3b8">{esc_display['sublabel']}</div>
        </div>
        """, unsafe_allow_html=True)

        # Confirmed badge
        staging = oncocase.get("staging_confidence", "UNKNOWN")
        st.markdown(f"""
        <div style="text-align:center; margin-bottom:12px">
            {format_staging_badge(staging)}
        </div>
        """, unsafe_allow_html=True)

        # Transcript
        with st.expander("📝 Transcript", expanded=False):
            transcript = session.transcript or "No audio transcription available."
            st.markdown(f"""
            <div style="font-size:0.82rem; color:#cbd5e1; line-height:1.6;
                        max-height:200px; overflow-y:auto">
                {transcript}
            </div>
            """, unsafe_allow_html=True)

        # Entity Chips
        clinical_frame = session.clinical_frame or {}
        symptoms   = clinical_frame.get("symptoms", [])
        meds       = clinical_frame.get("medications", [])
        conditions = clinical_frame.get("conditions", [])

        if symptoms or meds or conditions:
            st.markdown('<div class="section-header" style="font-size:0.85rem">🏷️ Entities</div>',
                       unsafe_allow_html=True)
            chips_html = ""
            for s in symptoms[:6]:
                chips_html += f'<span style="background:rgba(239,68,68,0.15); color:#fca5a5; padding:3px 8px; border-radius:12px; font-size:0.72rem; margin:2px; display:inline-block">💉 {s}</span> '
            for m in meds[:6]:
                chips_html += f'<span style="background:rgba(59,130,246,0.15); color:#93c5fd; padding:3px 8px; border-radius:12px; font-size:0.72rem; margin:2px; display:inline-block">💊 {m}</span> '
            for c in conditions[:4]:
                chips_html += f'<span style="background:rgba(168,85,247,0.15); color:#c4b5fd; padding:3px 8px; border-radius:12px; font-size:0.72rem; margin:2px; display:inline-block">🏥 {c}</span> '
            st.markdown(chips_html, unsafe_allow_html=True)

        # Risk
        st.markdown('<div class="section-header" style="font-size:0.85rem; margin-top:12px">⚡ Risk</div>',
                   unsafe_allow_html=True)
        risk_class = f"risk-{risk.get('overall_risk_level', 'green').lower()}"
        st.markdown(f"""
        <div class="{risk_class}" style="font-size:0.82rem">
            <strong>TB:</strong> {risk.get('tb_risk_level', 'LOW')} ({risk.get('tb_risk_score', 0):.0%})<br>
            <strong>HIV:</strong> {risk.get('hiv_risk_score', 0):.0%}
        </div>
        """, unsafe_allow_html=True)

        # Evidence Chips
        st.markdown('<div class="section-header" style="font-size:0.85rem; margin-top:12px">🔬 Evidence</div>',
                   unsafe_allow_html=True)
        for ev in session.evidence_pool:
            model  = ev.get("model", "Unknown")
            status = ev.get("status", "UNKNOWN")
            badge  = get_badge_html(model)
            if status == "MISSING_DATA":
                st.markdown(f"""
                <div style="display:flex; gap:6px; align-items:center; padding:4px 0;
                            border-bottom:1px solid rgba(148,163,184,0.05)">
                    {badge} <span class="status-pill status-missing" style="font-size:0.65rem">MISSING</span>
                </div>""", unsafe_allow_html=True)
            else:
                conf = ev.get("confidence")
                conf_str = f" · {conf:.0%}" if conf else ""
                st.markdown(f"""
                <div style="display:flex; gap:6px; align-items:center; padding:4px 0;
                            border-bottom:1px solid rgba(148,163,184,0.05)">
                    {badge} <span style="color:#94a3b8; font-size:0.7rem">{conf_str}</span>
                </div>""", unsafe_allow_html=True)

        # Uncertainty Flags
        flags = risk.get("uncertainty_flags", [])
        if flags:
            st.markdown('<div class="section-header" style="font-size:0.85rem; margin-top:12px">⚠️ Flags</div>',
                       unsafe_allow_html=True)
            for flag in flags:
                st.markdown(f'<div class="nba-item" style="font-size:0.78rem; padding:6px 10px">⚠ {flag}</div>',
                           unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════
    # COL 2 — 4 Tabs: Tumor Board | Patient Handout | Evidence Trace | Similar Cases
    # ══════════════════════════════════════════════════════════
    with col_main:
        tab1, tab2, tab3, tab4 = st.tabs([
            "🧠 Tumor Board",
            "📋 Patient Handout",
            "🔬 Evidence Trace",
            "💡 Similar Cases",
        ])

        # ── Tab 1: Tumor Board ──────────────────────────────────
        with tab1:
            st.markdown('<div class="section-header">🏥 Virtual Molecular Tumor Board</div>',
                       unsafe_allow_html=True)

            personas = [
                ("🔬 Virtual Pathologist",       "pass1_pathologist", "#a78bfa"),
                ("🫁 Virtual Radiologist",        "pass2_radiologist", "#3b82f6"),
                ("💊 Virtual Oncologist",         "pass3_oncologist",  "#f472b6"),
                ("🩺 Chief Physician Synthesizer","pass4_chief",       "#22c55e"),
            ]

            for emoji_name, key, color in personas:
                output = debate.get(key, "")
                if output:
                    tagged = render_badges_in_text(output)
                    st.markdown(f"""
                    <div class="persona-card" style="border-left: 3px solid {color}">
                        <div class="persona-name" style="color:{color}">{emoji_name}</div>
                        <div class="persona-output">{tagged}</div>
                    </div>
                    """, unsafe_allow_html=True)

            # ── Drug Interactions ──
            tx_result    = getattr(session, "tx_result", None) or oncocase.get("tx_analysis", {}) or {}
            interactions = tx_result.get("interaction_flags", [])

            # Detect weak/empty pipeline output — use demo data when demo mode active
            def _is_weak_output(lst):
                """True if list is empty or contains only trivially empty/placeholder items."""
                if not lst:
                    return True
                for item in lst:
                    if isinstance(item, dict):
                        vals = [safe_render_ddi_text(str(v)) for v in item.values()]
                        if any(v and len(v) > 3 for v in vals):
                            return False
                    elif isinstance(item, str) and len(safe_render_ddi_text(item)) > 5:
                        return False
                return True

            use_demo = st.session_state.demo_mode

            # Normalise all pipeline interactions; detect if result is genuinely useful
            normalised_interactions = [_normalise_ddi_entry(ix) for ix in interactions]
            normalised_interactions = [e for e in normalised_interactions if e.get("drug_a")]
            if use_demo and _is_weak_output(normalised_interactions):
                normalised_interactions = DEMO_DDI_INTERACTIONS

            if normalised_interactions:
                demo_label = ' <span style="font-size:0.65rem; color:#f59e0b; background:rgba(245,158,11,0.1); padding:2px 6px; border-radius:6px; vertical-align:middle">DEMO</span>' if (use_demo and _is_weak_output(interactions)) else ""
                st.markdown(f'<div class="section-header" style="margin-top:16px">💊 Drug Interactions{demo_label}</div>',
                           unsafe_allow_html=True)
                source_badge = get_badge_html("TxGemma")
                for entry in normalised_interactions:
                    _render_single_ddi_card(entry, source_badge)

            # ── Inventory Alerts ──
            inv_alerts = tx_result.get("inventory_alerts", [])
            if use_demo and not inv_alerts:
                inv_alerts = DEMO_INVENTORY_ALERTS
                _inv_demo = True
            else:
                _inv_demo = False

            if inv_alerts:
                demo_label = ' <span style="font-size:0.65rem; color:#f59e0b; background:rgba(245,158,11,0.1); padding:2px 6px; border-radius:6px; vertical-align:middle">DEMO</span>' if _inv_demo else ""
                st.markdown(f'<div class="section-header" style="margin-top:16px">📦 Inventory Alerts{demo_label}</div>',
                           unsafe_allow_html=True)
                for alert in inv_alerts:
                    drug   = safe_render_ddi_text(alert.get("drug", "Unknown"))
                    status = alert.get("status", "UNAVAILABLE")
                    msg    = safe_render_ddi_text(alert.get("message", ""))
                    sub    = safe_render_ddi_text(alert.get("substitute", ""))
                    tagged = alert.get("tagged", msg)
                    status_icon = "🚫" if status in ("UNAVAILABLE", "OUT_OF_STOCK") else "⚠️"
                    stock_color = "#f87171" if status in ("UNAVAILABLE", "OUT_OF_STOCK") else "#fbbf24"
                    inv_badge = get_badge_html("Local_Inventory_JSON")
                    sub_html = f'<div style="font-size:0.78rem; color:#6ee7b7; margin-top:5px; padding-top:5px; border-top:1px solid rgba(148,163,184,0.1)">💡 <strong>Substitute:</strong> {sub}</div>' if sub else ""
                    st.markdown(f"""
                    <div class="inventory-alert">
                        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:5px">
                            <span style="font-weight:700; color:{stock_color}">{status_icon} {drug}</span>
                            {inv_badge}
                        </div>
                        <div style="font-size:0.80rem; color:#94a3b8; line-height:1.5">{render_badges_in_text(str(tagged))}</div>
                        {sub_html}
                    </div>
                    """, unsafe_allow_html=True)

            # ── Substitution Recommendations ──
            substitutions = tx_result.get("substitutions", [])
            # Filter out trivially empty subs like "No substitutions needed" with no real content
            real_subs = [
                s for s in substitutions
                if len(safe_render_ddi_text(s.get("text", "") if isinstance(s, dict) else str(s))) > 20
                and "no substitution" not in safe_render_ddi_text(
                    s.get("text", "") if isinstance(s, dict) else str(s)).lower()
            ]
            if use_demo and not real_subs:
                real_subs = DEMO_SUBSTITUTIONS
                _sub_demo = True
            else:
                _sub_demo = False

            if real_subs:
                demo_label = ' <span style="font-size:0.65rem; color:#f59e0b; background:rgba(245,158,11,0.1); padding:2px 6px; border-radius:6px; vertical-align:middle">DEMO</span>' if _sub_demo else ""
                st.markdown(f'<div class="section-header" style="margin-top:16px">🔄 Substitution Recommendations{demo_label}</div>',
                           unsafe_allow_html=True)
                urgency_colors = {
                    "HIGH":      ("#ef4444", "rgba(239,68,68,0.08)",  "rgba(239,68,68,0.2)",  "#fca5a5"),
                    "MODERATE":  ("#f59e0b", "rgba(245,158,11,0.08)", "rgba(245,158,11,0.2)", "#fcd34d"),
                    "CONFIRMED": ("#22c55e", "rgba(34,197,94,0.08)",  "rgba(34,197,94,0.2)",  "#6ee7b7"),
                    "LOW":       ("#3b82f6", "rgba(59,130,246,0.08)", "rgba(59,130,246,0.2)", "#93c5fd"),
                }
                sub_icons = {"HIGH": "🔴", "MODERATE": "🟡", "CONFIRMED": "✅", "LOW": "🔵"}
                for sub in real_subs:
                    if isinstance(sub, dict):
                        text    = safe_render_ddi_text(sub.get("text", ""))
                        urgency = sub.get("urgency", "LOW").upper()
                    else:
                        text    = safe_render_ddi_text(str(sub))
                        urgency = "LOW"
                    tagged = render_badges_in_text(text)
                    border_c, bg_c, border_alpha, text_c = urgency_colors.get(urgency, urgency_colors["LOW"])
                    icon = sub_icons.get(urgency, "🔄")
                    st.markdown(f"""
                    <div style="background:{bg_c}; border:1px solid {border_alpha};
                                border-left:4px solid {border_c}; border-radius:8px;
                                padding:12px 14px; margin-bottom:8px">
                        <div style="display:flex; align-items:flex-start; gap:8px">
                            <span style="font-size:1rem; flex-shrink:0; margin-top:1px">{icon}</span>
                            <span style="color:{text_c}; font-size:0.83rem; line-height:1.6">{tagged}</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

            # ── NBA Missing Workup Checklist ──
            nba_list = oncocase.get("nba_list", [])
            if nba_list:
                st.markdown('<div class="section-header" style="margin-top:16px">📋 Missing Workup Checklist</div>',
                           unsafe_allow_html=True)
                for nba in nba_list:
                    st.markdown(f"""
                    <div class="nba-item">
                        <strong>{nba.get('model', '')}</strong>: {nba.get('nba', '')}<br>
                        <span style="font-size:0.75rem; color:#94a3b8">Cost: INR {nba.get('cost_inr', 'N/A')}</span>
                    </div>
                    """, unsafe_allow_html=True)

            # ── Download + Expanders ──
            st.markdown("<br>", unsafe_allow_html=True)
            if session.evidence_pool:
                try:
                    _report_oncocase = {
                        "clinical_frame": session.clinical_frame or {},
                        "evidence_pool": session.evidence_pool,
                        "staging_confidence": getattr(session, "staging_confidence", "PROVISIONAL"),
                    }
                    debate_result  = getattr(session, "debate_result", {}) or {}
                    txgemma_result = getattr(session, "txgemma_result", {}) or {}
                    report_html = generate_report_html(
                        oncocase=_report_oncocase,
                        debate_result=debate_result,
                        txgemma_result=txgemma_result,
                        evidence_trace=trace,
                    )
                    st.download_button(
                        label="📥 Download Clinical Report (HTML)",
                        data=report_html,
                        file_name="aegis_sphere_report.html",
                        mime="text/html",
                        width='stretch',
                    )
                except Exception as e:
                    st.warning(f"Report generation failed: {e}")

            with st.expander("📋 Clinical Frame"):
                if session.clinical_frame:
                    st.json(session.clinical_frame)
                else:
                    st.info("No clinical frame extracted yet.")

            with st.expander("🗂️ Full Evidence Pool"):
                for ev in session.evidence_pool:
                    model   = ev.get("model", "Unknown")
                    status  = ev.get("status", "UNKNOWN")
                    finding = ev.get("finding", "N/A")
                    badge   = get_badge_html(model)
                    status_class = ("status-ok" if status == "OK"
                                    else "status-missing" if status == "MISSING_DATA"
                                    else "status-blocked")
                    st.markdown(f"""
                    <div style="display:flex; gap:10px; align-items:center; padding:8px 0;
                                border-bottom:1px solid rgba(148,163,184,0.06)">
                        {badge}
                        <span class="status-pill {status_class}">{status}</span>
                        <span style="color:#cbd5e1; font-size:0.85rem">{finding or 'No data'}</span>
                    </div>
                    """, unsafe_allow_html=True)

        # ── Tab 2: Patient Handout ──────────────────────────────
        with tab2:
            st.markdown('<div class="section-header">💌 Patient-Friendly Summary</div>',
                       unsafe_allow_html=True)

            patient_text = debate.get("pass5_patient", "Patient handout not generated yet.")
            st.markdown(f"""
            <div class="patient-letter">
                {patient_text.replace(chr(10), '<br>')}
            </div>
            """, unsafe_allow_html=True)

            # Next Steps in patient language
            nba_list_pt = oncocase.get("nba_list", [])
            if nba_list_pt:
                st.markdown('<div class="section-header" style="margin-top:16px">☐ Your Next Steps</div>',
                           unsafe_allow_html=True)
                for nba in nba_list_pt:
                    patient_lang = nba.get("patient_language", nba.get("nba", ""))
                    st.markdown(f"""
                    <div style="background:rgba(16,185,129,0.08); border:1px solid rgba(16,185,129,0.15);
                                border-radius:8px; padding:10px 14px; margin-bottom:6px;
                                color:#6ee7b7; font-size:0.9rem">
                        ☐ {patient_lang}
                    </div>
                    """, unsafe_allow_html=True)

            st.markdown("""
            <div style="margin-top:20px; padding:12px; background:rgba(239,68,68,0.08);
                        border:1px solid rgba(239,68,68,0.15); border-radius:8px;
                        color:#fca5a5; font-size:0.75rem">
                ⚠️ <strong>Important:</strong> This letter was generated by an AI assistant
                and reviewed by virtual medical personas. It is NOT a substitute for direct
                consultation with your healthcare provider.
            </div>
            """, unsafe_allow_html=True)

        # ── Tab 3: Evidence Trace ───────────────────────────────
        with tab3:
            st.markdown('<div class="section-header">🔬 Evidence Grounding Trace</div>',
                       unsafe_allow_html=True)

            if trace:
                # Build a clean table manually from the trace dict
                rows_html = ""
                for source, claims in trace.items():
                    badge = get_badge_html(source)
                    if isinstance(claims, list):
                        claims_text = "".join(
                            f'<div style="margin-bottom:3px">• {safe_render_ddi_text(str(c))}</div>'
                            for c in claims
                        )
                    else:
                        claims_text = f'<div>{safe_render_ddi_text(str(claims))}</div>'
                    rows_html += f"""
                    <tr>
                        <td style="white-space:nowrap; vertical-align:top; padding-right:16px">{badge}</td>
                        <td style="color:#cbd5e1; font-size:0.82rem; line-height:1.6">{claims_text}</td>
                    </tr>"""

                st.markdown(f"""
                <table class="ev-table">
                    <thead><tr><th>Source</th><th>Claims</th></tr></thead>
                    <tbody>{rows_html}</tbody>
                </table>
                """, unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)

                # Source Coverage grid
                st.markdown('<div class="section-header">📊 Source Coverage</div>',
                           unsafe_allow_html=True)
                all_possible = [
                    "Path_Foundation", "CXR_Foundation", "HeAR", "Derm_Foundation",
                    "TxGemma", "Local_Inventory_JSON", "MedSigLIP_CaseLibrary",
                    "MedASR", "Clinical_Frame",
                ]
                cc1, cc2 = st.columns(2)
                for i, source in enumerate(all_possible):
                    col_c = cc1 if i % 2 == 0 else cc2
                    with col_c:
                        found = source in trace
                        count = len(trace.get(source, []))
                        badge = get_badge_html(source)
                        if found:
                            status_span = f'<span class="status-pill status-ok">{count} claims</span>'
                        else:
                            status_span = '<span class="status-pill status-missing">No data</span>'
                        st.markdown(
                            f'<div style="display:flex; justify-content:space-between; align-items:center; '
                            f'padding:6px 0; border-bottom:1px solid rgba(148,163,184,0.06)">'
                            f'{badge} {status_span}</div>',
                            unsafe_allow_html=True,
                        )
            else:
                st.markdown("""
                <div style="color:#94a3b8; text-align:center; padding:40px;
                            background:rgba(30,41,59,0.3); border-radius:12px;
                            border:1px dashed rgba(148,163,184,0.15)">
                    <div style="font-size:1.5rem; margin-bottom:8px">🔬</div>
                    Run the pipeline to see evidence grounding trace.
                </div>
                """, unsafe_allow_html=True)

        # ── Tab 4: Similar Cases + Override ────────────────────
        with tab4:
            sim_col, override_col = st.columns([3, 2])

            with sim_col:
                st.markdown('<div class="section-header">📚 Similar Cases</div>',
                           unsafe_allow_html=True)

                sim_cases = getattr(session, "similar_cases", []) or oncocase.get("similar_cases", [])
                if sim_cases:
                    st.markdown(f"""
                    <div style="color:#94a3b8; font-size:0.72rem; margin-bottom:12px">
                        <strong style="color:#60a5fa">{len(sim_cases)}</strong> cases retrieved from MedSigLIP case library
                    </div>
                    """, unsafe_allow_html=True)

                    for case in sim_cases:
                        case_id   = case.get("case_id", "N/A")
                        diagnosis = case.get("diagnosis", "N/A")
                        staging   = case.get("staging", "N/A")
                        treatment = case.get("treatment", "N/A")
                        score     = case.get("similarity_score", 0)
                        rank      = case.get("rank", 0)
                        modality  = case.get("modality", "Unknown")
                        hiv_status = case.get("hiv_status", False)
                        cd4       = case.get("cd4", None)

                        if score >= 0.85:
                            sc_color, sc_bg = "#22c55e", "rgba(34,197,94,0.12)"
                        elif score >= 0.7:
                            sc_color, sc_bg = "#f59e0b", "rgba(245,158,11,0.12)"
                        else:
                            sc_color, sc_bg = "#ef4444", "rgba(239,68,68,0.12)"
                        bar_w = int(score * 100)

                        mod_icons = {"CXR": "🫁", "Derm": "🔬", "Histopathology": "🧬", "MRI": "🧲"}
                        mod_icon  = mod_icons.get(modality, "📋")

                        hiv_badge = ""
                        if hiv_status:
                            cd4_s = f" · CD4: {cd4}" if cd4 is not None else ""
                            hiv_badge = f'<span style="background:rgba(239,68,68,0.15); color:#fca5a5; padding:1px 6px; border-radius:8px; font-size:0.6rem; margin-left:4px">HIV+{cd4_s}</span>'

                        st.markdown(f"""
                        <div style="background:rgba(30,41,59,0.5); border:1px solid rgba(148,163,184,0.1);
                                    border-left:3px solid {sc_color}; border-radius:10px;
                                    padding:12px 14px; margin-bottom:10px">
                            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:6px">
                                <div style="display:flex; align-items:center; gap:5px; flex-wrap:wrap">
                                    <span style="background:{sc_bg}; color:{sc_color}; padding:2px 7px;
                                                border-radius:10px; font-size:0.62rem; font-weight:700">#{rank}</span>
                                    <span style="font-size:0.78rem; font-weight:600; color:#e2e8f0">{case_id}</span>
                                </div>
                                <span style="color:{sc_color}; font-weight:700; font-size:0.82rem">{score:.0%}</span>
                            </div>
                            <div style="display:flex; align-items:center; gap:4px; margin-bottom:5px">
                                <span style="color:#64748b; font-size:0.68rem">{mod_icon} {modality}</span>
                                {hiv_badge}
                            </div>
                            <div style="font-size:0.78rem; color:#f1f5f9; margin-bottom:4px">🩺 {diagnosis}</div>
                            <div style="font-size:0.68rem; color:#94a3b8; margin-bottom:8px">
                                📊 <strong style="color:#cbd5e1">Stage:</strong> {staging}
                                &nbsp;·&nbsp;
                                💊 <strong style="color:#cbd5e1">Tx:</strong> {treatment}
                            </div>
                            <div style="background:rgba(148,163,184,0.05); border-radius:4px; height:4px; overflow:hidden">
                                <div style="width:{bar_w}%; height:100%; border-radius:4px;
                                            background:linear-gradient(90deg, {sc_color}88, {sc_color})"></div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div style="color:#94a3b8; text-align:center; padding:30px 20px;
                                background:rgba(30,41,59,0.3); border-radius:12px;
                                border:1px dashed rgba(148,163,184,0.15)">
                        <div style="font-size:2rem; margin-bottom:10px">📁</div>
                        Upload imaging data for personalised similar-case retrieval.
                    </div>
                    """, unsafe_allow_html=True)

            with override_col:
                st.markdown('<div class="section-header">🖊️ Override & Flag</div>',
                           unsafe_allow_html=True)

                override_field = st.selectbox(
                    "Field to override",
                    ["staging", "treatment", "risk_level", "diagnosis", "other"],
                    key="override_field",
                )
                override_note = st.text_area(
                    "Clinician note",
                    placeholder="Reason for override...",
                    height=100,
                    key="override_note",
                )
                override_value = st.text_input(
                    "New value",
                    key="override_value",
                )

                if st.button("📝 Submit Override", key="submit_override", type="primary"):
                    if override_note and override_value:
                        original = oncocase.get(override_field, "N/A")
                        record = log_override(
                            session_id=session.session_id,
                            clinician_note=override_note,
                            field_overridden=override_field,
                            original_value=str(original),
                            new_value=override_value,
                        )
                        st.success(f"✅ Override logged (ID: {record['record_id']})")
                    else:
                        st.warning("Please provide both a note and new value.")

                # Override stats
                sync_stats_main = get_override_stats()
                if sync_stats_main["total"] > 0:
                    st.markdown(f"""
                    <div style="margin-top:12px; padding:10px; background:rgba(30,41,59,0.5);
                                border-radius:8px; border:1px solid rgba(148,163,184,0.1)">
                        <span style="color:#94a3b8; font-size:0.7rem; text-transform:uppercase">Sync Engine</span><br>
                        <span style="color:#e2e8f0; font-size:0.85rem">
                            📦 {sync_stats_main['total']} overrides
                            · 🔄 {sync_stats_main['pending']} pending
                        </span>
                    </div>
                    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# Welcome / Landing State
# ═══════════════════════════════════════════════════════════════
else:
    st.markdown("""
    <div style="text-align:center; padding:48px 20px 16px 20px" class="fade-in">
        <div style="display:inline-block; padding:6px 20px; border-radius:100px; font-size:0.65rem;
                    font-weight:600; letter-spacing:2px; text-transform:uppercase;
                    background:rgba(99,102,241,0.08); color:#818cf8; border:1px solid rgba(99,102,241,0.15);
                    margin-bottom:16px; font-family:'JetBrains Mono',monospace">
            ▸ CLINICAL INTELLIGENCE PLATFORM
        </div>
        <div style="font-size:2rem; font-weight:800; color:#f1f5f9; margin-bottom:10px; letter-spacing:-1px">
            Welcome to <span style="background:linear-gradient(135deg,#60a5fa,#a78bfa);
            -webkit-background-clip:text; -webkit-text-fill-color:transparent">Aegis-Sphere</span>
        </div>
        <div style="font-size:0.88rem; color:#64748b; max-width:760px; margin:0 auto; line-height:1.8">
            An offline, dual-mode clinical intelligence platform that listens to TB/HIV consultations
            in real time, auto-detects malignancy signals, escalates to a multi-agent virtual tumor board
            where <strong style="color:#a78bfa">MedGemma 1.5</strong> instances run sequential
            persona passes as a <em style="color:#94a3b8">Pathologist, Radiologist, and Oncologist</em>
            before reaching consensus, dynamically routes treatment plans around real drug shortages,
            and presents empathetic patient-facing handouts — all on
            <strong style="color:#60a5fa">8 GB VRAM</strong> in an LMIC clinic.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Pipeline Architecture Strip ──
    st.markdown("""
    <div class="glass-card" style="max-width:1000px; margin:0 auto; padding:20px 28px">
        <div style="text-align:center; margin-bottom:16px">
            <span style="font-size:0.62rem; color:#475569; text-transform:uppercase; letter-spacing:2px;
                        font-family:'JetBrains Mono',monospace">PIPELINE ARCHITECTURE</span>
        </div>
        <div style="display:flex; flex-wrap:wrap; justify-content:center; gap:6px; align-items:center">
            <span style="background:rgba(99,102,241,0.12); color:#a5b4fc; padding:5px 12px; border-radius:8px;
                        font-size:0.72rem; font-weight:600; border:1px solid rgba(99,102,241,0.15)">🎤 MedASR</span>
            <span style="color:#334155; font-size:0.7rem">→</span>
            <span style="background:rgba(168,85,247,0.12); color:#c4b5fd; padding:5px 12px; border-radius:8px;
                        font-size:0.72rem; font-weight:600; border:1px solid rgba(168,85,247,0.15)">🧠 NER Extract</span>
            <span style="color:#334155; font-size:0.7rem">→</span>
            <span style="background:rgba(239,68,68,0.12); color:#fca5a5; padding:5px 12px; border-radius:8px;
                        font-size:0.72rem; font-weight:600; border:1px solid rgba(239,68,68,0.15)">🚨 Mode Bridge</span>
            <span style="color:#334155; font-size:0.7rem">→</span>
            <span style="background:rgba(34,197,94,0.12); color:#86efac; padding:5px 12px; border-radius:8px;
                        font-size:0.72rem; font-weight:600; border:1px solid rgba(34,197,94,0.15)">🫁 Vision AI</span>
            <span style="color:#334155; font-size:0.7rem">→</span>
            <span style="background:rgba(245,158,11,0.12); color:#fcd34d; padding:5px 12px; border-radius:8px;
                        font-size:0.72rem; font-weight:600; border:1px solid rgba(245,158,11,0.15)">⚗️ TxGemma</span>
            <span style="color:#334155; font-size:0.7rem">→</span>
            <span style="background:rgba(244,114,182,0.12); color:#f9a8d4; padding:5px 12px; border-radius:8px;
                        font-size:0.72rem; font-weight:600; border:1px solid rgba(244,114,182,0.15)">🧬 Tumor Board</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Dr. Priya's Day Before/After
    st.markdown("""
    <div class="glass-card" style="max-width:1000px; margin:0 auto">
        <div style="text-align:center; margin-bottom:16px">
            <span style="font-size:0.62rem; color:#475569; text-transform:uppercase; letter-spacing:2px;
                        font-family:'JetBrains Mono',monospace">THE PROBLEM</span>
            <div style="font-size:1.05rem; font-weight:700; color:#e2e8f0; margin-top:6px">
                🏥 Dr. Priya's Day — Nagpur District HIV Clinic
            </div>
        </div>
        <div style="font-size:0.82rem; color:#64748b; text-align:center; margin-bottom:16px; line-height:1.7">
            Dr. Priya sees 40 patients daily. When a 38-year-old HIV+ man presents with a 4-week wet cough,
            weight loss, and cervical lymphadenopathy, she correctly suspects TB — but misses that
            HIV+ patients have an <strong style="color:#f472b6">11.5× standardised incidence ratio for NHL</strong>.
        </div>
        <table style="width:100%; border-collapse:collapse; font-size:0.78rem">
            <thead>
                <tr>
                    <th style="text-align:left; padding:10px 14px; color:#ef4444; border-bottom:1px solid rgba(239,68,68,0.15);
                              font-size:0.68rem; letter-spacing:0.5px; font-family:'JetBrains Mono',monospace">
                        ❌ BEFORE
                    </th>
                    <th style="text-align:left; padding:10px 14px; color:#22c55e; border-bottom:1px solid rgba(34,197,94,0.15);
                              font-size:0.68rem; letter-spacing:0.5px; font-family:'JetBrains Mono',monospace">
                        ✅ AFTER
                    </th>
                </tr>
            </thead>
            <tbody style="color:#94a3b8">
                <tr><td style="padding:8px 14px; border-bottom:1px solid rgba(148,163,184,0.04)">Suspects TB, starts empiric RHEZ therapy</td>
                    <td style="padding:8px 14px; border-bottom:1px solid rgba(148,163,184,0.04); color:#cbd5e1">Ambient system detects oncology signals within 60s</td></tr>
                <tr><td style="padding:8px 14px; border-bottom:1px solid rgba(148,163,184,0.04)">Patient misclassified on TB therapy for 4–7 weeks</td>
                    <td style="padding:8px 14px; border-bottom:1px solid rgba(148,163,184,0.04); color:#cbd5e1">Escalation: "HIV-related malignancy detected. Activate OncoSphere?"</td></tr>
                <tr><td style="padding:8px 14px; border-bottom:1px solid rgba(148,163,184,0.04)">Lymphoma diagnosis delayed by months → Stage IV</td>
                    <td style="padding:8px 14px; border-bottom:1px solid rgba(148,163,184,0.04); color:#cbd5e1">Virtual tumor board convened. Staging + pathways generated.</td></tr>
                <tr><td style="padding:8px 14px; border-bottom:1px solid rgba(148,163,184,0.04)">R-CHOP prescribed — Rituximab is out of stock</td>
                    <td style="padding:8px 14px; border-bottom:1px solid rgba(148,163,184,0.04); color:#cbd5e1">TxGemma checks inventory → CHOP + Liposomal Dox auto-substituted</td></tr>
                <tr><td style="padding:8px 14px; border-bottom:1px solid rgba(148,163,184,0.04)">Patient leaves with no explanation</td>
                    <td style="padding:8px 14px; border-bottom:1px solid rgba(148,163,184,0.04); color:#cbd5e1">Grade-5 empathetic patient handout generated by MedGemma Pass 5</td></tr>
                <tr><td style="padding:8px 14px">No audit trail, no data, no specialist</td>
                    <td style="padding:8px 14px; color:#cbd5e1">Override records synced to big-center board for annotation</td></tr>
            </tbody>
        </table>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Impact Metrics
    im1, im2, im3, im4 = st.columns(4)
    impact_metrics = [
        ("🎯", "7,500",    "Early diagnoses/yr",  "at 500 pilot clinics",  "#6366f1"),
        ("📈", "+30–35%",  "Survival delta",       "Stage IIB vs IV NHL",  "#22c55e"),
        ("💊", "−20%",     "Drug waste",           "Blocked Rx prevented", "#f59e0b"),
        ("🌍", "75K",      "5-yr scale",           "India + SSA projection", "#a78bfa"),
    ]
    for col_im, (icon, value, label, sub, accent) in zip([im1, im2, im3, im4], impact_metrics):
        with col_im:
            st.markdown(f"""
            <div class="glass-card" style="text-align:center; min-height:140px; position:relative; overflow:hidden">
                <div style="position:absolute; top:0; left:0; right:0; height:2px;
                            background:linear-gradient(90deg, transparent, {accent}66, transparent)"></div>
                <div style="font-size:1.6rem; margin-bottom:6px">{icon}</div>
                <div style="font-size:1.5rem; font-weight:800; color:{accent};
                            font-family:'JetBrains Mono',monospace">{value}</div>
                <div style="font-size:0.72rem; font-weight:600; color:#e2e8f0; margin-top:4px">{label}</div>
                <div style="font-size:0.62rem; color:#475569; font-family:'JetBrains Mono',monospace">{sub}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Upload prompt
    st.markdown("""
    <div style="text-align:center; padding:12px 20px">
        <div style="font-size:0.92rem; color:#64748b; line-height:1.8">
            Upload patient data in the sidebar (audio, chest X-ray, skin lesion, pathology)
            and click <strong style="color:#a5b4fc">Run Aegis Pipeline</strong> to generate a full OncoCase analysis.<br>
            The system gracefully handles missing data — designed for LMIC clinics where not
            every modality is available.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Feature Cards — 5 columns with gradient top-border accents
    f1, f2, f3, f4, f5 = st.columns(5)
    features = [
        ("🧠", "8 AI Models",       "MedGemma, TxGemma, HeAR, CXR/Derm/Path, MedSigLIP, MedASR", "#6366f1"),
        ("📈", "VRAM Telemetry",    "Live GPU monitoring with sawtooth phase tracking", "#22c55e"),
        ("🏷️", "Evidence Tags",    "[Source: X] citations grounding every clinical claim", "#f59e0b"),
        ("🚨", "Mode Bridge",       "Auto-escalation from TB triage to OncoSphere tumor board", "#ef4444"),
        ("💌", "Patient Letters",   "Grade-5 empathetic handouts with next-step checklists", "#a78bfa"),
    ]
    for col_f, (icon, title, desc, accent) in zip([f1, f2, f3, f4, f5], features):
        with col_f:
            st.markdown(f"""
            <div class="glass-card" style="text-align:center; min-height:150px; position:relative; overflow:hidden;
                        transition:all 0.3s cubic-bezier(.4,0,.2,1)">
                <div style="position:absolute; top:0; left:0; right:0; height:2px;
                            background:linear-gradient(90deg, transparent, {accent}, transparent)"></div>
                <div style="font-size:1.6rem; margin-bottom:8px; margin-top:4px">{icon}</div>
                <div style="font-size:0.82rem; font-weight:700; color:#e2e8f0; margin-bottom:6px">{title}</div>
                <div style="font-size:0.68rem; color:#64748b; line-height:1.5">{desc}</div>
            </div>
            """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# Footer
# ═══════════════════════════════════════════════════════════════
st.markdown("""
<div style="text-align:center; padding:24px 0; margin-top:48px; position:relative">
    <div style="position:absolute; top:0; left:10%; right:10%; height:1px;
                background:linear-gradient(90deg, transparent, rgba(99,102,241,0.2), rgba(168,85,247,0.2), transparent)"></div>
    <div style="display:inline-block; padding:4px 16px; border-radius:100px; font-size:0.6rem;
                font-weight:600; letter-spacing:1.5px; background:rgba(99,102,241,0.06);
                color:#475569; border:1px solid rgba(99,102,241,0.08); margin-bottom:8px;
                font-family:'JetBrains Mono',monospace">
        AEGIS-SPHERE v1.0
    </div>
    <div style="color:#334155; font-size:0.7rem; margin-top:6px">
        AI-Assisted Oncology Decision Support · DPDP Act 2023 Compliant · Not a substitute for clinical judgment
    </div>
</div>
""", unsafe_allow_html=True)