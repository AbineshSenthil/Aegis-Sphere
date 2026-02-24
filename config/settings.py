"""
Aegis-Sphere — Application Settings & Constants
Paths, thresholds, NBA catalog, and degradation rules.
"""

import os
from pathlib import Path

# ─────────────────────────────────────────────────────────────
# App Identity
# ─────────────────────────────────────────────────────────────
APP_TITLE = "Aegis-Sphere"
APP_SUBTITLE = "AI-Assisted Oncology Decision Support for LMIC Settings"

# ─────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DEMO_CASE_DIR = DATA_DIR / "demo_case"
DEMO_CASE_DEGRADED_DIR = DATA_DIR / "demo_case_degraded"
FAISS_DIR = DATA_DIR / "faiss_case_library"
DB_PATH = PROJECT_ROOT / "db" / "aegis_sphere.db"
VRAM_LOG_PATH = PROJECT_ROOT / "evaluation" / "results" / "vram_log.csv"
LORA_OUTPUT_DIR = PROJECT_ROOT / "medgemma_lora_checkpoints"
HEAR_PROBE_PATH = DATA_DIR / "hear_tb_probe.pkl"
LOCAL_INVENTORY_PATH = DATA_DIR / "local_inventory.json"

# ─────────────────────────────────────────────────────────────
# GPU / VRAM
# ─────────────────────────────────────────────────────────────
MAX_VRAM_MB = 8192
VRAM_SAFE_ZONE = 4000       # green: 0–4000 MB
VRAM_LOADED_ZONE = 7000     # amber: 4000–7000 MB
VRAM_DANGER_ZONE = 8192     # red:   7000–8192 MB

# ─────────────────────────────────────────────────────────────
# Audio
# ─────────────────────────────────────────────────────────────
AUDIO_SAMPLE_RATE = 16000
COUGH_CLIP_DURATION_S = 2.0

# ─────────────────────────────────────────────────────────────
# Image
# ─────────────────────────────────────────────────────────────
CXR_INPUT_SIZE = (224, 224)
PATH_INPUT_SIZE = (224, 224)
DERM_INPUT_SIZE = (224, 224)

# ─────────────────────────────────────────────────────────────
# Degradation thresholds
# ─────────────────────────────────────────────────────────────
class DegradationLevel:
    FULL = "FULL"                   # 0 missing items
    REDUCED = "REDUCED"             # 1 missing item
    PROVISIONAL = "PROVISIONAL"     # 2 missing items
    MINIMAL = "MINIMAL"             # 3+ missing items
    NO_DATA = "NO_DATA"             # all 5 modalities missing


# ─────────────────────────────────────────────────────────────
# NBA (Next Best Action) Catalog — hardcoded LMIC fallbacks
# ─────────────────────────────────────────────────────────────
NBA_CATALOG = {
    "Path_Foundation": {
        "nba": (
            "Recommend immediate fine-needle aspiration cytology (FNAC) of the "
            "cervical lymph node — cost: INR 300–500, available at most district "
            "hospitals. This is the single highest-yield next step."
        ),
        "cost_inr": "300–500",
        "patient_language": (
            "Get a small tissue sample taken from your neck lump "
            "(it's a quick procedure done with a thin needle)"
        ),
    },
    "CXR_Foundation": {
        "nba": (
            "Recommend portable chest X-ray before chemotherapy. Do not proceed "
            "with anthracycline-based regimen without cardiopulmonary baseline."
        ),
        "cost_inr": "200–400",
        "patient_language": "Get a chest X-ray",
    },
    "HeAR": {
        "nba": (
            "Recommend recording a 10-second forced cough on any smartphone and "
            "re-uploading for HeAR analysis. Alternatively, refer for sputum AFB "
            "smear if TB is suspected."
        ),
        "cost_inr": "0 (smartphone) / 100–200 (AFB smear)",
        "patient_language": (
            "Record a short cough sound on your phone and bring it to your next visit"
        ),
    },
    "Derm_Foundation": {
        "nba": (
            "Recommend clinical photograph of skin lesion under good lighting. "
            "If Kaposi sarcoma is suspected clinically, a 4mm punch biopsy "
            "(INR 200–400) provides definitive diagnosis."
        ),
        "cost_inr": "0 (photo) / 200–400 (biopsy)",
        "patient_language": (
            "Take a clear photo of the skin spot in good light and show your doctor"
        ),
    },
    "MedASR": {
        "nba": (
            "Audio consultation data unavailable. Recommend recording next "
            "consultation using any recording device at 16kHz."
        ),
        "cost_inr": "0",
        "patient_language": "Your doctor will record your next conversation",
    },
}

# ─────────────────────────────────────────────────────────────
# Uncertainty propagation flags
# ─────────────────────────────────────────────────────────────
UNCERTAINTY_FLAGS = {
    "LOW_AUDIO_CONFIDENCE":       "ASR transcript has low-confidence segments",
    "NO_RESPIRATORY_DATA":        "HeAR cough analysis unavailable",
    "NO_CXR_DATA":                "CXR imaging unavailable — staging provisional",
    "NO_PATH_DATA":               "Histopathology unavailable — staging requires pathology",
    "NO_DERM_DATA":               "Dermatology imaging unavailable",
    "RECOMMENDATION_ONLY":        "TxGemma in recommendation-only mode",
    "INSUFFICIENT_DATA":          "3+ modalities missing — minimal data mode",
}

# ─────────────────────────────────────────────────────────────
# Modality list (for counting missing items)
# ─────────────────────────────────────────────────────────────
ALL_MODALITIES = [
    "audio",          # MedASR
    "cough",          # HeAR
    "cxr",            # CXR Foundation
    "histopathology", # Path Foundation
    "derm",           # Derm Foundation
]

# ─────────────────────────────────────────────────────────────
# Smart Sync Engine (DPDP Act 2023 compliance)
# ─────────────────────────────────────────────────────────────
SYNC_DIR = PROJECT_ROOT / "sync"
OVERRIDE_LOG_PATH = SYNC_DIR / "remote_board" / "override_log.jsonl"
SYNC_INTERVAL_SECONDS = 30

# ─────────────────────────────────────────────────────────────
# Mode Bridge — Oncology Trigger Keywords
# ─────────────────────────────────────────────────────────────
ONCOLOGY_TRIGGERS = [
    "lymphoma", "malignancy", "cancer", "tumor", "tumour",
    "metastasis", "metastatic", "carcinoma", "sarcoma",
    "kaposi", "mass", "neoplasm", "neoplastic", "oncology",
    "adenocarcinoma", "leukemia", "myeloma", "hodgkin",
    "non-hodgkin", "staging", "biopsy",
]
