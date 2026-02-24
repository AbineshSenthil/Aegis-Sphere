"""
Aegis-Sphere — Centralized Model ID Registry
All 8 Google Health AI models with HF paths, load configs, and VRAM estimates.
"""

# ─────────────────────────────────────────────────────────────────
# 1. MedGemma 1.5 4B — The Brain (Passes 1-5)
# ─────────────────────────────────────────────────────────────────
MEDGEMMA = {
    "id": "google/medgemma-1.5-4b-it",
    "type": "causal_lm",
    "quantization": "int4",
    "peak_vram_mb": 2800,
    "max_tokens_by_pass": {
        1: 200,   # Virtual Pathologist
        2: 200,   # Virtual Radiologist
        3: 200,   # Virtual Oncologist
        4: 600,   # Chief Physician Synthesizer
        5: 300,   # Empathetic Translator
    },
    "description": "MedGemma 1.5 4B — 5-pass persona debate engine",
}

# ─────────────────────────────────────────────────────────────────
# 2. MedASR — Ambient Transcription (Phase 1)
# ─────────────────────────────────────────────────────────────────
MEDASR = {
    "id": "google/medasr",
    "type": "asr_pipeline",
    "quantization": None,
    "peak_vram_mb": 800,
    "description": "Medical ASR — consultation transcription",
}

# ─────────────────────────────────────────────────────────────────
# 3. HeAR — Cough / Health Acoustic Representations (Phase 3A)
# ─────────────────────────────────────────────────────────────────
HEAR = {
    "id": "google/hear-pytorch",
    "id_tf": "google/hear",
    "type": "embedding",
    "quantization": None,
    "peak_vram_mb": 600,
    "embedding_dim": 512,
    "description": "HeAR — cough audio embedding for TB screening",
}

# ─────────────────────────────────────────────────────────────────
# 4. CXR Foundation — Chest X-ray Encoder (Phase 4.2)
# ─────────────────────────────────────────────────────────────────
CXR_FOUNDATION = {
    "id": "google/cxr-foundation",
    "type": "keras_encoder",
    "quantization": None,
    "peak_vram_mb": 500,
    "input_size": (224, 224),
    "description": "CXR Foundation — chest X-ray embedding encoder",
}

# ─────────────────────────────────────────────────────────────────
# 5. Derm Foundation — Skin Lesion Encoder (Phase 4.3)
# ─────────────────────────────────────────────────────────────────
DERM_FOUNDATION = {
    "id": "google/derm-foundation",
    "type": "keras_encoder",
    "quantization": None,
    "peak_vram_mb": 500,
    "embedding_dim": 1280,
    "description": "Derm Foundation — skin lesion embedding encoder",
}

# ─────────────────────────────────────────────────────────────────
# 6. Path Foundation — Histopathology Encoder (Phase 4.1)
# ─────────────────────────────────────────────────────────────────
PATH_FOUNDATION = {
    "id": "google/path-foundation",
    "type": "keras_encoder",
    "quantization": None,
    "peak_vram_mb": 500,
    "embedding_dim": 768,
    "input_size": (224, 224),
    "description": "Path Foundation — histopathology patch encoder",
}

# ─────────────────────────────────────────────────────────────────
# 7. MedSigLIP — Multimodal Embeddings + k-NN (Phase 4.4)
# ─────────────────────────────────────────────────────────────────
MEDSIGLIP = {
    "id": "google/medsiglip-448",
    "type": "siglip_encoder",
    "quantization": None,
    "peak_vram_mb": 0,  # runs on CPU
    "runs_on_cpu": True,
    "description": "MedSigLIP — multimodal embeddings for k-NN case retrieval",
}

# ─────────────────────────────────────────────────────────────────
# 8. TxGemma — Drug Safety + Inventory Routing (Phase 4.6)
# ─────────────────────────────────────────────────────────────────
TXGEMMA = {
    "id": "google/txgemma-9b-chat",
    "type": "causal_lm",
    "quantization": "int4",
    "peak_vram_mb": 5000,
    "description": "TxGemma 9B — drug interaction checking + inventory routing",
}

# ─────────────────────────────────────────────────────────────────
# Pipeline execution order (for VRAM chart annotations)
# ─────────────────────────────────────────────────────────────────
PIPELINE_ORDER = [
    ("Phase_1_MedASR",       MEDASR),
    ("Phase_3A_HeAR",        HEAR),
    ("Phase_4.1_Path",       PATH_FOUNDATION),
    ("Phase_4.2_CXR",        CXR_FOUNDATION),
    ("Phase_4.3_Derm",       DERM_FOUNDATION),
    ("Phase_4.4_MedSigLIP",  MEDSIGLIP),
    ("Phase_4.6_TxGemma",    TXGEMMA),
    ("Phase_6_MedGemma",     MEDGEMMA),
]
