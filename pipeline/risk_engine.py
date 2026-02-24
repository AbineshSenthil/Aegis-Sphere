"""
Aegis-Sphere — Risk Engine (Phase 5)
TB/HIV risk scoring with uncertainty propagation.
"""

import os
import sys

# ── Ensure project root is on sys.path for standalone execution ──
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import UNCERTAINTY_FLAGS


def compute_risk(
    clinical_frame: dict,
    evidence_pool: list,
    asr_confidence_flags: list = None,
) -> dict:
    """
    Compute TB/HIV risk scores and propagate uncertainty from missing data.

    Returns:
        {
            "tb_risk_score": float (0-1),
            "tb_risk_level": "HIGH" | "MODERATE" | "LOW",
            "hiv_risk_score": float (0-1),
            "overall_risk_level": "RED" | "AMBER" | "GREEN",
            "uncertainty_flags": [...],
            "staging_override": str | None,
            "treatment_override": str | None,
        }
    """
    uncertainty_flags = []
    asr_confidence_flags = asr_confidence_flags or []

    # ── Collect flags from ASR confidence ──
    if "LOW_AUDIO_CONFIDENCE" in asr_confidence_flags:
        uncertainty_flags.append("LOW_AUDIO_CONFIDENCE")

    # ── Propagate uncertainty from evidence pool ──
    missing_models = set()
    for ev in evidence_pool:
        if ev.get("status") == "MISSING_DATA":
            model = ev.get("model", "")
            missing_models.add(model)
            if model == "HeAR":
                uncertainty_flags.append("NO_RESPIRATORY_DATA")
            elif model == "CXR_Foundation":
                uncertainty_flags.append("NO_CXR_DATA")
            elif model == "Path_Foundation":
                uncertainty_flags.append("NO_PATH_DATA")
            elif model == "Derm_Foundation":
                uncertainty_flags.append("NO_DERM_DATA")

    # ── Count total missing ──
    missing_count = sum(1 for e in evidence_pool if e.get("status") == "MISSING_DATA")
    if missing_count >= 3:
        uncertainty_flags.append("INSUFFICIENT_DATA")

    # ── TB risk scoring ──
    tb_risk_score = _compute_tb_score(clinical_frame, evidence_pool)

    # ── HIV risk scoring ──
    hiv_risk_score = _compute_hiv_score(clinical_frame)

    # ── Combined risk level ──
    max_risk = max(tb_risk_score, hiv_risk_score)
    if max_risk > 0.7:
        overall_risk_level = "RED"
    elif max_risk > 0.4:
        overall_risk_level = "AMBER"
    else:
        overall_risk_level = "GREEN"

    # ── TB risk level ──
    if tb_risk_score > 0.7:
        tb_risk_level = "HIGH"
    elif tb_risk_score > 0.4:
        tb_risk_level = "MODERATE"
    else:
        tb_risk_level = "LOW"

    # ── Staging and treatment overrides ──
    staging_override = None
    treatment_override = None

    if "NO_CXR_DATA" in uncertainty_flags:
        staging_override = "PROVISIONAL"
    if "NO_PATH_DATA" in uncertainty_flags:
        staging_override = "PROVISIONAL — PATHOLOGY REQUIRED"
    if missing_count >= 3:
        staging_override = "INSUFFICIENT_DATA"

    # Check TxGemma status
    tx_items = [e for e in evidence_pool if e.get("model") == "TxGemma"]
    if tx_items and tx_items[0].get("status") == "BLOCKED":
        treatment_override = "RECOMMENDATION_ONLY — NOT PRESCRIPTION"
        uncertainty_flags.append("RECOMMENDATION_ONLY")

    return {
        "tb_risk_score": round(tb_risk_score, 3),
        "tb_risk_level": tb_risk_level,
        "hiv_risk_score": round(hiv_risk_score, 3),
        "overall_risk_level": overall_risk_level,
        "uncertainty_flags": list(set(uncertainty_flags)),
        "staging_override": staging_override,
        "treatment_override": treatment_override,
        "missing_count": missing_count,
    }


def _compute_tb_score(clinical_frame: dict, evidence_pool: list) -> float:
    """Compute TB risk score from clinical indicators."""
    score = 0.0

    symptoms = [s.lower() for s in clinical_frame.get("symptoms", [])]
    conditions = [c.lower() for c in clinical_frame.get("conditions", [])]

    # Symptom-based scoring
    tb_symptoms = {
        "cough": 0.15,
        "night sweats": 0.15,
        "weight loss": 0.15,
        "fever": 0.10,
        "fatigue": 0.05,
    }
    for symptom, weight in tb_symptoms.items():
        if any(symptom in s for s in symptoms):
            score += weight

    # Condition-based
    if any("tb" in c or "tuberculosis" in c for c in conditions):
        score += 0.25
    if any("hiv" in c for c in conditions):
        score += 0.10  # HIV increases TB risk

    # HeAR score
    for ev in evidence_pool:
        if ev.get("model") == "HeAR" and ev.get("status") == "OK":
            conf = ev.get("confidence")
            if conf and conf > 0.5:
                score += 0.15

    # CXR findings
    for ev in evidence_pool:
        if ev.get("model") == "CXR_Foundation" and ev.get("status") == "OK":
            finding = (ev.get("finding") or "").lower()
            if "infiltrate" in finding or "opacity" in finding:
                score += 0.10

    return min(score, 1.0)


def _compute_hiv_score(clinical_frame: dict) -> float:
    """Compute HIV risk score from clinical indicators."""
    score = 0.0

    conditions = [c.lower() for c in clinical_frame.get("conditions", [])]
    lab_values = clinical_frame.get("lab_values", [])

    if any("hiv" in c for c in conditions):
        score += 0.5

    # CD4 count
    for lab in lab_values:
        lab_lower = lab.lower()
        if "cd4" in lab_lower:
            import re
            numbers = re.findall(r'\d+', lab)
            if numbers:
                cd4 = int(numbers[-1])
                if cd4 < 100:
                    score += 0.35
                elif cd4 < 200:
                    score += 0.25
                elif cd4 < 350:
                    score += 0.10

    # HIV-associated conditions
    if any("lymphoma" in c for c in conditions):
        score += 0.10
    if any("kaposi" in c for c in conditions):
        score += 0.15

    return min(score, 1.0)
