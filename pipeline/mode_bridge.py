"""
Aegis-Sphere — Mode Bridge (Phase 3B)
Escalation logic: determines whether a case stays in TB/HIV triage
or escalates to full OncoSphere oncology workup.

This is the "Wow Moment" — a tuberculosis consult that automatically
triggers a comprehensive oncology pipeline when suspicious keywords
(lymphoma, mass, malignancy, etc.) are detected in the clinical frame.
"""

from typing import Optional

# ─────────────────────────────────────────────────────────────
# Oncology trigger keywords (case-insensitive matching)
# ─────────────────────────────────────────────────────────────
ONCOLOGY_TRIGGERS = [
    "lymphoma", "malignancy", "cancer", "tumor", "tumour",
    "metastasis", "metastatic", "carcinoma", "sarcoma",
    "kaposi", "mass", "neoplasm", "neoplastic", "oncology",
    "adenocarcinoma", "leukemia", "myeloma", "hodgkin",
    "non-hodgkin", "staging", "biopsy",
]

# Keywords that increase uncertainty even in TB-only mode
TB_HIV_COINFECTION_KEYWORDS = [
    "hiv", "cd4", "art", "antiretroviral", "viral load",
    "immunocompromised", "immunosuppressed",
]


def evaluate_escalation(
    clinical_frame: dict,
    asr_result: Optional[dict] = None,
) -> dict:
    """
    Evaluate whether a case should escalate from TB triage to OncoSphere.

    Decision tree:
      1. If ASR result is MISSING_DATA → uncertainty = CRITICAL
      2. Scan conditions, symptoms, medications for oncology triggers
      3. If any trigger found → mode = ONCOSPHERE
      4. If TB/HIV coinfection detected → increase uncertainty
      5. Otherwise → mode = TB_TRIAGE

    Args:
        clinical_frame: Extracted NER clinical frame dict
        asr_result: ASR worker result dict (may be None or MISSING_DATA)

    Returns:
        {
            "mode": "TB_TRIAGE" | "ONCOSPHERE",
            "triggers": [list of matched trigger keywords],
            "uncertainty": "LOW" | "MEDIUM" | "HIGH" | "CRITICAL",
            "rationale": str,
            "coinfection_flags": [list of HIV/TB flags detected],
        }
    """
    triggers_found = []
    coinfection_flags = []
    uncertainty = "LOW"
    rationale_parts = []

    # ── Check if ASR data is available ──
    asr_missing = False
    if asr_result is None:
        asr_missing = True
    elif isinstance(asr_result, dict):
        if asr_result.get("status") == "MISSING_DATA":
            asr_missing = True
        elif asr_result.get("evidence_item", {}).get("status") == "MISSING_DATA":
            asr_missing = True

    if asr_missing:
        uncertainty = "CRITICAL"
        rationale_parts.append(
            "Audio data unavailable — escalation assessment based on "
            "uploaded data and clinical history only."
        )

    # ── Collect all text fields from clinical frame ──
    searchable_texts = []
    for field in ["conditions", "symptoms", "medications", "lab_values"]:
        values = clinical_frame.get(field, [])
        if isinstance(values, list):
            searchable_texts.extend([str(v).lower() for v in values])
        elif isinstance(values, str):
            searchable_texts.append(values.lower())

    # Include demographics text if present
    demographics = clinical_frame.get("demographics", {})
    if isinstance(demographics, dict):
        for v in demographics.values():
            searchable_texts.append(str(v).lower())

    combined_text = " ".join(searchable_texts)

    # ── Scan for oncology triggers ──
    for trigger in ONCOLOGY_TRIGGERS:
        if trigger in combined_text:
            triggers_found.append(trigger)

    # ── Scan for TB/HIV coinfection ──
    for keyword in TB_HIV_COINFECTION_KEYWORDS:
        if keyword in combined_text:
            coinfection_flags.append(keyword)

    # ── Determine mode ──
    if triggers_found:
        mode = "ONCOSPHERE"
        rationale_parts.append(
            f"Oncology triggers detected: {', '.join(triggers_found)}. "
            f"Escalating from TB/HIV triage to full OncoSphere workup."
        )
        # Adjust uncertainty based on trigger strength
        if not asr_missing:
            if len(triggers_found) >= 3:
                uncertainty = "LOW"
            elif len(triggers_found) >= 2:
                uncertainty = "MEDIUM"
            else:
                uncertainty = "MEDIUM"
    else:
        mode = "TB_TRIAGE"
        rationale_parts.append(
            "No oncology triggers detected. Remaining in TB/HIV triage mode."
        )

    # ── Coinfection escalation factor ──
    if coinfection_flags and mode == "TB_TRIAGE":
        rationale_parts.append(
            f"TB/HIV coinfection indicators detected: {', '.join(coinfection_flags)}. "
            f"Monitor for opportunistic malignancies."
        )
        if uncertainty == "LOW":
            uncertainty = "MEDIUM"

    # ── Build final rationale ──
    rationale = " ".join(rationale_parts) if rationale_parts else "Standard triage assessment."

    return {
        "mode": mode,
        "triggers": triggers_found,
        "uncertainty": uncertainty,
        "rationale": rationale,
        "coinfection_flags": coinfection_flags,
    }


def format_escalation_display(result: dict) -> dict:
    """
    Format escalation result for UI display.

    Returns dict with icon, color, label, and description for rendering.
    """
    if result["mode"] == "ONCOSPHERE":
        return {
            "icon": "🚨",
            "color": "#ef4444",
            "bg_color": "rgba(239, 68, 68, 0.12)",
            "border_color": "rgba(239, 68, 68, 0.3)",
            "label": "ONCOSPHERE ESCALATION",
            "sublabel": f"Triggers: {', '.join(result['triggers'][:3])}",
            "description": result["rationale"],
        }
    else:
        return {
            "icon": "🩺",
            "color": "#22c55e",
            "bg_color": "rgba(34, 197, 94, 0.12)",
            "border_color": "rgba(34, 197, 94, 0.3)",
            "label": "TB/HIV TRIAGE",
            "sublabel": "Standard assessment mode",
            "description": result["rationale"],
        }
