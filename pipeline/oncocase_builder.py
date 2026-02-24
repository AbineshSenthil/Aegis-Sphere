"""
Aegis-Sphere — OncoCase Builder (Phase 4.7)
Builds the OncoCase JSON from the evidence pool.
Counts MISSING_DATA items, sets degradation level, builds NBA list.
"""

import os
import sys

# ── Ensure project root is on sys.path for standalone execution ──
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import (
    DegradationLevel,
    NBA_CATALOG,
    ALL_MODALITIES,
)


def build_oncocase(
    clinical_frame: dict,
    evidence_pool: list,
    risk_result: dict,
    txgemma_result: dict = None,
    similar_cases: list = None,
) -> dict:
    """
    Assemble the OncoCase JSON that drives Passes 1-5.

    Implements the Graceful Degradation decision tree:
      0 missing → FULL pipeline
      1 missing → REDUCED, inject NBA
      2 missing → PROVISIONAL staging
      3+ missing → MINIMAL (workup plan only)
      5 missing → NO_DATA (skip AI entirely)
    """
    # ── Count missing items ──
    missing_items = [e for e in evidence_pool if e.get("status") == "MISSING_DATA"]
    ok_items = [e for e in evidence_pool if e.get("status") == "OK"]
    missing_count = len(missing_items)
    total_modalities = len(ALL_MODALITIES)

    # ── Determine degradation level ──
    if missing_count == 0:
        degradation = DegradationLevel.FULL
        staging_confidence = "CONFIRMED"
    elif missing_count == 1:
        degradation = DegradationLevel.REDUCED
        staging_confidence = "CONFIRMED"
    elif missing_count == 2:
        degradation = DegradationLevel.PROVISIONAL
        staging_confidence = "PROVISIONAL"
    elif missing_count >= total_modalities:
        degradation = DegradationLevel.NO_DATA
        staging_confidence = "NO_DATA"
    else:
        degradation = DegradationLevel.MINIMAL
        staging_confidence = "INSUFFICIENT_DATA"

    # ── Path Foundation special case ──
    path_missing = any(
        e.get("model") == "Path_Foundation" and e.get("status") == "MISSING_DATA"
        for e in evidence_pool
    )
    if path_missing and staging_confidence == "CONFIRMED":
        staging_confidence = "PROVISIONAL — PATHOLOGY REQUIRED"

    # ── Build NBA list ──
    nba_list = []
    for item in missing_items:
        model = item.get("model", "")
        nba_entry = item.get("nba") or NBA_CATALOG.get(model, {}).get("nba", "")
        cost = NBA_CATALOG.get(model, {}).get("cost_inr", "N/A")
        patient_lang = NBA_CATALOG.get(model, {}).get("patient_language", "")
        if nba_entry:
            nba_list.append({
                "model": model,
                "nba": nba_entry,
                "cost_inr": cost,
                "patient_language": patient_lang,
                "priority": _get_nba_priority(model),
            })

    # Sort NBA by priority (lower = more urgent)
    nba_list.sort(key=lambda x: x["priority"])

    # ── Build findings summary ──
    findings = {}
    for item in ok_items:
        model = item.get("model", "unknown")
        findings[model] = item.get("finding", "No finding available.")

    # ── Assemble OncoCase ──
    oncocase = {
        "session_id": risk_result.get("session_id", "demo"),
        "clinical_frame": clinical_frame,
        "evidence_pool": evidence_pool,
        "findings": findings,
        "risk_assessment": risk_result,
        "degradation_level": degradation,
        "staging_confidence": staging_confidence,
        "missing_count": missing_count,
        "missing_modalities": [e.get("model", "") for e in missing_items],
        "nba_list": nba_list,
        "path_missing": path_missing,
        "conditions": clinical_frame.get("conditions", []),
        "medications": clinical_frame.get("medications", []),
        "demographics": clinical_frame.get("demographics", {}),
        "proposed_regimen": _suggest_regimen(clinical_frame, findings),
        "proposed_drugs": _suggest_drugs(clinical_frame, findings),
        "pass_config": _get_pass_config(degradation),
    }

    # ── TxGemma results ──
    if txgemma_result:
        oncocase["tx_analysis"] = txgemma_result
        oncocase["interaction_flags"] = txgemma_result.get("interaction_flags", [])
        oncocase["inventory_alerts"] = txgemma_result.get("inventory_alerts", [])
    else:
        oncocase["tx_analysis"] = None
        oncocase["interaction_flags"] = []
        oncocase["inventory_alerts"] = []

    # ── Similar cases ──
    oncocase["similar_cases"] = similar_cases or []

    return oncocase


def _get_pass_config(degradation: str) -> dict:
    """Determine which MedGemma passes to run based on degradation level."""
    if degradation == DegradationLevel.NO_DATA:
        return {
            "run_passes": [],
            "mode": "NO_AI",
            "note": "No clinical data available. No AI inference runs.",
        }
    elif degradation == DegradationLevel.MINIMAL:
        return {
            "run_passes": [4, 5],  # Chief Physician (workup only) + Patient Handout
            "mode": "WORKUP_ONLY",
            "note": "Minimal data mode. Workup plan generated, not a treatment plan.",
        }
    else:
        return {
            "run_passes": [1, 2, 3, 4, 5],  # All passes
            "mode": "FULL" if degradation in (DegradationLevel.FULL, DegradationLevel.REDUCED) else "PROVISIONAL",
            "note": None,
        }


def _get_nba_priority(model: str) -> int:
    """Priority for NBA ordering (lower = more urgent)."""
    priorities = {
        "Path_Foundation": 1,   # Most critical — can't stage without pathology
        "CXR_Foundation": 2,    # Needed for cardiopulmonary baseline
        "HeAR": 3,              # TB screening
        "Derm_Foundation": 4,   # Skin confirmation
        "MedASR": 5,            # Least urgent
    }
    return priorities.get(model, 10)


def _suggest_regimen(clinical_frame: dict, findings: dict) -> str:
    """Suggest a regimen based on clinical frame (heuristic for prompt injection)."""
    conditions = [c.lower() for c in clinical_frame.get("conditions", [])]
    if any("lymphoma" in c for c in conditions):
        return "CHOP"
    elif any("kaposi" in c for c in conditions):
        return "Liposomal Doxorubicin + ART optimization"
    elif any("cervical" in c for c in conditions):
        return "Cisplatin + RT"
    elif any("lung" in c or "adenocarcinoma" in c for c in conditions):
        return "Carboplatin + Paclitaxel"
    return "CHOP"  # default for HIV-associated lymphoma


def _suggest_drugs(clinical_frame: dict, findings: dict) -> list:
    """Suggest drugs for the proposed regimen."""
    conditions = [c.lower() for c in clinical_frame.get("conditions", [])]
    if any("lymphoma" in c for c in conditions):
        return ["cyclophosphamide", "doxorubicin", "vincristine", "prednisone"]
    elif any("kaposi" in c for c in conditions):
        return ["liposomal_doxorubicin"]
    return ["cyclophosphamide", "doxorubicin", "vincristine", "prednisone"]
