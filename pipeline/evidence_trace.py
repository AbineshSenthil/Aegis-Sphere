"""
Aegis-Sphere — Evidence Trace Builder
Builds structured evidence_trace dict from all tagged outputs AND
synthesises claims from evidence pool, clinical frame, and worker results.
"""

from __future__ import annotations

import re
from typing import Optional


# ── Model-name normalization map ──
_MODEL_ALIASES = {
    "TxGemma_DDI": "TxGemma",
    "MedASR": "MedASR_Transcript",
    "MedSigLIP": "MedSigLIP_CaseLibrary",
    "Clinical_Frame": "Clinical_Frame_JSON",
    "Local_Inventory": "Local_Inventory_JSON",
}


def _normalize(model_name: str) -> str:
    """Normalize a model name to its canonical evidence-trace key."""
    return _MODEL_ALIASES.get(model_name, model_name)


def build_evidence_trace(outputs: dict) -> dict:
    """
    Build evidence_trace dict: {model_name: [claim1, claim2, ...]}

    Args:
        outputs: dict with keys like 'pass1_pathologist', 'pass4_chief', 'tagged_output'

    Returns:
        Dict mapping source model names to lists of claims.
    """
    trace = {}
    pattern = r'([^.!?\n]*?\[Source:\s*(\w+(?:_\w+)*)\][^.!?\n]*[.!?\n]?)'

    # Collect all text outputs
    all_text = []
    for key, value in outputs.items():
        if isinstance(value, str) and value:
            all_text.append(value)
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict) and "output" in item:
                    all_text.append(item["output"])
                elif isinstance(item, str):
                    all_text.append(item)

    # Parse tags from all text
    for text in all_text:
        for match in re.finditer(pattern, text):
            full_claim = match.group(1).strip()
            source = _normalize(match.group(2))
            clean_claim = re.sub(r'\[Source:\s*\w+(?:_\w+)*\]', '', full_claim).strip()
            if not clean_claim:
                continue
            if source not in trace:
                trace[source] = []
            if clean_claim not in trace[source]:
                trace[source].append(clean_claim)

    return trace


def build_comprehensive_trace(
    debate_outputs: Optional[dict] = None,
    evidence_pool: Optional[list] = None,
    tx_result: Optional[dict] = None,
    transcript: Optional[str] = None,
    clinical_frame: Optional[dict] = None,
    oncocase: Optional[dict] = None,
) -> dict:
    """
    Build a comprehensive evidence trace that covers ALL 9 model sources.

    Combines:
    1. [Source: X] tags parsed from debate + TxGemma text (existing logic)
    2. Structured claims synthesised from evidence pool items
    3. Transcript → MedASR_Transcript
    4. Clinical frame → Clinical_Frame_JSON
    5. TxGemma interactions → TxGemma
    6. Inventory alerts → Local_Inventory_JSON
    7. Similar cases → MedSigLIP_CaseLibrary
    """
    # ── Step 1: tag-based extraction from text ──
    trace = build_evidence_trace(debate_outputs or {})

    # ── Step 2: evidence pool → model claims ──
    for ev in (evidence_pool or []):
        model = ev.get("model", "")
        status = ev.get("status", "")
        finding = ev.get("finding")
        canonical = _normalize(model)

        if status == "OK" and finding:
            _add_claim(trace, canonical, finding)
        elif status == "MISSING_DATA":
            nba = ev.get("nba")
            if nba:
                _add_claim(trace, canonical, f"[MISSING] {nba}")

    # ── Step 3: transcript → MedASR_Transcript ──
    if transcript:
        snippet = transcript[:200].strip()
        if snippet:
            _add_claim(trace, "MedASR_Transcript", snippet)

    # ── Step 4: clinical frame → Clinical_Frame_JSON ──
    if clinical_frame:
        for field in ("symptoms", "medications", "conditions"):
            items = clinical_frame.get(field, [])
            if items:
                _add_claim(trace, "Clinical_Frame_JSON",
                           f"{field.title()}: {', '.join(items[:6])}")
        demographics = clinical_frame.get("demographics", {})
        if demographics:
            demo_parts = [f"{k}: {v}" for k, v in demographics.items() if v]
            if demo_parts:
                _add_claim(trace, "Clinical_Frame_JSON",
                           f"Demographics: {', '.join(demo_parts[:4])}")

    # ── Step 5: TxGemma interactions → TxGemma ──
    if tx_result:
        tagged_output = tx_result.get("tagged_output", "")
        if tagged_output:
            # Parse tags from tagged output (maps TxGemma_DDI → TxGemma)
            _merge_traces(trace, build_evidence_trace({"txgemma": tagged_output}))

        for ix in tx_result.get("interaction_flags", []):
            if isinstance(ix, dict):
                severity = ix.get("severity", "")
                drugs = ix.get("drugs", "")
                detail = ix.get("detail", ix.get("text", ""))
                claim = f"[{severity}] {drugs}: {detail}" if drugs else f"[{severity}] {detail}"
                _add_claim(trace, "TxGemma", claim[:200])

    # ── Step 6: inventory alerts → Local_Inventory_JSON ──
    if tx_result:
        for alert in tx_result.get("inventory_alerts", []):
            if isinstance(alert, dict):
                msg = alert.get("message", "")
                if msg:
                    _add_claim(trace, "Local_Inventory_JSON", msg)

    # ── Step 7: similar cases → MedSigLIP_CaseLibrary ──
    if oncocase:
        similar = oncocase.get("similar_cases", [])
        for case in similar[:5]:
            if isinstance(case, dict):
                case_id = case.get("case_id", "Unknown")
                diagnosis = case.get("diagnosis", "N/A")
                score = case.get("similarity_score", 0)
                _add_claim(trace, "MedSigLIP_CaseLibrary",
                           f"Case {case_id}: {diagnosis} (similarity: {score:.2f})")

    # ── Step 8: ensure ALL evidence-pool models appear in the trace ──
    #    This prevents the Source Coverage panel from showing "No data" for
    #    models that processed data but whose outputs didn't contain [Source:] tags.
    _canonical_models = [
        "Path_Foundation", "CXR_Foundation", "HeAR", "Derm_Foundation",
        "TxGemma", "Local_Inventory_JSON", "MedSigLIP_CaseLibrary",
        "MedASR_Transcript", "Clinical_Frame_JSON",
    ]
    for ev in (evidence_pool or []):
        model = _normalize(ev.get("model", ""))
        if model and model not in trace:
            status = ev.get("status", "")
            finding = ev.get("finding")
            nba = ev.get("nba")
            if status == "OK" and finding:
                _add_claim(trace, model, finding)
            elif status == "MISSING_DATA" and nba:
                _add_claim(trace, model, f"[MISSING] {nba}")
            elif status == "MISSING_DATA":
                _add_claim(trace, model, f"[MISSING] No data available for {model}")
            else:
                _add_claim(trace, model, f"Data processed by {model}")

    return trace


def _add_claim(trace: dict, source: str, claim: str):
    """Add a claim to the trace if not already present."""
    if not claim:
        return
    if source not in trace:
        trace[source] = []
    clean = claim.strip()
    if clean and clean not in trace[source]:
        trace[source].append(clean)


def _merge_traces(target: dict, new: dict):
    """Merge new trace into target, deduplicating claims."""
    for source, claims in new.items():
        for claim in claims:
            _add_claim(target, source, claim)


def get_source_counts(trace: dict) -> dict:
    """Get count of claims per source model."""
    return {model: len(claims) for model, claims in trace.items()}


def get_all_sources(trace: dict) -> list:
    """Get sorted list of all source model names."""
    return sorted(trace.keys())
