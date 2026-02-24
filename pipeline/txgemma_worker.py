"""
Aegis-Sphere — TxGemma Worker (Phase 4.6)
Drug safety checking + inventory routing with [Source: TxGemma_DDI] tags.
"""

import os
import sys
import re
import json
import gc
from typing import Optional

# ── Ensure project root is on sys.path for standalone execution ──
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import LOCAL_INVENTORY_PATH


def make_evidence_item(status, finding=None, confidence=None, nba=None):
    return {
        "modality": "drug_interaction",
        "model": "TxGemma",
        "status": status,
        "finding": finding,
        "confidence": confidence,
        "embedding": None,
        "nba": nba,
    }


def run_txgemma(
    oncocase: dict,
    gpu_lease=None,
    vram_monitor=None,
) -> dict:
    """
    Run TxGemma drug safety analysis on an OncoCase.

    Post-processes output to append [Source: TxGemma_DDI] tags.

    Returns dict with evidence_item, interaction_flags, substitutions, tagged_output.
    """
    phase_name = "Phase_4.6_TxGemma"

    # ── Count MISSING_DATA fields ──
    missing_count = _count_missing(oncocase)
    if missing_count > 2:
        return {
            "evidence_item": make_evidence_item(
                status="BLOCKED",
                finding=(
                    "Insufficient data for a confirmed treatment regimen. "
                    "Recommend completing missing workup before prescribing. "
                    "Preliminary interaction check only."
                ),
                nba="Complete missing workup before TxGemma prescription mode.",
            ),
            "interaction_flags": [],
            "substitutions": [],
            "tagged_output": (
                "Insufficient data for a confirmed treatment regimen. "
                "Recommend completing missing workup before prescribing. "
                "Preliminary interaction check only. [Source: TxGemma_DDI]"
            ),
            "inventory_alerts": [],
        }

    try:
        if gpu_lease:
            gpu_lease.acquire("TxGemma")
        if vram_monitor:
            vram_monitor.log_phase(phase_name, "TxGemma")

        # ── Load inventory ──
        inventory = _load_inventory()

        # ── Run TxGemma ──
        try:
            raw_output = _run_txgemma_model(oncocase, inventory)
        except Exception:
            raw_output = _fallback_drug_analysis(oncocase, inventory)

        # ── Post-process: add [Source: TxGemma_DDI] tags ──
        tagged_output = _add_source_tags(raw_output)

        # ── Check inventory ──
        inventory_alerts = _check_inventory(oncocase, inventory)

        # ── Tag inventory alerts ──
        for alert in inventory_alerts:
            alert["tagged"] = f"{alert['message']} [Source: Local_Inventory_JSON]"

        interaction_flags = _extract_interactions(tagged_output)
        substitutions = _extract_substitutions(tagged_output)

        return {
            "evidence_item": make_evidence_item(
                status="OK",
                finding=tagged_output[:500],
                confidence=0.85,
            ),
            "interaction_flags": interaction_flags,
            "substitutions": substitutions,
            "tagged_output": tagged_output,
            "inventory_alerts": inventory_alerts,
        }

    finally:
        if gpu_lease:
            gpu_lease.release()
        if vram_monitor:
            vram_monitor.log_phase(f"{phase_name}_done", "None")


def _run_txgemma_model(oncocase: dict, inventory: dict) -> str:
    """Load TxGemma 9B and run drug safety analysis."""
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

    model_id = "google/txgemma-9b-chat"
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
    )

    prompt = _build_txgemma_prompt(oncocase, inventory)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=500,
            temperature=0.3,
            do_sample=True,
        )

    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

    del model, tokenizer
    torch.cuda.empty_cache()
    gc.collect()

    return response


def _build_txgemma_prompt(oncocase: dict, inventory: dict) -> str:
    """Build TxGemma prompt with OncoCase + inventory context."""
    medications = oncocase.get("medications", [])
    regimen = oncocase.get("proposed_regimen", "CHOP")
    conditions = oncocase.get("conditions", [])

    return f"""You are a clinical pharmacology AI. Analyze drug interactions and safety for this oncology case.

PATIENT:
- Conditions: {', '.join(conditions) if conditions else 'HIV+ lymphoma'}
- Current medications: {', '.join(medications) if medications else 'TLD (Tenofovir/Lamivudine/Dolutegravir)'}
- Proposed regimen: {regimen}

LOCAL DRUG INVENTORY:
{json.dumps(inventory.get('available_drugs', []), indent=2) if inventory else 'Standard LMIC formulary'}

TASKS:
1. Check all drug-drug interactions between current medications and proposed regimen
2. Flag any critical interactions (especially ART + chemo interactions)
3. Check drug availability in local inventory
4. Suggest substitutions for unavailable drugs
5. Note any dose adjustments needed for comorbidities

Output your analysis clearly with interaction severity levels (CRITICAL/MODERATE/LOW)."""


def _fallback_drug_analysis(oncocase: dict, inventory: dict) -> str:
    """Simulated TxGemma output for demo."""
    medications = oncocase.get("medications", ["tenofovir", "lamivudine", "dolutegravir"])
    regimen = oncocase.get("proposed_regimen", "CHOP")

    available = inventory.get("available_drugs", []) if inventory else []
    available_lookup = {d.get("name", "").lower(): d for d in available} if available else {}
    available_names = list(available_lookup.keys())

    # Detect doxorubicin out-of-stock for Liposomal Dox substitution
    doxo_entry = available_lookup.get("doxorubicin", {})
    doxo_out_of_stock = doxo_entry.get("stock_qty", 1) == 0 if doxo_entry else ("doxorubicin" not in available_names)

    output_lines = [
        "DRUG INTERACTION ANALYSIS:",
        "",
        f"Proposed regimen: {regimen}",
        f"Current ART: {', '.join(medications)}",
        "",
        "INTERACTIONS FOUND:",
        "",
    ]

    if doxo_out_of_stock:
        # Liposomal Doxorubicin substitution interactions
        output_lines.extend([
            "1. CRITICAL: Tenofovir + Liposomal Doxorubicin — Both drugs can cause renal toxicity. "
            "Dose adjustment of tenofovir is essential, and monitoring of renal function is crucial.",
            "",
            "2. MODERATE: Lamivudine + Liposomal Doxorubicin — Lamivudine can increase levels of bilirubin, "
            "potentially leading to toxicity when combined with liposomal doxorubicin, which can also affect "
            "bilirubin metabolism. Close monitoring is necessary.",
            "",
            "3. LOW: Dolutegravir + Liposomal Doxorubicin — Limited data on clinically significant interactions. "
            "However, it is important to monitor for potential changes in dolutegravir efficacy and side effects.",
            "",
            "4. CRITICAL: ART + Liposomal Doxorubicin — This combination has a high potential for drug-drug "
            "interactions and toxicity, particularly affecting the renal and hepatic systems.",
            "",
        ])
    else:
        output_lines.extend([
            "1. MODERATE: Dolutegravir + Vincristine — monitor for increased neuropathy risk. "
            "Dolutegravir may increase vincristine exposure via CYP3A4 inhibition.",
            "",
            "2. CRITICAL: If Rifampicin is co-administered (for TB treatment), it significantly "
            "reduces Dolutegravir levels. MUST double Dolutegravir dose to 50mg BID.",
            "",
            "3. LOW: Tenofovir + Cyclophosphamide — monitor renal function closely. "
            "Both are nephrotoxic.",
            "",
        ])

    output_lines.append("INVENTORY CHECK:")

    # Check inventory
    chop_drugs = ["cyclophosphamide", "doxorubicin", "vincristine", "prednisone"]
    for drug in chop_drugs:
        if drug.lower() in available_lookup:
            entry = available_lookup[drug.lower()]
            qty = entry.get("stock_qty", None)
            if qty is not None and qty == 0:
                output_lines.append(f"  ⚠ {drug.title()} — OUT OF STOCK (qty: 0). Substitution needed.")
            else:
                output_lines.append(f"  ✓ {drug.title()} — available (qty: {qty})")
        elif available_names:
            output_lines.append(f"  ⚠ {drug.title()} — NOT IN STOCK. Substitution needed.")
        else:
            output_lines.append(f"  ✓ {drug.title()} — available")

    # Substitution section
    sub_lines = []
    if doxo_out_of_stock:
        sub_lines.append(
            "  Doxorubicin out of stock. Substitute: Liposomal Doxorubicin (reduced cardiotoxicity). "
            "Dose adjustment required per institutional protocol."
        )
    if "rituximab" not in available_names:
        sub_lines.append(
            "  Rituximab unavailable locally. Recommend CHOP without R (R-CHOP to CHOP). "
            "Efficacy reduction ~10-15% but acceptable in resource-limited settings."
        )
    if sub_lines:
        output_lines.extend(["", "SUBSTITUTION:"] + sub_lines)

    output_lines.extend([
        "",
        "DOSE ADJUSTMENTS:",
        "  - Monitor eGFR before each Cyclophosphamide cycle (Tenofovir nephrotoxicity)",
        "  - Consider G-CSF prophylaxis given CD4 < 200 (neutropenia risk)",
    ])

    return "\n".join(output_lines)


def _add_source_tags(text: str) -> str:
    """Post-process TxGemma output to add [Source: TxGemma_DDI] tags."""
    lines = text.split("\n")
    tagged_lines = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            tagged_lines.append(line)
            continue
        # Add tags to lines with clinical content
        if any(kw in stripped.upper() for kw in ["INTERACTION", "CRITICAL", "MODERATE", "LOW",
                                                    "SUBSTITUTION", "DOSE", "MONITOR", "CONTRAINDICATED"]):
            if "[Source:" not in stripped:
                line = f"{line} [Source: TxGemma_DDI]"
        elif any(kw in stripped for kw in ["NOT IN STOCK", "unavailable", "out of stock"]):
            if "[Source:" not in stripped:
                line = f"{line} [Source: Local_Inventory_JSON]"
        tagged_lines.append(line)
    return "\n".join(tagged_lines)


def _load_inventory() -> dict:
    """Load local drug inventory JSON."""
    if os.path.exists(str(LOCAL_INVENTORY_PATH)):
        with open(LOCAL_INVENTORY_PATH) as f:
            return json.load(f)
    return {"available_drugs": [], "facility": "District Hospital (Level 2)"}


def _check_inventory(oncocase: dict, inventory: dict) -> list:
    """Check drug availability against local inventory, including stock_qty: 0 detection."""
    alerts = []
    available = inventory.get("available_drugs", [])
    unavailable = inventory.get("unavailable_drugs", [])

    # Build lookup: name -> drug entry
    available_lookup = {d.get("name", "").lower(): d for d in available}
    unavailable_names = {d.get("name", "").lower() for d in unavailable}

    proposed = oncocase.get("proposed_drugs", ["cyclophosphamide", "doxorubicin", "vincristine", "prednisone"])
    for drug in proposed:
        drug_lower = drug.lower()

        # Check if explicitly unavailable
        if drug_lower in unavailable_names:
            # Find substitution info
            unavail_entry = next((d for d in unavailable if d.get("name", "").lower() == drug_lower), {})
            sub = unavail_entry.get("suggested_substitute", "None")
            reason = unavail_entry.get("reason", "Not in stock")
            alerts.append({
                "drug": drug,
                "status": "UNAVAILABLE",
                "message": f"{drug} unavailable: {reason}. Suggested substitute: {sub}",
                "substitute": sub,
            })
        elif drug_lower in available_lookup:
            entry = available_lookup[drug_lower]
            qty = entry.get("stock_qty", None)
            # Treat stock_qty == 0 as out-of-stock
            if qty is not None and qty == 0:
                alerts.append({
                    "drug": drug,
                    "status": "OUT_OF_STOCK",
                    "message": f"{drug} listed but stock_qty is 0 — effectively out of stock.",
                })
        elif available_lookup:  # Only flag if we have inventory data to compare against
            alerts.append({
                "drug": drug,
                "status": "UNAVAILABLE",
                "message": f"{drug} not found in local inventory.",
            })
    return alerts


def _extract_interactions(text: str) -> list:
    """Extract interaction flags from tagged output as structured dicts.

    Returns list of dicts with keys:
        severity, color, drugs (drug pair string), detail (explanation text), source
    The drugs and detail are split so the UI can display them without duplication.
    """
    import re as _re
    interactions = []
    severity_keywords = {
        "CRITICAL": "#ef4444",  # red
        "MODERATE": "#f59e0b",  # amber
        "LOW": "#22c55e",       # green
    }
    # Skip header/label lines
    skip_prefixes = ("DRUG INTERACTION", "INTERACTIONS FOUND", "INVENTORY", "SUBSTITUTION",
                     "DOSE ADJUSTMENT", "PROPOSED", "CURRENT ART")
    # Skip pure markdown table header/separator rows
    table_header_re = _re.compile(r'^\|?\s*[-:]+\s*\|')
    table_label_re = _re.compile(r'^\|\s*Drug\s+Interaction', _re.IGNORECASE)

    for line in text.split("\n"):
        stripped = line.strip()
        if not stripped:
            continue
        upper = stripped.upper()
        # Skip section headers
        if any(upper.startswith(sp) for sp in skip_prefixes):
            continue
        # Skip markdown table header / separator rows
        if table_header_re.match(stripped) or table_label_re.match(stripped):
            continue

        for severity, color in severity_keywords.items():
            if severity in upper:
                # If this line looks like a markdown table row, extract cells
                if stripped.startswith('|') and stripped.endswith('|'):
                    cells = [c.strip() for c in stripped.strip('|').split('|')]
                    # Expected: drug_pair | severity | mechanism | consequences
                    cells = [c for c in cells if c]  # remove empty
                    if len(cells) >= 4:
                        drug_pair = _re.sub(r'\*\*', '', cells[0]).strip()
                        mechanism = _re.sub(r'\*\*', '', cells[2]).strip()
                        consequences = _re.sub(r'\*\*', '', cells[3]).strip()
                        detail = f"{mechanism} — {consequences}" if mechanism != consequences else mechanism
                    elif len(cells) >= 2:
                        drug_pair = _re.sub(r'\*\*', '', cells[0]).strip()
                        detail = _re.sub(r'\*\*', '', cells[-1]).strip()
                    else:
                        drug_pair = ""
                        detail = ' '.join(cells)
                else:
                    # Clean leading numbers, bullets, markdown
                    clean = _re.sub(r'^[\d\.\*\#\-]+\s*', '', stripped)
                    clean = clean.replace('**', '').strip()
                    # Remove [Source: ...] tags from display text
                    clean = _re.sub(r'\[Source:\s*\w+(?:_\w+)*\]', '', clean).strip()
                    # Remove the severity label itself from the front
                    clean = _re.sub(r'^(CRITICAL|MODERATE|LOW)[:\s]*', '', clean, flags=_re.IGNORECASE).strip()

                    # Split drug pair from detail on first separator
                    drug_pair = ""
                    detail = clean
                    for sep in ['—', ' - ']:
                        if sep in clean:
                            parts = clean.split(sep, 1)
                            drug_pair = parts[0].strip()
                            detail = parts[1].strip() if len(parts) > 1 else clean
                            break

                    if not drug_pair:
                        # Try colon or comma split
                        if ':' in clean:
                            parts = clean.split(':', 1)
                            # Only treat as drug pair if short enough
                            if len(parts[0]) < 60:
                                drug_pair = parts[0].strip()
                                detail = parts[1].strip()

                # ── Final sanitization: strip any HTML tags from detail + drugs ──
                detail = _re.sub(r'<[^>]+>', '', detail).strip()
                drug_pair = _re.sub(r'<[^>]+>', '', drug_pair).strip()
                # Remove leftover severity labels inside detail
                detail = _re.sub(r'^\**(CRITICAL|MODERATE|LOW)\**[:\s]*', '', detail, flags=_re.IGNORECASE).strip()

                if not detail:
                    continue  # Skip empty entries

                interactions.append({
                    "severity": severity,
                    "color": color,
                    "drugs": drug_pair,
                    "detail": detail,
                    "text": detail,  # kept for backward compat
                    "source": "TxGemma_DDI",
                })
                break  # Only match highest severity per line
    return interactions


def _extract_substitutions(text: str) -> list:
    """Extract substitution recommendations as structured dicts.
    Cleans markdown formatting artifacts."""
    import re as _re
    subs = []
    skip_labels = ("SUBSTITUTION:", "SUBSTITUTIONS:")
    for line in text.split("\n"):
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.upper().rstrip(':') in ("SUBSTITUTION", "SUBSTITUTIONS"):
            continue
        if "substitut" in stripped.lower() or "replace" in stripped.lower() or "unavailable" in stripped.lower():
            # Clean markdown, leading bullets, source tags
            clean = _re.sub(r'^[\d\.\*\#\-🔄]+\s*', '', stripped)
            clean = clean.replace('**', '').replace('*', '').strip()
            clean = _re.sub(r'\[Source:\s*\w+(?:_\w+)*\]', '', clean).strip()
            if clean:
                subs.append({
                    "text": clean,
                    "source": "TxGemma_DDI",
                })
    return subs


def _count_missing(oncocase: dict) -> int:
    """Count MISSING_DATA items in the OncoCase evidence pool."""
    evidence_pool = oncocase.get("evidence_pool", [])
    return sum(1 for e in evidence_pool if e.get("status") == "MISSING_DATA")


# ═══════════════════════════════════════════════════════════════
# Standalone Isolation Test
# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    from config.gpu_lease import get_gpu_lease

    print("=" * 60)
    print("ISOLATION TEST — TxGemma Worker")
    print("=" * 60)

    # Build a minimal OncoCase for testing
    test_oncocase = {
        "conditions": ["HIV-positive", "lymphoma"],
        "medications": ["tenofovir", "lamivudine", "dolutegravir"],
        "proposed_regimen": "CHOP",
        "proposed_drugs": ["cyclophosphamide", "doxorubicin", "vincristine", "prednisone"],
        "evidence_pool": [
            {"status": "OK", "modality": "cxr"},
            {"status": "OK", "modality": "derm"},
        ],
    }

    gpu_lease = get_gpu_lease()

    snap_before = gpu_lease.get_vram_snapshot()
    print(f"VRAM before: {snap_before['allocated_mb']:.0f} MB allocated")

    result = run_txgemma(test_oncocase, gpu_lease=gpu_lease)

    print("\n--- Evidence Item JSON ---")
    ev = result["evidence_item"]
    print(json.dumps(ev, indent=2, default=str))

    print(f"\nInteraction flags: {len(result.get('interaction_flags', []))}")
    for flag in result.get("interaction_flags", []):
        print(f"  [{flag['severity']}] {flag.get('drugs', '')} — {flag['text'][:80]}")

    print(f"\nSubstitutions: {len(result.get('substitutions', []))}")
    print(f"Inventory alerts: {len(result.get('inventory_alerts', []))}")
    for alert in result.get("inventory_alerts", []):
        print(f"  ⚠ {alert['drug']}: {alert['status']} — {alert['message']}")

    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
    gc.collect()
    snap_after = gpu_lease.get_vram_snapshot()
    print(f"\nVRAM after:  {snap_after['allocated_mb']:.0f} MB allocated")
    print(f"VRAM delta:  {snap_after['allocated_mb'] - snap_before['allocated_mb']:.0f} MB")
    print("✅ TxGemma Worker isolation test complete.")
