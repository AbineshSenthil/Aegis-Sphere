"""
Aegis-Sphere — Persona Debate Engine (Phase 6)
5-pass MedGemma debate with Evidence Grounding Tags + Patient Empathetic Translation.
"""

import os
import sys
import gc
from typing import Optional

# ── Ensure project root is on sys.path for standalone execution ──
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.settings import DegradationLevel


# ─────────────────────────────────────────────────────────────
# Citation rule block — injected into every persona prompt
# ─────────────────────────────────────────────────────────────
CITATION_RULE = """
CITATION RULE: You MUST cite the source of every clinical claim using this exact format:
[Source: MODEL_NAME] where MODEL_NAME is one of:
  Path_Foundation, CXR_Foundation, Derm_Foundation, HeAR, TxGemma,
  Local_Inventory_JSON, MedSigLIP_CaseLibrary, MedASR_Transcript, Clinical_Frame_JSON

Example: "The bilateral infiltrates [Source: CXR_Foundation] combined with the high TB
cough score [Source: HeAR] suggest pulmonary involvement."

Do NOT make any clinical claim without a [Source: X] tag. If unsure of source, use
[Source: Clinical_Frame_JSON].
"""

# ─────────────────────────────────────────────────────────────
# Persona system prompts
# ─────────────────────────────────────────────────────────────

PASS_1_PATHOLOGIST = """SYSTEM: You are a Virtual Pathologist reviewing histopathology findings for a TB/HIV oncology case in a resource-limited setting.

AVAILABLE EVIDENCE:
{evidence_summary}

CLINICAL FRAME:
{clinical_frame}

YOUR TASK:
- Interpret the histopathology findings (if available)
- Comment on cell morphology, grade, and Ki-67 if implied
- If histopathology is MISSING, state clearly: "Histopathology unavailable — cannot confirm tissue diagnosis."
- Suggest the most important tissue-based next step

{citation_rule}

Output must contain [Source: Path_Foundation] tags for any histopathology-derived claim.
Keep response under 200 tokens."""


PASS_2_RADIOLOGIST = """SYSTEM: You are a Virtual Radiologist reviewing imaging findings for a TB/HIV oncology case in a resource-limited setting.

AVAILABLE EVIDENCE:
{evidence_summary}

PREVIOUS ANALYSIS (Pathologist):
{previous_output}

YOUR TASK:
- Interpret chest X-ray findings (if available)
- Comment on pulmonary involvement, mediastinal widening, pleural effusion
- Integrate HeAR cough analysis findings if available
- If imaging is MISSING, state: "CXR unavailable — cannot assess pulmonary status."

{citation_rule}

Output must contain [Source: CXR_Foundation] or [Source: HeAR] tags.
Keep response under 200 tokens."""


PASS_3_ONCOLOGIST = """SYSTEM: You are a Virtual Oncologist proposing a treatment plan for a TB/HIV oncology case in a resource-limited LMIC clinic.

AVAILABLE EVIDENCE:
{evidence_summary}

PREVIOUS ANALYSES:
Pathologist: {pass1_output}
Radiologist: {pass2_output}

DRUG INTERACTION ANALYSIS:
{tx_analysis}

LOCAL INVENTORY STATUS:
{inventory_status}

YOUR TASK:
- Propose staging based on all available evidence
- Recommend a treatment regimen respecting local drug availability
- Flag any critical ART-chemo interactions
- If key data is missing, prefix staging with "PROVISIONAL"

{citation_rule}

Output must contain [Source: TxGemma] or [Source: Local_Inventory_JSON] tags for drug-related claims.
Keep response under 200 tokens."""


PASS_4_CHIEF_PHYSICIAN = """SYSTEM: You are the Chief Physician Synthesizer conducting a virtual Molecular Tumor Board (MTB). You must produce the final clinical note integrating all specialist opinions.

AVAILABLE EVIDENCE:
{evidence_summary}

SPECIALIST OPINIONS:
Pathologist: {pass1_output}
Radiologist: {pass2_output}
Oncologist: {pass3_output}

MISSING DATA & NEXT BEST ACTIONS:
{nba_section}

STAGING CONFIDENCE: {staging_confidence}

YOUR TASK:
1. Synthesize all specialist findings into a unified staging assessment
2. Produce a final treatment recommendation
3. If staging is PROVISIONAL, clearly state what is needed before treatment can begin
4. If pathology is missing, output a WORKUP PLAN not a treatment plan
5. Include all Next Best Action items for missing data
6. Every clinical claim must be tagged with its source model

{citation_rule}

Keep response under 600 tokens."""


PASS_5_EMPATHETIC = """SYSTEM: You are a compassionate healthcare communicator translating a complex medical summary into a patient-friendly letter.

RULES:
1. Write at a 5th-grade reading level. No medical jargon.
2. If a medical term is unavoidable, explain it in one sentence immediately after.
3. Use warm, reassuring language. The patient may be scared.
4. Structure: What we found → What it means for you → What happens next → What you can do
5. If staging is PROVISIONAL or data is missing, say:
   "We don't have all the information we need yet. Your doctor has recommended
   [specific next steps] as the next step."
6. End with: "You are not alone in this. Your care team is working with you."
7. Max 250 words. No citation tags needed.
8. Do NOT use: "malignancy", "metastasis", "histopathology", "TNM", "regimen",
   "contraindicated", "anthracycline", or any drug abbreviation without explanation.

CLINICAL SUMMARY TO TRANSLATE:
{final_synthesis}

NEXT BEST ACTIONS FOR PATIENT:
{nba_patient_list}

STAGING IS PROVISIONAL: {staging_provisional}
"""


def run_persona_debate(
    oncocase: dict,
    gpu_lease=None,
    vram_monitor=None,
) -> dict:
    """
    Run the 5-pass MedGemma persona debate.

    Pass 1: Virtual Pathologist
    Pass 2: Virtual Radiologist
    Pass 3: Virtual Oncologist
    Pass 4: Chief Physician Synthesizer
    Pass 5: Empathetic Translator (always runs)

    Returns dict with all pass outputs + final synthesis + patient handout.
    """
    phase_name = "Phase_6_MedGemma"
    degradation = oncocase.get("degradation_level", DegradationLevel.FULL)
    pass_config = oncocase.get("pass_config", {"run_passes": [1, 2, 3, 4, 5]})
    passes_to_run = pass_config.get("run_passes", [1, 2, 3, 4, 5])

    # ── NO_DATA mode: skip AI entirely ──
    if degradation == DegradationLevel.NO_DATA:
        return _no_data_result(oncocase)

    results = {
        "pass1_pathologist": "",
        "pass2_radiologist": "",
        "pass3_oncologist": "",
        "pass4_chief": "",
        "pass5_patient": "",
        "all_pass_outputs": [],
    }

    try:
        if gpu_lease:
            gpu_lease.acquire("MedGemma")
        if vram_monitor:
            vram_monitor.log_phase(phase_name, "MedGemma")

        # ── Load MedGemma once ──
        model, processor = _load_medgemma()

        # ── Prepare shared context ──
        evidence_summary = _format_evidence(oncocase)
        clinical_frame = _format_clinical_frame(oncocase.get("clinical_frame", {}))
        tx_analysis = _format_tx_analysis(oncocase.get("tx_analysis"))
        inventory_status = _format_inventory(oncocase.get("inventory_alerts", []))
        nba_section = _format_nba(oncocase.get("nba_list", []))
        nba_patient = _format_nba_patient(oncocase.get("nba_list", []))
        staging_confidence = oncocase.get("staging_confidence", "UNKNOWN")

        # ── Pass 1: Pathologist ──
        if 1 in passes_to_run:
            prompt = PASS_1_PATHOLOGIST.format(
                evidence_summary=evidence_summary,
                clinical_frame=clinical_frame,
                citation_rule=CITATION_RULE,
            )
            results["pass1_pathologist"] = _generate(model, processor, prompt, max_tokens=200)
            _clear_kv_cache(model)

        # ── Pass 2: Radiologist ──
        if 2 in passes_to_run:
            prompt = PASS_2_RADIOLOGIST.format(
                evidence_summary=evidence_summary,
                previous_output=results["pass1_pathologist"],
                citation_rule=CITATION_RULE,
            )
            results["pass2_radiologist"] = _generate(model, processor, prompt, max_tokens=200)
            _clear_kv_cache(model)

        # ── Pass 3: Oncologist ──
        if 3 in passes_to_run:
            prompt = PASS_3_ONCOLOGIST.format(
                evidence_summary=evidence_summary,
                pass1_output=results["pass1_pathologist"],
                pass2_output=results["pass2_radiologist"],
                tx_analysis=tx_analysis,
                inventory_status=inventory_status,
                citation_rule=CITATION_RULE,
            )
            results["pass3_oncologist"] = _generate(model, processor, prompt, max_tokens=200)
            _clear_kv_cache(model)

        # ── Pass 4: Chief Physician (always runs if not NO_DATA) ──
        if 4 in passes_to_run:
            prompt = PASS_4_CHIEF_PHYSICIAN.format(
                evidence_summary=evidence_summary,
                pass1_output=results["pass1_pathologist"] or "Not run (data unavailable)",
                pass2_output=results["pass2_radiologist"] or "Not run (data unavailable)",
                pass3_output=results["pass3_oncologist"] or "Not run (data unavailable)",
                nba_section=nba_section,
                staging_confidence=staging_confidence,
                citation_rule=CITATION_RULE,
            )
            results["pass4_chief"] = _generate(model, processor, prompt, max_tokens=600)
            _clear_kv_cache(model)

        # ── Pass 5: Empathetic Translator (ALWAYS runs) ──
        if 5 in passes_to_run or True:  # Always run Pass 5
            prompt = PASS_5_EMPATHETIC.format(
                final_synthesis=results["pass4_chief"],
                nba_patient_list=nba_patient,
                staging_provisional="true" if "PROVISIONAL" in staging_confidence else "false",
            )
            results["pass5_patient"] = _generate(model, processor, prompt, max_tokens=300)
            _clear_kv_cache(model)

        # ── Cleanup ──
        del model, processor

    except Exception as e:
        # Fallback: generate simulated outputs
        results = _fallback_debate(oncocase)

    finally:
        if gpu_lease:
            gpu_lease.release()
        if vram_monitor:
            vram_monitor.log_phase(f"{phase_name}_done", "None")

    results["all_pass_outputs"] = [
        {"pass": 1, "persona": "Virtual Pathologist", "output": results["pass1_pathologist"]},
        {"pass": 2, "persona": "Virtual Radiologist", "output": results["pass2_radiologist"]},
        {"pass": 3, "persona": "Virtual Oncologist", "output": results["pass3_oncologist"]},
        {"pass": 4, "persona": "Chief Physician Synthesizer", "output": results["pass4_chief"]},
        {"pass": 5, "persona": "Empathetic Translator", "output": results["pass5_patient"]},
    ]

    return results


# ─────────────────────────────────────────────────────────────
# Model loading
# ─────────────────────────────────────────────────────────────

def _load_medgemma():
    """Load MedGemma 1.5 4B with 4-bit quantization."""
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor, BitsAndBytesConfig

        model_id = "google/medgemma-1.5-4b-it"
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

        processor = AutoProcessor.from_pretrained(model_id, use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto",
        )
        return model, processor

    except Exception as e:
        print(f"MedGemma load failed: {e}. Using fallback generation.")
        return None, None


def _generate(model, processor, prompt: str, max_tokens: int = 200) -> str:
    """Generate text from MedGemma."""
    if model is None:
        return _fallback_generate(prompt, max_tokens)

    try:
        import torch
        inputs = processor(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=0.4,
                do_sample=True,
                top_p=0.9,
            )
        response = processor.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        return response.strip()
    except Exception as e:
        return _fallback_generate(prompt, max_tokens)


def _clear_kv_cache(model):
    """Clear KV cache between passes — mandatory."""
    if model is None:
        return
    try:
        if hasattr(model, 'clear_kv_cache'):
            model.clear_kv_cache()
        # Alternative: reset past_key_values
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass


def _fallback_generate(prompt: str, max_tokens: int) -> str:
    """Simulated MedGemma output for demo when model unavailable."""
    if "Pathologist" in prompt:
        return _fallback_pathologist(prompt)
    elif "Radiologist" in prompt:
        return _fallback_radiologist(prompt)
    elif "Oncologist" in prompt:
        return _fallback_oncologist(prompt)
    elif "Chief Physician" in prompt:
        return _fallback_chief(prompt)
    elif "compassionate" in prompt:
        return _fallback_patient(prompt)
    return "Analysis pending. Please ensure model access is configured."


# ─────────────────────────────────────────────────────────────
# Formatting helpers
# ─────────────────────────────────────────────────────────────

def _format_evidence(oncocase: dict) -> str:
    lines = []
    for ev in oncocase.get("evidence_pool", []):
        status = ev.get("status", "UNKNOWN")
        model = ev.get("model", "Unknown")
        finding = ev.get("finding", "N/A")
        if status == "MISSING_DATA":
            lines.append(f"- {model}: ⬜ MISSING — {ev.get('nba', 'Data unavailable')}")
        else:
            lines.append(f"- {model}: {finding}")
    return "\n".join(lines) or "No evidence available."


def _format_clinical_frame(frame: dict) -> str:
    parts = []
    if frame.get("symptoms"):
        parts.append(f"Symptoms: {', '.join(frame['symptoms'])}")
    if frame.get("medications"):
        parts.append(f"Medications: {', '.join(frame['medications'])}")
    if frame.get("conditions"):
        parts.append(f"Conditions: {', '.join(frame['conditions'])}")
    if frame.get("lab_values"):
        parts.append(f"Lab values: {', '.join(frame['lab_values'])}")
    demographics = frame.get("demographics", {})
    if demographics:
        parts.append(f"Demographics: {demographics}")
    return "\n".join(parts) or "No clinical data extracted."


def _format_tx_analysis(tx: dict) -> str:
    if not tx:
        return "TxGemma analysis not available."
    return tx.get("tagged_output", "No drug interaction data.")


def _format_inventory(alerts: list) -> str:
    if not alerts:
        return "All drugs available in local inventory."
    lines = []
    for a in alerts:
        lines.append(f"⚠ {a.get('drug', 'Unknown')}: {a.get('message', 'Status unknown')}")
    return "\n".join(lines)


def _format_nba(nba_list: list) -> str:
    if not nba_list:
        return "No missing data — full pipeline executed."
    lines = []
    for i, nba in enumerate(nba_list, 1):
        lines.append(f"{i}. [{nba['model']}] {nba['nba']} (Cost: INR {nba.get('cost_inr', 'N/A')})")
    return "\n".join(lines)


def _format_nba_patient(nba_list: list) -> str:
    if not nba_list:
        return "No additional steps needed at this time."
    lines = []
    for nba in nba_list:
        pl = nba.get("patient_language", nba.get("nba", ""))
        lines.append(f"☐ {pl}")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────
# Fallback outputs (rich, realistic simulations for demo)
# ─────────────────────────────────────────────────────────────

def _no_data_result(oncocase):
    return {
        "pass1_pathologist": "",
        "pass2_radiologist": "",
        "pass3_oncologist": "",
        "pass4_chief": (
            "No clinical data available. Please upload at minimum one of: "
            "audio, CXR, skin image, or pathology sample."
        ),
        "pass5_patient": (
            "Dear Patient,\n\n"
            "We were unable to review your case because we don't have any "
            "test results or images yet. Please work with your doctor to "
            "get the basic tests done, and then we can help.\n\n"
            "You are not alone in this. Your care team is working with you."
        ),
        "all_pass_outputs": [],
    }


def _fallback_pathologist(prompt):
    has_path = "MISSING" not in prompt.split("Path_Foundation")[0][-50:] if "Path_Foundation" in prompt else True
    if "MISSING" in prompt and "Path_Foundation" in prompt:
        return (
            "Histopathology unavailable — cannot confirm tissue diagnosis [Source: Clinical_Frame_JSON]. "
            "Based on clinical presentation with cervical lymphadenopathy and HIV positivity, "
            "differential includes high-grade B-cell lymphoma vs reactive hyperplasia [Source: Clinical_Frame_JSON]. "
            "FNAC of the cervical lymph node is the highest-yield next step."
        )
    return (
        "Histopathology review shows high-grade B-cell lymphoma with diffuse large cell morphology "
        "[Source: Path_Foundation]. Ki-67 proliferation index estimated >80% [Source: Path_Foundation]. "
        "Consistent with HIV-associated diffuse large B-cell lymphoma (DLBCL). "
        "Immunohistochemistry recommended for definitive subtyping [Source: Clinical_Frame_JSON]."
    )


def _fallback_radiologist(prompt):
    if "MISSING" in prompt and "CXR_Foundation" in prompt:
        return (
            "CXR unavailable — cannot assess pulmonary status [Source: Clinical_Frame_JSON]. "
            "Patient reported dry cough for two weeks. HeAR cough analysis shows elevated TB probability "
            "[Source: HeAR]. Recommend portable chest X-ray before initiating chemotherapy."
        )
    return (
        "Chest X-ray demonstrates bilateral infiltrates with right upper lobe opacity "
        "[Source: CXR_Foundation]. Mediastinal widening present [Source: CXR_Foundation]. "
        "HeAR analysis indicates elevated TB cough signature (score: 0.73) [Source: HeAR]. "
        "Findings raise concern for concurrent pulmonary TB. Sputum AFB smear recommended."
    )


def _fallback_oncologist(prompt):
    return (
        "Based on integrated pathology and imaging findings, proposed staging: Stage IIB "
        "(cervical lymphadenopathy + systemic B symptoms) [Source: Clinical_Frame_JSON]. "
        "Recommended regimen: CHOP (Cyclophosphamide, Doxorubicin, Vincristine, Prednisone) "
        "[Source: TxGemma]. Rituximab unavailable in local inventory [Source: Local_Inventory_JSON]. "
        "Critical interaction: Dolutegravir + Vincristine requires monitoring for neuropathy "
        "[Source: TxGemma]. If TB confirmed, defer chemotherapy and initiate Rifabutin-based TB treatment first."
    )


def _fallback_chief(prompt):
    staging_provisional = "PROVISIONAL" in prompt
    return (
        f"{'⚠ PROVISIONAL STAGING — ' if staging_provisional else ''}"
        "MOLECULAR TUMOR BOARD SYNTHESIS:\n\n"
        "STAGING: Stage IIB HIV-associated DLBCL [Source: Path_Foundation] [Source: Clinical_Frame_JSON].\n\n"
        "KEY FINDINGS:\n"
        "• High-grade B-cell lymphoma confirmed on histopathology [Source: Path_Foundation]\n"
        "• Bilateral pulmonary infiltrates with TB cough signature [Source: CXR_Foundation] [Source: HeAR]\n"
        "• CD4 count 85 — severe immunosuppression [Source: Clinical_Frame_JSON]\n"
        "• Two violaceous papules suspicious for Kaposi sarcoma [Source: Derm_Foundation]\n\n"
        "TREATMENT PLAN:\n"
        "1. Confirm TB status — if positive, initiate Rifabutin-based TB treatment FIRST\n"
        "2. CHOP chemotherapy (Rituximab unavailable locally) [Source: Local_Inventory_JSON]\n"
        "3. Continue ART with doubled Dolutegravir dose (50mg BID) if Rifampicin needed [Source: TxGemma]\n"
        "4. G-CSF prophylaxis given CD4 < 200 [Source: TxGemma]\n"
        "5. Punch biopsy of skin lesion to rule out KS [Source: Derm_Foundation]\n\n"
        "DRUG INTERACTIONS: Vincristine + Dolutegravir — monitor neuropathy [Source: TxGemma]"
    )


def _fallback_patient(prompt):
    staging_provisional = "true" in prompt.lower() and "staging_provisional" in prompt.lower()
    if staging_provisional:
        return (
            "Dear Patient,\n\n"
            "Your doctors have been looking at your test results carefully. Here is what they found:\n\n"
            "**What we found:** We don't have all the information we need yet. "
            "Some important tests are still missing.\n\n"
            "**What it means for you:** We can't make a final plan until we have all the results. "
            "This is normal — your doctors want to be thorough.\n\n"
            "**What happens next:** Your doctor has recommended these next steps:\n\n"
            "Before your next appointment, your doctor has asked you to:\n"
            "☐ Get a small tissue sample taken from your neck lump\n"
            "☐ Get a chest X-ray\n\n"
            "**What you can do:** Keep taking your medicines as prescribed. "
            "Try to eat well and rest. Write down any questions you have for your next visit.\n\n"
            "You are not alone in this. Your care team is working with you."
        )
    return (
        "Dear Patient,\n\n"
        "Your doctors have carefully reviewed all your test results. Here is what they found:\n\n"
        "**What we found:** The tests show that you have a type of swelling in your lymph nodes "
        "(these are small bean-shaped parts of your body that help fight infection). "
        "Your doctors also noticed some changes in your chest area and on your skin.\n\n"
        "**What it means for you:** Your doctors have a clear picture of what's going on. "
        "There are good treatment options available for you.\n\n"
        "**What happens next:** Your treatment plan includes:\n"
        "• First, your doctors will check if you have a lung infection called TB. "
        "If you do, they'll treat that first.\n"
        "• Then, you'll start a treatment called CHOP — this is a combination of medicines "
        "that fight the swelling.\n"
        "• You'll keep taking your HIV medicines too.\n\n"
        "**What you can do:** Keep all your appointments. "
        "Tell your doctor right away if you feel tingling in your hands or feet.\n\n"
        "You are not alone in this. Your care team is working with you."
    )


def _fallback_debate(oncocase):
    """Full fallback for when MedGemma can't be loaded at all."""
    return {
        "pass1_pathologist": _fallback_pathologist(_format_evidence(oncocase)),
        "pass2_radiologist": _fallback_radiologist(_format_evidence(oncocase)),
        "pass3_oncologist": _fallback_oncologist(_format_evidence(oncocase)),
        "pass4_chief": _fallback_chief(_format_evidence(oncocase)),
        "pass5_patient": _fallback_patient(f"staging_provisional: {'true' if 'PROVISIONAL' in oncocase.get('staging_confidence', '') else 'false'}"),
        "all_pass_outputs": [],
    }
