"""
Aegis-Sphere — Cortex Controller (Mode Bridge + Orchestrator)
Main pipeline orchestrator: runs all phases in sequence.
Updates VRAM chart after each step.
"""

import os
import sys
import time
from typing import Optional, Callable

# ── Ensure project root is on sys.path for standalone execution ──
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.session_manager import Session
from pipeline.asr_worker import run_asr
from pipeline.lang_extract import extract_clinical_frame
from pipeline.mode_bridge import evaluate_escalation
from pipeline.hear_worker import run_hear
from pipeline.path_worker import run_path
from pipeline.cxr_worker import run_cxr
from pipeline.derm_worker import run_derm
from pipeline.medsig_worker import run_medsig
from pipeline.txgemma_worker import run_txgemma
from pipeline.oncocase_builder import build_oncocase
from pipeline.risk_engine import compute_risk
from pipeline.persona_debate import run_persona_debate
from pipeline.evidence_trace import build_comprehensive_trace
from config.gpu_lease import get_gpu_lease


def run_pipeline(
    session: Session,
    vram_monitor=None,
    on_phase_complete: Optional[Callable] = None,
) -> Session:
    """
    Run the complete Aegis-Sphere pipeline.

    Phases:
        1. MedASR — transcription
        2. LangExtract — NER
        3A. HeAR — cough analysis
        4.1 Path Foundation — histopathology
        4.2 CXR Foundation — chest X-ray
        4.3 Derm Foundation — skin lesion
        4.4 MedSigLIP — similar case retrieval (CPU)
        5. Risk Engine — TB/HIV scoring
        4.7 OncoCase Builder — evidence assembly + degradation
        4.6 TxGemma — drug safety
        6. Persona Debate — 5-pass MedGemma
        7. Evidence Trace — tag extraction
    """
    gpu_lease = get_gpu_lease(
        vram_callback=lambda phase, model: (
            vram_monitor.log_phase(phase, model) if vram_monitor else None
        )
    )

    session.status = "RUNNING"

    def _notify(phase):
        session.mark_phase(phase)
        if on_phase_complete:
            on_phase_complete(phase)

    # ══════════════════════════════════════════════════════════
    # Phase 1: MedASR Transcription
    # ══════════════════════════════════════════════════════════
    try:
        asr_result = run_asr(
            audio_path=session.audio_path,
            gpu_lease=gpu_lease,
            vram_monitor=vram_monitor,
        )
        session.transcript = asr_result.get("transcript")
        session.evidence_pool.append(asr_result["evidence_item"])
        asr_confidence_flags = asr_result.get("confidence_flags", [])
        _notify("Phase_1_MedASR")
    except Exception as e:
        session.add_error("Phase_1_MedASR", str(e))
        asr_confidence_flags = ["ASR_ERROR"]
        _notify("Phase_1_MedASR")

    # ══════════════════════════════════════════════════════════
    # Phase 2: Language Extraction (NER)
    # ══════════════════════════════════════════════════════════
    try:
        session.clinical_frame = extract_clinical_frame(session.transcript)
        _notify("Phase_2_LangExtract")
    except Exception as e:
        session.add_error("Phase_2_LangExtract", str(e))
        session.clinical_frame = {"symptoms": [], "medications": [], "durations": [],
                                   "conditions": [], "lab_values": [], "vitals": [],
                                   "demographics": {}}
        _notify("Phase_2_LangExtract")

    # ══════════════════════════════════════════════════════════
    # Phase 3B: Mode Bridge — Escalation
    # ══════════════════════════════════════════════════════════
    try:
        asr_ev = next(
            (e for e in session.evidence_pool if e.get("model") == "MedASR"),
            None,
        )
        session.escalation_result = evaluate_escalation(
            clinical_frame=session.clinical_frame,
            asr_result=asr_ev,
        )
        _notify("Phase_3B_ModeBridge")
    except Exception as e:
        session.add_error("Phase_3B_ModeBridge", str(e))
        session.escalation_result = {
            "mode": "ONCOSPHERE", "triggers": [], "uncertainty": "CRITICAL",
            "rationale": "Mode bridge error — defaulting to full OncoSphere.",
            "coinfection_flags": [],
        }
        _notify("Phase_3B_ModeBridge")

    # ══════════════════════════════════════════════════════════
    # Phase 3A: HeAR Cough Analysis
    # ══════════════════════════════════════════════════════════
    try:
        hear_result = run_hear(
            cough_audio_path=session.audio_path,  # reuse consultation audio for cough
            gpu_lease=gpu_lease,
            vram_monitor=vram_monitor,
        )
        session.evidence_pool.append(hear_result["evidence_item"])
        _notify("Phase_3A_HeAR")
    except Exception as e:
        session.add_error("Phase_3A_HeAR", str(e))
        _notify("Phase_3A_HeAR")

    # ══════════════════════════════════════════════════════════
    # Phase 4.1: Path Foundation
    # ══════════════════════════════════════════════════════════
    try:
        path_result = run_path(
            path_image_path=session.path_path,
            gpu_lease=gpu_lease,
            vram_monitor=vram_monitor,
        )
        session.evidence_pool.append(path_result["evidence_item"])
        _notify("Phase_4.1_Path")
    except Exception as e:
        session.add_error("Phase_4.1_Path", str(e))
        _notify("Phase_4.1_Path")

    # ══════════════════════════════════════════════════════════
    # Phase 4.2: CXR Foundation
    # ══════════════════════════════════════════════════════════
    try:
        cxr_result = run_cxr(
            cxr_image_path=session.cxr_path,
            gpu_lease=gpu_lease,
            vram_monitor=vram_monitor,
        )
        session.evidence_pool.append(cxr_result["evidence_item"])
        _notify("Phase_4.2_CXR")
    except Exception as e:
        session.add_error("Phase_4.2_CXR", str(e))
        _notify("Phase_4.2_CXR")

    # ══════════════════════════════════════════════════════════
    # Phase 4.3: Derm Foundation
    # ══════════════════════════════════════════════════════════
    try:
        derm_result = run_derm(
            derm_image_path=session.derm_path,
            gpu_lease=gpu_lease,
            vram_monitor=vram_monitor,
        )
        session.evidence_pool.append(derm_result["evidence_item"])
        _notify("Phase_4.3_Derm")
    except Exception as e:
        session.add_error("Phase_4.3_Derm", str(e))
        _notify("Phase_4.3_Derm")

    # ══════════════════════════════════════════════════════════
    # Phase 4.4: MedSigLIP (CPU — no GPU lease)
    # ══════════════════════════════════════════════════════════
    try:
        medsig_result = run_medsig(
            image_paths={
                "cxr": session.cxr_path,
                "derm": session.derm_path,
                "path": session.path_path,
            },
            gpu_lease=None,  # CPU only
            vram_monitor=vram_monitor,
        )
        similar_cases = medsig_result.get("similar_cases", [])
        session.similar_cases = similar_cases
        _notify("Phase_4.4_MedSigLIP")
    except Exception as e:
        session.add_error("Phase_4.4_MedSigLIP", str(e))
        similar_cases = []
        session.similar_cases = []
        _notify("Phase_4.4_MedSigLIP")

    # ══════════════════════════════════════════════════════════
    # Phase 5: Risk Engine
    # ══════════════════════════════════════════════════════════
    try:
        session.risk_result = compute_risk(
            clinical_frame=session.clinical_frame,
            evidence_pool=session.evidence_pool,
            asr_confidence_flags=asr_confidence_flags,
        )
        _notify("Phase_5_RiskEngine")
    except Exception as e:
        session.add_error("Phase_5_RiskEngine", str(e))
        session.risk_result = {
            "tb_risk_score": 0.0, "tb_risk_level": "LOW",
            "hiv_risk_score": 0.0, "overall_risk_level": "GREEN",
            "uncertainty_flags": [], "staging_override": None,
            "treatment_override": None, "missing_count": 0,
        }
        _notify("Phase_5_RiskEngine")

    # ══════════════════════════════════════════════════════════
    # Phase 4.7: OncoCase Builder
    # ══════════════════════════════════════════════════════════
    try:
        session.oncocase = build_oncocase(
            clinical_frame=session.clinical_frame,
            evidence_pool=session.evidence_pool,
            risk_result=session.risk_result,
            similar_cases=similar_cases,
        )
        _notify("Phase_4.7_OncoCase")
    except Exception as e:
        session.add_error("Phase_4.7_OncoCase", str(e))
        _notify("Phase_4.7_OncoCase")

    # ══════════════════════════════════════════════════════════
    # Phase 4.6: TxGemma Drug Safety
    # ══════════════════════════════════════════════════════════
    try:
        if session.oncocase:
            txgemma_result = run_txgemma(
                oncocase=session.oncocase,
                gpu_lease=gpu_lease,
                vram_monitor=vram_monitor,
            )
            # Update oncocase with TxGemma results
            session.oncocase["tx_analysis"] = txgemma_result
            session.oncocase["interaction_flags"] = txgemma_result.get("interaction_flags", [])
            session.oncocase["inventory_alerts"] = txgemma_result.get("inventory_alerts", [])
            session.tx_result = txgemma_result
            _notify("Phase_4.6_TxGemma")
    except Exception as e:
        session.add_error("Phase_4.6_TxGemma", str(e))
        _notify("Phase_4.6_TxGemma")

    # ══════════════════════════════════════════════════════════
    # Phase 6: Persona Debate (5-pass MedGemma)
    # ══════════════════════════════════════════════════════════
    try:
        if session.oncocase:
            session.debate_results = run_persona_debate(
                oncocase=session.oncocase,
                gpu_lease=gpu_lease,
                vram_monitor=vram_monitor,
            )
            _notify("Phase_6_PersonaDebate")
    except Exception as e:
        session.add_error("Phase_6_PersonaDebate", str(e))
        _notify("Phase_6_PersonaDebate")

    # ══════════════════════════════════════════════════════════
    # Phase 7: Evidence Trace
    # ══════════════════════════════════════════════════════════
    try:
        session.evidence_trace = build_comprehensive_trace(
            debate_outputs=session.debate_results,
            evidence_pool=session.evidence_pool,
            tx_result=getattr(session, 'tx_result', None),
            transcript=session.transcript,
            clinical_frame=session.clinical_frame,
            oncocase=session.oncocase,
        )
        _notify("Phase_7_EvidenceTrace")
    except Exception as e:
        session.add_error("Phase_7_EvidenceTrace", str(e))
        session.evidence_trace = {}
        _notify("Phase_7_EvidenceTrace")

    # ── Export VRAM log ──
    if vram_monitor:
        try:
            from config.settings import VRAM_LOG_PATH
            VRAM_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
            vram_monitor.export_csv(str(VRAM_LOG_PATH))
        except Exception:
            pass

    session.status = "COMPLETED"
    return session
