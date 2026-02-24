"""
Aegis-Sphere — Session Manager
Session lifecycle, patient metadata, and state persistence.
"""

import uuid
import time
import json
from pathlib import Path
from typing import Optional


class Session:
    """Represents a single patient analysis session."""

    def __init__(self, patient_id: str = None, session_id: str = None):
        self.session_id = session_id or str(uuid.uuid4())[:8]
        self.patient_id = patient_id or f"PT-{self.session_id}"
        self.created_at = time.time()
        self.status = "INITIALIZED"

        # Input files
        self.audio_path: Optional[str] = None
        self.cxr_path: Optional[str] = None
        self.derm_path: Optional[str] = None
        self.path_path: Optional[str] = None
        self.inventory_path: Optional[str] = None

        # Pipeline results (populated as phases complete)
        self.transcript: Optional[str] = None
        self.clinical_frame: Optional[dict] = None
        self.evidence_pool: list = []
        self.risk_result: Optional[dict] = None
        self.oncocase: Optional[dict] = None
        self.tx_result: Optional[dict] = None
        self.debate_results: Optional[dict] = None
        self.evidence_trace: Optional[dict] = None
        self.similar_cases: list = []
        self.escalation_result: Optional[dict] = None

        # Metadata
        self.phases_completed: list = []
        self.vram_log: list = []
        self.errors: list = []

    def mark_phase(self, phase: str, status: str = "DONE"):
        self.phases_completed.append({
            "phase": phase,
            "status": status,
            "timestamp": time.time(),
        })

    def add_error(self, phase: str, error: str):
        self.errors.append({
            "phase": phase,
            "error": str(error),
            "timestamp": time.time(),
        })

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "patient_id": self.patient_id,
            "created_at": self.created_at,
            "status": self.status,
            "phases_completed": self.phases_completed,
            "errors": self.errors,
        }

    def __repr__(self):
        return f"Session({self.session_id}, status={self.status})"
