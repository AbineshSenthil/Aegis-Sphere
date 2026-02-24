"""
Aegis-Sphere — Override Logger (DPDP Act 2023 Compliant)
Creates & stores anonymized clinician override records locally.

All patient identifiers are SHA-256 hashed before storage.
Records are stored in append-only JSONL format for audit trail.
"""

import json
import hashlib
import time
import uuid
from pathlib import Path
from typing import Optional


# Default log path (relative to project root)
_DEFAULT_LOG_DIR = Path(__file__).resolve().parent / "remote_board"


def _anonymize(value: str) -> str:
    """SHA-256 hash a value for DPDP-compliant anonymization."""
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def log_override(
    session_id: str,
    clinician_note: str,
    field_overridden: str,
    original_value: str,
    new_value: str,
    log_dir: Optional[Path] = None,
) -> dict:
    """
    Log a clinician override with anonymized patient ID.

    The clinician may disagree with the AI's staging, treatment plan,
    or any other output. This function creates an auditable record
    that can later be used for:
      - LoRA fine-tuning (learning from corrections)
      - DPDP Act 2023 compliance audit trail
      - Remote tumor board review via SmartSync

    Args:
        session_id: Current session identifier
        clinician_note: Free-text note from the clinician
        field_overridden: Which field was changed (e.g., "staging", "treatment")
        original_value: AI-generated value
        new_value: Clinician-corrected value
        log_dir: Override log directory (defaults to sync/remote_board/)

    Returns:
        The created override record dict.
    """
    log_dir = log_dir or _DEFAULT_LOG_DIR
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "override_log.jsonl"

    record = {
        "record_id": str(uuid.uuid4())[:12],
        "timestamp": time.time(),
        "timestamp_iso": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "session_hash": _anonymize(session_id),
        "field": field_overridden,
        "original_value": original_value,
        "new_value": new_value,
        "clinician_note": clinician_note,
        "sync_status": "PENDING",
    }

    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return record


def get_pending_overrides(log_dir: Optional[Path] = None) -> list:
    """
    Retrieve all override records with sync_status == 'PENDING'.

    Returns:
        List of pending override record dicts.
    """
    log_dir = log_dir or _DEFAULT_LOG_DIR
    log_path = log_dir / "override_log.jsonl"

    if not log_path.exists():
        return []

    pending = []
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                if record.get("sync_status") == "PENDING":
                    pending.append(record)
            except json.JSONDecodeError:
                continue

    return pending


def mark_synced(record_ids: list, log_dir: Optional[Path] = None):
    """
    Mark override records as synced by updating their sync_status.

    Re-writes the JSONL file with updated statuses. In production,
    this would use a proper database — JSONL is for demo/LMIC offline use.
    """
    log_dir = log_dir or _DEFAULT_LOG_DIR
    log_path = log_dir / "override_log.jsonl"

    if not log_path.exists():
        return

    updated_lines = []
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                if record.get("record_id") in record_ids:
                    record["sync_status"] = "SYNCED"
                    record["synced_at"] = time.time()
                updated_lines.append(json.dumps(record, ensure_ascii=False))
            except json.JSONDecodeError:
                updated_lines.append(line)

    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(updated_lines) + "\n")


def get_override_stats(log_dir: Optional[Path] = None) -> dict:
    """Get summary stats of override records."""
    log_dir = log_dir or _DEFAULT_LOG_DIR
    log_path = log_dir / "override_log.jsonl"

    if not log_path.exists():
        return {"total": 0, "pending": 0, "synced": 0}

    total = 0
    pending = 0
    synced = 0

    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                total += 1
                if record.get("sync_status") == "PENDING":
                    pending += 1
                elif record.get("sync_status") == "SYNCED":
                    synced += 1
            except json.JSONDecodeError:
                continue

    return {"total": total, "pending": pending, "synced": synced}
