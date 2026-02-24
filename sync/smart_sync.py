"""
Aegis-Sphere — Smart Sync Engine
Background thread: checks connectivity & uploads override records securely.

Designed for LMIC offline-first clinics:
  - Works without internet (records accumulate locally)
  - Syncs when connectivity is detected
  - All data is anonymized before transmission
  - Mock endpoint writes to sync/remote_board/ for local testing
"""

import json
import time
import threading
from pathlib import Path
from typing import Optional

from sync.override_logger import get_pending_overrides, mark_synced


# Default paths
_REMOTE_BOARD_DIR = Path(__file__).resolve().parent / "remote_board"
_SYNCED_LOG_PATH = _REMOTE_BOARD_DIR / "synced_records.jsonl"


class SmartSyncEngine:
    """
    Background sync engine for offline-first clinic deployments.

    Usage:
        engine = SmartSyncEngine()
        engine.start()
        # ... app runs ...
        engine.stop()

    The engine runs a background thread that periodically:
      1. Checks for pending override records
      2. Attempts to "upload" them (mock: writes to remote_board/)
      3. Marks records as SYNCED on success
    """

    def __init__(
        self,
        sync_interval: int = 30,
        remote_dir: Optional[Path] = None,
    ):
        self.sync_interval = sync_interval
        self.remote_dir = remote_dir or _REMOTE_BOARD_DIR
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._sync_count = 0
        self._last_sync_time: Optional[float] = None
        self._errors: list = []

    def start(self):
        """Start the background sync thread."""
        if self._thread and self._thread.is_alive():
            return  # Already running

        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run_loop,
            daemon=True,
            name="AegisSyncEngine",
        )
        self._thread.start()

    def stop(self):
        """Stop the background sync thread gracefully."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    @property
    def sync_count(self) -> int:
        return self._sync_count

    @property
    def last_sync_time(self) -> Optional[float]:
        return self._last_sync_time

    def get_status(self) -> dict:
        """Get current sync engine status."""
        return {
            "running": self.is_running,
            "sync_count": self._sync_count,
            "last_sync_time": self._last_sync_time,
            "last_sync_ago": (
                f"{time.time() - self._last_sync_time:.0f}s ago"
                if self._last_sync_time else "Never"
            ),
            "errors": len(self._errors),
        }

    def attempt_sync(self) -> dict:
        """
        Manually trigger a sync attempt.

        Returns dict with sync results.
        """
        pending = get_pending_overrides()
        if not pending:
            return {"synced": 0, "status": "NO_PENDING"}

        try:
            # ── "Upload" to mock remote endpoint ──
            self.remote_dir.mkdir(parents=True, exist_ok=True)
            with open(_SYNCED_LOG_PATH, "a", encoding="utf-8") as f:
                for record in pending:
                    # Add sync metadata
                    synced_record = {
                        **record,
                        "sync_status": "SYNCED",
                        "synced_at": time.time(),
                        "synced_at_iso": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                        "sync_endpoint": "mock://remote_board",
                    }
                    f.write(json.dumps(synced_record, ensure_ascii=False) + "\n")

            # ── Mark local records as synced ──
            record_ids = [r["record_id"] for r in pending]
            mark_synced(record_ids)

            self._sync_count += len(pending)
            self._last_sync_time = time.time()

            return {"synced": len(pending), "status": "SUCCESS"}

        except Exception as e:
            self._errors.append({
                "time": time.time(),
                "error": str(e),
            })
            return {"synced": 0, "status": "ERROR", "error": str(e)}

    def _check_connectivity(self) -> bool:
        """
        Check if network connectivity is available.

        For the mock/demo version, this always returns True.
        In production, this would ping the actual remote server.
        """
        # Mock: always "connected" for local testing
        return True

    def _run_loop(self):
        """Background thread loop."""
        while not self._stop_event.is_set():
            try:
                if self._check_connectivity():
                    self.attempt_sync()
            except Exception as e:
                self._errors.append({
                    "time": time.time(),
                    "error": str(e),
                })

            # Wait for the sync interval, but check stop_event frequently
            self._stop_event.wait(timeout=self.sync_interval)
