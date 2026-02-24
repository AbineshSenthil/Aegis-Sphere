"""
Aegis-Sphere — GPU Lease Manager
Thread-safe sequential model loading. Only ONE model on GPU at a time.
Integrates with VRAMMonitor for live telemetry.
"""

import gc
import time
import threading
from typing import Optional, Callable


def _get_torch():
    """Lazy-import torch to avoid crashing if PyTorch is not installed or broken."""
    try:
        import torch
        return torch
    except (ImportError, OSError):
        return None


class GPULeaseManager:
    """
    Enforces strict sequential GPU access:
      lease.acquire("MedASR")   → loads model onto GPU
      ... inference ...
      lease.release()           → unloads, frees VRAM, runs gc

    Only one model can hold the lease at a time (thread-safe via Lock).
    """

    def __init__(self, vram_callback: Optional[Callable] = None):
        self._lock = threading.Lock()
        self._current_model: Optional[str] = None
        self._current_objects: list = []  # refs to model/processor for cleanup
        self._vram_callback = vram_callback  # called after load/unload for telemetry
        self._session_start = time.time()

    @property
    def current_model(self) -> Optional[str]:
        return self._current_model

    @property
    def is_busy(self) -> bool:
        return self._current_model is not None

    def acquire(self, model_name: str):
        """Acquire the GPU lease for a named model. Blocks until lease is free."""
        self._lock.acquire()
        if self._current_model is not None:
            # Safety: auto-release previous if somehow still held
            self._do_release()
        self._current_model = model_name
        self._log(f"GPU lease ACQUIRED by: {model_name}")
        if self._vram_callback:
            self._vram_callback(f"{model_name}_loaded", model_name)

    def register_objects(self, *objects):
        """Register model/processor objects for cleanup on release."""
        self._current_objects.extend(objects)

    def release(self):
        """Release the GPU lease, unload model, free VRAM."""
        if self._current_model is None:
            self._log("WARNING: release() called but no lease is held")
            try:
                self._lock.release()
            except RuntimeError:
                pass
            return
        model_name = self._current_model
        self._do_release()
        self._log(f"GPU lease RELEASED by: {model_name}")
        if self._vram_callback:
            self._vram_callback(f"{model_name}_unloaded", "None")
        self._lock.release()

    def _do_release(self):
        """Internal: delete registered objects and force VRAM cleanup."""
        for obj in self._current_objects:
            try:
                del obj
            except Exception:
                pass
        self._current_objects.clear()
        self._current_model = None

        torch = _get_torch()
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        gc.collect()

    def get_vram_snapshot(self) -> dict:
        """Return current VRAM usage snapshot."""
        torch = _get_torch()
        if torch is not None and torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / (1024 ** 2)
            reserved = torch.cuda.memory_reserved() / (1024 ** 2)
        else:
            allocated = 0.0
            reserved = 0.0

        return {
            "timestamp": time.time(),
            "elapsed_s": time.time() - self._session_start,
            "allocated_mb": round(allocated, 1),
            "reserved_mb": round(reserved, 1),
            "model_active": self._current_model or "None",
        }

    def _log(self, msg: str):
        elapsed = time.time() - self._session_start
        vram = self.get_vram_snapshot()
        print(
            f"[GPULease {elapsed:.1f}s] {msg} "
            f"| VRAM alloc={vram['allocated_mb']:.0f}MB "
            f"reserved={vram['reserved_mb']:.0f}MB"
        )


# ── Singleton instance ──
_gpu_lease: Optional[GPULeaseManager] = None


def get_gpu_lease(vram_callback: Optional[Callable] = None) -> GPULeaseManager:
    """Get or create the global GPU lease manager singleton."""
    global _gpu_lease
    if _gpu_lease is None:
        _gpu_lease = GPULeaseManager(vram_callback=vram_callback)
    return _gpu_lease
