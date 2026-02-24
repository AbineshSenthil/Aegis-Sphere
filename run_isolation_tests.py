"""
Aegis-Sphere — Isolation Test Runner
Run all worker scripts sequentially and verify Evidence Item JSON output + VRAM release.

Usage:
    python run_isolation_tests.py
"""

import subprocess
import sys
import time
import os

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

WORKERS = [
    ("MedASR",           "pipeline/asr_worker.py"),
    ("HeAR",             "pipeline/hear_worker.py"),
    ("CXR Foundation",   "pipeline/cxr_worker.py"),
    ("Path Foundation",  "pipeline/path_worker.py"),
    ("Derm Foundation",  "pipeline/derm_worker.py"),
    ("MedSigLIP",        "pipeline/medsig_worker.py"),
    ("TxGemma",          "pipeline/txgemma_worker.py"),
]


def run_worker(name: str, script_path: str) -> dict:
    """Run a single worker and capture output."""
    full_path = os.path.join(PROJECT_ROOT, script_path)
    start = time.time()

    try:
        result = subprocess.run(
            [sys.executable, full_path],
            capture_output=True,
            text=True,
            timeout=120,
            cwd=PROJECT_ROOT,
        )
        elapsed = time.time() - start

        has_json = "Evidence Item JSON" in result.stdout
        passed = result.returncode == 0 and has_json

        return {
            "name": name,
            "script": script_path,
            "passed": passed,
            "return_code": result.returncode,
            "elapsed_s": round(elapsed, 1),
            "has_evidence_json": has_json,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }

    except subprocess.TimeoutExpired:
        return {
            "name": name,
            "script": script_path,
            "passed": False,
            "return_code": -1,
            "elapsed_s": 120,
            "has_evidence_json": False,
            "stdout": "",
            "stderr": "TIMEOUT after 120s",
        }
    except Exception as e:
        return {
            "name": name,
            "script": script_path,
            "passed": False,
            "return_code": -1,
            "elapsed_s": 0,
            "has_evidence_json": False,
            "stdout": "",
            "stderr": str(e),
        }


def main():
    print("=" * 70)
    print("  AEGIS-SPHERE — COMPONENT ISOLATION TEST SUITE")
    print("=" * 70)
    print()

    results = []
    for name, script in WORKERS:
        print(f"▶ Running {name} ({script})...")
        r = run_worker(name, script)
        results.append(r)

        status = "✅ PASS" if r["passed"] else "❌ FAIL"
        print(f"  {status}  ({r['elapsed_s']}s, rc={r['return_code']})")

        if not r["passed"] and r["stderr"]:
            # Print first 3 lines of stderr
            for line in r["stderr"].strip().split("\n")[:3]:
                print(f"    ⚠ {line}")
        print()

    # ── Summary Table ──
    print("=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"{'Worker':<20} {'Status':<10} {'Time':<8} {'Evidence JSON':<15}")
    print("-" * 60)
    for r in results:
        status = "PASS" if r["passed"] else "FAIL"
        json_ok = "Yes" if r["has_evidence_json"] else "No"
        print(f"{r['name']:<20} {status:<10} {r['elapsed_s']:<8} {json_ok:<15}")

    passed = sum(1 for r in results if r["passed"])
    total = len(results)
    print("-" * 60)
    print(f"Total: {passed}/{total} passed")
    print()

    if passed == total:
        print("🎉 All isolation tests passed!")
    else:
        print(f"⚠ {total - passed} test(s) failed. Check output above for details.")
        sys.exit(1)


if __name__ == "__main__":
    main()
