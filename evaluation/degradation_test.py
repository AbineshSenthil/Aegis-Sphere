"""
Aegis-Sphere — Graceful Degradation Test Suite
Runs pipeline on 5 scenarios with 0–5 missing items.
"""

import sys
import os
import json

# Fix Windows console encoding
sys.stdout.reconfigure(encoding='utf-8')

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipeline.oncocase_builder import build_oncocase
from config.settings import DegradationLevel


def make_evidence(model, modality, status="OK", finding="Test finding", nba=None):
    return {
        "modality": modality,
        "model": model,
        "status": status,
        "finding": finding if status == "OK" else None,
        "confidence": 0.85 if status == "OK" else None,
        "embedding": None,
        "nba": nba,
    }


CLINICAL_FRAME = {
    "symptoms": ["night sweats", "weight loss", "cough"],
    "medications": ["tenofovir", "lamivudine", "dolutegravir"],
    "conditions": ["HIV-positive", "lymphoma"],
    "lab_values": ["CD4 85"],
    "demographics": {"age": 38, "sex": "male"},
}

RISK_RESULT = {
    "tb_risk_score": 0.7,
    "tb_risk_level": "HIGH",
    "hiv_risk_score": 0.85,
    "overall_risk_level": "RED",
    "uncertainty_flags": [],
    "staging_override": None,
    "treatment_override": None,
    "missing_count": 0,
}


def test_scenario(name, evidence_pool, expected_degradation, expected_staging_contains):
    """Run a degradation scenario and validate output."""
    oncocase = build_oncocase(
        clinical_frame=CLINICAL_FRAME,
        evidence_pool=evidence_pool,
        risk_result=RISK_RESULT,
    )

    degradation = oncocase["degradation_level"]
    staging = oncocase["staging_confidence"]
    missing = oncocase["missing_count"]
    nba_count = len(oncocase["nba_list"])
    passes = oncocase["pass_config"]["run_passes"]

    passed = True
    errors = []

    if degradation != expected_degradation:
        errors.append(f"Expected degradation={expected_degradation}, got={degradation}")
        passed = False

    if expected_staging_contains and expected_staging_contains not in staging:
        errors.append(f"Expected staging to contain '{expected_staging_contains}', got='{staging}'")
        passed = False

    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"\n{status} — {name}")
    print(f"  Degradation: {degradation}")
    print(f"  Staging: {staging}")
    print(f"  Missing: {missing}")
    print(f"  NBA items: {nba_count}")
    print(f"  Passes to run: {passes}")

    if errors:
        for e in errors:
            print(f"  ERROR: {e}")

    return passed


def run_all_tests():
    print("=" * 60)
    print("AEGIS-SPHERE — Graceful Degradation Test Suite")
    print("=" * 60)

    results = []

    # ── Scenario 1: 0 missing (FULL) ──
    pool = [
        make_evidence("MedASR", "audio"),
        make_evidence("HeAR", "cough"),
        make_evidence("Path_Foundation", "histopathology"),
        make_evidence("CXR_Foundation", "cxr"),
        make_evidence("Derm_Foundation", "derm"),
    ]
    results.append(test_scenario(
        "Scenario 1: All data present (0 missing)",
        pool, DegradationLevel.FULL, "CONFIRMED",
    ))

    # ── Scenario 2: 1 missing (REDUCED) ──
    pool = [
        make_evidence("MedASR", "audio"),
        make_evidence("HeAR", "cough"),
        make_evidence("Path_Foundation", "histopathology"),
        make_evidence("CXR_Foundation", "cxr"),
        make_evidence("Derm_Foundation", "derm", status="MISSING_DATA",
                      nba="Recommend clinical photograph"),
    ]
    results.append(test_scenario(
        "Scenario 2: Derm missing (1 missing)",
        pool, DegradationLevel.REDUCED, "CONFIRMED",
    ))

    # ── Scenario 3: 2 missing (PROVISIONAL) ──
    pool = [
        make_evidence("MedASR", "audio"),
        make_evidence("HeAR", "cough"),
        make_evidence("Path_Foundation", "histopathology", status="MISSING_DATA",
                      nba="Recommend FNAC"),
        make_evidence("CXR_Foundation", "cxr"),
        make_evidence("Derm_Foundation", "derm", status="MISSING_DATA",
                      nba="Recommend photo"),
    ]
    results.append(test_scenario(
        "Scenario 3: Path + Derm missing (2 missing)",
        pool, DegradationLevel.PROVISIONAL, "PROVISIONAL",
    ))

    # ── Scenario 4: 3 missing (MINIMAL) ──
    pool = [
        make_evidence("MedASR", "audio"),
        make_evidence("HeAR", "cough", status="MISSING_DATA"),
        make_evidence("Path_Foundation", "histopathology", status="MISSING_DATA"),
        make_evidence("CXR_Foundation", "cxr", status="MISSING_DATA"),
        make_evidence("Derm_Foundation", "derm"),
    ]
    results.append(test_scenario(
        "Scenario 4: HeAR + Path + CXR missing (3 missing)",
        pool, DegradationLevel.MINIMAL, "INSUFFICIENT",
    ))

    # ── Scenario 5: All 5 missing (NO_DATA) ──
    pool = [
        make_evidence("MedASR", "audio", status="MISSING_DATA"),
        make_evidence("HeAR", "cough", status="MISSING_DATA"),
        make_evidence("Path_Foundation", "histopathology", status="MISSING_DATA"),
        make_evidence("CXR_Foundation", "cxr", status="MISSING_DATA"),
        make_evidence("Derm_Foundation", "derm", status="MISSING_DATA"),
    ]
    results.append(test_scenario(
        "Scenario 5: All data missing (5 missing)",
        pool, DegradationLevel.NO_DATA, "NO_DATA",
    ))

    # ── Summary ──
    print("\n" + "=" * 60)
    passed = sum(results)
    total = len(results)
    print(f"RESULTS: {passed}/{total} scenarios passed")
    if passed == total:
        print("✅ All degradation scenarios working correctly!")
    else:
        print("❌ Some scenarios failed — review output above.")

    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
