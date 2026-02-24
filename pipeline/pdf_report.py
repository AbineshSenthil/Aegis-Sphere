"""
Aegis-Sphere — HTML Report Generator
Generates a styled, self-contained HTML report that can be saved as PDF.
Uses only Python stdlib (no external PDF libraries needed).
"""

import html
import json
from datetime import datetime
from typing import Optional


def _esc(text: str) -> str:
    """HTML-escape text."""
    return html.escape(str(text)) if text else ""


def _severity_color(severity: str) -> str:
    return {"CRITICAL": "#ef4444", "MODERATE": "#f59e0b", "LOW": "#22c55e"}.get(severity.upper(), "#94a3b8")


def generate_report_html(
    oncocase: dict,
    debate_result: dict,
    txgemma_result: dict,
    evidence_trace: Optional[dict] = None,
    similar_cases: Optional[list] = None,
) -> str:
    """
    Generate a complete, self-contained HTML report from pipeline results.

    Args:
        oncocase:       The OncoCase dict (clinical_frame, evidence_pool, etc.)
        debate_result:  Output from persona_debate (pass_1..pass_4, patient_handout, etc.)
        txgemma_result: Output from txgemma_worker (interaction_flags, inventory_alerts, substitutions, etc.)
        evidence_trace: Optional dict from report_formatter.build_evidence_trace
        similar_cases:  Optional list from medsig_worker

    Returns:
        Complete HTML string ready for download or browser print-to-PDF.
    """
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    # ── Extract clinical frame ──
    clinical_frame = oncocase.get("clinical_frame", {})
    patient_id = clinical_frame.get("patient_id", "DEMO-001")
    age = clinical_frame.get("age", "38")
    gender = clinical_frame.get("gender", "Male")
    conditions = clinical_frame.get("conditions", ["HIV+", "suspected lymphoma"])
    medications = clinical_frame.get("medications", ["TLD (Tenofovir/Lamivudine/Dolutegravir)"])

    # ── Extract staging ──
    staging = oncocase.get("staging_confidence", debate_result.get("staging_confidence", "PROVISIONAL"))

    # ── Persona outputs ──
    pass_names = [
        ("Pass 1 — Virtual Pathologist", "pass_1"),
        ("Pass 2 — Virtual Radiologist", "pass_2"),
        ("Pass 3 — Virtual Oncologist", "pass_3"),
        ("Pass 4 — Chief Physician Synthesis", "pass_4"),
    ]

    # ── Drug interactions ──
    interactions = txgemma_result.get("interaction_flags", [])
    inv_alerts = txgemma_result.get("inventory_alerts", [])
    substitutions = txgemma_result.get("substitutions", [])

    # ── Patient handout ──
    patient_handout = debate_result.get("patient_handout", "")

    # ── Build HTML ──
    sections = []

    # Header
    sections.append(f"""
    <div style="text-align:center; margin-bottom:30px; border-bottom:2px solid #a78bfa; padding-bottom:20px">
        <h1 style="margin:0; color:#818cf8; font-size:1.6rem">🩺 Aegis-Sphere Clinical Report</h1>
        <p style="color:#94a3b8; font-size:0.85rem; margin-top:6px">
            Generated: {now} | Patient: {_esc(patient_id)} | Staging: {_esc(staging)}
        </p>
    </div>
    """)

    # Patient Summary
    sections.append(f"""
    <div style="background:#1e293b; border-radius:10px; padding:16px; margin-bottom:20px">
        <h2 style="color:#60a5fa; font-size:1.1rem; margin-top:0">📋 Patient Summary</h2>
        <table style="width:100%; font-size:0.85rem; color:#cbd5e1">
            <tr><td style="padding:4px 8px; font-weight:600; width:140px">Patient ID</td><td>{_esc(patient_id)}</td></tr>
            <tr><td style="padding:4px 8px; font-weight:600">Age / Gender</td><td>{_esc(str(age))} / {_esc(gender)}</td></tr>
            <tr><td style="padding:4px 8px; font-weight:600">Conditions</td><td>{_esc(', '.join(conditions) if isinstance(conditions, list) else str(conditions))}</td></tr>
            <tr><td style="padding:4px 8px; font-weight:600">Medications</td><td>{_esc(', '.join(medications) if isinstance(medications, list) else str(medications))}</td></tr>
            <tr><td style="padding:4px 8px; font-weight:600">Staging</td><td>{_esc(staging)}</td></tr>
        </table>
    </div>
    """)

    # Tumor Board Synthesis
    for title, key in pass_names:
        output = debate_result.get(key, "")
        if output:
            sections.append(f"""
            <div style="background:#1e293b; border-radius:10px; padding:16px; margin-bottom:14px">
                <h3 style="color:#a78bfa; font-size:0.95rem; margin-top:0">{_esc(title)}</h3>
                <p style="color:#cbd5e1; font-size:0.82rem; line-height:1.7; white-space:pre-wrap">{_esc(str(output))}</p>
            </div>
            """)

    # Drug Interactions
    if interactions:
        rows = []
        for ix in interactions:
            if isinstance(ix, dict):
                sev = ix.get("severity", "LOW")
                color = _severity_color(sev)
                drugs = ix.get("drugs", "")
                detail = ix.get("detail", ix.get("text", ""))
                rows.append(f"""
                <tr>
                    <td style="padding:6px 8px"><span style="color:{color}; font-weight:700">{_esc(sev)}</span></td>
                    <td style="padding:6px 8px; font-weight:600">{_esc(drugs)}</td>
                    <td style="padding:6px 8px">{_esc(detail)}</td>
                </tr>""")

        sections.append(f"""
        <div style="background:#1e293b; border-radius:10px; padding:16px; margin-bottom:14px">
            <h2 style="color:#f59e0b; font-size:1.1rem; margin-top:0">💊 Drug Interactions</h2>
            <table style="width:100%; font-size:0.82rem; color:#cbd5e1; border-collapse:collapse">
                <thead>
                    <tr style="border-bottom:1px solid #334155">
                        <th style="text-align:left; padding:6px 8px; color:#94a3b8; width:90px">Severity</th>
                        <th style="text-align:left; padding:6px 8px; color:#94a3b8; width:200px">Drugs</th>
                        <th style="text-align:left; padding:6px 8px; color:#94a3b8">Detail</th>
                    </tr>
                </thead>
                <tbody>{''.join(rows)}</tbody>
            </table>
        </div>
        """)

    # Inventory Alerts
    if inv_alerts:
        alert_html = []
        for alert in inv_alerts:
            if isinstance(alert, dict):
                status = alert.get("status", "")
                msg = alert.get("message", "")
                icon = "🚫" if status in ("UNAVAILABLE", "OUT_OF_STOCK") else "⚠️"
                alert_html.append(f"<div style='padding:6px 0; border-bottom:1px solid #1e293b'>{icon} {_esc(msg)}</div>")

        sections.append(f"""
        <div style="background:#1e293b; border-radius:10px; padding:16px; margin-bottom:14px">
            <h2 style="color:#ef4444; font-size:1.1rem; margin-top:0">📦 Inventory Alerts</h2>
            <div style="font-size:0.82rem; color:#fca5a5">{''.join(alert_html)}</div>
        </div>
        """)

    # Substitutions
    if substitutions:
        sub_items = []
        for sub in substitutions:
            text = sub.get("text", str(sub)) if isinstance(sub, dict) else str(sub)
            sub_items.append(f"<div style='padding:4px 0'>🔄 {_esc(text)}</div>")

        sections.append(f"""
        <div style="background:#1e293b; border-radius:10px; padding:16px; margin-bottom:14px">
            <h2 style="color:#60a5fa; font-size:1.1rem; margin-top:0">🔄 Substitution Recommendations</h2>
            <div style="font-size:0.82rem; color:#93c5fd">{''.join(sub_items)}</div>
        </div>
        """)

    # Evidence Trace
    if evidence_trace:
        trace_rows = []
        for model, claims in evidence_trace.items():
            for claim in claims[:5]:
                trace_rows.append(f"""
                <tr>
                    <td style="padding:4px 8px; font-weight:600">{_esc(model)}</td>
                    <td style="padding:4px 8px">{_esc(claim[:120])}</td>
                </tr>""")

        if trace_rows:
            sections.append(f"""
            <div style="background:#1e293b; border-radius:10px; padding:16px; margin-bottom:14px">
                <h2 style="color:#a78bfa; font-size:1.1rem; margin-top:0">🏷️ Evidence Trace</h2>
                <table style="width:100%; font-size:0.82rem; color:#cbd5e1; border-collapse:collapse">
                    <thead>
                        <tr style="border-bottom:1px solid #334155">
                            <th style="text-align:left; padding:4px 8px; color:#94a3b8; width:180px">Source</th>
                            <th style="text-align:left; padding:4px 8px; color:#94a3b8">Claim</th>
                        </tr>
                    </thead>
                    <tbody>{''.join(trace_rows)}</tbody>
                </table>
            </div>
            """)

    # Patient Handout
    if patient_handout:
        sections.append(f"""
        <div style="background:#0f172a; border:2px solid #818cf8; border-radius:10px; padding:20px; margin-bottom:20px">
            <h2 style="color:#818cf8; font-size:1.1rem; margin-top:0">💌 Patient Handout</h2>
            <div style="color:#e2e8f0; font-size:0.85rem; line-height:1.8; white-space:pre-wrap">{_esc(str(patient_handout))}</div>
        </div>
        """)

    # Footer
    sections.append("""
    <div style="text-align:center; margin-top:30px; padding-top:16px; border-top:1px solid #334155">
        <p style="color:#64748b; font-size:0.72rem; line-height:1.5">
            ⚠️ <strong>Disclaimer:</strong> This report is generated by Aegis-Sphere, an AI-assisted clinical
            decision-support system. It is NOT a substitute for professional medical advice, diagnosis, or treatment.
            All clinical decisions must be reviewed and approved by a qualified healthcare professional.<br>
            <em>Aegis-Sphere v1.0 • Google AI Hackathon 2025</em>
        </p>
    </div>
    """)

    # Assemble
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Aegis-Sphere Clinical Report — {_esc(patient_id)}</title>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
            background: #0f172a;
            color: #e2e8f0;
            padding: 40px;
            max-width: 900px;
            margin: 0 auto;
        }}
        @media print {{
            body {{ background: white; color: #1e293b; padding: 20px; }}
            div[style*="background:#1e293b"] {{ background: #f8fafc !important; border: 1px solid #e2e8f0; }}
            h1, h2, h3 {{ color: #1e293b !important; }}
            p, td, div {{ color: #334155 !important; }}
        }}
    </style>
</head>
<body>
{''.join(sections)}
</body>
</html>"""
