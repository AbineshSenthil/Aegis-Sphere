"""
Aegis-Sphere — Report Formatter
Tag parser, badge renderer, and NBA formatter for the UI.
"""

import os
import sys
import re

# ── Ensure project root is on sys.path for standalone execution ──
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.badge_colors import get_badge_html, BADGE_MAP


def parse_source_tags(text: str) -> tuple:
    """
    Parse all [Source: X] tags from text.

    Returns:
        (clean_text_with_tags, list of unique source names)
    """
    if not text:
        return "", []

    pattern = r'\[Source:\s*(\w+(?:_\w+)*)\]'
    sources = re.findall(pattern, text)
    unique_sources = list(dict.fromkeys(sources))

    return text, unique_sources


def build_evidence_trace(all_outputs: list) -> dict:
    """
    Build evidence_trace dict from all MedGemma and TxGemma outputs.

    Returns:
        {"model_name": ["claim 1", "claim 2"], ...}
    """
    trace = {}
    pattern = r'([^.!?\n]*?\[Source:\s*(\w+(?:_\w+)*)\][^.!?\n]*[.!?\n]?)'

    for output in all_outputs:
        text = output.get("output", "") if isinstance(output, dict) else str(output)
        for match in re.finditer(pattern, text):
            claim = match.group(1).strip()
            source = match.group(2)
            if source not in trace:
                trace[source] = []
            # Clean the claim text (remove the tag for display)
            clean_claim = re.sub(r'\[Source:\s*\w+(?:_\w+)*\]', '', claim).strip()
            if clean_claim and clean_claim not in trace[source]:
                trace[source].append(clean_claim)

    return trace


def render_badges_in_text(text: str) -> str:
    """
    Replace all [Source: X] tags in text with styled HTML badges.
    """
    if not text:
        return ""

    def replace_tag(match):
        source_name = match.group(1)
        return get_badge_html(source_name)

    pattern = r'\[Source:\s*(\w+(?:_\w+)*)\]'
    return re.sub(pattern, replace_tag, text)


def format_evidence_trace_table(trace: dict) -> str:
    """Format evidence trace as an HTML table for the sidebar."""
    if not trace:
        return "<p style='color:#94a3b8'>No evidence tags found.</p>"

    rows = []
    for model, claims in trace.items():
        badge = get_badge_html(model)
        claim_list = "<br>".join(f"• {c[:80]}{'...' if len(c) > 80 else ''}" for c in claims[:5])
        rows.append(f"<tr><td style='vertical-align:top; padding:6px'>{badge}</td>"
                     f"<td style='padding:6px; color:#e2e8f0; font-size:0.85em'>{claim_list}</td></tr>")

    return (
        "<table style='width:100%; border-collapse:collapse'>"
        "<thead><tr><th style='text-align:left; padding:6px; color:#94a3b8; font-size:0.8em'>Source</th>"
        "<th style='text-align:left; padding:6px; color:#94a3b8; font-size:0.8em'>Claims</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def format_nba_checklist(nba_list: list, for_patient: bool = False) -> str:
    """Format NBA items as a checklist."""
    if not nba_list:
        return ""

    items = []
    for nba in nba_list:
        if for_patient:
            text = nba.get("patient_language", nba.get("nba", ""))
            items.append(f"☐ {text}")
        else:
            model = nba.get("model", "")
            text = nba.get("nba", "")
            cost = nba.get("cost_inr", "N/A")
            items.append(f"☐ **{model}**: {text} *(Cost: INR {cost})*")

    return "\n".join(items)


def format_staging_badge(staging: str) -> str:
    """Return a styled HTML badge for the staging confidence level."""
    if not staging:
        return ""

    if "CONFIRMED" in staging:
        bg = "#22c55e"
        icon = "✅"
    elif "PROVISIONAL" in staging:
        bg = "#f59e0b"
        icon = "⚠️"
    elif "INSUFFICIENT" in staging:
        bg = "#ef4444"
        icon = "🔴"
    elif "NO_DATA" in staging:
        bg = "#6b7280"
        icon = "⬜"
    else:
        bg = "#94a3b8"
        icon = "❓"

    return (
        f'<div style="display:inline-block; background:{bg}; color:white; '
        f'padding:6px 16px; border-radius:8px; font-weight:700; '
        f'font-size:1em; letter-spacing:0.5px">'
        f'{icon} {staging}</div>'
    )


def format_risk_badge(risk_level: str, risk_score: float) -> str:
    """Return styled HTML for the risk level badge."""
    colors = {
        "RED": ("#ef4444", "🔴"),
        "AMBER": ("#f59e0b", "🟡"),
        "GREEN": ("#22c55e", "🟢"),
    }
    bg, icon = colors.get(risk_level, ("#94a3b8", "❓"))

    return (
        f'<div style="display:inline-block; background:{bg}; color:white; '
        f'padding:4px 12px; border-radius:8px; font-weight:600; font-size:0.9em">'
        f'{icon} {risk_level} ({risk_score:.0%})</div>'
    )
