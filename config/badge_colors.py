"""
Aegis-Sphere — Evidence Grounding Badge Colors & Icons
Maps each source model to its UI badge styling.
"""

BADGE_MAP = {
    "Path_Foundation": {
        "color": "#7c3aed",
        "text_color": "white",
        "icon": "🔬",
        "label": "Path_Foundation",
    },
    "CXR_Foundation": {
        "color": "#1a73e8",
        "text_color": "white",
        "icon": "📷",
        "label": "CXR_Foundation",
    },
    "Derm_Foundation": {
        "color": "#0d9488",
        "text_color": "white",
        "icon": "🩺",
        "label": "Derm_Foundation",
    },
    "HeAR": {
        "color": "#34a853",
        "text_color": "white",
        "icon": "🎙️",
        "label": "HeAR",
    },
    "TxGemma_DDI": {
        "color": "#fbbc04",
        "text_color": "black",
        "icon": "⚗️",
        "label": "TxGemma",
    },
    "TxGemma": {
        "color": "#fbbc04",
        "text_color": "black",
        "icon": "⚗️",
        "label": "TxGemma",
    },
    "Local_Inventory_JSON": {
        "color": "#ea4335",
        "text_color": "white",
        "icon": "💊",
        "label": "Inventory",
    },
    "MedSigLIP_CaseLibrary": {
        "color": "#f97316",
        "text_color": "white",
        "icon": "🗂️",
        "label": "MedSigLIP",
    },
    "MedASR_Transcript": {
        "color": "#6b7280",
        "text_color": "white",
        "icon": "📝",
        "label": "MedASR",
    },
    "Clinical_Frame_JSON": {
        "color": "#475569",
        "text_color": "white",
        "icon": "🗒️",
        "label": "Clinical_Frame",
    },
}


def get_badge_html(source_name: str) -> str:
    """Return styled HTML <span> badge for a given source model name."""
    cfg = BADGE_MAP.get(source_name, {
        "color": "#94a3b8",
        "text_color": "white",
        "icon": "🏷️",
        "label": source_name,
    })
    return (
        f'<span style="'
        f"background:{cfg['color']}; "
        f"color:{cfg['text_color']}; "
        f"padding:2px 8px; "
        f"border-radius:12px; "
        f"font-size:0.75em; "
        f"font-weight:600; "
        f"letter-spacing:0.3px; "
        f"white-space:nowrap; "
        f"display:inline-block; "
        f"margin:1px 2px; "
        f"cursor:pointer"
        f'">{cfg["icon"]} {cfg["label"]}</span>'
    )
