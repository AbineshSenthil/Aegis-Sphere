"""
Aegis-Sphere — VRAM Telemetry Monitor
Live GPU memory tracking with Plotly chart generation.
"""

import time
import csv
from pathlib import Path
from typing import Optional, List
import plotly.graph_objects as go

from config.settings import MAX_VRAM_MB, VRAM_SAFE_ZONE, VRAM_LOADED_ZONE


class VRAMMonitor:
    """
    Monitors GPU VRAM usage across the pipeline.

    Usage:
        monitor = VRAMMonitor()
        monitor.log_phase("Phase_1_MedASR", "MedASR")
        ... (run phase) ...
        monitor.log_phase("Phase_1_done", "None")
        fig = monitor.generate_chart()
        monitor.export_csv("vram_log.csv")
    """

    def __init__(self):
        self._log: List[dict] = []
        self._session_start = time.time()
        self._annotations: List[dict] = []
        self._peak_allocated = 0.0
        self._peak_reserved = 0.0

    def log_phase(self, phase_name: str, model_name: str):
        """Record a VRAM snapshot for a pipeline phase."""
        snapshot = self._take_snapshot(phase_name, model_name)
        self._log.append(snapshot)

        # Track peak
        if snapshot["allocated_mb"] > self._peak_allocated:
            self._peak_allocated = snapshot["allocated_mb"]
        if snapshot["reserved_mb"] > self._peak_reserved:
            self._peak_reserved = snapshot["reserved_mb"]

        # Add annotation for model load/unload events
        if "loaded" in phase_name.lower() or "unloaded" in phase_name.lower():
            self._annotations.append({
                "x": snapshot["elapsed_s"],
                "y": snapshot["allocated_mb"],
                "text": phase_name.replace("_", " "),
            })
        elif phase_name.endswith("_done"):
            self._annotations.append({
                "x": snapshot["elapsed_s"],
                "y": snapshot["allocated_mb"],
                "text": f"✓ {phase_name.replace('_done', '').replace('_', ' ')}",
            })
        else:
            self._annotations.append({
                "x": snapshot["elapsed_s"],
                "y": snapshot["allocated_mb"],
                "text": model_name,
            })

    def _take_snapshot(self, phase: str, model: str) -> dict:
        """Take a VRAM usage snapshot."""
        try:
            import torch
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / (1024 ** 2)
                reserved = torch.cuda.memory_reserved() / (1024 ** 2)
            else:
                allocated = 0.0
                reserved = 0.0
        except ImportError:
            allocated = 0.0
            reserved = 0.0

        return {
            "timestamp": time.time(),
            "elapsed_s": round(time.time() - self._session_start, 2),
            "phase": phase,
            "allocated_mb": round(allocated, 1),
            "reserved_mb": round(reserved, 1),
            "model_active": model,
        }

    @property
    def peak_allocated_mb(self) -> float:
        return self._peak_allocated

    @property
    def peak_reserved_mb(self) -> float:
        return self._peak_reserved

    def get_log(self) -> List[dict]:
        return self._log.copy()

    def generate_chart(self) -> go.Figure:
        """Generate the live VRAM telemetry Plotly chart."""
        if not self._log:
            return self._empty_chart()

        elapsed = [s["elapsed_s"] for s in self._log]
        allocated = [s["allocated_mb"] for s in self._log]
        reserved = [s["reserved_mb"] for s in self._log]

        fig = go.Figure()

        # ── Zone background shading (higher opacity for visibility) ──
        # Safe zone (green)
        fig.add_hrect(
            y0=0, y1=VRAM_SAFE_ZONE,
            fillcolor="rgba(34, 197, 94, 0.12)",
            line_width=0,
            annotation_text="SAFE",
            annotation_position="top left",
            annotation=dict(font=dict(size=9, color="rgba(34,197,94,0.5)", family="Inter, monospace")),
        )
        # Loaded zone (amber)
        fig.add_hrect(
            y0=VRAM_SAFE_ZONE, y1=VRAM_LOADED_ZONE,
            fillcolor="rgba(245, 158, 11, 0.10)",
            line_width=0,
            annotation_text="LOADED",
            annotation_position="top left",
            annotation=dict(font=dict(size=9, color="rgba(245,158,11,0.5)", family="Inter, monospace")),
        )
        # Danger zone (red)
        fig.add_hrect(
            y0=VRAM_LOADED_ZONE, y1=MAX_VRAM_MB,
            fillcolor="rgba(239, 68, 68, 0.10)",
            line_width=0,
            annotation_text="DANGER",
            annotation_position="top left",
            annotation=dict(font=dict(size=9, color="rgba(239,68,68,0.5)", family="Inter, monospace")),
        )

        # ── Allocated line (bold neon blue with gradient fill) ──
        fig.add_trace(go.Scatter(
            x=elapsed, y=allocated,
            mode="lines+markers",
            name="Allocated",
            line=dict(color="#60a5fa", width=3, shape="spline"),
            marker=dict(size=6, color="#60a5fa", line=dict(width=1, color="#1e40af")),
            fill="tozeroy",
            fillcolor="rgba(96, 165, 250, 0.15)",
            hovertemplate="<b>%{y:.0f} MB</b> at %{x:.1f}s<extra>Allocated</extra>",
        ))

        # ── Reserved line (brighter dashed) ──
        fig.add_trace(go.Scatter(
            x=elapsed, y=reserved,
            mode="lines",
            name="Reserved",
            line=dict(color="#64748b", width=2, dash="dot"),
            hovertemplate="<b>%{y:.0f} MB</b> at %{x:.1f}s<extra>Reserved</extra>",
        ))

        # ── 8GB Redline ──
        fig.add_hline(
            y=MAX_VRAM_MB,
            line=dict(color="#ef4444", width=2.5, dash="dash"),
            annotation_text="⬤ 8 GB LIMIT",
            annotation_position="top right",
            annotation_font=dict(color="#ef4444", size=11, family="Inter, monospace"),
        )

        # ── Phase annotations (bigger, more visible) ──
        for ann in self._annotations:
            if ann["text"] and "None" not in ann["text"]:
                fig.add_annotation(
                    x=ann["x"], y=ann["y"],
                    text=ann["text"],
                    showarrow=True,
                    arrowhead=2,
                    arrowsize=0.8,
                    arrowwidth=1.5,
                    arrowcolor="#94a3b8",
                    font=dict(size=9, color="#e2e8f0", family="Inter, monospace"),
                    bgcolor="rgba(15, 23, 42, 0.9)",
                    bordercolor="#475569",
                    borderwidth=1,
                    borderpad=4,
                )

        # ── Layout (taller, bolder, premium) ──
        fig.update_layout(
            title=dict(
                text="<b>VRAM TELEMETRY</b>  ·  Sawtooth Phase Tracker",
                font=dict(size=13, color="#94a3b8", family="Inter, monospace"),
                x=0.01, y=0.98,
            ),
            xaxis_title="Elapsed Time (s)",
            yaxis_title="VRAM (MB)",
            xaxis=dict(
                showgrid=True,
                gridcolor="rgba(71, 85, 105, 0.3)",
                gridwidth=1,
                zeroline=False,
                title_font=dict(size=11, color="#94a3b8"),
                tickfont=dict(size=10, color="#cbd5e1"),
            ),
            yaxis=dict(
                range=[0, MAX_VRAM_MB + 500],
                showgrid=True,
                gridcolor="rgba(71, 85, 105, 0.25)",
                gridwidth=1,
                zeroline=False,
                title_font=dict(size=11, color="#94a3b8"),
                tickfont=dict(size=10, color="#cbd5e1"),
                dtick=1000,
            ),
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15, 23, 42, 0.85)",
            font=dict(family="Inter, monospace", size=11, color="#e2e8f0"),
            margin=dict(l=50, r=16, t=36, b=44),
            height=380,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                font=dict(size=10, color="#e2e8f0"),
                bgcolor="rgba(15, 23, 42, 0.6)",
                bordercolor="rgba(71, 85, 105, 0.3)",
                borderwidth=1,
            ),
            showlegend=True,
            hoverlabel=dict(
                bgcolor="rgba(15, 23, 42, 0.95)",
                bordercolor="#475569",
                font_size=11,
                font_color="#e2e8f0",
                font_family="Inter, monospace",
            ),
        )

        return fig

    def _empty_chart(self) -> go.Figure:
        """Return an empty chart placeholder."""
        fig = go.Figure()
        fig.add_hline(
            y=MAX_VRAM_MB,
            line=dict(color="#ef4444", width=2.5, dash="dash"),
            annotation_text="⬤ 8 GB LIMIT",
            annotation_position="top right",
            annotation_font=dict(color="#ef4444", size=11, family="Inter, monospace"),
        )
        # Add a subtle "waiting for data" annotation
        fig.add_annotation(
            x=0.5, y=0.5, xref="paper", yref="paper",
            text="Run pipeline to see VRAM telemetry",
            showarrow=False,
            font=dict(size=14, color="rgba(148, 163, 184, 0.5)", family="Inter"),
        )
        fig.update_layout(
            title=dict(
                text="<b>VRAM TELEMETRY</b>  ·  Awaiting Pipeline",
                font=dict(size=13, color="#64748b", family="Inter, monospace"),
                x=0.01, y=0.98,
            ),
            xaxis_title="Elapsed Time (s)",
            yaxis_title="VRAM (MB)",
            xaxis=dict(
                showgrid=True,
                gridcolor="rgba(71, 85, 105, 0.2)",
                zeroline=False,
                title_font=dict(size=11, color="#64748b"),
                tickfont=dict(size=10, color="#64748b"),
            ),
            yaxis=dict(
                range=[0, MAX_VRAM_MB + 500],
                showgrid=True,
                gridcolor="rgba(71, 85, 105, 0.2)",
                zeroline=False,
                title_font=dict(size=11, color="#64748b"),
                tickfont=dict(size=10, color="#64748b"),
                dtick=1000,
            ),
            template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15, 23, 42, 0.85)",
            font=dict(family="Inter, monospace", size=11, color="#94a3b8"),
            margin=dict(l=50, r=16, t=36, b=44),
            height=380,
        )
        return fig

    def generate_demo_chart(self) -> go.Figure:
        """Generate a realistic demo chart showing typical pipeline VRAM usage."""
        # Simulated VRAM profile for the full pipeline
        phases = [
            ("Start",           0,   50,  100),
            ("MedASR load",     3,   800, 900),
            ("MedASR infer",    8,   820, 950),
            ("MedASR unload",   10,  80,  200),
            ("HeAR load",       12,  650, 750),
            ("HeAR infer",      15,  680, 800),
            ("HeAR unload",     17,  60,  180),
            ("Path load",       19,  520, 620),
            ("Path infer",      22,  550, 660),
            ("Path unload",     24,  50,  170),
            ("CXR load",        26,  500, 600),
            ("CXR infer",       29,  530, 640),
            ("CXR unload",      31,  45,  160),
            ("Derm load",       33,  510, 610),
            ("Derm infer",      36,  540, 650),
            ("Derm unload",     38,  40,  150),
            ("MedSigLIP CPU",   40,  40,  150),
            ("TxGemma load",    45,  4800, 5200),
            ("TxGemma infer",   55,  5100, 5500),
            ("TxGemma unload",  58,  60,  200),
            ("MedGemma load",   62,  2700, 3100),
            ("MedGemma P1",     68,  2850, 3200),
            ("MedGemma P2",     74,  2830, 3180),
            ("MedGemma P3",     80,  2860, 3220),
            ("MedGemma P4",     88,  2900, 3300),
            ("MedGemma P5",     94,  2880, 3250),
            ("MedGemma unload", 97,  50,  180),
            ("Done",            100, 30,  100),
        ]

        self._log = []
        self._annotations = []
        self._peak_allocated = 0

        for label, elapsed, alloc, reserved in phases:
            entry = {
                "timestamp": self._session_start + elapsed,
                "elapsed_s": elapsed,
                "phase": label,
                "allocated_mb": alloc,
                "reserved_mb": reserved,
                "model_active": label.split(" ")[0] if "load" in label or "infer" in label else "None",
            }
            self._log.append(entry)
            if alloc > self._peak_allocated:
                self._peak_allocated = alloc
            if reserved > self._peak_reserved:
                self._peak_reserved = reserved

            # Add annotations for key events
            if "load" in label.lower() or "unload" in label.lower():
                self._annotations.append({
                    "x": elapsed,
                    "y": alloc,
                    "text": label,
                })

        return self.generate_chart()

    def export_csv(self, path: str):
        """Export VRAM log to CSV."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        if not self._log:
            return

        with open(path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                "timestamp", "elapsed_s", "phase",
                "allocated_mb", "reserved_mb", "model_active",
            ])
            writer.writeheader()
            writer.writerows(self._log)
