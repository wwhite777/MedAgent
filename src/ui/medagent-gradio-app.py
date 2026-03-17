"""
MedAgent Gradio demo — interactive clinical decision support interface.

Features:
- Text input for clinical vignettes with example case buttons
- Real-time graph visualization showing active node
- Tabbed output: Triage, Literature, Differential Dx, SOAP Note
- Human-in-the-loop approval for emergency (ESI 1-2) cases
"""

from __future__ import annotations

import asyncio
import importlib
import json
import logging
import os
import time
import uuid
from pathlib import Path

import gradio as gr

# ---------------------------------------------------------------------------
# Imports (hyphenated filenames)
# ---------------------------------------------------------------------------
_graph_mod = importlib.import_module("src.graph.medagent-clinical-graph")
_state_mod = importlib.import_module("src.graph.medagent-state-define")

build_graph = _graph_mod.build_graph
ClinicalState = _state_mod.ClinicalState

logger = logging.getLogger("medagent.ui")
logging.basicConfig(level=logging.INFO)

# ---------------------------------------------------------------------------
# Load test cases for example buttons
# ---------------------------------------------------------------------------
_DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "test_cases.json"


def _load_examples() -> list[dict]:
    if _DATA_PATH.exists():
        with open(_DATA_PATH) as f:
            return json.load(f)
    return []


EXAMPLES = _load_examples()

# Example cases for quick buttons (one per ESI level)
QUICK_EXAMPLES = {
    "ESI 1 — STEMI": next((c["vignette"] for c in EXAMPLES if c["id"] == "TC-001"), ""),
    "ESI 2 — Stroke": next((c["vignette"] for c in EXAMPLES if c["id"] == "TC-003"), ""),
    "ESI 3 — Pneumonia": next((c["vignette"] for c in EXAMPLES if c["id"] == "TC-006"), ""),
    "ESI 4 — UTI": next((c["vignette"] for c in EXAMPLES if c["id"] == "TC-024"), ""),
    "ESI 5 — Med Refill": next((c["vignette"] for c in EXAMPLES if c["id"] == "TC-038"), ""),
}


# ---------------------------------------------------------------------------
# Graph execution SVG (Mermaid-style status visualization)
# ---------------------------------------------------------------------------
GRAPH_NODES = ["triage", "human_review", "literature", "reasoning", "report"]

NODE_LABELS = {
    "triage": "Triage Agent",
    "human_review": "Human Review",
    "literature": "Literature Agent",
    "reasoning": "Reasoning Agent",
    "report": "Report Agent",
}


def _make_graph_svg(active_node: str = "", completed: list[str] | None = None) -> str:
    """Generate an SVG visualization of the clinical workflow graph."""
    completed = completed or []
    nodes_svg = []
    x_start, y = 50, 30
    spacing = 130
    box_w, box_h = 120, 40

    for i, node in enumerate(GRAPH_NODES):
        x = x_start + i * spacing
        if node == active_node:
            fill, stroke, text_fill = "#2563eb", "#1d4ed8", "white"
        elif node in completed:
            fill, stroke, text_fill = "#22c55e", "#16a34a", "white"
        else:
            fill, stroke, text_fill = "#f1f5f9", "#94a3b8", "#334155"

        nodes_svg.append(
            f'<rect x="{x}" y="{y}" width="{box_w}" height="{box_h}" rx="8" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="2"/>'
        )
        label = NODE_LABELS.get(node, node)
        nodes_svg.append(
            f'<text x="{x + box_w // 2}" y="{y + box_h // 2 + 5}" '
            f'text-anchor="middle" fill="{text_fill}" font-size="11" font-family="sans-serif">{label}</text>'
        )

        # Arrow to next node
        if i < len(GRAPH_NODES) - 1:
            ax = x + box_w
            ay = y + box_h // 2
            nodes_svg.append(
                f'<line x1="{ax}" y1="{ay}" x2="{ax + spacing - box_w}" y2="{ay}" '
                f'stroke="#94a3b8" stroke-width="2" marker-end="url(#arrow)"/>'
            )

    svg = f"""<svg width="720" height="100" xmlns="http://www.w3.org/2000/svg">
  <defs>
    <marker id="arrow" markerWidth="10" markerHeight="7" refX="10" refY="3.5" orient="auto">
      <polygon points="0 0, 10 3.5, 0 7" fill="#94a3b8"/>
    </marker>
  </defs>
  {''.join(nodes_svg)}
</svg>"""
    return svg



def run_workflow_sync(vignette: str, approve_emergency: bool):
    """Synchronous wrapper for Gradio."""
    if not vignette.strip():
        return ("", "Please enter a clinical vignette.", "", "", "", "")

    import traceback

    try:
        app = build_graph(checkpointer=None)
        input_state = {"patient_input": vignette}

        # Run synchronously — Gradio handles threading
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            final_state = loop.run_until_complete(app.ainvoke(input_state))
        finally:
            loop.close()

        final = dict(final_state)

        # Format triage output
        esi = final.get("esi_level", "?")
        triage_text = (
            f"**ESI Level: {esi}**\n\n"
            f"**Chief Complaint:** {final.get('chief_complaint', 'N/A')}\n\n"
            f"**Reasoning:** {final.get('triage_reasoning', 'N/A')}\n\n"
            f"**Vitals:** {json.dumps(final.get('vitals', {}), indent=2)}\n\n"
            f"**Medications:** {', '.join(final.get('medications', []))}\n\n"
            f"**Labs:** {json.dumps(final.get('lab_results', []), indent=2)}"
        )

        # Format literature output
        lit = final.get("literature_results", [])
        summary = final.get("literature_summary", "")
        lit_items = "\n".join(
            f"- **{r.get('title', 'N/A')}** (relevance: {r.get('relevance_score', 'N/A')})\n  {r.get('snippet', '')}"
            for r in lit
        )
        literature_text = f"**Evidence Summary:**\n{summary}\n\n**Sources:**\n{lit_items}"

        # Format reasoning output
        ddx = final.get("differential_diagnosis", [])
        workup = final.get("recommended_workup", [])
        ddx_items = "\n".join(
            f"**{i+1}. {d.get('diagnosis', '?')}** — {d.get('probability', 0):.0%}\n"
            f"   {d.get('reasoning', '')}"
            for i, d in enumerate(ddx)
        )
        reasoning_text = (
            f"**Differential Diagnosis:**\n{ddx_items}\n\n"
            f"**Recommended Workup:** {', '.join(workup) if workup else 'None additional'}"
        )

        # Report
        report_text = final.get("soap_note", "No report generated.")

        svg = _make_graph_svg(active_node="report", completed=["triage", "literature", "reasoning", "report"])
        status = f"Workflow completed. ESI Level: {esi}"

        return (svg, triage_text, literature_text, reasoning_text, report_text, status)

    except Exception as exc:
        logger.exception("Workflow error")
        tb = traceback.format_exc()
        return (_make_graph_svg(), f"Error: {exc}", "", "", "", f"ERROR:\n{tb}")


# ---------------------------------------------------------------------------
# Gradio interface
# ---------------------------------------------------------------------------
def create_demo() -> gr.Blocks:
    with gr.Blocks(
        title="MedAgent — Multi-Agent Clinical Decision Support",
    ) as demo:
        gr.Markdown(
            "# MedAgent — Multi-Agent Clinical Decision Support\n"
            "A LangGraph-powered multi-agent system for clinical triage, literature search, "
            "differential diagnosis, and SOAP note generation."
        )

        with gr.Row():
            # ── Left: Input ─────────────────────────────────────────
            with gr.Column(scale=2):
                vignette_input = gr.Textbox(
                    label="Clinical Vignette",
                    placeholder="Enter a clinical vignette or select an example below...",
                    lines=10,
                )
                with gr.Row():
                    for label, text in QUICK_EXAMPLES.items():
                        gr.Button(label, size="sm").click(
                            fn=lambda t=text: t, outputs=vignette_input
                        )

                approve_checkbox = gr.Checkbox(
                    label="Auto-approve emergency cases (human-in-the-loop)",
                    value=True,
                )
                run_btn = gr.Button("Run Clinical Workflow", variant="primary", size="lg")

            # ── Right: Graph visualization ──────────────────────────
            with gr.Column(scale=3):
                graph_viz = gr.HTML(
                    value=_make_graph_svg(),
                    label="Workflow Progress",
                    elem_classes=["graph-container"],
                )
                status_box = gr.Textbox(
                    label="Execution Log",
                    lines=5,
                    interactive=False,
                    elem_classes=["status-box"],
                )

        # ── Tabbed output ───────────────────────────────────────────
        with gr.Tabs():
            with gr.Tab("Triage"):
                triage_output = gr.Markdown(value="*Waiting for input...*")
            with gr.Tab("Literature"):
                literature_output = gr.Markdown(value="*Waiting for input...*")
            with gr.Tab("Differential Diagnosis"):
                reasoning_output = gr.Markdown(value="*Waiting for input...*")
            with gr.Tab("SOAP Note"):
                report_output = gr.Markdown(value="*Waiting for input...*")

        # ── Wire up ─────────────────────────────────────────────────
        run_btn.click(
            fn=run_workflow_sync,
            inputs=[vignette_input, approve_checkbox],
            outputs=[
                graph_viz,
                triage_output,
                literature_output,
                reasoning_output,
                report_output,
                status_box,
            ],
        )

    return demo


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    demo = create_demo()
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("GRADIO_PORT", "7860")),
        share=True,
    )
