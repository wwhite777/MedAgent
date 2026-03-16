"""
MedAgent clinical workflow — LangGraph StateGraph definition.

Orchestrates four agents (triage → literature → reasoning → report) with
conditional routing based on ESI level and human-in-the-loop for emergencies.
"""

from __future__ import annotations

import importlib
import logging
from pathlib import Path

import yaml
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

# ---------------------------------------------------------------------------
# Imports via importlib (hyphenated filenames)
# ---------------------------------------------------------------------------
_state_mod = importlib.import_module("src.graph.medagent-state-define")
_routing_mod = importlib.import_module("src.graph.medagent-routing-logic")
_triage_mod = importlib.import_module("src.agents.medagent-triage-agent")
_literature_mod = importlib.import_module("src.agents.medagent-literature-agent")
_reasoning_mod = importlib.import_module("src.agents.medagent-reasoning-agent")
_report_mod = importlib.import_module("src.agents.medagent-report-agent")

ClinicalState = _state_mod.ClinicalState
route_by_esi = _routing_mod.route_by_esi
route_by_complexity = _routing_mod.route_by_complexity

triage_agent = _triage_mod.run
literature_agent = _literature_mod.run
reasoning_agent = _reasoning_mod.run
report_agent = _report_mod.run

logger = logging.getLogger("medagent.graph")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
_CFG_PATH = Path(__file__).resolve().parents[2] / "configs" / "medagent-agents-config.yaml"


def _load_config() -> dict:
    if _CFG_PATH.exists():
        with open(_CFG_PATH) as f:
            return yaml.safe_load(f)
    return {}


CONFIG = _load_config()


# ---------------------------------------------------------------------------
# Human-in-the-loop node
# ---------------------------------------------------------------------------
def human_review_node(state: dict) -> dict:
    """Placeholder node for human approval on emergency cases.

    LangGraph pauses execution here via ``interrupt_before``. When the
    clinician approves, execution resumes and this node simply passes
    the state through with ``human_approved = True``.
    """
    logger.info("Human review checkpoint — ESI %s", state.get("esi_level"))
    return {"human_approved": True}


# ---------------------------------------------------------------------------
# Build the graph
# ---------------------------------------------------------------------------
def build_graph(*, checkpointer: MemorySaver | None = None) -> StateGraph:
    """Construct and compile the clinical workflow graph.

    Parameters
    ----------
    checkpointer : MemorySaver | None
        If provided, enables human-in-the-loop interrupts and state
        persistence. Pass ``None`` for stateless (testing) usage.

    Returns
    -------
    Compiled LangGraph application.
    """
    workflow = StateGraph(ClinicalState)

    # ── Nodes ──────────────────────────────────────────────────────────
    workflow.add_node("triage", triage_agent)
    workflow.add_node("human_review", human_review_node)
    workflow.add_node("literature", literature_agent)
    workflow.add_node("reasoning", reasoning_agent)
    workflow.add_node("report", report_agent)

    # ── Edges ──────────────────────────────────────────────────────────
    workflow.set_entry_point("triage")

    # After triage: ESI 1-2 → human review, ESI 3-5 → literature
    workflow.add_conditional_edges(
        "triage",
        route_by_esi,
        {
            "emergency": "human_review",
            "standard": "literature",
        },
    )

    workflow.add_edge("human_review", "literature")

    # After literature: ESI 1-3 → full reasoning, ESI 4-5 → straight to report
    workflow.add_conditional_edges(
        "literature",
        route_by_complexity,
        {
            "full_reasoning": "reasoning",
            "simple_report": "report",
        },
    )

    workflow.add_edge("reasoning", "report")
    workflow.add_edge("report", END)

    # ── Compile ────────────────────────────────────────────────────────
    compile_kwargs: dict = {}
    if checkpointer is not None:
        compile_kwargs["checkpointer"] = checkpointer
        compile_kwargs["interrupt_before"] = ["human_review"]

    app = workflow.compile(**compile_kwargs)
    logger.info("Clinical graph compiled (checkpointer=%s)", checkpointer is not None)
    return app


# ---------------------------------------------------------------------------
# Convenience: default app with in-memory checkpointer
# ---------------------------------------------------------------------------
def get_default_app():
    """Return a compiled graph with in-memory checkpointer for the demo."""
    return build_graph(checkpointer=MemorySaver())
