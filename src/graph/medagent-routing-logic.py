"""
Conditional edge functions for the MedAgent clinical workflow graph.

These functions inspect ClinicalState and return the name of the next node
to execute. Used by LangGraph's ``add_conditional_edges``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import importlib

    _state_mod = importlib.import_module("src.graph.medagent-state-define")
    ClinicalState = _state_mod.ClinicalState


def route_by_esi(state: dict) -> str:
    """Route after triage based on ESI level.

    - ESI 1-2 (emergent): require human review before proceeding
    - ESI 3-5 (urgent/routine): go straight to literature search
    """
    esi = state.get("esi_level", 5)
    if esi <= 2:
        return "emergency"
    return "standard"


def route_by_complexity(state: dict) -> str:
    """Route after literature based on case complexity.

    - ESI 1-3: full reasoning with differential diagnosis
    - ESI 4-5: skip deep reasoning, go straight to report
    """
    esi = state.get("esi_level", 5)
    if esi <= 3:
        return "full_reasoning"
    return "simple_report"
