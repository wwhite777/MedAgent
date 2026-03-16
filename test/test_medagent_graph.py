"""
Tests for the MedAgent clinical workflow graph.

Uses stub agent functions to test graph structure, routing logic, and
state propagation without requiring API keys.
"""

from __future__ import annotations

import asyncio
import importlib
from unittest.mock import patch

import pytest

_state_mod = importlib.import_module("src.graph.medagent-state-define")
_routing_mod = importlib.import_module("src.graph.medagent-routing-logic")

ClinicalState = _state_mod.ClinicalState
route_by_esi = _routing_mod.route_by_esi
route_by_complexity = _routing_mod.route_by_complexity


class TestRouting:
    """Test conditional edge routing functions."""

    def test_route_emergency_esi1(self):
        assert route_by_esi({"esi_level": 1}) == "emergency"

    def test_route_emergency_esi2(self):
        assert route_by_esi({"esi_level": 2}) == "emergency"

    def test_route_standard_esi3(self):
        assert route_by_esi({"esi_level": 3}) == "standard"

    def test_route_standard_esi4(self):
        assert route_by_esi({"esi_level": 4}) == "standard"

    def test_route_standard_esi5(self):
        assert route_by_esi({"esi_level": 5}) == "standard"

    def test_route_default_no_esi(self):
        assert route_by_esi({}) == "standard"

    def test_complexity_full_esi1(self):
        assert route_by_complexity({"esi_level": 1}) == "full_reasoning"

    def test_complexity_full_esi3(self):
        assert route_by_complexity({"esi_level": 3}) == "full_reasoning"

    def test_complexity_simple_esi4(self):
        assert route_by_complexity({"esi_level": 4}) == "simple_report"

    def test_complexity_simple_esi5(self):
        assert route_by_complexity({"esi_level": 5}) == "simple_report"


# Stub async functions that LangGraph can inspect
async def _stub_triage(state: dict) -> dict:
    return {"esi_level": 3, "chief_complaint": "test", "triage_reasoning": "test"}


async def _stub_literature(state: dict) -> dict:
    return {"literature_summary": "test", "literature_results": []}


async def _stub_reasoning(state: dict) -> dict:
    return {"differential_diagnosis": [], "recommended_workup": []}


async def _stub_report(state: dict) -> dict:
    return {"soap_note": "test note"}


class TestGraphBuild:
    """Test that the graph compiles without errors."""

    def test_build_graph_no_checkpointer(self):
        _graph_mod = importlib.import_module("src.graph.medagent-clinical-graph")

        with patch.object(_graph_mod, "triage_agent", _stub_triage), \
             patch.object(_graph_mod, "literature_agent", _stub_literature), \
             patch.object(_graph_mod, "reasoning_agent", _stub_reasoning), \
             patch.object(_graph_mod, "report_agent", _stub_report):
            app = _graph_mod.build_graph(checkpointer=None)
            assert app is not None

    def test_build_graph_with_checkpointer(self):
        from langgraph.checkpoint.memory import MemorySaver

        _graph_mod = importlib.import_module("src.graph.medagent-clinical-graph")

        with patch.object(_graph_mod, "triage_agent", _stub_triage), \
             patch.object(_graph_mod, "literature_agent", _stub_literature), \
             patch.object(_graph_mod, "reasoning_agent", _stub_reasoning), \
             patch.object(_graph_mod, "report_agent", _stub_report):
            app = _graph_mod.build_graph(checkpointer=MemorySaver())
            assert app is not None

    def test_graph_execution_standard_case(self):
        """Test end-to-end graph execution with a standard (ESI 4) case."""
        _graph_mod = importlib.import_module("src.graph.medagent-clinical-graph")

        with patch.object(_graph_mod, "triage_agent", _stub_triage), \
             patch.object(_graph_mod, "literature_agent", _stub_literature), \
             patch.object(_graph_mod, "reasoning_agent", _stub_reasoning), \
             patch.object(_graph_mod, "report_agent", _stub_report):
            app = _graph_mod.build_graph(checkpointer=None)
            result = asyncio.get_event_loop().run_until_complete(
                app.ainvoke({"patient_input": "Test patient with headache"})
            )
            assert result.get("esi_level") == 3
            assert result.get("soap_note") == "test note"
