"""
Triage agent — classifies patient ESI level and extracts structured clinical data.

Uses an LLM with structured output to parse a clinical vignette into ESI level,
chief complaint, vital signs, medications, and lab results.
"""

from __future__ import annotations

import importlib
import logging
import os
from pathlib import Path

import yaml
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

logger = logging.getLogger("medagent.agents.triage")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
_CFG_PATH = Path(__file__).resolve().parents[2] / "configs" / "medagent-agents-config.yaml"


def _load_prompt() -> str:
    if _CFG_PATH.exists():
        with open(_CFG_PATH) as f:
            cfg = yaml.safe_load(f)
        return cfg.get("agents", {}).get("triage", {}).get("system_prompt", "")
    return "You are an emergency medicine triage specialist."


SYSTEM_PROMPT = _load_prompt()


# ---------------------------------------------------------------------------
# Structured output schema
# ---------------------------------------------------------------------------
class TriageOutput(BaseModel):
    """Structured triage classification result."""

    esi_level: int = Field(ge=1, le=5, description="Emergency Severity Index 1-5")
    triage_reasoning: str = Field(description="Detailed reasoning for the ESI classification")
    chief_complaint: str = Field(description="Primary chief complaint extracted from vignette")
    vitals: dict = Field(
        default_factory=dict,
        description="Vital signs: heart_rate, blood_pressure, temperature, respiratory_rate, spo2",
    )
    medications: list[str] = Field(default_factory=list, description="Current medications")
    lab_results: list[dict] = Field(
        default_factory=list, description="Lab results as [{test_name, value, unit}]"
    )


# ---------------------------------------------------------------------------
# Agent function
# ---------------------------------------------------------------------------
def _get_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=os.getenv("MEDAGENT_MODEL", "gpt-4o-mini"),
        temperature=0.0,
        api_key=os.getenv("OPENAI_API_KEY"),
    )


async def run(state: dict) -> dict:
    """Execute triage classification on the patient input.

    Reads ``patient_input`` from state, returns triage fields.
    """
    patient_input = state.get("patient_input", "")
    if not patient_input:
        return {"error": "No patient input provided for triage"}

    llm = _get_llm().with_structured_output(TriageOutput)

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"Classify the following clinical vignette:\n\n{patient_input}"),
    ]

    logger.info("Running triage agent on input (%d chars)", len(patient_input))
    result: TriageOutput = await llm.ainvoke(messages)

    return {
        "esi_level": result.esi_level,
        "triage_reasoning": result.triage_reasoning,
        "chief_complaint": result.chief_complaint,
        "vitals": result.vitals,
        "medications": result.medications,
        "lab_results": result.lab_results,
    }
