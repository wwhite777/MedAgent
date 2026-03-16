"""
Reasoning agent — generates differential diagnosis using chain-of-thought.

Combines triage data, literature evidence, drug interactions, and lab
interpretations to produce a ranked differential diagnosis with recommended
workup.
"""

from __future__ import annotations

import logging
import os

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

logger = logging.getLogger("medagent.agents.reasoning")


# ---------------------------------------------------------------------------
# Structured output schema
# ---------------------------------------------------------------------------
class DiagnosisItem(BaseModel):
    diagnosis: str
    probability: float = Field(ge=0.0, le=1.0)
    reasoning: str
    supporting_evidence: list[str] = Field(default_factory=list)


class ReasoningOutput(BaseModel):
    """Differential diagnosis and recommended workup."""

    differential_diagnosis: list[DiagnosisItem] = Field(
        description="Ranked differential diagnoses with probabilities summing to ~1.0"
    )
    recommended_workup: list[str] = Field(
        default_factory=list,
        description="Additional labs, imaging, or tests to consider",
    )


SYSTEM_PROMPT = (
    "You are a clinical reasoning specialist. Given triage data, literature evidence, "
    "and clinical context, produce a ranked differential diagnosis.\n\n"
    "For each differential:\n"
    "- State the diagnosis\n"
    "- Assign a probability (0.0–1.0, must sum to ~1.0)\n"
    "- Explain your reasoning with supporting evidence\n"
    "- Reference relevant literature findings\n\n"
    "Also recommend additional workup (labs, imaging) if needed."
)


# ---------------------------------------------------------------------------
# Agent function
# ---------------------------------------------------------------------------
def _get_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=os.getenv("MEDAGENT_MODEL", "gpt-4o-mini"),
        temperature=0.2,
        api_key=os.getenv("OPENAI_API_KEY"),
    )


def _format_clinical_context(state: dict) -> str:
    """Build a clinical context string from state for the reasoning prompt."""
    parts = []

    parts.append(f"Chief Complaint: {state.get('chief_complaint', 'N/A')}")
    parts.append(f"ESI Level: {state.get('esi_level', 'N/A')}")
    parts.append(f"Triage Reasoning: {state.get('triage_reasoning', 'N/A')}")

    vitals = state.get("vitals", {})
    if vitals:
        parts.append(f"Vital Signs: {vitals}")

    meds = state.get("medications", [])
    if meds:
        parts.append(f"Current Medications: {', '.join(meds)}")

    labs = state.get("lab_results", [])
    if labs:
        lab_strs = [f"{l.get('test_name', '?')}: {l.get('value', '?')} {l.get('unit', '')}" for l in labs]
        parts.append(f"Lab Results: {'; '.join(lab_strs)}")

    lit = state.get("literature_summary", "")
    if lit:
        parts.append(f"Literature Evidence:\n{lit}")

    interactions = state.get("drug_interactions", [])
    if interactions:
        parts.append(f"Drug Interactions: {interactions}")

    lab_interps = state.get("lab_interpretations", [])
    if lab_interps:
        parts.append(f"Lab Interpretations: {lab_interps}")

    return "\n\n".join(parts)


async def run(state: dict) -> dict:
    """Generate differential diagnosis from accumulated clinical data."""
    context = _format_clinical_context(state)

    llm = _get_llm().with_structured_output(ReasoningOutput)

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(
            content=f"Analyze this clinical case and provide a differential diagnosis:\n\n{context}"
        ),
    ]

    logger.info("Running reasoning agent — ESI %s", state.get("esi_level"))
    result: ReasoningOutput = await llm.ainvoke(messages)

    return {
        "differential_diagnosis": [d.model_dump() for d in result.differential_diagnosis],
        "recommended_workup": result.recommended_workup,
    }
