"""
Report agent — generates a structured SOAP note from the clinical workflow.

Aggregates triage, literature, reasoning, and interaction data into a
clinical report following the SOAP (Subjective, Objective, Assessment, Plan)
format.
"""

from __future__ import annotations

import logging
import os

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

logger = logging.getLogger("medagent.agents.report")


# ---------------------------------------------------------------------------
# Structured output schema
# ---------------------------------------------------------------------------
class ReportOutput(BaseModel):
    """SOAP note and summary."""

    soap_note: str = Field(description="Complete SOAP note in markdown format")


SYSTEM_PROMPT = (
    "You are a clinical documentation specialist. Generate a structured SOAP note "
    "from the clinical workflow results.\n\n"
    "Format:\n"
    "**Subjective**: Chief complaint, history of present illness\n"
    "**Objective**: Vital signs, lab results, relevant findings\n"
    "**Assessment**: Differential diagnosis with probabilities, drug interactions, lab flags\n"
    "**Plan**: Recommended workup, treatment considerations, follow-up\n\n"
    "Include inline citations to retrieved literature where applicable."
)


# ---------------------------------------------------------------------------
# Agent function
# ---------------------------------------------------------------------------
def _get_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=os.getenv("MEDAGENT_MODEL", "gpt-4o-mini"),
        temperature=0.1,
        api_key=os.getenv("OPENAI_API_KEY"),
    )


def _format_full_context(state: dict) -> str:
    """Build complete clinical context for SOAP note generation."""
    parts = []

    parts.append(f"Patient Vignette:\n{state.get('patient_input', 'N/A')}")
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

    ddx = state.get("differential_diagnosis", [])
    if ddx:
        ddx_strs = [f"- {d.get('diagnosis', '?')} ({d.get('probability', 0):.0%}): {d.get('reasoning', '')}" for d in ddx]
        parts.append(f"Differential Diagnosis:\n" + "\n".join(ddx_strs))

    interactions = state.get("drug_interactions", [])
    if interactions:
        parts.append(f"Drug Interactions: {interactions}")

    lab_interps = state.get("lab_interpretations", [])
    if lab_interps:
        parts.append(f"Lab Interpretations: {lab_interps}")

    workup = state.get("recommended_workup", [])
    if workup:
        parts.append(f"Recommended Workup: {', '.join(workup)}")

    return "\n\n".join(parts)


async def run(state: dict) -> dict:
    """Generate SOAP note from accumulated clinical data."""
    context = _format_full_context(state)

    llm = _get_llm().with_structured_output(ReportOutput)

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"Generate a SOAP note for this case:\n\n{context}"),
    ]

    logger.info("Running report agent — generating SOAP note")
    result: ReportOutput = await llm.ainvoke(messages)

    return {"soap_note": result.soap_note}
