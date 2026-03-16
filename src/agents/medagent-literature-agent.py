"""
Literature agent — searches medical knowledge base and synthesizes evidence.

Calls the MedLlama RAG MCP tool (or falls back to LLM-only) to retrieve
relevant literature for the patient's clinical scenario.
"""

from __future__ import annotations

import logging
import os

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

logger = logging.getLogger("medagent.agents.literature")


# ---------------------------------------------------------------------------
# Structured output schema
# ---------------------------------------------------------------------------
class LiteratureOutput(BaseModel):
    """Synthesized literature search results."""

    literature_results: list[dict] = Field(
        default_factory=list,
        description="Retrieved articles: [{title, snippet, relevance_score, source}]",
    )
    literature_summary: str = Field(description="Synthesized evidence summary with key findings")


SYSTEM_PROMPT = (
    "You are a medical literature research specialist. Given a clinical scenario, "
    "synthesize relevant medical evidence. Focus on recent clinical guidelines, "
    "diagnostic criteria, and treatment protocols. Provide a concise evidence summary."
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


async def run(state: dict) -> dict:
    """Search literature for the patient's clinical scenario.

    Uses chief_complaint and triage_reasoning from state to form the query.
    """
    chief = state.get("chief_complaint", "")
    reasoning = state.get("triage_reasoning", "")
    patient_input = state.get("patient_input", "")

    query = f"Chief complaint: {chief}\nClinical context: {reasoning}\nFull vignette: {patient_input}"

    # TODO: Replace with MCP tool call to medllama_rag server (Phase 2)
    # For now, use LLM-only synthesis
    llm = _get_llm().with_structured_output(LiteratureOutput)

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"Search for relevant medical literature for this case:\n\n{query}\n\n"
                "Return structured results with titles, snippets, and a synthesis."
            )
        ),
    ]

    logger.info("Running literature agent for: %s", chief)
    result: LiteratureOutput = await llm.ainvoke(messages)

    return {
        "literature_results": result.literature_results,
        "literature_summary": result.literature_summary,
    }
