"""
LangChain tool wrappers for MedAgent MCP servers.

These tools can be bound to LLMs via LangGraph's tool-calling interface.
Each tool starts its corresponding MCP server as a subprocess, invokes
the tool, and returns the result.
"""

from __future__ import annotations

import asyncio
import json
import logging
import subprocess
import sys
from pathlib import Path

import httpx
from langchain_core.tools import tool

logger = logging.getLogger("medagent.tools")

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------
# Direct HTTP wrappers (simpler than spawning MCP subprocesses for each call)
# ---------------------------------------------------------------------------

@tool
async def search_literature(query: str, top_k: int = 5) -> str:
    """Search the MedLlama medical knowledge base for relevant literature.

    Args:
        query: Medical search query
        top_k: Number of results to return
    """
    import os

    url = os.getenv("MEDLLAMA_API_URL", "http://localhost:8090")
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(f"{url}/retrieve", json={"query": query, "top_k": top_k})
            resp.raise_for_status()
            return json.dumps(resp.json(), indent=2)
    except httpx.HTTPError as exc:
        logger.warning("MedLlama /retrieve unavailable: %s — returning empty results", exc)
        return json.dumps({"sources": [], "error": str(exc), "fallback": True})


@tool
async def ask_medllama(question: str, use_rag: bool = True) -> str:
    """Ask the MedLlama fine-tuned model a medical question with optional RAG.

    Args:
        question: Medical question to ask
        use_rag: Whether to use RAG retrieval (default True)
    """
    import os

    url = os.getenv("MEDLLAMA_API_URL", "http://localhost:8090")
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(f"{url}/chat", json={"query": question, "use_rag": use_rag})
            resp.raise_for_status()
            return json.dumps(resp.json(), indent=2)
    except httpx.HTTPError as exc:
        logger.warning("MedLlama /chat unavailable: %s — returning fallback", exc)
        return json.dumps({"response": "", "error": str(exc), "fallback": True})


@tool
async def check_drug_interactions(drugs: list[str]) -> str:
    """Check for drug-drug interactions using the NLM RxNorm API.

    Args:
        drugs: List of drug names to check for interactions
    """
    if len(drugs) < 2:
        return json.dumps({"interactions": [], "note": "Need at least 2 drugs to check interactions"})

    rxnorm_base = "https://rxnav.nlm.nih.gov/REST"

    async with httpx.AsyncClient(timeout=30.0) as client:
        # Resolve drug names to RxCUI
        rxcui_map = {}
        for drug in drugs:
            try:
                resp = await client.get(f"{rxnorm_base}/rxcui.json", params={"name": drug, "search": 1})
                if resp.status_code == 200:
                    ids = resp.json().get("idGroup", {}).get("rxnormId", [])
                    if ids:
                        rxcui_map[drug] = ids[0]
            except httpx.HTTPError:
                pass

        if len(rxcui_map) < 2:
            return json.dumps({"interactions": [], "resolved": rxcui_map, "note": "Could not resolve enough drugs"})

        # Query interactions
        try:
            resp = await client.get(
                f"{rxnorm_base}/interaction/list.json",
                params={"rxcuis": "+".join(rxcui_map.values())},
            )
            if resp.status_code != 200:
                return json.dumps({"interactions": [], "error": f"API returned {resp.status_code}"})

            data = resp.json()
            interactions = []
            for group in data.get("fullInteractionTypeGroup", []):
                for itype in group.get("fullInteractionType", []):
                    for pair in itype.get("interactionPair", []):
                        interactions.append({
                            "drug_pair": [
                                c.get("minConceptItem", {}).get("name", "?")
                                for c in pair.get("interactionConcept", [])
                            ],
                            "severity": pair.get("severity", "N/A"),
                            "description": pair.get("description", ""),
                        })

            return json.dumps({"interactions": interactions, "total": len(interactions)}, indent=2)
        except httpx.HTTPError as exc:
            return json.dumps({"interactions": [], "error": str(exc)})


@tool
async def interpret_labs(labs: list[dict]) -> str:
    """Interpret lab values against normal reference ranges.

    Args:
        labs: List of lab results as [{test_name, value, unit}]
    """
    ranges_path = _PROJECT_ROOT / "data" / "lab_ranges.json"
    if ranges_path.exists():
        with open(ranges_path) as f:
            lab_ranges = json.load(f)
    else:
        return json.dumps({"error": "Lab ranges database not found"})

    results = []
    for lab in labs:
        test = lab.get("test_name", "").lower().strip()
        value = lab.get("value", 0)
        unit = lab.get("unit", "")

        ref = lab_ranges.get(test)
        if ref is None:
            results.append({"test_name": test, "value": value, "unit": unit, "flag": "unknown", "clinical_note": f"No reference range for '{test}'"})
            continue

        low, high = ref.get("low", float("-inf")), ref.get("high", float("inf"))
        crit_low, crit_high = ref.get("critical_low", float("-inf")), ref.get("critical_high", float("inf"))

        if value <= crit_low or value >= crit_high:
            flag = "critical"
        elif value < low:
            flag = "low"
        elif value > high:
            flag = "high"
        else:
            flag = "normal"

        note_key = f"{flag}_note" if flag != "normal" else None
        note = ref.get(note_key, f"{test}: {flag}") if note_key else f"{test} within normal range ({low}-{high} {ref.get('unit', unit)})"

        results.append({"test_name": test, "value": value, "unit": unit, "flag": flag, "reference_range": f"{low}-{high} {ref.get('unit', unit)}", "clinical_note": note})

    return json.dumps(results, indent=2)


# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------
ALL_TOOLS = [search_literature, ask_medllama, check_drug_interactions, interpret_labs]
