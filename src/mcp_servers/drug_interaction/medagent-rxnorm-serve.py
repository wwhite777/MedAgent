"""
MCP server for drug-drug interaction checking via the NLM RxNorm REST API.

Exposes a single tool ``check_drug_interactions`` that resolves drug names to
RxCUI codes and queries the interaction API.

API docs: https://lhncbc.nlm.nih.gov/RxNav/APIs/InteractionAPIs.html
"""

from __future__ import annotations

import asyncio
import json
import logging

import httpx
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

logger = logging.getLogger("medagent.mcp.drug_interaction")

RXNORM_BASE = "https://rxnav.nlm.nih.gov/REST"

server = Server("drug-interaction-server")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
async def _resolve_rxcui(client: httpx.AsyncClient, drug_name: str) -> str | None:
    """Resolve a drug name to an RxCUI identifier."""
    resp = await client.get(
        f"{RXNORM_BASE}/rxcui.json",
        params={"name": drug_name, "search": 1},
    )
    if resp.status_code != 200:
        return None
    data = resp.json()
    ids = data.get("idGroup", {}).get("rxnormId", [])
    return ids[0] if ids else None


async def _get_interactions(client: httpx.AsyncClient, rxcuis: list[str]) -> list[dict]:
    """Query RxNorm interaction API for a list of RxCUI codes."""
    if len(rxcuis) < 2:
        return []

    resp = await client.get(
        f"{RXNORM_BASE}/interaction/list.json",
        params={"rxcuis": "+".join(rxcuis)},
    )
    if resp.status_code != 200:
        return []

    data = resp.json()
    interactions = []

    for group in data.get("fullInteractionTypeGroup", []):
        for itype in group.get("fullInteractionType", []):
            for pair in itype.get("interactionPair", []):
                desc = pair.get("description", "")
                severity = pair.get("severity", "N/A")
                drugs = [
                    c.get("minConceptItem", {}).get("name", "?")
                    for c in pair.get("interactionConcept", [])
                ]
                interactions.append({
                    "drug_pair": drugs,
                    "severity": severity,
                    "description": desc,
                })

    return interactions


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------
@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="check_drug_interactions",
            description=(
                "Check for drug-drug interactions using the NLM RxNorm API. "
                "Provide a list of drug names; returns known interactions with severity."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "drugs": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of drug names to check for interactions",
                    },
                },
                "required": ["drugs"],
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    if name != "check_drug_interactions":
        return [TextContent(type="text", text=f"Unknown tool: {name}")]

    drugs = arguments.get("drugs", [])
    if len(drugs) < 2:
        return [TextContent(type="text", text=json.dumps({"interactions": [], "note": "Need at least 2 drugs"}))]

    async with httpx.AsyncClient(timeout=30.0) as client:
        # Resolve drug names to RxCUI codes
        rxcui_map = {}
        for drug in drugs:
            rxcui = await _resolve_rxcui(client, drug)
            if rxcui:
                rxcui_map[drug] = rxcui
            else:
                logger.warning("Could not resolve RxCUI for: %s", drug)

        if len(rxcui_map) < 2:
            return [TextContent(
                type="text",
                text=json.dumps({
                    "interactions": [],
                    "resolved": rxcui_map,
                    "note": "Could not resolve enough drugs to check interactions",
                }),
            )]

        # Query interaction API
        interactions = await _get_interactions(client, list(rxcui_map.values()))

        result = {
            "resolved_drugs": rxcui_map,
            "interactions": interactions,
            "total_interactions": len(interactions),
        }

    return [TextContent(type="text", text=json.dumps(result, indent=2))]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
