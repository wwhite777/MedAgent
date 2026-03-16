"""
MCP server bridging MedAgent to the MedLlama RAG API (Project 1).

Exposes two tools:
- ``search_literature``: retrieves relevant documents from the vector DB
- ``ask_medllama``: sends a query to the fine-tuned MedLlama chat endpoint

Connects to MedLlama's FastAPI at MEDLLAMA_API_URL (default localhost:8090).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os

import httpx
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

logger = logging.getLogger("medagent.mcp.medllama_rag")

MEDLLAMA_URL = os.getenv("MEDLLAMA_API_URL", "http://localhost:8090")

server = Server("medllama-rag-bridge")


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------
@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="search_literature",
            description=(
                "Search the MedLlama medical knowledge base for relevant literature. "
                "Returns document snippets with relevance scores."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Medical search query"},
                    "top_k": {"type": "integer", "description": "Number of results", "default": 5},
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="ask_medllama",
            description=(
                "Ask the MedLlama fine-tuned model a medical question. "
                "Optionally uses RAG to ground the response in retrieved literature."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "question": {"type": "string", "description": "Medical question"},
                    "use_rag": {"type": "boolean", "description": "Enable RAG retrieval", "default": True},
                },
                "required": ["question"],
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    async with httpx.AsyncClient(timeout=60.0) as client:
        if name == "search_literature":
            return await _search_literature(client, arguments)
        elif name == "ask_medllama":
            return await _ask_medllama(client, arguments)
        else:
            return [TextContent(type="text", text=f"Unknown tool: {name}")]


async def _search_literature(client: httpx.AsyncClient, args: dict) -> list[TextContent]:
    """Call MedLlama /retrieve endpoint."""
    query = args.get("query", "")
    top_k = args.get("top_k", 5)

    try:
        resp = await client.post(
            f"{MEDLLAMA_URL}/retrieve",
            json={"query": query, "top_k": top_k},
        )
        resp.raise_for_status()
        data = resp.json()
        return [TextContent(type="text", text=json.dumps(data, indent=2))]
    except httpx.HTTPError as exc:
        logger.warning("MedLlama /retrieve failed: %s", exc)
        return [TextContent(type="text", text=json.dumps({"error": str(exc), "fallback": True}))]


async def _ask_medllama(client: httpx.AsyncClient, args: dict) -> list[TextContent]:
    """Call MedLlama /chat endpoint."""
    question = args.get("question", "")
    use_rag = args.get("use_rag", True)

    try:
        resp = await client.post(
            f"{MEDLLAMA_URL}/chat",
            json={"query": question, "use_rag": use_rag},
        )
        resp.raise_for_status()
        data = resp.json()
        return [TextContent(type="text", text=json.dumps(data, indent=2))]
    except httpx.HTTPError as exc:
        logger.warning("MedLlama /chat failed: %s", exc)
        return [TextContent(type="text", text=json.dumps({"error": str(exc), "fallback": True}))]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
