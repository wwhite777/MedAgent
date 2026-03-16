"""
MCP server for lab value interpretation.

Uses a local JSON database of normal reference ranges to flag abnormal results
and provide clinical significance notes.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool

logger = logging.getLogger("medagent.mcp.lab_interpreter")

_DATA_PATH = Path(__file__).resolve().parents[3] / "data" / "lab_ranges.json"

server = Server("lab-interpreter-server")


# ---------------------------------------------------------------------------
# Reference range database
# ---------------------------------------------------------------------------
def _load_lab_ranges() -> dict:
    if _DATA_PATH.exists():
        with open(_DATA_PATH) as f:
            return json.load(f)
    logger.warning("Lab ranges file not found at %s — using empty DB", _DATA_PATH)
    return {}


LAB_RANGES: dict = _load_lab_ranges()


def _interpret_single(test_name: str, value: float, unit: str) -> dict:
    """Interpret a single lab value against reference ranges."""
    key = test_name.lower().strip()
    ref = LAB_RANGES.get(key)

    if ref is None:
        return {
            "test_name": test_name,
            "value": value,
            "unit": unit,
            "flag": "unknown",
            "clinical_note": f"No reference range available for '{test_name}'",
        }

    low = ref.get("low", float("-inf"))
    high = ref.get("high", float("inf"))
    critical_low = ref.get("critical_low", float("-inf"))
    critical_high = ref.get("critical_high", float("inf"))
    ref_unit = ref.get("unit", unit)

    if value <= critical_low or value >= critical_high:
        flag = "critical"
    elif value < low:
        flag = "low"
    elif value > high:
        flag = "high"
    else:
        flag = "normal"

    notes = {
        "critical": ref.get("critical_note", f"CRITICAL: {test_name} = {value} {unit}"),
        "high": ref.get("high_note", f"{test_name} elevated above normal range ({low}-{high} {ref_unit})"),
        "low": ref.get("low_note", f"{test_name} below normal range ({low}-{high} {ref_unit})"),
        "normal": f"{test_name} within normal range ({low}-{high} {ref_unit})",
    }

    return {
        "test_name": test_name,
        "value": value,
        "unit": unit,
        "flag": flag,
        "reference_range": f"{low}-{high} {ref_unit}",
        "clinical_note": notes[flag],
    }


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------
@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="interpret_labs",
            description=(
                "Interpret lab values against normal reference ranges. "
                "Returns flags (normal/high/low/critical) and clinical notes."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "labs": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "test_name": {"type": "string"},
                                "value": {"type": "number"},
                                "unit": {"type": "string"},
                            },
                            "required": ["test_name", "value", "unit"],
                        },
                        "description": "Lab results to interpret",
                    },
                },
                "required": ["labs"],
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    if name != "interpret_labs":
        return [TextContent(type="text", text=f"Unknown tool: {name}")]

    labs = arguments.get("labs", [])
    results = [
        _interpret_single(lab["test_name"], lab["value"], lab["unit"])
        for lab in labs
    ]

    return [TextContent(type="text", text=json.dumps(results, indent=2))]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
