"""
Trace analyzer — analyzes LangGraph execution traces for debugging and
performance profiling.

Reads evaluation result files from result/eval/ and produces summary statistics
on tool calls, token usage, error patterns, and latency distribution.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path

logger = logging.getLogger("medagent.eval.trace")

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_EVAL_DIR = _PROJECT_ROOT / "result" / "eval"


def load_latest_eval() -> dict | None:
    """Load the most recent evaluation result file."""
    if not _EVAL_DIR.exists():
        return None
    files = sorted(_EVAL_DIR.glob("eval_*.json"), reverse=True)
    if not files:
        return None
    with open(files[0]) as f:
        return json.load(f)


def analyze_traces(report: dict) -> dict:
    """Analyze evaluation results for patterns and performance insights."""
    results = report.get("results", [])
    if not results:
        return {"error": "No results to analyze"}

    # ESI distribution
    esi_distribution = Counter(r.get("predicted_esi") for r in results)

    # Error analysis
    errors = [r for r in results if r.get("error")]
    error_types = Counter(r["error"].split(":")[0] for r in errors if r["error"])

    # Latency by ESI level
    latency_by_esi = {}
    for r in results:
        esi = r.get("expected_esi")
        if esi and r.get("latency_seconds") and not r.get("error"):
            latency_by_esi.setdefault(esi, []).append(r["latency_seconds"])

    latency_summary = {
        esi: {"avg": round(sum(lats) / len(lats), 2), "count": len(lats)}
        for esi, lats in sorted(latency_by_esi.items())
    }

    # SOAP section coverage
    soap_coverage = Counter()
    soap_total = 0
    for r in results:
        sections = r.get("soap_sections", {})
        if sections:
            soap_total += 1
            for section, present in sections.items():
                if present:
                    soap_coverage[section] += 1

    soap_rates = {
        section: round(count / soap_total, 3) if soap_total > 0 else 0
        for section, count in soap_coverage.items()
    }

    # Triage confusion matrix (expected vs predicted ESI)
    confusion = {}
    for r in results:
        exp = r.get("expected_esi")
        pred = r.get("predicted_esi")
        if exp is not None and pred is not None:
            key = f"expected_{exp}_predicted_{pred}"
            confusion[key] = confusion.get(key, 0) + 1

    return {
        "total_cases": len(results),
        "total_errors": len(errors),
        "esi_distribution": dict(esi_distribution),
        "error_types": dict(error_types),
        "latency_by_esi": latency_summary,
        "soap_section_coverage": soap_rates,
        "triage_confusion": confusion,
    }


def print_analysis(analysis: dict) -> None:
    """Pretty-print trace analysis results."""
    print("\n" + "=" * 60)
    print("TRACE ANALYSIS")
    print("=" * 60)

    print(f"\nTotal cases: {analysis['total_cases']}")
    print(f"Total errors: {analysis['total_errors']}")

    print("\nESI Distribution (predicted):")
    for esi, count in sorted(analysis["esi_distribution"].items()):
        print(f"  ESI {esi}: {count}")

    print("\nLatency by ESI Level:")
    for esi, stats in analysis["latency_by_esi"].items():
        print(f"  ESI {esi}: avg {stats['avg']}s (n={stats['count']})")

    print("\nSOAP Section Coverage:")
    for section, rate in analysis["soap_section_coverage"].items():
        print(f"  {section}: {rate:.1%}")

    if analysis["error_types"]:
        print("\nError Types:")
        for etype, count in analysis["error_types"].items():
            print(f"  {etype}: {count}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    report = load_latest_eval()
    if report is None:
        print("No evaluation results found. Run medagent-agent-evaluate.py first.")
    else:
        analysis = analyze_traces(report)
        print_analysis(analysis)
