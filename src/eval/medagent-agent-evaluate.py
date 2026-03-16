"""
MedAgent evaluation framework — runs test cases through the clinical graph
and measures key metrics.

Metrics:
- Triage accuracy (ESI level vs. ground truth)
- Differential diagnosis top-3 accuracy
- Report completeness (SOAP section presence)
- End-to-end latency
"""

from __future__ import annotations

import asyncio
import importlib
import json
import logging
import time
from pathlib import Path

logger = logging.getLogger("medagent.eval")

_graph_mod = importlib.import_module("src.graph.medagent-clinical-graph")
build_graph = _graph_mod.build_graph

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_TEST_CASES_PATH = _PROJECT_ROOT / "data" / "test_cases.json"


def load_test_cases() -> list[dict]:
    with open(_TEST_CASES_PATH) as f:
        return json.load(f)


async def run_single_case(app, case: dict) -> dict:
    """Run a single test case and collect results."""
    start = time.time()
    result_state = {}
    error = None

    try:
        final_state = await app.ainvoke({"patient_input": case["vignette"]})
        result_state = dict(final_state)
    except Exception as exc:
        error = str(exc)
        logger.error("Case %s failed: %s", case["id"], exc)

    elapsed = time.time() - start

    return {
        "case_id": case["id"],
        "expected_esi": case.get("esi_expected"),
        "predicted_esi": result_state.get("esi_level"),
        "expected_diagnosis": case.get("expected_diagnosis", ""),
        "predicted_diagnoses": [
            d.get("diagnosis", "") for d in result_state.get("differential_diagnosis", [])
        ],
        "has_soap_note": bool(result_state.get("soap_note")),
        "soap_sections": _check_soap_sections(result_state.get("soap_note", "")),
        "latency_seconds": round(elapsed, 2),
        "error": error,
    }


def _check_soap_sections(note: str) -> dict:
    """Check which SOAP sections are present in the note."""
    note_lower = note.lower()
    return {
        "subjective": "subjective" in note_lower or "chief complaint" in note_lower,
        "objective": "objective" in note_lower or "vital" in note_lower,
        "assessment": "assessment" in note_lower or "differential" in note_lower,
        "plan": "plan" in note_lower or "recommend" in note_lower,
    }


def compute_metrics(results: list[dict]) -> dict:
    """Compute aggregate metrics from evaluation results."""
    total = len(results)
    if total == 0:
        return {}

    # Triage accuracy
    triage_correct = sum(
        1 for r in results
        if r["expected_esi"] is not None and r["predicted_esi"] == r["expected_esi"]
    )
    triage_total = sum(1 for r in results if r["expected_esi"] is not None)

    # Differential diagnosis top-3 accuracy
    dx_correct = 0
    dx_total = 0
    for r in results:
        if r["expected_diagnosis"]:
            dx_total += 1
            expected_lower = r["expected_diagnosis"].lower()
            top3 = [d.lower() for d in r["predicted_diagnoses"][:3]]
            if any(expected_lower in d or d in expected_lower for d in top3):
                dx_correct += 1

    # Report completeness
    soap_complete = sum(
        1 for r in results
        if r["has_soap_note"] and all(r["soap_sections"].values())
    )

    # Latency
    latencies = [r["latency_seconds"] for r in results if r["error"] is None]
    errors = sum(1 for r in results if r["error"] is not None)

    return {
        "total_cases": total,
        "errors": errors,
        "triage_accuracy": round(triage_correct / triage_total, 3) if triage_total > 0 else 0,
        "differential_top3_accuracy": round(dx_correct / dx_total, 3) if dx_total > 0 else 0,
        "report_completeness": round(soap_complete / total, 3),
        "avg_latency_seconds": round(sum(latencies) / len(latencies), 2) if latencies else 0,
        "max_latency_seconds": round(max(latencies), 2) if latencies else 0,
        "min_latency_seconds": round(min(latencies), 2) if latencies else 0,
    }


async def run_evaluation(max_cases: int | None = None) -> dict:
    """Run the full evaluation suite."""
    cases = load_test_cases()
    if max_cases:
        cases = cases[:max_cases]

    app = build_graph(checkpointer=None)
    logger.info("Running evaluation on %d cases", len(cases))

    results = []
    for i, case in enumerate(cases):
        logger.info("Evaluating case %d/%d: %s", i + 1, len(cases), case["id"])
        result = await run_single_case(app, case)
        results.append(result)

    metrics = compute_metrics(results)

    report = {
        "metrics": metrics,
        "results": results,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    # Save results
    output_path = _PROJECT_ROOT / "result" / "eval"
    output_path.mkdir(parents=True, exist_ok=True)
    report_file = output_path / f"eval_{time.strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2)

    logger.info("Evaluation complete. Metrics: %s", json.dumps(metrics, indent=2))
    logger.info("Results saved to %s", report_file)

    return report


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run MedAgent evaluation")
    parser.add_argument("--max-cases", type=int, default=None, help="Limit number of test cases")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    report = asyncio.run(run_evaluation(max_cases=args.max_cases))

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    for k, v in report["metrics"].items():
        print(f"  {k}: {v}")
