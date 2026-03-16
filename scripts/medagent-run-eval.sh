#!/usr/bin/env bash
# Run MedAgent evaluation suite
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

echo "=== MedAgent Evaluation ==="
echo "Running from: $PROJECT_DIR"
echo ""

# Run evaluation
python -m src.eval.medagent-agent-evaluate "$@"

echo ""
echo "=== Trace Analysis ==="
python -m src.eval.medagent-trace-analyze
