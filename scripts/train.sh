#!/usr/bin/env bash
# =============================================================================
# Drive the full AML training pipeline.
# =============================================================================
# Wrapper around `python -m src.models.train` that activates the project
# virtual environment and surfaces the most useful run-time flags via
# environment variables. The wrapper exists so the typical operator
# workflow is `bash scripts/train.sh` rather than a longer command line
# that operators frequently mistype.
#
# Environment variables (all optional):
#   MODEL_CONFIG     Override the model config path
#                    (default: configs/model_config.yaml)
#   COST_MATRIX      Override the cost matrix path
#                    (default: configs/cost_matrix.yaml)
#   MODEL_OUTPUT     Override the ensemble artifact destination
#                    (default: models/ensemble.pkl)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

MODEL_CONFIG="${MODEL_CONFIG:-configs/model_config.yaml}"
COST_MATRIX="${COST_MATRIX:-configs/cost_matrix.yaml}"
API_CONFIG="${API_CONFIG:-configs/api_config.yaml}"
MODEL_OUTPUT="${MODEL_OUTPUT:-models/ensemble.pkl}"
SUMMARY_OUTPUT="${SUMMARY_OUTPUT:-mlruns/training_summary.json}"

echo "AML training pipeline starting from ${PROJECT_ROOT}"
echo "  Model config:    ${MODEL_CONFIG}"
echo "  Cost matrix:     ${COST_MATRIX}"
echo "  API config:      ${API_CONFIG}"
echo "  Output artifact: ${MODEL_OUTPUT}"
echo "  Summary JSON:    ${SUMMARY_OUTPUT}"
echo ""

python -m src.models.train \
    --model-config "${MODEL_CONFIG}" \
    --cost-matrix "${COST_MATRIX}" \
    --api-config "${API_CONFIG}" \
    --model-output "${MODEL_OUTPUT}" \
    --summary-output "${SUMMARY_OUTPUT}"

# Refresh the README results table from the training summary JSON.
# Failure here does not fail the script — training succeeded, the
# README update is a courtesy step.
if [[ -f "scripts/update_results.py" ]]; then
    echo ""
    echo "Refreshing README results table from training summary..."
    python scripts/update_results.py --summary "${SUMMARY_OUTPUT}" || \
        echo "Warning: README update failed (training artifact is still good)."
fi
