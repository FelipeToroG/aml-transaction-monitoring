#!/usr/bin/env bash
# =============================================================================
# Drive the full AML training pipeline.
# =============================================================================
# Wrapper around `python -m src.models.train` that calls the project venv
# interpreter directly and surfaces the most useful run-time flags via
# environment variables. The wrapper exists so the typical operator
# workflow is `bash scripts/train.sh` rather than a longer command line
# that operators frequently mistype.
#
# Environment variables (all optional):
#   MODEL_CONFIG     Override the model config path
#                    (default: configs/model_config.yaml)
#   COST_MATRIX      Override the cost matrix path
#                    (default: configs/cost_matrix.yaml)
#   API_CONFIG       Override the API config path
#                    (default: configs/api_config.yaml)
#   MODEL_OUTPUT     Override the ensemble artifact destination
#                    (default: models/ensemble.pkl)
#   SUMMARY_OUTPUT   Override the training summary JSON destination
#                    (default: mlruns/training_summary.json)
#   SAMPLE_FRACTION  Train on a contiguous early time slice of this
#                    fraction of rows (e.g. 0.02) for a fast smoke run.
#                    Unset trains on the full dataset.
#   PYTHON           Override the interpreter
#                    (default: .venv/bin/python)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

# The venv is not auto-activated in non-interactive shells (cron, CI, a fresh
# terminal), so resolve the interpreter explicitly rather than trusting that a
# bare `python` is on PATH. Override with PYTHON=... for a non-default venv.
PYTHON="${PYTHON:-.venv/bin/python}"
if [[ ! -x "${PYTHON}" ]]; then
    echo "Error: ${PYTHON} not found. Create the venv (uv venv) or set PYTHON=..." >&2
    exit 1
fi

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

TRAIN_ARGS=(
    --model-config "${MODEL_CONFIG}"
    --cost-matrix "${COST_MATRIX}"
    --api-config "${API_CONFIG}"
    --model-output "${MODEL_OUTPUT}"
    --summary-output "${SUMMARY_OUTPUT}"
)
# Only forward --sample-fraction when set so the full-data default path stays
# free of the flag (train.py treats an absent flag as "train on everything").
if [[ -n "${SAMPLE_FRACTION:-}" ]]; then
    TRAIN_ARGS+=(--sample-fraction "${SAMPLE_FRACTION}")
fi

"${PYTHON}" -m src.models.train "${TRAIN_ARGS[@]}"

# Refresh the README results table from the training summary JSON.
# Failure here does not fail the script — training succeeded, the
# README update is a courtesy step.
if [[ -f "scripts/update_results.py" ]]; then
    echo ""
    echo "Refreshing README results table from training summary..."
    "${PYTHON}" scripts/update_results.py --summary "${SUMMARY_OUTPUT}" || \
        echo "Warning: README update failed (training artifact is still good)."
fi
