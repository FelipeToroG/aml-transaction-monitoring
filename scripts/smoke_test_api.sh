#!/usr/bin/env bash
# Smoke-test the AML scoring API end-to-end.
# Verifies that /health responds, /score accepts the canonical request
# shape, and /alerts is queryable. Returns non-zero on the first
# failure with a clear log line identifying the failing step.

set -euo pipefail

API_URL="${API_URL:-http://localhost:8000}"

echo "Smoke testing ${API_URL}"

# ----- /health ---------------------------------------------------------
echo ""
echo "1. GET /health"
HEALTH=$(curl -sS -w "\n%{http_code}" "${API_URL}/health")
STATUS_CODE=$(echo "${HEALTH}" | tail -n 1)
BODY=$(echo "${HEALTH}" | sed '$d')
if [[ "${STATUS_CODE}" != "200" ]]; then
    echo "  FAIL: expected 200, got ${STATUS_CODE}"
    echo "  ${BODY}"
    exit 1
fi
echo "  OK: ${BODY}"

# ----- /score ----------------------------------------------------------
echo ""
echo "2. POST /score"
SCORE_RESPONSE=$(curl -sS -w "\n%{http_code}" -X POST \
    -H "Content-Type: application/json" \
    -d '{
        "transaction_id": "smoke_001",
        "timestamp": "2026-05-01T14:23:11Z",
        "source_bank": 11,
        "source_account": "8000A1B2",
        "dest_bank": 22,
        "dest_account": "8000C3D4",
        "amount_received": 9850.00,
        "receiving_currency": "USD",
        "amount_paid": 9850.00,
        "payment_currency": "USD",
        "payment_format": "Wire"
    }' \
    "${API_URL}/score")
SCORE_CODE=$(echo "${SCORE_RESPONSE}" | tail -n 1)
SCORE_BODY=$(echo "${SCORE_RESPONSE}" | sed '$d')
if [[ "${SCORE_CODE}" != "200" ]]; then
    echo "  FAIL: expected 200, got ${SCORE_CODE}"
    echo "  ${SCORE_BODY}"
    exit 1
fi
echo "  OK: ${SCORE_BODY}"

# ----- /alerts ---------------------------------------------------------
echo ""
echo "3. GET /alerts?limit=5"
ALERTS=$(curl -sS -w "\n%{http_code}" "${API_URL}/alerts?limit=5")
ALERTS_CODE=$(echo "${ALERTS}" | tail -n 1)
ALERTS_BODY=$(echo "${ALERTS}" | sed '$d')
if [[ "${ALERTS_CODE}" != "200" ]]; then
    echo "  FAIL: expected 200, got ${ALERTS_CODE}"
    echo "  ${ALERTS_BODY}"
    exit 1
fi
echo "  OK: ${ALERTS_BODY}"

echo ""
echo "Smoke test passed."
