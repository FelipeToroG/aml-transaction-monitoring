#!/usr/bin/env bash
# Minimal load test for the /score endpoint.
# Sends N requests concurrently to measure latency and throughput.
# Output mirrors a basic Apache Bench run so the numbers are
# interpretable without external tooling.
#
# For production benchmarking, replace this with Locust or k6 — this
# script is the floor: it verifies the service handles a modest
# concurrent load without crashing.

set -euo pipefail

API_URL="${API_URL:-http://localhost:8000}"
N_REQUESTS="${N_REQUESTS:-200}"
CONCURRENCY="${CONCURRENCY:-10}"

if ! command -v ab >/dev/null 2>&1; then
    echo "ERROR: Apache Bench (ab) not installed." >&2
    echo "Install with: brew install httpd  (macOS)  /  apt install apache2-utils  (Debian)" >&2
    exit 1
fi

PAYLOAD_FILE="$(mktemp)"
trap 'rm -f "${PAYLOAD_FILE}"' EXIT

cat > "${PAYLOAD_FILE}" <<'JSON'
{
    "transaction_id": "load_test_001",
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
}
JSON

echo "Load testing ${API_URL}/score"
echo "  Requests:    ${N_REQUESTS}"
echo "  Concurrency: ${CONCURRENCY}"
echo ""

ab -n "${N_REQUESTS}" \
   -c "${CONCURRENCY}" \
   -p "${PAYLOAD_FILE}" \
   -T "application/json" \
   "${API_URL}/score"
