# Deployment guide

## Local — Docker Compose

The fastest path from clone to running service.

```bash
git clone https://github.com/FelipeToroG/aml-transaction-monitoring.git
cd aml-transaction-monitoring

# Provide your Anthropic API key.
cp .env.example .env
# Edit .env: set ANTHROPIC_API_KEY=sk-ant-...

# Ensure the model artifact exists.
bash scripts/download_data.sh   # ~5 minutes
bash scripts/train.sh           # ~30–45 minutes on M-series Mac

# Bring up the stack.
mkdir -p data/runtime
docker-compose up --build
```

Then:

- API:    http://localhost:8000/docs
- UI:     http://localhost:8501
- Health: http://localhost:8000/health
- Smoke:  `bash scripts/smoke_test_api.sh`

## Local — virtual environment (no Docker)

```bash
python3.11 -m venv venv
source venv/bin/activate

pip install -r requirements.txt
pip install -r requirements-api.txt
pip install -r requirements-ui.txt

# Editable install so `src.*` resolves for pytest, notebooks, and scripts.
pip install -e .

cp .env.example .env  # then edit

# Same data + training prerequisites as above.
bash scripts/download_data.sh
bash scripts/train.sh

# Run API and UI in two terminals.
uvicorn src.api.main:app --reload --port 8000
streamlit run ui/app.py
```

## Cloud — Render / Fly.io

Both providers accept the multi-stage Dockerfile in this repo with no modification.

### Render

```yaml
# render.yaml (place at repo root if using Render Blueprint)
services:
  - type: web
    name: aml-api
    runtime: docker
    dockerfilePath: ./Dockerfile
    plan: standard
    healthCheckPath: /health
    envVars:
      - key: ANTHROPIC_API_KEY
        sync: false
      - key: DATABASE_URL
        sync: false
      - key: ENVIRONMENT
        value: production-render
```

For the UI, repeat with `dockerfilePath: ./Dockerfile.ui` and add `API_BASE_URL` pointing at the API service's internal URL.

### Fly.io

```bash
fly launch --dockerfile Dockerfile --name aml-api --no-deploy
fly secrets set ANTHROPIC_API_KEY=sk-ant-... DATABASE_URL=postgres://...
fly deploy
```

## Production checklist

Items the demo stack handles loosely that a production deployment must tighten:

| Concern | Demo | Production action |
|---|---|---|
| Persistence | SQLite | Postgres via `DATABASE_URL=postgresql+psycopg2://...`; no code change |
| Schema migrations | `init_db()` at startup | Replace with Alembic; run migrations before app start |
| Secrets | `.env` file | Use the platform's secrets manager (AWS SSM, GCP Secret Manager, Fly secrets) |
| Webhook | Slack URL in `.env` | Pager rotation via your incident tool; rate-limit per alert |
| LLM cost ceiling | Unbounded | Prometheus alert on `aml_llm_cost_usd_total` daily growth |
| Model artifact | Bundled in image | Pull from object storage at startup; version-pin via env var |
| Authentication | None | Front the API with a service-mesh auth proxy or API gateway |
| TLS | None | Terminate at the platform's load balancer; redirect 80→443 |

## Monitoring setup

The `/metrics` endpoint exposes 9 metrics under the `aml_*` namespace. Recommended alerts:

| Alert | Condition |
|---|---|
| `aml_alerts_created_total` rate drops > 50% | Possible upstream feed stoppage |
| `aml_llm_cost_usd_total` daily growth > $X | Budget ceiling exceeded |
| `aml_llm_refusals_total{code="schema_failure"}` rate > 1% | Prompt regression; investigate |
| `aml_webhook_deliveries_total{outcome="failed"}` rate > 5% | Webhook target unhealthy |
| `aml_drift_events_total{severity="regulator-relevant"}` > 0 | Escalate to model risk |
| API p99 latency > 200 ms | Investigate scoring path |

Langfuse traces are searchable by `alert_id`, prompt version, and refusal code. Recommended dashboards:

- Cost per day, grouped by model and prompt version
- p95 latency per model
- Refusal rate over time

## Rollback

The model artifact is versioned via `EnsembleMetadata.schema_version`. To roll back:

1. Replace `models/ensemble.pkl` with the previous artifact.
2. Restart the API. The schema-version check at load time rejects a mismatched artifact loudly rather than silently using the wrong one.
3. Verify via `GET /health` that `model_version` reports the rollback version.

## Backup and recovery

For SQLite deployments, the database file is the entire backup target. Snapshot `data/runtime/aml_alerts.db` on the cadence your retention policy requires (5-year minimum for SAR-adjacent records per `audit_retention_days`).

For Postgres, follow your platform's standard pg_dump or continuous-WAL backup configuration. The audit-log table is the highest-value table for recovery purposes; consider pointing it at a separate retention policy.
