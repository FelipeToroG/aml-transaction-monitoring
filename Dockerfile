# Multi-stage Dockerfile for the AML scoring API.
# Stage 1 (builder) installs dependencies into a temporary venv that is
# copied wholesale into the runtime stage. This keeps the runtime image
# free of pip, build tools, and source-distribution artifacts that would
# expand the attack surface and the image size.

# =============================================================================
# Stage 1 — builder
# =============================================================================
FROM python:3.11-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# System packages required to compile the ML dependencies. lightgbm and
# xgboost ship pre-built wheels for linux/amd64 and linux/arm64, so the
# build essentials are only needed if pip falls back to a source build.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Build the virtual environment in a known location so the runtime stage
# can copy it as a single layer.
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /build
COPY requirements-api.txt .
RUN pip install --upgrade pip \
    && pip install -r requirements-api.txt

# =============================================================================
# Stage 2 — runtime
# =============================================================================
FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONPATH=/app

# Runtime needs only libgomp for the boosting libraries. No build tools,
# no compilers, no pip cache.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 10001 aml \
    && useradd --uid 10001 --gid aml --shell /bin/false --create-home aml

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
COPY src/ ./src/
COPY configs/ ./configs/
COPY models/ ./models/

# Drop privileges. The runtime user owns nothing it does not need to.
RUN chown -R aml:aml /app
USER aml

EXPOSE 8000

# The healthcheck calls the same /health endpoint Kubernetes / ECS would
# use for liveness probes. Failure here marks the container unhealthy
# and triggers a restart.
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import httpx, sys; r = httpx.get('http://localhost:8000/health', timeout=4); sys.exit(0 if r.status_code == 200 else 1)" \
    || exit 1

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
