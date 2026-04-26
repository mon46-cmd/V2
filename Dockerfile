# syntax=docker/dockerfile:1.6
# ---------------------------------------------------------------------------
# V8 Production Image -- multi-stage, non-root, slim runtime.
#
# Stages:
#   builder  -- compiles wheels in an isolated venv (build tools available)
#   runtime  -- copies only the venv and source; no build tools, no cache.
#
# The same image is used for BOTH services in docker-compose:
#   * engine    -- runs `python -m loops.runner`
#   * dashboard -- runs `python -m web_api.app`  (HTTP /health probed)
# ---------------------------------------------------------------------------

# =============================================================================
# Stage 1 -- builder
# =============================================================================
FROM python:3.12-slim AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive

# Build deps for any wheels that need compilation. Stripped in stage 2.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Isolated venv -- copied verbatim into the runtime image.
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Cache pip downloads across rebuilds when BuildKit is enabled.
COPY requirements.txt ./
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip \
 && pip install -r requirements.txt


# =============================================================================
# Stage 2 -- runtime
# =============================================================================
FROM python:3.12-slim AS runtime

# tini reaps zombies and forwards SIGTERM correctly: critical for fast,
# graceful shutdown when systemd / orchestrator sends SIGTERM.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PATH="/opt/venv/bin:$PATH" \
    TZ=UTC \
    DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        tini \
        wget \
    && rm -rf /var/lib/apt/lists/* \
    && update-ca-certificates

WORKDIR /app

# Copy the prebuilt venv and the application source.
COPY --from=builder /opt/venv /opt/venv
COPY . .

# Non-root user -- principle of least privilege.
RUN addgroup --system --gid 10001 appgroup \
 && adduser  --system --uid 10001 --ingroup appgroup --home /app appuser \
 && mkdir -p /app/data \
 && chown -R appuser:appgroup /app

USER appuser

EXPOSE 8765

# Default healthcheck targets the dashboard (HTTP /health). The engine
# service overrides this with a process-based check in docker-compose.yml.
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD wget --quiet --tries=1 --spider http://localhost:8765/health || exit 1

# tini as PID 1 ensures clean SIGTERM forwarding to the Python child.
ENTRYPOINT ["/usr/bin/tini", "--"]

# Default to engine; docker-compose overrides this for the dashboard service.
CMD ["python", "-m", "loops.runner"]
