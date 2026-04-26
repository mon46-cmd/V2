# syntax=docker/dockerfile:1.6
# ---------------------------------------------------------------------------
# V8 Production Image -- single-stage, system Python 3.12, no venv.
#
# Design choices (optimised for fast rebuilds on a small VPS):
#   * `python:3.12-slim` ships Python 3.12 + pip; we install straight into
#     the system site-packages -- no venv layer to copy, no PATH gymnastics.
#   * No `build-essential`. Every runtime dep (numpy, pandas, pyarrow,
#     pydantic-core, orjson, aiohttp, uvloop) ships manylinux wheels for
#     cp312, so apt downloads stay tiny and the build is mostly network.
#   * Single stage -> no inter-stage COPY of /opt/venv.
#   * BuildKit pip cache mount -> instant reinstalls when requirements.txt
#     hasn't changed.
#   * requirements.txt is COPYed before the source tree so editing app
#     code doesn't bust the dependency layer.
# ---------------------------------------------------------------------------

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_ROOT_USER_ACTION=ignore \
    TZ=UTC \
    DEBIAN_FRONTEND=noninteractive

# Minimal apt: tini for clean PID 1 signal handling, wget for the
# dashboard healthcheck, ca-certificates for HTTPS to Bybit/OpenRouter.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        ca-certificates \
        tini \
        wget \
 && rm -rf /var/lib/apt/lists/* \
 && update-ca-certificates

WORKDIR /app

# Dependency layer (cached when requirements.txt is unchanged).
COPY requirements.txt ./
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip \
 && pip install -r requirements.txt

# Application source.
COPY . .

# Non-root runtime user. Owns /app/data so the named volume mounts cleanly.
RUN groupadd --system --gid 10001 appgroup \
 && useradd  --system --uid 10001 --gid appgroup --home-dir /app --shell /usr/sbin/nologin appuser \
 && mkdir -p /app/data \
 && chown -R appuser:appgroup /app
USER appuser

EXPOSE 8765

# Default healthcheck targets the dashboard's HTTP /health endpoint.
# The engine service overrides this with a process check in compose.
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD wget --quiet --tries=1 --spider http://localhost:8765/health || exit 1

# tini as PID 1 forwards SIGTERM cleanly to Python so the engine's
# shutdown handler can flush positions to disk.
ENTRYPOINT ["/usr/bin/tini", "--"]

# Default command -- compose overrides this for the dashboard service.
CMD ["python", "-m", "loops.runner"]
