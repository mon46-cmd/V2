# syntax=docker/dockerfile:1.6
# ---------------------------------------------------------------------------
# V8 Production Image -- single-stage, system Python 3.12, no venv, no apt.
#
# Why no apt-get?
#   * Skipping `apt-get update` removes the slowest layer entirely (and the
#     dependency on deb.debian.org being reachable, which has been flaky).
#   * `python:3.12-slim` already ships ca-certificates and Python.
#   * `tini` -> replaced by Docker's built-in `--init` flag (set in compose
#     via `init: true`). Docker uses tini under the hood for that.
#   * `wget` -> replaced by a one-line urllib healthcheck (Python is here).
#
# Result: build is essentially `pip install -r requirements.txt`, fully
# cached on rebuilds when requirements.txt is unchanged.
# ---------------------------------------------------------------------------

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_ROOT_USER_ACTION=ignore \
    TZ=UTC

WORKDIR /app

# Dependency layer (cached when requirements.txt is unchanged).
COPY requirements.txt ./
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip \
 && pip install -r requirements.txt

# Application source.
COPY . .

# Non-root runtime user. Owns /app/data so the named volume mounts cleanly.
# We use the Python-shipped `useradd`-style fallback via `adduser` from
# python:3.12-slim base (login.defs is present). If absent, a chown-only
# approach still works because we set USER by uid below.
RUN groupadd --system --gid 10001 appgroup \
 && useradd  --system --uid 10001 --gid appgroup --home-dir /app --shell /usr/sbin/nologin appuser \
 && mkdir -p /app/data \
 && chown -R appuser:appgroup /app
USER appuser

EXPOSE 8765

# Healthcheck without wget: a 5s urllib request to /health.
# Exit 0 on HTTP 200, non-zero otherwise. The engine service overrides
# this with a process check in docker-compose.yml.
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; \
sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8765/health', timeout=5).status==200 else 1)" \
        || exit 1

# No ENTRYPOINT: Docker's `--init` (compose `init: true`) provides a
# tini-equivalent PID 1 that reaps zombies and forwards SIGTERM cleanly,
# so the engine's shutdown hook runs and positions are persisted.
CMD ["python", "-m", "loops.runner"]
