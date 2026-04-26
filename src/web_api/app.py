"""V8 Dashboard — FastAPI application factory and server entry point."""
from __future__ import annotations

import argparse
import logging
import sys
from collections import deque
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from core.config import load_config

STATIC_DIR = Path(__file__).parent / "static"
log = logging.getLogger(__name__)

# ── In-memory log ring buffer exposed via /api/logs ──────────────────────
LOG_BUFFER: deque[dict] = deque(maxlen=2000)

_LEVEL_COLOR = {
    "DEBUG": "muted", "INFO": "normal", "WARNING": "warn",
    "ERROR": "error", "CRITICAL": "error",
}


class _RingHandler(logging.Handler):
    """Captures every log record into LOG_BUFFER."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            LOG_BUFFER.append({
                "ts":    record.created,
                "ts_h":  self.formatTime(record, "%H:%M:%S"),
                "level": record.levelname,
                "color": _LEVEL_COLOR.get(record.levelname, "normal"),
                "name":  record.name,
                "msg":   record.getMessage(),
            })
        except Exception:
            pass


def create_app() -> FastAPI:
    cfg = load_config()

    app = FastAPI(title="Trading Dashboard", docs_url="/docs")
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
    app.state.cfg = cfg

    # Liveness probe (used by Docker HEALTHCHECK / VPS monitors).
    # Intentionally lightweight: returns 200 as long as the FastAPI process
    # is responsive. Deeper health (engine running, broker state) is exposed
    # under /api/status.
    @app.get("/health")
    def _health() -> dict:
        return {"status": "ok"}

    from web_api.routes_api import router as api_router
    app.include_router(api_router, prefix="/api")

    STATIC_DIR.mkdir(parents=True, exist_ok=True)
    app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="static")
    return app


def main() -> None:
    # Attach ring buffer to root logger so ALL log output is captured
    ring = _RingHandler()
    ring.setLevel(logging.DEBUG)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout), ring],
    )

    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8765)
    args = ap.parse_args()

    log.info("Starting Trading Dashboard on %s:%d", args.host, args.port)
    uvicorn.run(
        "web_api.app:create_app",
        factory=True,
        host=args.host,
        port=args.port,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    main()
