"""V8 trading engine entry point.

Run::

    cd V8
    python -m loops.runner             # live mode with env var config
    python -m loops.runner --dry-run   # force AI dry run (no real AI calls)
    python -m loops.runner --once      # run exactly one tick then exit (debug)
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import signal
import sys
from pathlib import Path

from core.config import load_config
from loops.scanner import Scanner


def _setup_logging(level: str = "INFO") -> None:
    fmt = "%(asctime)s %(levelname)-7s %(name)s: %(message)s"
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=fmt,
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    # Quiet noisy third-party loggers
    for noisy in ("httpx", "httpcore", "openai", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="V8 Paper Trading Engine")
    ap.add_argument("--dry-run", action="store_true", help="Force AI dry-run mode (log but don't call AI)")
    ap.add_argument("--once",    action="store_true", help="Run one tick then exit")
    ap.add_argument("--log",     default="INFO",      help="Log level (DEBUG/INFO/WARNING)")
    return ap.parse_args()


async def _run_once(scanner: Scanner) -> None:
    """Run a single tick, then exit cleanly."""
    await scanner._tick()  # noqa: SLF001
    scanner._save()        # noqa: SLF001


async def _run_forever(scanner: Scanner) -> None:
    """Run until SIGINT/SIGTERM.

    Cancels the scanner task cleanly and waits for it to drain so that
    state (positions / cooldowns) is persisted before exit.
    """
    loop = asyncio.get_running_loop()
    log  = logging.getLogger(__name__)

    main_task: asyncio.Task[None] = asyncio.create_task(scanner.run(), name="scanner.run")

    def _request_shutdown(sig_name: str) -> None:
        log.info("received %s -- shutting down", sig_name)
        if not main_task.done():
            main_task.cancel()

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, lambda s=sig.name: _request_shutdown(s))
        except NotImplementedError:
            # Windows: fall back to KeyboardInterrupt path below.
            pass

    try:
        await main_task
    except asyncio.CancelledError:
        # Expected on shutdown -- scanner.run() already saved state in its
        # except CancelledError branch.
        pass
    except KeyboardInterrupt:
        log.info("KeyboardInterrupt -- shutting down")
        if not main_task.done():
            main_task.cancel()
            try:
                await main_task
            except (asyncio.CancelledError, KeyboardInterrupt):
                pass


def main() -> None:
    args = _parse_args()
    _setup_logging(args.log)

    log = logging.getLogger(__name__)

    cfg = load_config()

    if args.dry_run:
        cfg = cfg  # env var AI_DRY_RUN=1 is the canonical way; --dry-run sets it in memory
        import os
        os.environ["AI_DRY_RUN"] = "1"
        cfg = load_config()   # reload so the flag is picked up
        log.info("DRY-RUN mode active — AI calls will be mocked")

    # Ensure data dirs exist
    for d in (cfg.run_root, cfg.log_root):
        Path(d).mkdir(parents=True, exist_ok=True)

    log.info("V8 engine starting — equity=$%.2f  universe_size=%d  dry_run=%s",
             cfg.paper_equity_usd, cfg.universe_size, cfg.ai_dry_run)

    scanner = Scanner(cfg)

    if args.once:
        log.info("--once mode: running a single tick")
        asyncio.run(_run_once(scanner))
    else:
        asyncio.run(_run_forever(scanner))

    log.info("V8 engine stopped cleanly")


if __name__ == "__main__":
    main()
