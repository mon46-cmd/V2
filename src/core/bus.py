"""Lightweight in-process async event bus.

Allows loosely coupled components to emit and subscribe to named events
without importing each other directly.  Useful for feeding the dashboard,
Telegram notifier, or audit log without wiring them into the hot path.

Usage::

    from core.bus import bus

    # Subscribe (at startup):
    @bus.on("position.opened")
    async def on_open(payload: dict) -> None:
        await telegram.send(f"Opened {payload['symbol']}")

    # Publish (anywhere in the hot path):
    await bus.emit("position.opened", {"symbol": "BTCUSDT", "side": "long"})

Notes
-----
* All handlers are ``async`` -- synchronous callbacks are NOT supported.
* Exceptions in handlers are caught and logged; they never propagate to
  the emitter.
* The bus is a module-level singleton (``bus``).  You can also instantiate
  ``EventBus()`` directly for isolated unit tests.
"""
from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from typing import Any, Callable, Coroutine

log = logging.getLogger(__name__)

# Type alias for an async handler.
AsyncHandler = Callable[..., Coroutine[Any, Any, None]]


class EventBus:
    """Async publish / subscribe event bus."""

    def __init__(self) -> None:
        self._handlers: dict[str, list[AsyncHandler]] = defaultdict(list)

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def on(self, event: str) -> Callable[[AsyncHandler], AsyncHandler]:
        """Decorator: register an async handler for *event*.

        Example::

            @bus.on("trade.closed")
            async def notify(payload: dict) -> None:
                ...
        """
        def decorator(fn: AsyncHandler) -> AsyncHandler:
            self._handlers[event].append(fn)
            return fn
        return decorator

    def subscribe(self, event: str, handler: AsyncHandler) -> None:
        """Register *handler* for *event* imperatively."""
        self._handlers[event].append(handler)

    def unsubscribe(self, event: str, handler: AsyncHandler) -> None:
        """Remove a previously registered *handler* for *event*."""
        try:
            self._handlers[event].remove(handler)
        except ValueError:
            pass

    def clear(self, event: str | None = None) -> None:
        """Remove all handlers for *event*, or all handlers if None."""
        if event is None:
            self._handlers.clear()
        else:
            self._handlers.pop(event, None)

    # ------------------------------------------------------------------
    # Emission
    # ------------------------------------------------------------------

    async def emit(self, event: str, payload: Any = None) -> None:
        """Emit *event* and await all registered handlers concurrently.

        Exceptions in individual handlers are caught and logged so that
        a misbehaving subscriber can never block or crash the emitter.

        Args:
            event:   Event name (e.g. ``"position.opened"``).
            payload: Arbitrary data passed as the first argument to each
                     handler.  Prefer plain dicts for forward-compat.
        """
        handlers = list(self._handlers.get(event, []))
        if not handlers:
            return

        async def _call(h: AsyncHandler) -> None:
            try:
                await h(payload)
            except Exception as exc:  # noqa: BLE001
                log.error(
                    "event handler raised",
                    extra={"event": event, "handler": h.__qualname__, "exc": repr(exc)},
                    exc_info=True,
                )

        await asyncio.gather(*(_call(h) for h in handlers))

    def emit_nowait(self, event: str, payload: Any = None) -> None:
        """Schedule ``emit`` as a fire-and-forget task (non-blocking).

        Requires a running event loop.  Use this from synchronous contexts
        that cannot ``await``.
        """
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            log.warning("emit_nowait: no running event loop; event dropped", extra={"event": event})
            return
        loop.create_task(self.emit(event, payload))


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

#: Shared event bus.  Import this wherever you need pub/sub.
bus: EventBus = EventBus()
