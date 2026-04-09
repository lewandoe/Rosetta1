"""
data/feed.py — Real-time polling feed for all 10 universe symbols.

Architecture:
  - FeedManager runs one background thread per symbol (10 threads total).
  - Each thread polls its broker's get_quote() on the configured interval
    (settings.execution.signal_scan_interval_seconds).
  - Latest quote per symbol is stored in an in-memory snapshot dict.
  - Downstream consumers (signal engine, dashboard) call get_latest() or
    subscribe via callbacks — they never block on I/O.
  - On consecutive poll failures (>= MAX_CONSECUTIVE_ERRORS), the thread
    raises FeedError and the manager marks that symbol as degraded.
    The system logs loudly but continues on remaining symbols.

Thread model:
  - One Lock per symbol protects its snapshot entry.
  - FeedManager itself is safe to call from multiple threads.

Lifecycle:
  - start()  — launches all 10 poll threads (idempotent)
  - stop()   — signals threads to exit, joins them with a timeout
  - Designed to be started once in main.py and stopped on shutdown.
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Dict, List, Optional

from broker.base import BrokerInterface, BrokerError, Quote
from config.settings import UNIVERSE, settings

logger = logging.getLogger(__name__)

# Number of consecutive poll failures before a symbol is marked degraded
MAX_CONSECUTIVE_ERRORS = 5

# Callback type: receives the fresh Quote each time a symbol is updated
QuoteCallback = Callable[[Quote], None]


class FeedError(Exception):
    """Raised when a symbol's feed thread exceeds its error threshold."""


@dataclass
class SymbolState:
    """All per-symbol runtime state held by FeedManager."""
    symbol: str
    latest: Optional[Quote] = None
    last_updated: Optional[datetime] = None
    consecutive_errors: int = 0
    degraded: bool = False
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)


class FeedManager:
    """
    Manages real-time quote polling for every symbol in UNIVERSE.

    Usage:
        broker = PaperBroker()
        feed = FeedManager(broker)
        feed.start()

        quote = feed.get_latest("SPY")   # non-blocking snapshot read
        feed.subscribe("SPY", my_callback)

        feed.stop()
    """

    def __init__(
        self,
        broker: BrokerInterface,
        symbols: List[str] = UNIVERSE,
        poll_interval: Optional[float] = None,
    ) -> None:
        self._broker = broker
        self._symbols = symbols
        self._poll_interval: float = (
            poll_interval
            if poll_interval is not None
            else float(settings.execution.signal_scan_interval_seconds)
        )
        # Per-symbol state
        self._states: Dict[str, SymbolState] = {
            sym: SymbolState(symbol=sym) for sym in symbols
        }
        # Per-symbol subscriber callbacks
        self._callbacks: Dict[str, List[QuoteCallback]] = {sym: [] for sym in symbols}
        self._callback_lock = threading.Lock()

        # Thread handles
        self._threads: Dict[str, threading.Thread] = {}
        self._stop_event = threading.Event()
        self._started = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Launch one polling thread per symbol.  Idempotent."""
        if self._started:
            logger.warning("FeedManager.start() called but feed is already running")
            return

        self._stop_event.clear()
        for sym in self._symbols:
            t = threading.Thread(
                target=self._poll_loop,
                args=(sym,),
                name=f"feed-{sym}",
                daemon=True,   # dies automatically if main thread exits
            )
            self._threads[sym] = t
            t.start()

        self._started = True
        logger.info(
            "FeedManager started | %d symbols | poll_interval=%.1fs",
            len(self._symbols), self._poll_interval,
        )

    def stop(self, timeout: float = 5.0) -> None:
        """Signal all threads to exit and join them."""
        if not self._started:
            return
        self._stop_event.set()
        for sym, t in self._threads.items():
            t.join(timeout=timeout)
            if t.is_alive():
                logger.warning("FeedManager: poll thread for %s did not exit cleanly", sym)
        self._threads.clear()
        self._started = False
        logger.info("FeedManager stopped")

    # ------------------------------------------------------------------
    # Consumer API
    # ------------------------------------------------------------------

    def get_latest(self, symbol: str) -> Optional[Quote]:
        """
        Return the most recent Quote for *symbol*, or None if no poll has
        succeeded yet.  Non-blocking — reads from in-memory snapshot.
        """
        state = self._get_state(symbol)
        with state.lock:
            return state.latest

    def get_all_latest(self) -> Dict[str, Optional[Quote]]:
        """Return a snapshot dict of symbol → latest Quote for all symbols."""
        return {sym: self.get_latest(sym) for sym in self._symbols}

    def subscribe(self, symbol: str, callback: QuoteCallback) -> None:
        """
        Register *callback* to be called each time a new quote arrives for
        *symbol*.  Callback is invoked from the poll thread — keep it fast.
        """
        self._get_state(symbol)  # validates symbol
        with self._callback_lock:
            self._callbacks[symbol].append(callback)

    def unsubscribe(self, symbol: str, callback: QuoteCallback) -> None:
        """Remove a previously registered callback."""
        with self._callback_lock:
            try:
                self._callbacks[symbol].remove(callback)
            except ValueError:
                pass  # already removed — not an error

    def is_degraded(self, symbol: str) -> bool:
        """True if this symbol's poll thread has exceeded the error threshold."""
        return self._get_state(symbol).degraded

    def degraded_symbols(self) -> List[str]:
        """Return list of all symbols currently marked degraded."""
        return [sym for sym in self._symbols if self._states[sym].degraded]

    def status(self) -> Dict[str, dict]:
        """
        Return a health summary for each symbol — used by the dashboard.

        Example entry:
            "SPY": {
                "last_price": 512.34,
                "last_updated": "2024-01-08T10:30:01",
                "consecutive_errors": 0,
                "degraded": False,
            }
        """
        result = {}
        for sym, state in self._states.items():
            with state.lock:
                result[sym] = {
                    "last_price": state.latest.last if state.latest else None,
                    "last_updated": (
                        state.last_updated.isoformat() if state.last_updated else None
                    ),
                    "consecutive_errors": state.consecutive_errors,
                    "degraded": state.degraded,
                }
        return result

    # ------------------------------------------------------------------
    # Poll loop (runs in background thread)
    # ------------------------------------------------------------------

    def _poll_loop(self, symbol: str) -> None:
        """
        Continuously poll quotes for *symbol* until stop() is called.

        Error handling:
          - Transient BrokerError: log warning, increment counter.
          - If consecutive_errors >= MAX_CONSECUTIVE_ERRORS: mark degraded,
            log error, and exit the thread.  The system continues on other symbols.
        """
        logger.debug("FeedManager: poll thread started for %s", symbol)
        state = self._states[symbol]

        while not self._stop_event.is_set():
            poll_start = time.monotonic()
            try:
                quote = self._broker.get_quote(symbol)
                self._store_quote(state, quote)
            except BrokerError as exc:
                self._handle_poll_error(state, exc)
                if state.degraded:
                    logger.error(
                        "FeedManager: %s feed degraded after %d consecutive errors — "
                        "stopping poll thread",
                        symbol, MAX_CONSECUTIVE_ERRORS,
                    )
                    return
            except Exception as exc:
                # Unexpected exceptions are treated the same as BrokerError
                self._handle_poll_error(state, exc)
                if state.degraded:
                    return

            # Sleep for the remainder of the poll interval (accounts for fetch time)
            elapsed = time.monotonic() - poll_start
            sleep_for = max(0.0, self._poll_interval - elapsed)
            self._stop_event.wait(timeout=sleep_for)

        logger.debug("FeedManager: poll thread exiting for %s", symbol)

    def _store_quote(self, state: SymbolState, quote: Quote) -> None:
        """Update snapshot and fire callbacks.  Resets error counter on success."""
        with state.lock:
            state.latest = quote
            state.last_updated = datetime.utcnow()
            state.consecutive_errors = 0
            # Don't hold the lock while firing callbacks — could deadlock
            callbacks = list(self._callbacks.get(state.symbol, []))

        for cb in callbacks:
            try:
                cb(quote)
            except Exception as exc:
                logger.warning(
                    "FeedManager: callback error for %s: %s", state.symbol, exc
                )

    def _handle_poll_error(self, state: SymbolState, exc: Exception) -> None:
        with state.lock:
            state.consecutive_errors += 1
            count = state.consecutive_errors
            if count >= MAX_CONSECUTIVE_ERRORS:
                state.degraded = True

        logger.warning(
            "FeedManager: poll error for %s (%d/%d): %s",
            state.symbol, count, MAX_CONSECUTIVE_ERRORS, exc,
        )

    def _get_state(self, symbol: str) -> SymbolState:
        if symbol not in self._states:
            raise FeedError(
                f"{symbol} is not tracked by this FeedManager. "
                f"Tracked: {list(self._states.keys())}"
            )
        return self._states[symbol]
