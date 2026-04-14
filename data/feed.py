"""
data/feed.py — Real-time polling feed for all universe symbols.

Architecture:
  - FeedManager runs a single coordinator thread that fires every
    settings.feed.poll_interval_seconds.
  - Each scan fans out to a ThreadPoolExecutor (max_workers=settings.feed.max_workers).
  - Every worker acquires a shared semaphore, sleeps request_delay_seconds
    (rate-limit courtesy), fetches a single quote, then releases the semaphore.
  - This keeps concurrent Robinhood API calls at or below max_workers while
    completing a full 25-symbol scan in ~(n_symbols/max_workers * request_delay)s.
  - Latest quote per symbol is stored in an in-memory snapshot dict.
  - Subscribers (signal engine, dashboard) receive callbacks fired from the
    worker threads — callbacks must be fast and thread-safe.
  - On >= MAX_CONSECUTIVE_ERRORS for a symbol, that symbol is marked degraded;
    the system logs loudly but continues on remaining symbols.

Rate budget (default settings):
  25 symbols / 8 workers = 4 batches × 0.2 s delay ≈ 0.8 s scan time
  per 10 s interval → 2.5 req/s/symbol = ~62.5 req/min total — safely under
  Robinhood's unofficial ~100 req/min ceiling.

Lifecycle:
  - start()  — launches coordinator thread (idempotent)
  - stop()   — signals coordinator to exit, shuts down executor, joins
"""

from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Dict, List, Optional

from broker.base import BrokerInterface, BrokerError, Quote
from config.settings import settings

logger = logging.getLogger(__name__)

# Number of consecutive poll failures before a symbol is marked degraded
MAX_CONSECUTIVE_ERRORS = 5

# Callback type: receives the fresh Quote each time a symbol is updated
QuoteCallback = Callable[[Quote], None]


class FeedError(Exception):
    """Raised when a symbol's feed exceeds its error threshold."""


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
        feed = FeedManager(broker, symbols=universe)
        feed.start()

        quote = feed.get_latest("SPY")   # non-blocking snapshot read
        feed.subscribe("SPY", my_callback)

        feed.stop()
    """

    def __init__(
        self,
        broker: BrokerInterface,
        symbols: List[str] | None = None,
        poll_interval: Optional[float] = None,
    ) -> None:
        from data.universe import get_universe
        self._broker = broker
        self._symbols: List[str] = symbols if symbols is not None else get_universe()
        self._poll_interval: float = (
            poll_interval
            if poll_interval is not None
            else settings.feed.poll_interval_seconds
        )
        self._max_workers: int = settings.feed.max_workers
        self._request_delay: float = settings.feed.request_delay_seconds

        # Per-symbol state
        self._states: Dict[str, SymbolState] = {
            sym: SymbolState(symbol=sym) for sym in self._symbols
        }
        # Per-symbol subscriber callbacks
        self._callbacks: Dict[str, List[QuoteCallback]] = {
            sym: [] for sym in self._symbols
        }
        self._callback_lock = threading.Lock()

        # Rate-limiting semaphore shared across all worker threads in a scan
        self._rate_sem = threading.Semaphore(self._max_workers)

        # Coordinator thread
        self._coordinator: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._started = False

        # Executor is created fresh on start() and shut down on stop()
        self._executor: Optional[ThreadPoolExecutor] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Launch coordinator thread and worker executor.  Idempotent."""
        if self._started:
            logger.warning("FeedManager.start() called but feed is already running")
            return

        self._stop_event.clear()
        self._executor = ThreadPoolExecutor(
            max_workers=self._max_workers,
            thread_name_prefix="feed-worker",
        )
        self._coordinator = threading.Thread(
            target=self._coordinator_loop,
            name="feed-coordinator",
            daemon=True,
        )
        self._coordinator.start()
        self._started = True
        logger.info(
            "FeedManager started | %d symbols | poll_interval=%.1fs | "
            "workers=%d | request_delay=%.2fs",
            len(self._symbols),
            self._poll_interval,
            self._max_workers,
            self._request_delay,
        )

    def stop(self, timeout: float = 10.0) -> None:
        """Signal coordinator to exit, drain executor, and join."""
        if not self._started:
            return
        self._stop_event.set()
        if self._coordinator and self._coordinator.is_alive():
            self._coordinator.join(timeout=timeout)
            if self._coordinator.is_alive():
                logger.warning("FeedManager: coordinator thread did not exit cleanly")
        if self._executor:
            self._executor.shutdown(wait=True, cancel_futures=True)
            self._executor = None
        self._started = False
        logger.info("FeedManager stopped")

    # ------------------------------------------------------------------
    # Consumer API
    # ------------------------------------------------------------------

    def get_latest(self, symbol: str) -> Optional[Quote]:
        """Return the most recent Quote for *symbol*, or None.  Non-blocking."""
        state = self._get_state(symbol)
        with state.lock:
            return state.latest

    def get_all_latest(self) -> Dict[str, Optional[Quote]]:
        """Return a snapshot dict of symbol → latest Quote for all symbols."""
        return {sym: self.get_latest(sym) for sym in self._symbols}

    def subscribe(self, symbol: str, callback: QuoteCallback) -> None:
        """
        Register *callback* to be called each time a new quote arrives for
        *symbol*.  Callback is invoked from a worker thread — keep it fast.
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
                pass

    def is_degraded(self, symbol: str) -> bool:
        """True if this symbol has exceeded its consecutive-error threshold."""
        return self._get_state(symbol).degraded

    def degraded_symbols(self) -> List[str]:
        """Return list of all symbols currently marked degraded."""
        return [sym for sym in self._symbols if self._states[sym].degraded]

    def status(self) -> Dict[str, dict]:
        """Return a health summary for each symbol — used by the dashboard."""
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
    # Coordinator loop
    # ------------------------------------------------------------------

    def _coordinator_loop(self) -> None:
        """
        Runs in the coordinator thread.  Every poll_interval seconds,
        submits all non-degraded symbols to the executor for parallel fetching.
        """
        logger.debug("FeedManager: coordinator started")

        while not self._stop_event.is_set():
            scan_start = time.monotonic()
            self._run_scan()
            elapsed = time.monotonic() - scan_start
            sleep_for = max(0.0, self._poll_interval - elapsed)
            self._stop_event.wait(timeout=sleep_for)

        logger.debug("FeedManager: coordinator exiting")

    def _run_scan(self) -> None:
        """
        Fan out one quote fetch per non-degraded symbol using the executor.
        Blocks until all futures complete so the coordinator can accurately
        measure elapsed time for interval scheduling.
        """
        if self._executor is None or self._stop_event.is_set():
            return

        active_symbols = [
            sym for sym in self._symbols if not self._states[sym].degraded
        ]
        if not active_symbols:
            logger.warning("FeedManager: all symbols degraded — nothing to scan")
            return

        futures = {
            self._executor.submit(self._fetch_one, sym): sym
            for sym in active_symbols
        }

        for future in as_completed(futures):
            sym = futures[future]
            exc = future.exception()
            if exc is not None:
                # _fetch_one catches all exceptions internally — this shouldn't fire,
                # but guard against unexpected raises from the executor itself.
                logger.error("FeedManager: unexpected future error for %s: %s", sym, exc)

    # ------------------------------------------------------------------
    # Per-symbol worker
    # ------------------------------------------------------------------

    def _fetch_one(self, symbol: str) -> None:
        """
        Worker task: rate-throttle → fetch → store/callback.

        Rate throttle: acquire semaphore (limits concurrency to max_workers),
        sleep request_delay_seconds, then release after fetch.  This staggers
        requests even when all workers are available simultaneously.
        """
        if self._stop_event.is_set():
            return

        state = self._states[symbol]

        # Acquire rate-limit slot
        acquired = self._rate_sem.acquire(timeout=self._poll_interval)
        if not acquired:
            logger.warning("FeedManager: semaphore timeout for %s — skipping", symbol)
            return

        try:
            # Courtesy delay before issuing the actual API call
            time.sleep(self._request_delay)

            if self._stop_event.is_set():
                return

            quote = self._broker.get_quote(symbol)
            self._store_quote(state, quote)

        except BrokerError as exc:
            self._handle_fetch_error(state, exc)
        except Exception as exc:
            self._handle_fetch_error(state, exc)
        finally:
            self._rate_sem.release()

    # ------------------------------------------------------------------
    # Quote storage and callbacks
    # ------------------------------------------------------------------

    def _store_quote(self, state: SymbolState, quote: Quote) -> None:
        """Update snapshot and fire callbacks.  Resets error counter on success."""
        with state.lock:
            state.latest = quote
            state.last_updated = datetime.utcnow()
            state.consecutive_errors = 0
            callbacks = list(self._callbacks.get(state.symbol, []))

        # Fire callbacks outside the lock to avoid deadlock
        for cb in callbacks:
            try:
                cb(quote)
            except Exception as exc:
                logger.warning(
                    "FeedManager: callback error for %s: %s", state.symbol, exc
                )

    def _handle_fetch_error(self, state: SymbolState, exc: Exception) -> None:
        with state.lock:
            state.consecutive_errors += 1
            count = state.consecutive_errors
            if count >= MAX_CONSECUTIVE_ERRORS:
                state.degraded = True

        if state.degraded:
            logger.error(
                "FeedManager: %s degraded after %d consecutive errors (%s) — "
                "symbol removed from scan",
                state.symbol, count, exc,
            )
        else:
            logger.warning(
                "FeedManager: fetch error for %s (%d/%d): %s",
                state.symbol, count, MAX_CONSECUTIVE_ERRORS, exc,
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_state(self, symbol: str) -> SymbolState:
        if symbol not in self._states:
            raise FeedError(
                f"{symbol} is not tracked by this FeedManager. "
                f"Tracked: {list(self._states.keys())}"
            )
        return self._states[symbol]
