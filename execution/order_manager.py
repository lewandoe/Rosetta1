"""
execution/order_manager.py — Entry, exit, stop-loss, and position monitoring.

Responsibilities:
  1. execute_signal()     — place entry, validate slippage, record open trade
  2. _monitor_loop()      — background thread: checks target/stop/EOD each tick
  3. _close_position()    — market exit, records P&L in RiskGuard
  4. force_close_all()    — EOD liquidation (called directly by main.py too)

Stop-loss strategy:
  The monitor loop is the primary stop mechanism for both paper and live.
  In live mode, a broker-side stop order is also placed as a belt-and-
  suspenders failsafe — if the monitor thread lags, the broker fires the stop.
  In paper mode we rely on the monitor loop only (paper broker simulates
  stop orders as immediate fills at current price, not conditional fills).

Partial fills:
  If the entry order is partially filled (filled_qty < requested_qty), the
  manager accepts the partial position and sizes the stop order accordingly.
  A WARN is logged. If filled_qty == 0 the trade is abandoned.

Thread safety:
  _open_trades is protected by _lock. All reads/writes go through the lock.
  Callbacks fire outside the lock to prevent deadlocks.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional

import pytz

from broker.base import (
    BrokerError,
    BrokerInterface,
    Order,
    OrderResult,
    OrderSide,
    OrderStatus,
    OrderType,
)
from config.settings import settings
from risk.guard import RiskDecision, RiskGuard
from signals.base import SignalResult

logger = logging.getLogger(__name__)

ET = pytz.timezone("America/New_York")

# Callback type: invoked after each trade closes
TradeClosedCallback = Callable[["ClosedTrade"], None]


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class OpenTrade:
    """State for a single live position managed by OrderManager."""
    trade_id: str
    symbol: str
    direction: str          # "long" or "short"
    shares: int             # actual filled quantity (may differ from requested)
    entry_price: float      # actual avg fill price
    target_price: float
    stop_price: float
    signal_type: str
    confidence: int
    entry_order_id: str
    opened_at: datetime
    signal: SignalResult
    # Broker-side stop order ID (live mode only)
    stop_order_id: Optional[str] = None
    # Original stop distance at entry — used by trailing stop logic
    initial_stop_distance: float = 0.0


@dataclass
class ClosedTrade:
    """Immutable record emitted when a trade closes."""
    trade_id: str
    symbol: str
    direction: str
    shares: int
    entry_price: float
    exit_price: float
    gross_pnl: float
    signal_type: str
    confidence: int
    entry_order_id: str
    exit_order_id: str
    opened_at: datetime
    closed_at: datetime
    exit_reason: str        # "target", "stop", "eod", "manual", "slippage"

    @property
    def hold_seconds(self) -> float:
        return (self.closed_at - self.opened_at).total_seconds()


# ---------------------------------------------------------------------------
# Order Manager
# ---------------------------------------------------------------------------

class OrderManager:
    """
    Manages the full trade lifecycle from signal → entry → exit.

    Usage:
        om = OrderManager(broker, risk_guard)
        om.start()

        decision = risk_guard.check(signal, ...)
        if decision.approved:
            om.execute_signal(signal, decision)

        om.stop()
    """

    def __init__(
        self,
        broker: BrokerInterface,
        risk_guard: RiskGuard,
    ) -> None:
        self._broker = broker
        self._risk = risk_guard

        self._lock = threading.Lock()
        self._open_trades: Dict[str, OpenTrade] = {}   # trade_id → OpenTrade
        self._closed_trades: List[ClosedTrade] = []

        self._stop_event = threading.Event()
        self._monitor_thread: Optional[threading.Thread] = None
        self._started = False

        self._on_trade_closed: List[TradeClosedCallback] = []

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Launch the position monitor background thread."""
        if self._started:
            logger.warning("OrderManager.start() called but already running")
            return
        self._stop_event.clear()
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            name="order-monitor",
            daemon=True,
        )
        self._monitor_thread.start()
        self._started = True
        logger.info(
            "OrderManager started | monitor_interval=%ds",
            settings.execution.position_monitor_interval_seconds,
        )

    def stop(self, timeout: float = 5.0) -> None:
        """Signal monitor thread to exit and join it."""
        if not self._started:
            return
        self._stop_event.set()
        if self._monitor_thread:
            self._monitor_thread.join(timeout=timeout)
            if self._monitor_thread.is_alive():
                logger.warning("OrderManager: monitor thread did not exit cleanly")
        self._started = False
        logger.info("OrderManager stopped")

    def on_trade_closed(self, callback: TradeClosedCallback) -> None:
        """Register a callback invoked after every trade close (from monitor thread)."""
        self._on_trade_closed.append(callback)

    # ------------------------------------------------------------------
    # Signal execution
    # ------------------------------------------------------------------

    def execute_signal(
        self,
        signal: SignalResult,
        decision: RiskDecision,
    ) -> Optional[OpenTrade]:
        """
        Place the entry order for an approved signal.

        Steps:
          1. Place market entry order.
          2. Validate fill slippage — if excessive, close immediately.
          3. Register open trade so the monitor loop tracks it.
          4. Place broker-side stop order (live mode only).

        Returns:
            OpenTrade if the entry was accepted, None if rejected post-fill.
        """
        if not decision.approved or decision.position_size < 1:
            logger.error(
                "OrderManager.execute_signal called with unapproved decision: %s",
                decision,
            )
            return None

        side = OrderSide.BUY if signal.direction == "long" else OrderSide.SELL
        entry_order = Order(
            symbol=signal.symbol,
            side=side,
            quantity=decision.position_size,
            order_type=OrderType.MARKET,
            time_in_force=settings.execution.default_time_in_force,
            client_order_id=str(uuid.uuid4()),
        )

        # ── Place entry ─────────────────────────────────────────────────────
        try:
            result = self._broker.place_order(entry_order)
        except BrokerError as exc:
            logger.error(
                "OrderManager [%s]: entry order failed — %s", signal.symbol, exc
            )
            return None

        if result.status == OrderStatus.REJECTED or result.status == OrderStatus.FAILED:
            logger.error(
                "OrderManager [%s]: entry order %s — %s",
                signal.symbol, result.status.value, result.raw,
            )
            return None

        # ── Handle partial fill ─────────────────────────────────────────────
        filled_qty = result.filled_quantity
        fill_price = result.avg_fill_price or signal.entry_price

        if filled_qty == 0:
            logger.warning(
                "OrderManager [%s]: zero fill — abandoning trade", signal.symbol
            )
            return None

        if filled_qty < decision.position_size:
            logger.warning(
                "OrderManager [%s]: partial fill %d/%d shares @ %.4f",
                signal.symbol, filled_qty, decision.position_size, fill_price,
            )

        # ── Slippage check ──────────────────────────────────────────────────
        slip = self._risk.check_slippage(fill_price, signal.entry_price)
        if not slip.acceptable:
            logger.warning(
                "OrderManager [%s]: slippage %.4f%% exceeds limit — closing immediately",
                signal.symbol, slip.slippage_pct * 100,
            )
            self._emergency_close(signal.symbol, filled_qty, signal.direction,
                                  fill_price, signal, result.order_id, "slippage")
            return None

        # ── Register open trade ─────────────────────────────────────────────
        trade_id = str(uuid.uuid4())
        trade = OpenTrade(
            trade_id=trade_id,
            symbol=signal.symbol,
            direction=signal.direction,
            shares=filled_qty,
            entry_price=fill_price,
            target_price=signal.target_price,
            stop_price=signal.stop_price,
            signal_type=signal.signal_type,
            confidence=signal.confidence,
            entry_order_id=result.order_id,
            opened_at=datetime.now(timezone.utc),
            signal=signal,
            initial_stop_distance=abs(fill_price - signal.stop_price),
        )

        with self._lock:
            self._open_trades[trade_id] = trade

        logger.info(
            "OrderManager ENTRY [%s] %s x%d @ %.4f | target=%.4f stop=%.4f | id=%s",
            signal.symbol, signal.direction.upper(), filled_qty, fill_price,
            signal.target_price, signal.stop_price, trade_id[:8],
        )

        # ── Place broker-side stop order (live only) ────────────────────────
        if settings.broker.trading_mode == "live":
            stop_order_id = self._place_broker_stop(trade)
            if stop_order_id:
                with self._lock:
                    self._open_trades[trade_id].stop_order_id = stop_order_id

        return trade

    # ------------------------------------------------------------------
    # Position monitor loop
    # ------------------------------------------------------------------

    def _monitor_loop(self) -> None:
        """
        Background thread: polls open positions and closes them when:
          - Target price reached (take profit)
          - Stop price reached (stop loss)
          - EOD liquidation cutoff hit
        """
        interval = settings.execution.position_monitor_interval_seconds
        logger.debug("OrderManager: monitor loop started")

        while not self._stop_event.is_set():
            tick_start = time.monotonic()
            try:
                self._monitor_tick()
            except Exception as exc:
                logger.error("OrderManager: monitor tick error: %s", exc, exc_info=True)

            elapsed = time.monotonic() - tick_start
            self._stop_event.wait(timeout=max(0.0, interval - elapsed))

        logger.debug("OrderManager: monitor loop exiting")

    def _monitor_tick(self) -> None:
        """Single monitor iteration — checks all open trades."""
        # Copy trade list outside the lock to minimise lock hold time
        with self._lock:
            trades = list(self._open_trades.values())

        if not trades:
            return

        # ── EOD check first — if past cutoff, close everything ─────────────
        now_et = datetime.now(ET)
        eod = now_et.replace(
            hour=settings.risk.eod_liquidation_hour,
            minute=settings.risk.eod_liquidation_minute,
            second=0, microsecond=0,
        )
        if now_et >= eod:
            logger.warning("OrderManager: EOD cutoff reached — liquidating all positions")
            for trade in trades:
                self._close_trade(trade, reason="eod")
            return

        # ── Price checks — single batch quote fetch for all symbols ──────────
        symbols = [t.symbol for t in trades]
        quotes = self._broker.get_quotes_batch(symbols)

        trades_to_close = []
        for trade in trades:
            quote = quotes.get(trade.symbol)
            if not quote:
                logger.warning("OrderManager: no quote in batch for %s", trade.symbol)
                continue
            current = quote.last

            # ── Trailing stop ───────────────────────────────────────────────
            stop_dist = trade.initial_stop_distance
            if stop_dist > 0:
                if trade.direction == "long":
                    unrealized_r = (current - trade.entry_price) / stop_dist
                    if unrealized_r >= 2.0:
                        new_stop = trade.entry_price + (current - trade.entry_price) * 0.5
                        if new_stop > trade.stop_price:
                            old_stop = trade.stop_price
                            trade.stop_price = new_stop
                            logger.info(
                                "OrderManager TRAIL [%s]: stop moved %.4f→%.4f (unrealized_r=%.2f)",
                                trade.symbol, old_stop, trade.stop_price, unrealized_r,
                            )
                    elif unrealized_r >= 1.0:
                        if trade.entry_price > trade.stop_price:
                            old_stop = trade.stop_price
                            trade.stop_price = trade.entry_price
                            logger.info(
                                "OrderManager TRAIL [%s]: stop moved %.4f→%.4f (unrealized_r=%.2f)",
                                trade.symbol, old_stop, trade.stop_price, unrealized_r,
                            )
                else:  # short
                    unrealized_r = (trade.entry_price - current) / stop_dist
                    if unrealized_r >= 2.0:
                        new_stop = trade.entry_price - (trade.entry_price - current) * 0.5
                        if new_stop < trade.stop_price:
                            old_stop = trade.stop_price
                            trade.stop_price = new_stop
                            logger.info(
                                "OrderManager TRAIL [%s]: stop moved %.4f→%.4f (unrealized_r=%.2f)",
                                trade.symbol, old_stop, trade.stop_price, unrealized_r,
                            )
                    elif unrealized_r >= 1.0:
                        if trade.entry_price < trade.stop_price:
                            old_stop = trade.stop_price
                            trade.stop_price = trade.entry_price
                            logger.info(
                                "OrderManager TRAIL [%s]: stop moved %.4f→%.4f (unrealized_r=%.2f)",
                                trade.symbol, old_stop, trade.stop_price, unrealized_r,
                            )

            # ── Target / stop exit checks ───────────────────────────────────
            if trade.direction == "long":
                if current >= trade.target_price:
                    trades_to_close.append((trade, "target"))
                elif current <= trade.stop_price:
                    trades_to_close.append((trade, "stop"))
            else:  # short
                if current <= trade.target_price:
                    trades_to_close.append((trade, "target"))
                elif current >= trade.stop_price:
                    trades_to_close.append((trade, "stop"))

        for trade, reason in trades_to_close:
            self._close_trade(trade, reason=reason)

    # ------------------------------------------------------------------
    # EOD forced liquidation (also callable from main.py)
    # ------------------------------------------------------------------

    def force_close_all(self, reason: str = "eod") -> List[ClosedTrade]:
        """
        Close every open position immediately at market price.
        Returns list of ClosedTrade records.
        """
        with self._lock:
            trades = list(self._open_trades.values())

        closed = []
        for trade in trades:
            ct = self._close_trade(trade, reason=reason)
            if ct:
                closed.append(ct)
        logger.info(
            "OrderManager.force_close_all: closed %d position(s) — reason=%s",
            len(closed), reason,
        )
        return closed

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def get_open_trades(self) -> List[OpenTrade]:
        with self._lock:
            return list(self._open_trades.values())

    def get_closed_trades(self) -> List[ClosedTrade]:
        with self._lock:
            return list(self._closed_trades)

    def open_trade_count(self) -> int:
        with self._lock:
            return len(self._open_trades)

    def update_trade_exits(
        self,
        symbol: str,
        new_target: float,
        new_stop: float,
    ) -> bool:
        """
        Update target_price and stop_price for the open trade in *symbol*.
        Called when a duplicate signal arrives — instead of opening a new
        position, the existing trade's exit levels are refreshed.

        Returns True if a matching trade was found and updated, False otherwise.
        """
        with self._lock:
            for trade in self._open_trades.values():
                if trade.symbol == symbol:
                    old_target = trade.target_price
                    old_stop   = trade.stop_price
                    trade.target_price = new_target
                    trade.stop_price   = new_stop
                    logger.info(
                        "OrderManager UPDATE [%s]: target %.4f→%.4f stop %.4f→%.4f",
                        symbol, old_target, new_target, old_stop, new_stop,
                    )
                    return True
        return False

    def exit_trade_by_symbol(self, symbol: str) -> bool:
        """
        Immediately close the open trade for *symbol* at market price.
        Called when a signal reversal is detected — the signal engine has
        flipped direction, so we exit the existing position and get flat.

        Returns True if a trade was found and closed, False otherwise.
        """
        with self._lock:
            trade = next(
                (t for t in self._open_trades.values() if t.symbol == symbol),
                None,
            )
        if trade is None:
            return False
        # _close_trade performs the atomic pop under lock before any network I/O
        result = self._close_trade(trade, reason="signal_reversal")
        return result is not None

    # ------------------------------------------------------------------
    # Internal close helpers
    # ------------------------------------------------------------------

    def _close_trade(
        self,
        trade: OpenTrade,
        reason: str,
    ) -> Optional[ClosedTrade]:
        """
        Place a market close order and record the trade result.
        Atomically removes the trade from open_trades first (under lock, before
        any network I/O) so concurrent monitor ticks or force-close calls cannot
        double-close the same position.
        """
        with self._lock:
            if self._open_trades.pop(trade.trade_id, None) is None:
                return None  # already closed by another path

        # Cancel any live broker-side stop order first to prevent double-close
        if trade.stop_order_id and settings.broker.trading_mode == "live":
            try:
                self._broker.cancel_order(trade.stop_order_id)
            except BrokerError as exc:
                logger.warning(
                    "OrderManager: could not cancel stop order %s: %s",
                    trade.stop_order_id, exc,
                )

        # Place market close
        close_side = OrderSide.SELL if trade.direction == "long" else OrderSide.BUY
        close_order = Order(
            symbol=trade.symbol,
            side=close_side,
            quantity=trade.shares,
            order_type=OrderType.MARKET,
            time_in_force=settings.execution.default_time_in_force,
        )

        try:
            close_result = self._broker.place_order(close_order)
        except BrokerError as exc:
            logger.error(
                "OrderManager [%s]: close order failed (%s): %s",
                trade.symbol, reason, exc,
            )
            return None

        if close_result.status not in (OrderStatus.FILLED, OrderStatus.PARTIALLY_FILLED):
            logger.error(
                "OrderManager [%s]: close order not filled (status=%s)",
                trade.symbol, close_result.status.value,
            )
            return None

        exit_price = close_result.avg_fill_price or trade.entry_price
        exit_qty   = close_result.filled_quantity or trade.shares

        # P&L: long profit = (exit - entry) × qty; short = (entry - exit) × qty
        if trade.direction == "long":
            gross_pnl = (exit_price - trade.entry_price) * exit_qty
        else:
            gross_pnl = (trade.entry_price - exit_price) * exit_qty

        closed_at = datetime.now(timezone.utc)
        ct = ClosedTrade(
            trade_id=trade.trade_id,
            symbol=trade.symbol,
            direction=trade.direction,
            shares=exit_qty,
            entry_price=trade.entry_price,
            exit_price=exit_price,
            gross_pnl=gross_pnl,
            signal_type=trade.signal_type,
            confidence=trade.confidence,
            entry_order_id=trade.entry_order_id,
            exit_order_id=close_result.order_id,
            opened_at=trade.opened_at,
            closed_at=closed_at,
            exit_reason=reason,
        )

        with self._lock:
            self._closed_trades.append(ct)

        # Record P&L with risk guard
        self._risk.record_trade_pnl(gross_pnl)

        logger.info(
            "OrderManager CLOSE [%s] %s x%d @ %.4f | P&L=%.2f | reason=%s | hold=%.0fs",
            trade.symbol, trade.direction.upper(), exit_qty, exit_price,
            gross_pnl, reason, ct.hold_seconds,
        )

        # Fire callbacks outside the lock
        for cb in self._on_trade_closed:
            try:
                cb(ct)
            except Exception as exc:
                logger.warning("OrderManager: trade-closed callback error: %s", exc)

        return ct

    def _emergency_close(
        self,
        symbol: str,
        shares: int,
        direction: str,
        fill_price: float,
        signal: SignalResult,
        entry_order_id: str,
        reason: str,
    ) -> None:
        """
        Immediately close a position that should not have been opened.
        Used when slippage check fails post-fill.
        """
        close_side = OrderSide.SELL if direction == "long" else OrderSide.BUY
        try:
            close_result = self._broker.place_order(Order(
                symbol=symbol,
                side=close_side,
                quantity=shares,
                order_type=OrderType.MARKET,
            ))
            exit_price = close_result.avg_fill_price or fill_price
        except BrokerError as exc:
            logger.critical(
                "OrderManager [%s]: EMERGENCY CLOSE FAILED — %s. "
                "Manual intervention required!", symbol, exc,
            )
            return

        pnl = (exit_price - fill_price) * shares * (1 if direction == "long" else -1)
        self._risk.record_trade_pnl(pnl)

        logger.warning(
            "OrderManager [%s]: emergency close | entry=%.4f exit=%.4f P&L=%.2f reason=%s",
            symbol, fill_price, exit_price, pnl, reason,
        )

    def _place_broker_stop(self, trade: OpenTrade) -> Optional[str]:
        """
        Place a broker-side stop-loss order (live mode belt-and-suspenders).
        Returns the order_id or None on failure.
        """
        stop_side = OrderSide.SELL if trade.direction == "long" else OrderSide.BUY
        stop_order = Order(
            symbol=trade.symbol,
            side=stop_side,
            quantity=trade.shares,
            order_type=OrderType.STOP,
            stop_price=trade.stop_price,
            time_in_force=settings.execution.default_time_in_force,
        )
        try:
            result = self._broker.place_order(stop_order)
            logger.info(
                "OrderManager [%s]: broker stop placed @ %.4f | id=%s",
                trade.symbol, trade.stop_price, result.order_id[:8],
            )
            return result.order_id
        except BrokerError as exc:
            # Non-fatal — monitor loop is the primary stop mechanism
            logger.warning(
                "OrderManager [%s]: could not place broker stop — %s. "
                "Monitor loop will handle stop-loss.",
                trade.symbol, exc,
            )
            return None
