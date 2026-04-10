"""
broker/paper.py — Paper trading broker implementation.

Simulates order fills against live quotes from yfinance without sending any
real orders.  This is the default mode and is always safe to run.

Fill model:
  - Market BUY  → fills at ask price
  - Market SELL → fills at bid price
  - Limit BUY   → fills immediately if ask ≤ limit_price, else PENDING
  - Limit SELL  → fills immediately if bid ≥ limit_price, else PENDING
  - Stop orders are treated as market orders once triggered (simplified)

State is in-memory only — resets on restart.  The trade log (Stage 8) is
the durable record.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime
from threading import Lock
from typing import Dict, List, Optional

import pytz
import yfinance as yf

from config.settings import UNIVERSE, settings
from broker.base import (
    AccountInfo,
    BrokerError,
    BrokerInterface,
    InsufficientFundsError,
    Order,
    OrderResult,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
    Quote,
    SymbolNotInUniverseError,
)

logger = logging.getLogger(__name__)

ET = pytz.timezone("America/New_York")


class PaperBroker(BrokerInterface):
    """
    In-memory paper trading broker.

    Thread-safe via a single mutex — the position monitor and order manager
    may call this from separate threads.
    """

    def __init__(self, starting_capital: Optional[float] = None) -> None:
        self._cash: float = (
            starting_capital
            if starting_capital is not None
            else settings.paper_starting_capital
        )
        # symbol → Position
        self._positions: Dict[str, Position] = {}
        # order_id → OrderResult
        self._orders: Dict[str, OrderResult] = {}
        # Rolling day-trade tracker: list of dates a day trade was recorded
        self._day_trade_dates: List[datetime] = []
        self._lock = Lock()

        logger.info(
            "PaperBroker initialised | starting_capital=$%.2f", self._cash
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fetch_quote_yf(self, symbol: str) -> Quote:
        """
        Pull a real-time quote from yfinance.

        yfinance `fast_info` returns a lightweight snapshot without pulling
        full history, which is appropriate for order simulation.
        """
        try:
            info = yf.Ticker(symbol).fast_info
            lp = getattr(info, "last_price", None)
            pc = getattr(info, "previous_close", None)
            op = getattr(info, "open", None)
            last = float(lp if lp is not None else pc if pc is not None else op if op is not None else 0.0)
            bid = last * 0.9995
            ask = last * 1.0005
            volume = int(getattr(info, "three_month_average_volume", None) or 0)
            return Quote(
                symbol=symbol,
                bid=bid,
                ask=ask,
                last=last,
                volume=volume,
                timestamp=datetime.utcnow(),
            )
        except Exception as exc:
            # Fail loudly — never silently proceed with a bad quote
            raise BrokerError(
                f"PaperBroker: failed to fetch quote for {symbol}: {exc}"
            ) from exc

    def _validate_symbol(self, symbol: str) -> None:
        if symbol not in UNIVERSE:
            raise SymbolNotInUniverseError(
                f"{symbol} is not in the trading universe: {UNIVERSE}"
            )

    def _simulate_fill(self, order: Order, quote: Quote) -> tuple[float, OrderStatus]:
        """
        Determine fill price and status for the given order + quote.

        Returns (avg_fill_price, status).
        """
        if order.order_type == OrderType.MARKET:
            price = quote.ask if order.side == OrderSide.BUY else quote.bid
            return price, OrderStatus.FILLED

        if order.order_type == OrderType.LIMIT:
            if order.limit_price is None:
                raise BrokerError("Limit order submitted without limit_price")
            if order.side == OrderSide.BUY and quote.ask <= order.limit_price:
                return quote.ask, OrderStatus.FILLED
            if order.side == OrderSide.SELL and quote.bid >= order.limit_price:
                return quote.bid, OrderStatus.FILLED
            # Price not yet reached — leave as pending
            return order.limit_price, OrderStatus.PENDING

        if order.order_type in (OrderType.STOP, OrderType.STOP_LIMIT):
            # Simplified: treat as market order (stop already triggered by caller)
            price = quote.ask if order.side == OrderSide.BUY else quote.bid
            return price, OrderStatus.FILLED

        raise BrokerError(f"Unsupported order type in paper mode: {order.order_type}")

    def _apply_fill(self, order: Order, fill_price: float) -> None:
        """
        Update internal cash and position state after a confirmed fill.
        Must be called while _lock is held.
        """
        cost = fill_price * order.quantity

        if order.side == OrderSide.BUY:
            if cost > self._cash:
                raise InsufficientFundsError(
                    f"Insufficient funds: need ${cost:.2f}, have ${self._cash:.2f}"
                )
            self._cash -= cost
            pos = self._positions.get(order.symbol)
            if pos is None:
                self._positions[order.symbol] = Position(
                    symbol=order.symbol,
                    quantity=order.quantity,
                    avg_cost=fill_price,
                    current_price=fill_price,
                )
            else:
                # Weighted average cost basis
                total_qty = pos.quantity + order.quantity
                pos.avg_cost = (
                    (pos.avg_cost * pos.quantity + fill_price * order.quantity)
                    / total_qty
                )
                pos.quantity = total_qty
                pos.current_price = fill_price

        else:  # SELL
            pos = self._positions.get(order.symbol)
            if pos is not None and pos.quantity > 0:
                # Closing an existing long position
                self._cash += fill_price * order.quantity
                pos.quantity -= order.quantity
                pos.current_price = fill_price
                if pos.quantity == 0:
                    self._day_trade_dates.append(datetime.now(ET))
                    del self._positions[order.symbol]
            else:
                # Opening a short position (negative quantity)
                if pos is None:
                    self._positions[order.symbol] = type('Position', (), {
                        'symbol': order.symbol,
                        'quantity': -order.quantity,
                        'avg_cost': fill_price,
                        'current_price': fill_price,
                    })()
                else:
                    pos.quantity -= order.quantity
                    pos.current_price = fill_price
                # Credit cash (short proceeds)
                self._cash += fill_price * order.quantity

    # ------------------------------------------------------------------
    # BrokerInterface implementation
    # ------------------------------------------------------------------

    def get_quote(self, symbol: str) -> Quote:
        self._validate_symbol(symbol)
        return self._fetch_quote_yf(symbol)

    def get_account(self) -> AccountInfo:
        with self._lock:
            portfolio_value = sum(
                p.quantity * p.current_price for p in self._positions.values()
            )
            # Count day trades in the rolling 5-business-day window
            now_et = datetime.now(ET)
            five_bdays_ago = self._five_business_days_ago(now_et)
            day_trades_used = sum(
                1 for dt in self._day_trade_dates if dt >= five_bdays_ago
            )
            return AccountInfo(
                buying_power=self._cash,
                portfolio_value=self._cash + portfolio_value,
                cash=self._cash,
                day_trades_used=day_trades_used,
                timestamp=datetime.utcnow(),
            )

    def place_order(self, order: Order) -> OrderResult:
        self._validate_symbol(order.symbol)

        if order.quantity <= 0:
            raise BrokerError(f"Order quantity must be > 0, got {order.quantity}")

        # Fetch quote outside the lock — network I/O should not block readers
        quote = self._fetch_quote_yf(order.symbol)

        with self._lock:
            fill_price, status = self._simulate_fill(order, quote)

            order_id = str(uuid.uuid4())
            now = datetime.utcnow()

            result = OrderResult(
                order_id=order_id,
                symbol=order.symbol,
                side=order.side,
                quantity=order.quantity,
                filled_quantity=order.quantity if status == OrderStatus.FILLED else 0,
                avg_fill_price=fill_price if status == OrderStatus.FILLED else None,
                status=status,
                created_at=now,
                updated_at=now,
                raw={"mode": "paper", "simulated_quote": {
                    "bid": quote.bid, "ask": quote.ask, "last": quote.last
                }},
            )

            if status == OrderStatus.FILLED:
                # _apply_fill may raise — let it propagate so caller knows the
                # order did NOT actually execute
                self._apply_fill(order, fill_price)

            self._orders[order_id] = result

            logger.info(
                "PaperBroker ORDER | %s %s x%d @ $%.4f | status=%s | id=%s",
                order.side.value.upper(),
                order.symbol,
                order.quantity,
                fill_price,
                status.value,
                order_id,
            )
            return result

    def cancel_order(self, order_id: str) -> bool:
        with self._lock:
            result = self._orders.get(order_id)
            if result is None:
                raise BrokerError(f"Unknown order_id: {order_id}")
            if result.is_complete:
                # Already in a terminal state — nothing to cancel
                logger.warning(
                    "PaperBroker: cancel requested for terminal order %s (status=%s)",
                    order_id,
                    result.status.value,
                )
                return False
            result.status = OrderStatus.CANCELLED
            result.updated_at = datetime.utcnow()
            logger.info("PaperBroker: cancelled order %s", order_id)
            return True

    def get_open_positions(self) -> list[Position]:
        with self._lock:
            # Return copies to prevent external mutation of internal state
            return list(self._positions.values())

    def get_order_status(self, order_id: str) -> OrderResult:
        with self._lock:
            result = self._orders.get(order_id)
            if result is None:
                raise BrokerError(f"Unknown order_id: {order_id}")
            return result

    def is_market_open(self) -> bool:
        """
        Returns True if current ET time is within regular trading hours:
        09:30–16:00, Monday–Friday.

        Does not account for market holidays — the Robinhood implementation
        queries the broker API for that.  Paper mode is conservative: it uses
        only clock checks.
        """
        now_et = datetime.now(ET)
        if now_et.weekday() >= 5:  # Saturday=5, Sunday=6
            return False
        market_open = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = now_et.replace(hour=16, minute=0, second=0, microsecond=0)
        return market_open <= now_et < market_close

    # ------------------------------------------------------------------
    # Utility — accessible to tests
    # ------------------------------------------------------------------

    def reset(self, starting_capital: Optional[float] = None) -> None:
        """Reset all state.  Useful in tests."""
        with self._lock:
            self._cash = (
                starting_capital
                if starting_capital is not None
                else settings.paper_starting_capital
            )
            self._positions.clear()
            self._orders.clear()
            self._day_trade_dates.clear()
        logger.info("PaperBroker reset | capital=$%.2f", self._cash)

    @staticmethod
    def _five_business_days_ago(now: datetime) -> datetime:
        """Return the datetime 5 business days before *now* (ET)."""
        import datetime as dt_mod
        count = 0
        d = now.date()
        while count < 5:
            d -= dt_mod.timedelta(days=1)
            if d.weekday() < 5:  # Mon–Fri
                count += 1
        return ET.localize(datetime.combine(d, datetime.min.time()))
