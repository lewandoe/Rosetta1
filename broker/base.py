"""
broker/base.py — Abstract BrokerInterface and shared data models.

All broker implementations (Robinhood, Paper) must subclass BrokerInterface.
Signal/strategy/risk code imports ONLY from this file — never from a concrete
broker implementation.  This enforces the clean interface boundary.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class OrderSide(str, Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class TimeInForce(str, Enum):
    GFD = "gfd"   # good-for-day (default for intraday)
    GTC = "gtc"   # good-till-cancelled
    IOC = "ioc"   # immediate-or-cancel
    FOK = "fok"   # fill-or-kill


class OrderStatus(str, Enum):
    PENDING = "pending"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    FAILED = "failed"


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class Quote:
    """Best bid/ask snapshot for a single symbol."""
    symbol: str
    bid: float
    ask: float
    last: float
    volume: int
    timestamp: datetime = field(default_factory=datetime.utcnow)

    @property
    def mid(self) -> float:
        return (self.bid + self.ask) / 2.0

    @property
    def spread_pct(self) -> float:
        """Bid-ask spread as a fraction of mid price.  Used by spread filter."""
        if self.mid == 0:
            return float("inf")
        return (self.ask - self.bid) / self.mid


@dataclass
class AccountInfo:
    """Snapshot of account state needed by risk and execution layers."""
    buying_power: float
    portfolio_value: float
    cash: float
    # Number of day trades used in the rolling 5-business-day window
    day_trades_used: int
    # Timestamp of this snapshot
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class Order:
    """Input spec for placing an order."""
    symbol: str
    side: OrderSide
    quantity: int
    order_type: OrderType = OrderType.MARKET
    limit_price: Optional[float] = None
    stop_price: Optional[float] = None
    time_in_force: TimeInForce = TimeInForce.GFD
    # Caller-supplied reference (e.g. signal UUID) for correlation in logs
    client_order_id: Optional[str] = None


@dataclass
class OrderResult:
    """Result returned after placing (or querying) an order."""
    order_id: str
    symbol: str
    side: OrderSide
    quantity: int
    filled_quantity: int
    avg_fill_price: Optional[float]
    status: OrderStatus
    created_at: datetime
    updated_at: datetime
    # Raw broker-specific payload preserved for debugging
    raw: dict = field(default_factory=dict)

    @property
    def is_complete(self) -> bool:
        return self.status in (OrderStatus.FILLED, OrderStatus.CANCELLED,
                               OrderStatus.REJECTED, OrderStatus.FAILED)

    @property
    def slippage_from(self) -> float:
        """
        Return |avg_fill - reference_price| / reference_price.
        Caller must supply reference_price; this is a helper for the risk guard.
        """
        raise NotImplementedError("Call slippage_pct(reference_price) instead")

    def slippage_pct(self, reference_price: float) -> float:
        """Fractional slippage relative to a reference (signal) price."""
        if not self.avg_fill_price or reference_price == 0:
            return 0.0
        return abs(self.avg_fill_price - reference_price) / reference_price


@dataclass
class Position:
    """Open position held in the account."""
    symbol: str
    quantity: int           # positive = long, negative = short
    avg_cost: float         # average cost basis per share
    current_price: float
    timestamp: datetime = field(default_factory=datetime.utcnow)

    @property
    def market_value(self) -> float:
        return self.quantity * self.current_price

    @property
    def unrealized_pnl(self) -> float:
        return self.quantity * (self.current_price - self.avg_cost)

    @property
    def unrealized_pnl_pct(self) -> float:
        if self.avg_cost == 0:
            return 0.0
        return (self.current_price - self.avg_cost) / self.avg_cost

    @property
    def side(self) -> str:
        return "long" if self.quantity > 0 else "short"


# ---------------------------------------------------------------------------
# Abstract interface
# ---------------------------------------------------------------------------

class BrokerInterface(abc.ABC):
    """
    Abstract base class every broker implementation must satisfy.

    Concrete implementations: broker/robinhood.py, broker/paper.py
    Consumers: execution/, risk/ — they import BrokerInterface only.

    Design contract:
    - All methods raise BrokerError (or subclass) on unrecoverable failures.
    - Never return None where a typed value is expected — raise instead.
    - Implementations are responsible for their own retry logic on transient
      network errors, up to settings.execution.max_order_retries attempts.
    """

    # ------------------------------------------------------------------
    # Market data
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def get_quote(self, symbol: str) -> Quote:
        """
        Return the latest bid/ask/last quote for *symbol*.

        Raises:
            BrokerError: if the symbol is invalid or the API call fails.
        """

    # ------------------------------------------------------------------
    # Account
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def get_account(self) -> AccountInfo:
        """
        Return current account state (buying power, cash, day-trade count).

        Raises:
            BrokerError: on API failure.
        """

    # ------------------------------------------------------------------
    # Orders
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def place_order(self, order: Order) -> OrderResult:
        """
        Submit *order* to the broker.

        Implementations MUST:
        - Validate that trading_mode == "live" before sending a real order.
        - Log the order attempt before submission.
        - Return an OrderResult even for rejected orders (status=REJECTED).

        Raises:
            BrokerError: on network failure or unexpected broker response.
            TradingModeError: if live order attempted in paper mode.
        """

    @abc.abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """
        Request cancellation of *order_id*.

        Returns:
            True if the cancellation was accepted, False if already terminal.

        Raises:
            BrokerError: on API failure.
        """

    # ------------------------------------------------------------------
    # Positions
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def get_open_positions(self) -> list[Position]:
        """
        Return all currently open positions.

        Raises:
            BrokerError: on API failure.
        """

    # ------------------------------------------------------------------
    # Order lifecycle
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def get_order_status(self, order_id: str) -> OrderResult:
        """
        Fetch the latest state of *order_id*.

        Raises:
            BrokerError: if order_id is unknown or API call fails.
        """

    # ------------------------------------------------------------------
    # Market hours
    # ------------------------------------------------------------------

    @abc.abstractmethod
    def is_market_open(self) -> bool:
        """
        Return True if the US equity market is currently open (9:30–16:00 ET,
        weekdays only, no holiday logic unless the broker API provides it).
        """


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class BrokerError(Exception):
    """Raised for all unrecoverable broker API errors.  Never silently caught."""


class TradingModeError(BrokerError):
    """Raised when a live order is attempted while in paper-trading mode."""


class InsufficientFundsError(BrokerError):
    """Raised when an order exceeds available buying power."""


class SymbolNotInUniverseError(BrokerError):
    """Raised when an order targets a symbol outside the allowed universe."""
