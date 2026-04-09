"""
tests/test_broker.py — Tests for the broker layer.

Tests PaperBroker exhaustively (no network required — yfinance calls are
mocked).  Tests RobinhoodBroker interface contract and safety checks only
(no real API calls — robin_stocks is fully mocked).

Run:
    pytest tests/test_broker.py -v
"""

from __future__ import annotations

import uuid
from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from broker.base import (
    AccountInfo,
    BrokerError,
    InsufficientFundsError,
    Order,
    OrderResult,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
    Quote,
    SymbolNotInUniverseError,
    TimeInForce,
    TradingModeError,
)
from broker.paper import PaperBroker


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_quote(symbol: str, bid: float = 99.90, ask: float = 100.10,
                last: float = 100.00) -> Quote:
    return Quote(symbol=symbol, bid=bid, ask=ask, last=last, volume=1_000_000)


@pytest.fixture
def broker() -> PaperBroker:
    """Fresh PaperBroker with $10,000 starting capital."""
    return PaperBroker(starting_capital=10_000.0)


@pytest.fixture
def mock_quote(broker: PaperBroker):
    """Patch _fetch_quote_yf so tests never hit the network."""
    with patch.object(
        broker, "_fetch_quote_yf",
        side_effect=lambda symbol: _make_quote(symbol)
    ):
        yield


# ---------------------------------------------------------------------------
# Quote
# ---------------------------------------------------------------------------

class TestGetQuote:
    def test_rejects_symbol_outside_universe(self, broker: PaperBroker):
        with pytest.raises(SymbolNotInUniverseError):
            broker.get_quote("FAKE")

    def test_returns_quote_for_valid_symbol(self, broker: PaperBroker):
        with patch.object(broker, "_fetch_quote_yf", return_value=_make_quote("SPY")):
            q = broker.get_quote("SPY")
        assert q.symbol == "SPY"
        assert q.bid < q.ask

    def test_spread_pct_calculation(self):
        q = _make_quote("AAPL", bid=199.80, ask=200.20, last=200.00)
        # spread = 0.40, mid = 200.0 → 0.2%
        assert abs(q.spread_pct - 0.002) < 1e-6

    def test_mid_price(self):
        q = _make_quote("TSLA", bid=250.00, ask=250.50)
        assert q.mid == pytest.approx(250.25)


# ---------------------------------------------------------------------------
# Account
# ---------------------------------------------------------------------------

class TestGetAccount:
    def test_initial_account_state(self, broker: PaperBroker):
        acct = broker.get_account()
        assert acct.cash == pytest.approx(10_000.0)
        assert acct.buying_power == pytest.approx(10_000.0)
        assert acct.portfolio_value == pytest.approx(10_000.0)
        assert acct.day_trades_used == 0

    def test_portfolio_value_reflects_open_position(self, broker: PaperBroker, mock_quote):
        # Buy 10 shares @ ask=$100.10 → cash: $9,000 - wait, let's check fill
        # Market buy fills at ask=100.10 for 10 shares → cost=$1001.00
        order = Order(symbol="SPY", side=OrderSide.BUY, quantity=10)
        broker.place_order(order)
        acct = broker.get_account()
        # cash reduced, portfolio_value = cash + (10 * 100.00 last)
        assert acct.cash == pytest.approx(10_000.0 - 10 * 100.10)


# ---------------------------------------------------------------------------
# Place order — market orders
# ---------------------------------------------------------------------------

class TestPlaceOrderMarket:
    def test_market_buy_fills_at_ask(self, broker: PaperBroker, mock_quote):
        order = Order(symbol="AAPL", side=OrderSide.BUY, quantity=5)
        result = broker.place_order(order)
        assert result.status == OrderStatus.FILLED
        assert result.avg_fill_price == pytest.approx(100.10)  # ask
        assert result.filled_quantity == 5

    def test_market_sell_fills_at_bid(self, broker: PaperBroker, mock_quote):
        # First buy so we have a position to sell
        broker.place_order(Order(symbol="AAPL", side=OrderSide.BUY, quantity=5))
        result = broker.place_order(Order(symbol="AAPL", side=OrderSide.SELL, quantity=5))
        assert result.status == OrderStatus.FILLED
        assert result.avg_fill_price == pytest.approx(99.90)  # bid

    def test_buy_reduces_cash(self, broker: PaperBroker, mock_quote):
        broker.place_order(Order(symbol="SPY", side=OrderSide.BUY, quantity=10))
        acct = broker.get_account()
        assert acct.cash == pytest.approx(10_000.0 - 10 * 100.10)

    def test_sell_increases_cash(self, broker: PaperBroker, mock_quote):
        broker.place_order(Order(symbol="SPY", side=OrderSide.BUY, quantity=10))
        cash_after_buy = broker.get_account().cash
        broker.place_order(Order(symbol="SPY", side=OrderSide.SELL, quantity=10))
        acct = broker.get_account()
        # Sell fills at bid=99.90
        assert acct.cash == pytest.approx(cash_after_buy + 10 * 99.90)

    def test_rejects_symbol_outside_universe(self, broker: PaperBroker):
        with pytest.raises(SymbolNotInUniverseError):
            broker.place_order(Order(symbol="GME", side=OrderSide.BUY, quantity=1))

    def test_rejects_zero_quantity(self, broker: PaperBroker):
        with pytest.raises(BrokerError):
            broker.place_order(Order(symbol="SPY", side=OrderSide.BUY, quantity=0))

    def test_insufficient_funds_raises(self, broker: PaperBroker, mock_quote):
        with pytest.raises(InsufficientFundsError):
            broker.place_order(Order(symbol="SPY", side=OrderSide.BUY, quantity=200))

    def test_sell_without_position_raises(self, broker: PaperBroker, mock_quote):
        with pytest.raises(BrokerError):
            broker.place_order(Order(symbol="AAPL", side=OrderSide.SELL, quantity=1))

    def test_sell_more_than_held_raises(self, broker: PaperBroker, mock_quote):
        broker.place_order(Order(symbol="AAPL", side=OrderSide.BUY, quantity=5))
        with pytest.raises(BrokerError):
            broker.place_order(Order(symbol="AAPL", side=OrderSide.SELL, quantity=10))


# ---------------------------------------------------------------------------
# Place order — limit orders
# ---------------------------------------------------------------------------

class TestPlaceOrderLimit:
    def test_limit_buy_fills_when_ask_le_limit(self, broker: PaperBroker, mock_quote):
        # ask=100.10, limit=101.00 → fills
        order = Order(
            symbol="SPY", side=OrderSide.BUY, quantity=1,
            order_type=OrderType.LIMIT, limit_price=101.00
        )
        result = broker.place_order(order)
        assert result.status == OrderStatus.FILLED

    def test_limit_buy_stays_pending_when_ask_gt_limit(self, broker: PaperBroker, mock_quote):
        # ask=100.10, limit=99.00 → not filled yet
        order = Order(
            symbol="SPY", side=OrderSide.BUY, quantity=1,
            order_type=OrderType.LIMIT, limit_price=99.00
        )
        result = broker.place_order(order)
        assert result.status == OrderStatus.PENDING

    def test_limit_sell_fills_when_bid_ge_limit(self, broker: PaperBroker, mock_quote):
        broker.place_order(Order(symbol="SPY", side=OrderSide.BUY, quantity=1))
        order = Order(
            symbol="SPY", side=OrderSide.SELL, quantity=1,
            order_type=OrderType.LIMIT, limit_price=99.00  # bid=99.90 ≥ 99.00
        )
        result = broker.place_order(order)
        assert result.status == OrderStatus.FILLED

    def test_limit_sell_stays_pending_when_bid_lt_limit(self, broker: PaperBroker, mock_quote):
        broker.place_order(Order(symbol="SPY", side=OrderSide.BUY, quantity=1))
        order = Order(
            symbol="SPY", side=OrderSide.SELL, quantity=1,
            order_type=OrderType.LIMIT, limit_price=110.00  # bid=99.90 < 110.00
        )
        result = broker.place_order(order)
        assert result.status == OrderStatus.PENDING

    def test_limit_order_without_price_raises(self, broker: PaperBroker, mock_quote):
        with pytest.raises(BrokerError):
            broker.place_order(Order(
                symbol="SPY", side=OrderSide.BUY, quantity=1,
                order_type=OrderType.LIMIT, limit_price=None
            ))


# ---------------------------------------------------------------------------
# Cancel order
# ---------------------------------------------------------------------------

class TestCancelOrder:
    def test_cancel_pending_order(self, broker: PaperBroker, mock_quote):
        result = broker.place_order(Order(
            symbol="SPY", side=OrderSide.BUY, quantity=1,
            order_type=OrderType.LIMIT, limit_price=50.00  # will be PENDING
        ))
        assert result.status == OrderStatus.PENDING
        cancelled = broker.cancel_order(result.order_id)
        assert cancelled is True
        status = broker.get_order_status(result.order_id)
        assert status.status == OrderStatus.CANCELLED

    def test_cancel_filled_order_returns_false(self, broker: PaperBroker, mock_quote):
        result = broker.place_order(Order(symbol="SPY", side=OrderSide.BUY, quantity=1))
        assert result.status == OrderStatus.FILLED
        cancelled = broker.cancel_order(result.order_id)
        assert cancelled is False

    def test_cancel_unknown_order_raises(self, broker: PaperBroker):
        with pytest.raises(BrokerError):
            broker.cancel_order(str(uuid.uuid4()))


# ---------------------------------------------------------------------------
# Open positions
# ---------------------------------------------------------------------------

class TestGetOpenPositions:
    def test_empty_initially(self, broker: PaperBroker):
        assert broker.get_open_positions() == []

    def test_position_added_after_buy(self, broker: PaperBroker, mock_quote):
        broker.place_order(Order(symbol="NVDA", side=OrderSide.BUY, quantity=3))
        positions = broker.get_open_positions()
        assert len(positions) == 1
        assert positions[0].symbol == "NVDA"
        assert positions[0].quantity == 3

    def test_position_removed_after_full_sell(self, broker: PaperBroker, mock_quote):
        broker.place_order(Order(symbol="NVDA", side=OrderSide.BUY, quantity=3))
        broker.place_order(Order(symbol="NVDA", side=OrderSide.SELL, quantity=3))
        assert broker.get_open_positions() == []

    def test_position_reduced_after_partial_sell(self, broker: PaperBroker, mock_quote):
        broker.place_order(Order(symbol="NVDA", side=OrderSide.BUY, quantity=5))
        broker.place_order(Order(symbol="NVDA", side=OrderSide.SELL, quantity=2))
        positions = broker.get_open_positions()
        assert positions[0].quantity == 3

    def test_multiple_symbols_tracked_independently(self, broker: PaperBroker, mock_quote):
        broker.place_order(Order(symbol="SPY", side=OrderSide.BUY, quantity=2))
        broker.place_order(Order(symbol="AAPL", side=OrderSide.BUY, quantity=3))
        positions = broker.get_open_positions()
        symbols = {p.symbol for p in positions}
        assert symbols == {"SPY", "AAPL"}

    def test_avg_cost_weighted_correctly(self, broker: PaperBroker):
        """Adding to a position must recompute weighted-average cost."""
        # First buy @ ask=100.10 (qty=2), second buy @ ask=110.10 (qty=3)
        with patch.object(broker, "_fetch_quote_yf",
                          side_effect=lambda s: _make_quote(s, bid=99.90, ask=100.10)):
            broker.place_order(Order(symbol="TSLA", side=OrderSide.BUY, quantity=2))
        with patch.object(broker, "_fetch_quote_yf",
                          side_effect=lambda s: _make_quote(s, bid=109.90, ask=110.10)):
            broker.place_order(Order(symbol="TSLA", side=OrderSide.BUY, quantity=3))

        pos = broker.get_open_positions()[0]
        expected_avg = (2 * 100.10 + 3 * 110.10) / 5
        assert pos.avg_cost == pytest.approx(expected_avg)


# ---------------------------------------------------------------------------
# Position data model
# ---------------------------------------------------------------------------

class TestPositionModel:
    def test_unrealized_pnl(self):
        pos = Position(symbol="SPY", quantity=10, avg_cost=100.0, current_price=110.0)
        assert pos.unrealized_pnl == pytest.approx(100.0)

    def test_unrealized_pnl_pct(self):
        pos = Position(symbol="SPY", quantity=10, avg_cost=100.0, current_price=110.0)
        assert pos.unrealized_pnl_pct == pytest.approx(0.10)

    def test_market_value(self):
        pos = Position(symbol="AAPL", quantity=5, avg_cost=150.0, current_price=160.0)
        assert pos.market_value == pytest.approx(800.0)

    def test_side_long(self):
        pos = Position(symbol="SPY", quantity=10, avg_cost=100.0, current_price=100.0)
        assert pos.side == "long"


# ---------------------------------------------------------------------------
# OrderResult slippage helper
# ---------------------------------------------------------------------------

class TestOrderResultSlippage:
    def test_slippage_pct(self):
        result = OrderResult(
            order_id="x", symbol="SPY", side=OrderSide.BUY,
            quantity=10, filled_quantity=10, avg_fill_price=100.20,
            status=OrderStatus.FILLED,
            created_at=datetime.utcnow(), updated_at=datetime.utcnow(),
        )
        # |100.20 - 100.00| / 100.00 = 0.002
        assert result.slippage_pct(100.00) == pytest.approx(0.002)

    def test_slippage_zero_fill_price(self):
        result = OrderResult(
            order_id="x", symbol="SPY", side=OrderSide.BUY,
            quantity=10, filled_quantity=0, avg_fill_price=None,
            status=OrderStatus.PENDING,
            created_at=datetime.utcnow(), updated_at=datetime.utcnow(),
        )
        assert result.slippage_pct(100.00) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Market hours (paper broker)
# ---------------------------------------------------------------------------

class TestMarketHours:
    def test_closed_on_weekend(self, broker: PaperBroker):
        # Saturday 2024-01-06 10:00 ET
        import pytz
        from unittest.mock import patch as p
        et = pytz.timezone("America/New_York")
        saturday = et.localize(datetime(2024, 1, 6, 10, 0, 0))
        with patch("broker.paper.datetime") as mock_dt:
            mock_dt.now.return_value = saturday
            mock_dt.utcnow.return_value = datetime.utcnow()
            mock_dt.combine = datetime.combine
            mock_dt.min = datetime.min
            assert broker.is_market_open() is False

    def test_closed_before_open(self, broker: PaperBroker):
        import pytz
        et = pytz.timezone("America/New_York")
        early = et.localize(datetime(2024, 1, 8, 9, 0, 0))  # Monday 9:00 AM ET
        with patch("broker.paper.datetime") as mock_dt:
            mock_dt.now.return_value = early
            mock_dt.utcnow.return_value = datetime.utcnow()
            mock_dt.combine = datetime.combine
            mock_dt.min = datetime.min
            assert broker.is_market_open() is False

    def test_open_during_session(self, broker: PaperBroker):
        import pytz
        et = pytz.timezone("America/New_York")
        midday = et.localize(datetime(2024, 1, 8, 12, 0, 0))  # Monday noon ET
        with patch("broker.paper.datetime") as mock_dt:
            mock_dt.now.return_value = midday
            mock_dt.utcnow.return_value = datetime.utcnow()
            mock_dt.combine = datetime.combine
            mock_dt.min = datetime.min
            assert broker.is_market_open() is True


# ---------------------------------------------------------------------------
# Reset utility
# ---------------------------------------------------------------------------

class TestReset:
    def test_reset_clears_positions_and_cash(self, broker: PaperBroker, mock_quote):
        broker.place_order(Order(symbol="SPY", side=OrderSide.BUY, quantity=5))
        broker.reset(starting_capital=5_000.0)
        assert broker.get_account().cash == pytest.approx(5_000.0)
        assert broker.get_open_positions() == []


# ---------------------------------------------------------------------------
# RobinhoodBroker — safety / interface contract (no live API calls)
# ---------------------------------------------------------------------------

class TestRobinhoodBrokerSafety:
    """
    These tests validate safety constraints only.
    robin_stocks is stubbed at the sys.modules level so the package need not
    be installed in the test environment.
    """

    @pytest.fixture(autouse=True)
    def _stub_robin_stocks(self):
        """Inject a fake robin_stocks module before any import of robinhood.py."""
        import sys
        import types

        fake_rh = types.ModuleType("robin_stocks")
        fake_rh.robinhood = types.ModuleType("robin_stocks.robinhood")
        fake_rh.robinhood.login = MagicMock(return_value=None)

        sys.modules.setdefault("robin_stocks", fake_rh)
        sys.modules.setdefault("robin_stocks.robinhood", fake_rh.robinhood)

        # Remove any cached real import so the stub takes effect
        sys.modules.pop("broker.robinhood", None)
        yield
        sys.modules.pop("broker.robinhood", None)

    def _make_broker(self, trading_mode: str = "paper"):
        from broker.robinhood import RobinhoodBroker
        with patch("broker.robinhood.settings") as ms:
            ms.broker.username = "u"
            ms.broker.password = "p"
            ms.broker.mfa_secret = ""
            ms.broker.trading_mode = trading_mode
            ms.execution.max_order_retries = 1
            b = RobinhoodBroker.__new__(RobinhoodBroker)
            b._authenticated = True
            b._settings_trading_mode = trading_mode
            return b, ms

    def test_place_order_raises_in_paper_mode(self):
        from broker.robinhood import RobinhoodBroker
        b = RobinhoodBroker.__new__(RobinhoodBroker)
        b._authenticated = True

        with patch("broker.robinhood.settings") as ms:
            ms.broker.trading_mode = "paper"
            ms.execution.max_order_retries = 1
            with pytest.raises(TradingModeError):
                b.place_order(Order(symbol="SPY", side=OrderSide.BUY, quantity=1))

    def test_place_order_raises_for_invalid_symbol(self):
        from broker.robinhood import RobinhoodBroker
        b = RobinhoodBroker.__new__(RobinhoodBroker)
        b._authenticated = True

        with patch("broker.robinhood.settings") as ms:
            ms.broker.trading_mode = "live"
            ms.execution.max_order_retries = 1
            with pytest.raises(SymbolNotInUniverseError):
                b.place_order(Order(symbol="FAKE", side=OrderSide.BUY, quantity=1))

    def test_requires_auth_raises_when_not_authenticated(self):
        from broker.robinhood import RobinhoodBroker
        b = RobinhoodBroker.__new__(RobinhoodBroker)
        b._authenticated = False

        with pytest.raises(BrokerError, match="not authenticated"):
            b.get_account()
