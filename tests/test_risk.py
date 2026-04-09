"""
tests/test_risk.py — Tests for risk/guard.py.

All broker and signal objects are constructed directly — no network calls.

Run:
    pytest tests/test_risk.py -v
"""
from __future__ import annotations

from datetime import datetime
from unittest.mock import patch

import pytest
import pytz

from broker.base import AccountInfo, Position, Quote
from risk.guard import RiskDecision, RiskGuard, SlippageDecision
from signals.base import SignalResult

ET = pytz.timezone("America/New_York")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _account(
    buying_power: float = 10_000.0,
    cash: float = 10_000.0,
    portfolio_value: float = 10_000.0,
    day_trades_used: int = 0,
) -> AccountInfo:
    return AccountInfo(
        buying_power=buying_power,
        portfolio_value=portfolio_value,
        cash=cash,
        day_trades_used=day_trades_used,
        timestamp=datetime.utcnow(),
    )


def _quote(
    symbol: str = "SPY",
    bid: float = 499.90,
    ask: float = 500.10,
    last: float = 500.00,
) -> Quote:
    return Quote(symbol=symbol, bid=bid, ask=ask, last=last, volume=2_000_000)


def _signal(
    symbol: str = "SPY",
    direction: str = "long",
    entry: float = 500.00,
    target: float = 501.50,
    stop: float = 499.00,
    confidence: int = 80,
    signal_type: str = "momentum",
) -> SignalResult:
    return SignalResult(
        symbol=symbol,
        signal_type=signal_type,
        direction=direction,
        entry_price=entry,
        target_price=target,
        stop_price=stop,
        confidence=confidence,
        estimated_hold_seconds=300,
    )


def _positions(n: int = 0) -> list[Position]:
    return [
        Position(symbol=f"SYM{i}", quantity=10, avg_cost=100.0, current_price=100.0)
        for i in range(n)
    ]


def _market_open_et(hour: int = 11, minute: int = 0) -> datetime:
    """Return a tz-aware ET datetime during market hours."""
    return ET.localize(datetime(2024, 1, 8, hour, minute, 0))


def _guard_with_market_time(hour: int = 11, minute: int = 0) -> RiskGuard:
    """Return a fresh guard; patch datetime.now so market-hours check passes."""
    return RiskGuard()


# ---------------------------------------------------------------------------
# Patch helper — makes datetime.now(ET) return a fixed market-hours time
# ---------------------------------------------------------------------------
PATCH_DT = "risk.guard.datetime"


def _patched_now(hour: int = 11, minute: int = 0):
    """Context manager: patch risk.guard.datetime.now to return market hours."""
    from unittest.mock import MagicMock
    mock_dt = MagicMock()
    mock_dt.now.return_value = ET.localize(datetime(2024, 1, 8, hour, minute, 0))
    return patch(PATCH_DT, mock_dt)


# ===========================================================================
# Approval — all checks passing
# ===========================================================================

class TestApproval:
    def test_approved_with_valid_inputs(self):
        guard = RiskGuard()
        with _patched_now():
            decision = guard.check(
                signal=_signal(),
                quote=_quote(),
                account=_account(),
                open_positions=_positions(0),
                is_market_open=True,
            )
        assert decision.approved is True
        assert decision.position_size > 0
        assert decision.capital_allocated > 0
        assert decision.reason == "ok"

    def test_position_size_respects_capital_limit(self):
        # 10% of $10,000 = $1,000 / $500 = 2 shares
        guard = RiskGuard()
        with _patched_now():
            decision = guard.check(
                signal=_signal(entry=500.0),
                quote=_quote(),
                account=_account(buying_power=10_000.0),
                open_positions=[],
                is_market_open=True,
            )
        assert decision.approved is True
        assert decision.position_size == 2
        assert decision.capital_allocated == pytest.approx(1000.0)

    def test_all_checks_in_passed_list(self):
        guard = RiskGuard()
        with _patched_now():
            decision = guard.check(
                signal=_signal(),
                quote=_quote(),
                account=_account(),
                open_positions=[],
                is_market_open=True,
            )
        expected_checks = [
            "emergency_halt", "market_hours", "daily_loss_limit",
            "max_positions", "pdt_limit", "spread_filter",
            "capital_limit", "risk_reward", "stop_distance",
        ]
        for c in expected_checks:
            assert c in decision.checks_passed, f"Expected '{c}' in checks_passed"
        assert decision.checks_failed == []


# ===========================================================================
# Emergency halt
# ===========================================================================

class TestEmergencyHalt:
    def test_halt_blocks_all_orders(self):
        guard = RiskGuard()
        guard.halt("test halt")
        with _patched_now():
            decision = guard.check(
                signal=_signal(), quote=_quote(),
                account=_account(), open_positions=[],
                is_market_open=True,
            )
        assert decision.approved is False
        assert "emergency_halt" in decision.checks_failed

    def test_resume_unblocks_orders(self):
        guard = RiskGuard()
        guard.halt("test halt")
        guard.resume()
        with _patched_now():
            decision = guard.check(
                signal=_signal(), quote=_quote(),
                account=_account(), open_positions=[],
                is_market_open=True,
            )
        assert decision.approved is True

    def test_is_halted_property(self):
        guard = RiskGuard()
        assert guard.is_halted is False
        guard.halt("x")
        assert guard.is_halted is True
        guard.resume()
        assert guard.is_halted is False

    def test_halt_reason_stored(self):
        guard = RiskGuard()
        guard.halt("specific reason")
        assert guard.halt_reason == "specific reason"


# ===========================================================================
# Market hours
# ===========================================================================

class TestMarketHours:
    def test_rejected_when_market_closed(self):
        guard = RiskGuard()
        decision = guard.check(
            signal=_signal(), quote=_quote(),
            account=_account(), open_positions=[],
            is_market_open=False,
        )
        assert decision.approved is False
        assert "market_hours" in decision.checks_failed

    def test_rejected_after_eod_cutoff(self):
        guard = RiskGuard()
        # 3:46 PM ET — past the 3:45 cutoff
        with _patched_now(hour=15, minute=46):
            decision = guard.check(
                signal=_signal(), quote=_quote(),
                account=_account(), open_positions=[],
                is_market_open=True,
            )
        assert decision.approved is False
        assert "market_hours" in decision.checks_failed

    def test_approved_just_before_eod_cutoff(self):
        guard = RiskGuard()
        # 3:44 PM ET — just before cutoff
        with _patched_now(hour=15, minute=44):
            decision = guard.check(
                signal=_signal(), quote=_quote(),
                account=_account(), open_positions=[],
                is_market_open=True,
            )
        assert decision.approved is True


# ===========================================================================
# Daily loss limit
# ===========================================================================

class TestDailyLossLimit:
    def test_rejected_when_limit_hit(self):
        guard = RiskGuard()
        # -200 triggers auto-halt, so check() fails at emergency_halt;
        # either way the trade is blocked — assert approved=False.
        guard.record_trade_pnl(-200.0)
        with _patched_now():
            decision = guard.check(
                signal=_signal(), quote=_quote(),
                account=_account(), open_positions=[],
                is_market_open=True,
            )
        assert decision.approved is False
        assert guard.is_halted is True

    def test_rejected_when_limit_exceeded(self):
        guard = RiskGuard()
        guard.record_trade_pnl(-250.0)
        with _patched_now():
            decision = guard.check(
                signal=_signal(), quote=_quote(),
                account=_account(), open_positions=[],
                is_market_open=True,
            )
        assert decision.approved is False

    def test_approved_when_under_limit(self):
        guard = RiskGuard()
        guard.record_trade_pnl(-100.0)   # only half the limit used
        with _patched_now():
            decision = guard.check(
                signal=_signal(), quote=_quote(),
                account=_account(), open_positions=[],
                is_market_open=True,
            )
        assert decision.approved is True

    def test_record_trade_pnl_accumulates(self):
        guard = RiskGuard()
        guard.record_trade_pnl(-80.0)
        guard.record_trade_pnl(-80.0)
        assert guard.get_daily_pnl() == pytest.approx(-160.0)

    def test_record_trade_pnl_auto_halts(self):
        guard = RiskGuard()
        guard.record_trade_pnl(-201.0)
        assert guard.is_halted is True

    def test_reset_daily_pnl(self):
        guard = RiskGuard()
        guard.record_trade_pnl(-150.0)
        guard.reset_daily_pnl()
        assert guard.get_daily_pnl() == pytest.approx(0.0)

    def test_daily_loss_remaining(self):
        guard = RiskGuard()
        guard.record_trade_pnl(-75.0)
        assert guard.daily_loss_remaining() == pytest.approx(125.0)


# ===========================================================================
# Max open positions
# ===========================================================================

class TestMaxPositions:
    def test_rejected_at_max_positions(self):
        guard = RiskGuard()
        with _patched_now():
            decision = guard.check(
                signal=_signal(), quote=_quote(),
                account=_account(),
                open_positions=_positions(3),   # 3 = max
                is_market_open=True,
            )
        assert decision.approved is False
        assert "max_positions" in decision.checks_failed

    def test_approved_below_max_positions(self):
        guard = RiskGuard()
        with _patched_now():
            decision = guard.check(
                signal=_signal(), quote=_quote(),
                account=_account(),
                open_positions=_positions(2),   # 2 < 3
                is_market_open=True,
            )
        assert decision.approved is True


# ===========================================================================
# PDT limit
# ===========================================================================

class TestPdtLimit:
    def test_rejected_at_pdt_limit(self):
        guard = RiskGuard()
        with _patched_now():
            decision = guard.check(
                signal=_signal(), quote=_quote(),
                account=_account(day_trades_used=3),   # at limit
                open_positions=[],
                is_market_open=True,
            )
        assert decision.approved is False
        assert "pdt_limit" in decision.checks_failed

    def test_approved_under_pdt_limit(self):
        guard = RiskGuard()
        with _patched_now():
            decision = guard.check(
                signal=_signal(), quote=_quote(),
                account=_account(day_trades_used=2),
                open_positions=[],
                is_market_open=True,
            )
        assert decision.approved is True


# ===========================================================================
# Spread filter
# ===========================================================================

class TestSpreadFilter:
    def test_rejected_when_spread_too_wide(self):
        guard = RiskGuard()
        # Spread = (ask - bid) / mid = 2.0 / 500.0 = 0.40% > 0.20% limit
        wide_quote = _quote(bid=499.0, ask=501.0)
        with _patched_now():
            decision = guard.check(
                signal=_signal(), quote=wide_quote,
                account=_account(), open_positions=[],
                is_market_open=True,
            )
        assert decision.approved is False
        assert "spread_filter" in decision.checks_failed

    def test_approved_when_spread_tight(self):
        guard = RiskGuard()
        tight_quote = _quote(bid=499.95, ask=500.05)   # 0.02%
        with _patched_now():
            decision = guard.check(
                signal=_signal(), quote=tight_quote,
                account=_account(), open_positions=[],
                is_market_open=True,
            )
        assert decision.approved is True


# ===========================================================================
# Capital limit + position sizing
# ===========================================================================

class TestCapitalLimit:
    def test_rejected_when_price_too_high_for_buying_power(self):
        guard = RiskGuard()
        # $100 buying power, entry = $500 → 0 shares affordable
        with _patched_now():
            decision = guard.check(
                signal=_signal(entry=500.0),
                quote=_quote(),
                account=_account(buying_power=100.0),
                open_positions=[],
                is_market_open=True,
            )
        assert decision.approved is False
        assert "capital_limit" in decision.checks_failed

    def test_position_size_floors_to_whole_shares(self):
        guard = RiskGuard()
        # 10% of $1,050 = $105 / $50 = 2.1 → floor = 2 shares
        with _patched_now():
            decision = guard.check(
                signal=_signal(entry=50.0, target=50.75, stop=49.50),
                quote=_quote(bid=49.95, ask=50.05, last=50.0),
                account=_account(buying_power=1_050.0),
                open_positions=[],
                is_market_open=True,
            )
        assert decision.approved is True
        assert decision.position_size == 2

    def test_capital_allocated_equals_shares_times_entry(self):
        guard = RiskGuard()
        with _patched_now():
            decision = guard.check(
                signal=_signal(entry=50.0, target=50.75, stop=49.50),
                quote=_quote(bid=49.95, ask=50.05, last=50.0),
                account=_account(buying_power=1_050.0),
                open_positions=[],
                is_market_open=True,
            )
        assert decision.capital_allocated == pytest.approx(
            decision.position_size * 50.0
        )


# ===========================================================================
# Risk/reward validation
# ===========================================================================

class TestRiskReward:
    def test_rejected_when_rr_too_high(self):
        guard = RiskGuard()
        # risk = 3.0, reward = 1.0 → R:R = 3.0 > max 2.0
        with _patched_now():
            decision = guard.check(
                signal=_signal(entry=100.0, target=101.0, stop=97.0),
                quote=_quote(bid=99.90, ask=100.10, last=100.0),
                account=_account(),
                open_positions=[],
                is_market_open=True,
            )
        assert decision.approved is False
        assert "risk_reward" in decision.checks_failed

    def test_approved_when_rr_within_limit(self):
        guard = RiskGuard()
        # risk = 1.0, reward = 1.5 → R:R = 0.67 < 2.0
        with _patched_now():
            decision = guard.check(
                signal=_signal(entry=100.0, target=101.5, stop=99.0),
                quote=_quote(bid=99.90, ask=100.10, last=100.0),
                account=_account(),
                open_positions=[],
                is_market_open=True,
            )
        assert decision.approved is True


# ===========================================================================
# Stop distance
# ===========================================================================

class TestStopDistance:
    def test_rejected_when_stop_too_far(self):
        guard = RiskGuard()
        # stop distance = 5% > 2% limit
        with _patched_now():
            decision = guard.check(
                signal=_signal(entry=100.0, target=108.0, stop=95.0),
                quote=_quote(bid=99.90, ask=100.10, last=100.0),
                account=_account(),
                open_positions=[],
                is_market_open=True,
            )
        assert decision.approved is False
        assert "stop_distance" in decision.checks_failed

    def test_approved_when_stop_within_limit(self):
        guard = RiskGuard()
        # stop distance = 1% < 2% limit
        with _patched_now():
            decision = guard.check(
                signal=_signal(entry=100.0, target=101.5, stop=99.0),
                quote=_quote(bid=99.90, ask=100.10, last=100.0),
                account=_account(),
                open_positions=[],
                is_market_open=True,
            )
        assert decision.approved is True


# ===========================================================================
# Slippage check
# ===========================================================================

class TestSlippageCheck:
    def test_acceptable_slippage(self):
        guard = RiskGuard()
        # 0.10% slippage < 0.15% limit
        result = guard.check_slippage(fill_price=100.10, signal_price=100.0)
        assert result.acceptable is True
        assert result.slippage_pct == pytest.approx(0.001)

    def test_unacceptable_slippage(self):
        guard = RiskGuard()
        # 0.20% slippage > 0.15% limit
        result = guard.check_slippage(fill_price=100.20, signal_price=100.0)
        assert result.acceptable is False
        assert result.slippage_pct == pytest.approx(0.002)

    def test_zero_signal_price_rejected(self):
        guard = RiskGuard()
        result = guard.check_slippage(fill_price=100.0, signal_price=0.0)
        assert result.acceptable is False

    def test_well_within_limit_is_acceptable(self):
        guard = RiskGuard()
        # 0.14% — clearly within the 0.15% limit
        result = guard.check_slippage(fill_price=100.14, signal_price=100.0)
        assert result.acceptable is True


# ===========================================================================
# RiskDecision string representation
# ===========================================================================

class TestRiskDecisionStr:
    def test_approved_str(self):
        d = RiskDecision(approved=True, reason="ok", position_size=5, capital_allocated=500.0)
        assert "APPROVED" in str(d)
        assert "5 shares" in str(d)

    def test_rejected_str(self):
        d = RiskDecision(approved=False, reason="Market is closed")
        assert "REJECTED" in str(d)
        assert "Market is closed" in str(d)
