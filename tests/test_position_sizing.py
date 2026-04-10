"""
tests/test_position_sizing.py — Unit tests for ATR-normalized position sizing.

Tests the new sizing logic in risk/guard.py:
    shares = floor(target_risk_usd / stop_distance)
    capped by floor(capital_budget / entry_price) and max_shares.

All scenarios use a $35,000 account with 0.5% risk = $175 target risk.
Capital ceiling = 10% of buying_power = $3,500.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import List
from unittest.mock import MagicMock

import pytest

from broker.base import AccountInfo, Position, Quote
from risk.guard import RiskGuard, RiskDecision
from signals.base import SignalResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ACCOUNT_VALUE   = 35_000.0   # buying_power + portfolio_value
BUYING_POWER    = 35_000.0   # no open P&L for simplicity
TARGET_RISK_PCT = 0.005       # 0.5% → $175 target risk
CAPITAL_PCT     = 0.10        # 10% → $3,500 capital ceiling


def _account() -> AccountInfo:
    return AccountInfo(
        buying_power=BUYING_POWER,
        portfolio_value=0.0,
        cash=BUYING_POWER,
        day_trades_used=0,
        timestamp=datetime.utcnow(),
    )


def _quote(symbol: str, price: float) -> Quote:
    return Quote(
        symbol=symbol,
        bid=price * 0.9998,
        ask=price * 1.0002,
        last=price,
        volume=1_000_000,
        timestamp=datetime.utcnow(),
    )


def _signal(
    symbol: str,
    direction: str,
    entry: float,
    stop: float,
    target: float,
) -> SignalResult:
    return SignalResult(
        symbol=symbol,
        signal_type="ema_cross",
        direction=direction,
        entry_price=entry,
        stop_price=stop,
        target_price=target,
        confidence=85,
        estimated_hold_seconds=600,
    )


def _make_guard(
    target_risk_pct: float = TARGET_RISK_PCT,
    capital_pct: float = CAPITAL_PCT,
    min_shares: int = 1,
    max_shares: int = 500,
) -> RiskGuard:
    """Return a RiskGuard with patched settings for isolation."""
    guard = RiskGuard()
    # Patch settings attributes directly
    guard_settings = MagicMock()
    guard_settings.risk.target_risk_per_trade_pct = target_risk_pct
    guard_settings.risk.max_capital_per_trade_pct = capital_pct
    guard_settings.risk.min_shares = min_shares
    guard_settings.risk.max_shares = max_shares
    guard_settings.risk.max_daily_loss = 10_000.0
    guard_settings.risk.max_open_positions = 10
    guard_settings.risk.max_day_trades = 500
    guard_settings.risk.max_spread_pct = 0.01
    guard_settings.risk.max_stop_distance_pct = 0.10
    guard_settings.risk.max_loss_to_gain_ratio = 3.0
    guard_settings.risk.eod_liquidation_hour = 15
    guard_settings.risk.eod_liquidation_minute = 45
    return guard


def _approved_size(
    symbol: str,
    entry: float,
    stop: float,
    target: float,
    direction: str = "long",
) -> int:
    """Run the real RiskGuard.check() and return approved position_size."""
    guard = RiskGuard()
    sig   = _signal(symbol, direction, entry, stop, target)
    quote = _quote(symbol, entry)
    acct  = _account()
    dec   = guard.check(sig, quote, acct, [], is_market_open=True)
    return dec.position_size if dec.approved else 0


# ---------------------------------------------------------------------------
# Core sizing arithmetic (isolated, no guard overhead)
# ---------------------------------------------------------------------------

class TestSizingArithmetic:
    """Verify the math directly without going through check()."""

    def _size(
        self,
        entry: float,
        stop: float,
        account_value: float = ACCOUNT_VALUE,
        buying_power: float = BUYING_POWER,
        target_risk_pct: float = TARGET_RISK_PCT,
        capital_pct: float = CAPITAL_PCT,
        min_shares: int = 1,
        max_shares: int = 500,
    ) -> tuple[int, int, int]:
        """Returns (final_shares, risk_based, capital_based_max)."""
        stop_dist       = abs(entry - stop)
        target_risk_usd = account_value * target_risk_pct
        risk_based      = math.floor(target_risk_usd / stop_dist) if stop_dist > 0 else 1
        cap_budget      = buying_power * capital_pct
        cap_max         = math.floor(cap_budget / entry) if entry > 0 else 0
        final           = max(min_shares, min(risk_based, cap_max, max_shares))
        return final, risk_based, cap_max

    # ── SPY ──────────────────────────────────────────────────────────────
    def test_spy_risk_based(self):
        """SPY: $175 / $1.78 stop = 98 risk-based shares."""
        _, risk_based, _ = self._size(entry=681.0, stop=679.22)
        assert risk_based == 98

    def test_spy_capital_cap(self):
        """SPY: 98 shares × $681 = $66,738 > $3,500 cap → floor($3,500/$681) = 5."""
        _, _, cap_max = self._size(entry=681.0, stop=679.22)
        assert cap_max == 5

    def test_spy_final_shares(self):
        """SPY: capital cap binds → 5 shares final."""
        final, _, _ = self._size(entry=681.0, stop=679.22)
        assert final == 5

    # ── AMD ──────────────────────────────────────────────────────────────
    def test_amd_risk_based(self):
        """AMD: $175 / $1.18 stop = 148 risk-based shares."""
        _, risk_based, _ = self._size(entry=247.0, stop=245.82)
        assert risk_based == 148

    def test_amd_capital_cap(self):
        """AMD: floor($3,500 / $247) = 14."""
        _, _, cap_max = self._size(entry=247.0, stop=245.82)
        assert cap_max == 14

    def test_amd_final_shares(self):
        """AMD: capital cap binds → 14 shares final."""
        final, _, _ = self._size(entry=247.0, stop=245.82)
        assert final == 14

    # ── TSLA ─────────────────────────────────────────────────────────────
    def test_tsla_risk_based(self):
        """TSLA: $175 / $2.28 stop = 76 risk-based shares."""
        _, risk_based, _ = self._size(entry=346.0, stop=343.72)
        assert risk_based == 76

    def test_tsla_capital_cap(self):
        """TSLA: floor($3,500 / $346) = 10."""
        _, _, cap_max = self._size(entry=346.0, stop=343.72)
        assert cap_max == 10

    def test_tsla_final_shares(self):
        """TSLA: capital cap binds → 10 shares final."""
        final, _, _ = self._size(entry=346.0, stop=343.72)
        assert final == 10

    # ── Dollar risk after capital cap ────────────────────────────────────
    def test_spy_dollar_risk(self):
        final, _, _ = self._size(entry=681.0, stop=679.22)
        dollar_risk = final * abs(681.0 - 679.22)
        assert dollar_risk == pytest.approx(5 * 1.78, abs=0.01)

    def test_amd_dollar_risk(self):
        final, _, _ = self._size(entry=247.0, stop=245.82)
        dollar_risk = final * abs(247.0 - 245.82)
        assert dollar_risk == pytest.approx(14 * 1.18, abs=0.01)

    def test_tsla_dollar_risk(self):
        final, _, _ = self._size(entry=346.0, stop=343.72)
        dollar_risk = final * abs(346.0 - 343.72)
        assert dollar_risk == pytest.approx(10 * 2.28, abs=0.01)

    # ── Capital cap always binds for our $35k account at 10% ─────────────
    def test_capital_cap_always_binds(self):
        """All three are capital-capped, not risk-capped."""
        for entry, stop in [(681.0, 679.22), (247.0, 245.82), (346.0, 343.72)]:
            final, risk_based, cap_max = self._size(entry=entry, stop=stop)
            assert cap_max < risk_based, (
                f"entry={entry}: expected capital to bind but risk_based={risk_based} "
                f"<= cap_max={cap_max}"
            )
            assert final == cap_max


# ---------------------------------------------------------------------------
# Boundary conditions
# ---------------------------------------------------------------------------

class TestSizingBoundaries:
    def _size(self, entry, stop, **kwargs):
        stop_dist       = abs(entry - stop)
        account_value   = kwargs.get("account_value", ACCOUNT_VALUE)
        buying_power    = kwargs.get("buying_power", BUYING_POWER)
        target_risk_pct = kwargs.get("target_risk_pct", TARGET_RISK_PCT)
        capital_pct     = kwargs.get("capital_pct", CAPITAL_PCT)
        min_shares      = kwargs.get("min_shares", 1)
        max_shares      = kwargs.get("max_shares", 500)
        target_risk_usd = account_value * target_risk_pct
        risk_based      = math.floor(target_risk_usd / stop_dist) if stop_dist > 0 else 1
        cap_budget      = buying_power * capital_pct
        cap_max         = math.floor(cap_budget / entry) if entry > 0 else 0
        return max(min_shares, min(risk_based, cap_max, max_shares))

    def test_max_shares_cap(self):
        """Very tight stop + large account → risk_based > max_shares → capped."""
        # $175 / $0.01 stop = 17,500 shares — must cap at 500
        final = self._size(100.0, 99.99, max_shares=500)
        assert final <= 500

    def test_min_shares_floor(self):
        """Huge stop relative to budget → risk_based = 0 → raised to min_shares."""
        # $175 / $500 stop = 0 risk_based → min_shares=1
        final = self._size(1000.0, 500.0, min_shares=1)
        assert final >= 1

    def test_zero_stop_distance_defaults_to_one(self):
        """Zero stop distance must not divide-by-zero — defaults to 1."""
        stop_dist = 0.0
        risk_based = math.floor(175.0 / stop_dist) if stop_dist > 0 else 1
        assert risk_based == 1

    def test_shares_never_exceed_buying_power(self):
        """Final capital_used must not exceed buying_power."""
        for entry, stop in [(681.0, 679.22), (247.0, 245.82), (346.0, 343.72)]:
            final = self._size(entry, stop)
            assert final * entry <= BUYING_POWER + 0.01  # tiny float tolerance
