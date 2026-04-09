"""
tests/test_analytics.py — Tests for analytics/logger.py and analytics/performance.py.

Run:
    pytest tests/test_analytics.py -v
"""
from __future__ import annotations

import math
import uuid
from datetime import datetime, timedelta
from typing import List

import pytest

from analytics.logger import TradeLogger
from analytics.performance import PerformanceMetrics, compute
from execution.order_manager import ClosedTrade


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _trade(
    symbol="SPY",
    direction="long",
    shares=2,
    entry=100.0,
    exit_=101.0,
    pnl: float | None = None,
    reason="target",
    hold=300.0,
    opened_at: datetime | None = None,
    closed_at: datetime | None = None,
) -> ClosedTrade:
    base = datetime(2024, 1, 8, 10, 0, 0)
    gross_pnl = pnl if pnl is not None else (exit_ - entry) * shares * (1 if direction == "long" else -1)
    return ClosedTrade(
        trade_id=str(uuid.uuid4()),
        symbol=symbol,
        direction=direction,
        shares=shares,
        entry_price=entry,
        exit_price=exit_,
        gross_pnl=gross_pnl,
        signal_type="momentum",
        confidence=75,
        entry_order_id=str(uuid.uuid4()),
        exit_order_id=str(uuid.uuid4()),
        opened_at=opened_at or base,
        closed_at=closed_at or (base + timedelta(seconds=hold)),
        exit_reason=reason,
    )


# ===========================================================================
# TradeLogger
# ===========================================================================

class TestTradeLogger:
    def test_log_and_retrieve(self, tmp_path):
        db = str(tmp_path / "test_trades.db")
        tl = TradeLogger(path=db)
        t = _trade(pnl=10.0)
        tl.log_trade(t)
        trades = tl.get_trades()
        assert len(trades) == 1
        assert trades[0].trade_id == t.trade_id
        tl.close()

    def test_trade_count(self, tmp_path):
        tl = TradeLogger(path=str(tmp_path / "tc.db"))
        for _ in range(3):
            tl.log_trade(_trade())
        assert tl.trade_count() == 3
        tl.close()

    def test_duplicate_trade_ignored(self, tmp_path):
        tl = TradeLogger(path=str(tmp_path / "dup.db"))
        t = _trade()
        tl.log_trade(t)
        tl.log_trade(t)  # second insert should be silently ignored
        assert tl.trade_count() == 1
        tl.close()

    def test_filter_by_symbol(self, tmp_path):
        tl = TradeLogger(path=str(tmp_path / "sym.db"))
        tl.log_trade(_trade(symbol="SPY"))
        tl.log_trade(_trade(symbol="AAPL"))
        tl.log_trade(_trade(symbol="SPY"))
        result = tl.get_trades(symbol="SPY")
        assert len(result) == 2
        assert all(t.symbol == "SPY" for t in result)
        tl.close()

    def test_filter_by_date(self, tmp_path):
        tl = TradeLogger(path=str(tmp_path / "date.db"))
        jan8 = datetime(2024, 1, 8, 10, 0, 0)
        jan9 = datetime(2024, 1, 9, 10, 0, 0)
        tl.log_trade(_trade(closed_at=jan8))
        tl.log_trade(_trade(closed_at=jan9))
        result = tl.get_trades(date_=jan8.date())
        assert len(result) == 1
        assert result[0].closed_at.date() == jan8.date()
        tl.close()

    def test_pnl_roundtrip(self, tmp_path):
        tl = TradeLogger(path=str(tmp_path / "pnl.db"))
        t = _trade(entry=100.0, exit_=102.5, pnl=5.0)
        tl.log_trade(t)
        retrieved = tl.get_trades()[0]
        assert retrieved.gross_pnl == pytest.approx(5.0)
        tl.close()

    def test_hold_seconds_roundtrip(self, tmp_path):
        tl = TradeLogger(path=str(tmp_path / "hold.db"))
        t = _trade(hold=180.0)
        tl.log_trade(t)
        retrieved = tl.get_trades()[0]
        assert retrieved.hold_seconds == pytest.approx(180.0)
        tl.close()

    def test_clear_removes_all_rows(self, tmp_path):
        tl = TradeLogger(path=str(tmp_path / "clr.db"))
        tl.log_trade(_trade())
        tl.log_trade(_trade())
        tl.clear()
        assert tl.trade_count() == 0
        tl.close()

    def test_empty_db_returns_empty_list(self, tmp_path):
        tl = TradeLogger(path=str(tmp_path / "empty.db"))
        assert tl.get_trades() == []
        tl.close()

    def test_all_exit_reasons_stored(self, tmp_path):
        tl = TradeLogger(path=str(tmp_path / "reasons.db"))
        for reason in ("target", "stop", "eod", "manual", "slippage"):
            tl.log_trade(_trade(reason=reason))
        reasons = {t.exit_reason for t in tl.get_trades()}
        assert reasons == {"target", "stop", "eod", "manual", "slippage"}
        tl.close()

    def test_multiple_symbols_filter_combined(self, tmp_path):
        tl = TradeLogger(path=str(tmp_path / "combo.db"))
        d = datetime(2024, 3, 1, 11, 0, 0)
        tl.log_trade(_trade(symbol="TSLA", closed_at=d))
        tl.log_trade(_trade(symbol="SPY",  closed_at=d))
        result = tl.get_trades(symbol="TSLA", date_=d.date())
        assert len(result) == 1
        assert result[0].symbol == "TSLA"
        tl.close()


# ===========================================================================
# PerformanceMetrics — compute()
# ===========================================================================

class TestPerformanceMetrics:
    def test_empty_trades_returns_zero_metrics(self):
        m = compute([])
        assert m.num_trades == 0
        assert m.total_pnl == 0.0
        assert m.win_rate == 0.0
        assert m.sharpe_ratio == 0.0

    def test_all_wins(self):
        trades = [_trade(pnl=10.0), _trade(pnl=5.0), _trade(pnl=8.0)]
        m = compute(trades)
        assert m.num_wins == 3
        assert m.num_losses == 0
        assert m.win_rate == pytest.approx(1.0)
        assert m.total_pnl == pytest.approx(23.0)
        assert m.avg_loss == pytest.approx(0.0)
        assert math.isinf(m.profit_factor)

    def test_all_losses(self):
        trades = [_trade(pnl=-5.0), _trade(pnl=-3.0)]
        m = compute(trades)
        assert m.num_wins == 0
        assert m.win_rate == pytest.approx(0.0)
        assert m.total_pnl == pytest.approx(-8.0)
        assert m.profit_factor == pytest.approx(0.0)
        assert m.avg_win == pytest.approx(0.0)

    def test_mixed_trades(self):
        trades = [_trade(pnl=10.0), _trade(pnl=-5.0), _trade(pnl=8.0), _trade(pnl=-2.0)]
        m = compute(trades)
        assert m.num_trades == 4
        assert m.num_wins == 2
        assert m.num_losses == 2
        assert m.win_rate == pytest.approx(0.5)
        assert m.total_pnl == pytest.approx(11.0)
        assert m.avg_win == pytest.approx(9.0)
        assert m.avg_loss == pytest.approx(-3.5)
        assert m.profit_factor == pytest.approx(18.0 / 7.0)

    def test_expectancy(self):
        trades = [_trade(pnl=10.0), _trade(pnl=-4.0)]
        m = compute(trades)
        assert m.expectancy == pytest.approx(3.0)   # (10 - 4) / 2

    def test_sharpe_single_trade_is_zero(self):
        m = compute([_trade(pnl=10.0)])
        assert m.sharpe_ratio == pytest.approx(0.0)

    def test_sharpe_positive_for_consistent_winners(self):
        # 10 identical wins → all same value → std=0 → sharpe=0
        trades = [_trade(pnl=5.0) for _ in range(10)]
        m = compute(trades)
        assert m.sharpe_ratio == pytest.approx(0.0)  # zero variance

    def test_sharpe_nonzero_for_varied_pnls(self):
        trades = [_trade(pnl=p) for p in [10.0, 5.0, 8.0, -2.0, 12.0]]
        m = compute(trades)
        assert m.sharpe_ratio != 0.0

    def test_max_drawdown_no_loss(self):
        trades = [_trade(pnl=5.0), _trade(pnl=3.0), _trade(pnl=2.0)]
        m = compute(trades)
        assert m.max_drawdown == pytest.approx(0.0)

    def test_max_drawdown_with_loss_sequence(self):
        # Equity: 10 → 15 → 8 → 12 — drawdown from 15 to 8 = -7
        trades = [_trade(pnl=10.0), _trade(pnl=5.0), _trade(pnl=-7.0), _trade(pnl=4.0)]
        m = compute(trades)
        assert m.max_drawdown == pytest.approx(-7.0)

    def test_avg_hold_seconds(self):
        trades = [_trade(hold=300.0), _trade(hold=600.0), _trade(hold=150.0)]
        m = compute(trades)
        assert m.avg_hold_seconds == pytest.approx(350.0)

    def test_str_representation(self):
        m = compute([_trade(pnl=10.0), _trade(pnl=-3.0)])
        s = str(m)
        assert "Trades=2" in s
        assert "WinRate=" in s
        assert "TotalPnL=" in s

    def test_profit_factor_no_losses(self):
        m = compute([_trade(pnl=5.0)])
        assert math.isinf(m.profit_factor)

    def test_profit_factor_no_wins(self):
        m = compute([_trade(pnl=-5.0)])
        assert m.profit_factor == pytest.approx(0.0)
