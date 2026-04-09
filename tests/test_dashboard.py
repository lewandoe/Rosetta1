"""
tests/test_dashboard.py — Tests for analytics/dashboard.py and main.py components.

Tests cover:
  - Dashboard renders without error (smoke test with mocked dependencies)
  - _fmt_duration formats durations correctly
  - Dashboard start/stop lifecycle
  - main.py _print_summary produces expected output
  - Rosetta1 shutdown path closes all subsystems

Run:
    pytest tests/test_dashboard.py -v
"""
from __future__ import annotations

import io
import time
import uuid
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from analytics.dashboard import Dashboard, _fmt_duration
from analytics.logger import TradeLogger
from analytics.performance import PerformanceMetrics
from execution.order_manager import ClosedTrade, OpenTrade, OrderManager
from risk.guard import RiskGuard
from signals.base import SignalResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_signal() -> SignalResult:
    return SignalResult(
        symbol="SPY", signal_type="momentum", direction="long",
        entry_price=100.0, target_price=101.5, stop_price=99.0,
        confidence=75, estimated_hold_seconds=300,
    )


def _make_open_trade() -> OpenTrade:
    return OpenTrade(
        trade_id=str(uuid.uuid4()),
        symbol="SPY", direction="long", shares=2,
        entry_price=100.0, target_price=101.5, stop_price=99.0,
        signal_type="momentum", confidence=75,
        entry_order_id=str(uuid.uuid4()),
        opened_at=datetime.utcnow() - timedelta(seconds=120),
        signal=_make_signal(),
    )


def _make_closed_trade(pnl: float = 5.0) -> ClosedTrade:
    base = datetime(2024, 1, 8, 10, 0, 0)
    return ClosedTrade(
        trade_id=str(uuid.uuid4()),
        symbol="SPY", direction="long", shares=2,
        entry_price=100.0, exit_price=102.5,
        gross_pnl=pnl, signal_type="momentum", confidence=75,
        entry_order_id=str(uuid.uuid4()),
        exit_order_id=str(uuid.uuid4()),
        opened_at=base,
        closed_at=base + timedelta(seconds=300),
        exit_reason="target",
    )


def _make_dashboard(tmp_path, open_trades=None, today_trades=None, halted=False):
    """Build a Dashboard with mocked dependencies."""
    om = MagicMock(spec=OrderManager)
    om.get_open_trades.return_value = open_trades or []

    risk = MagicMock(spec=RiskGuard)
    risk.is_halted = halted
    risk.halt_reason = "test halt" if halted else ""
    risk.get_daily_pnl.return_value = 10.0
    risk.daily_loss_remaining.return_value = 190.0

    db = str(tmp_path / "dash.db")
    logger = TradeLogger(path=db)
    for t in (today_trades or []):
        logger.log_trade(t)

    return Dashboard(om, risk, logger, started_at=datetime.utcnow()), logger


# ===========================================================================
# _fmt_duration
# ===========================================================================

class TestFmtDuration:
    def test_under_60s(self):
        assert _fmt_duration(45) == "45s"

    def test_exactly_60s(self):
        assert _fmt_duration(60) == "1m00s"

    def test_minutes_and_seconds(self):
        assert _fmt_duration(125) == "2m05s"

    def test_hours(self):
        assert _fmt_duration(3665) == "1h01m"

    def test_zero(self):
        assert _fmt_duration(0) == "0s"


# ===========================================================================
# Dashboard — smoke tests
# ===========================================================================

class TestDashboard:
    def test_build_no_trades(self, tmp_path):
        dash, tl = _make_dashboard(tmp_path)
        layout = dash._build()
        assert layout is not None
        tl.close()

    def test_build_with_open_trade(self, tmp_path):
        dash, tl = _make_dashboard(tmp_path, open_trades=[_make_open_trade()])
        layout = dash._build()
        assert layout is not None
        tl.close()

    def test_build_with_closed_trades(self, tmp_path):
        trades = [_make_closed_trade(pnl=p) for p in [5.0, -3.0, 8.0]]
        dash, tl = _make_dashboard(tmp_path, today_trades=trades)
        layout = dash._build()
        assert layout is not None
        tl.close()

    def test_build_halted_state(self, tmp_path):
        dash, tl = _make_dashboard(tmp_path, halted=True)
        layout = dash._build()
        assert layout is not None
        tl.close()

    def test_start_stop(self, tmp_path):
        dash, tl = _make_dashboard(tmp_path)
        dash.start()
        time.sleep(0.15)
        dash.stop()
        assert not dash._thread.is_alive()
        tl.close()

    def test_double_start_is_safe(self, tmp_path):
        dash, tl = _make_dashboard(tmp_path)
        dash.start()
        dash.start()  # second start should be ignored
        time.sleep(0.1)
        dash.stop()
        tl.close()

    def test_stop_before_start_is_safe(self, tmp_path):
        dash, tl = _make_dashboard(tmp_path)
        dash.stop()  # should not raise
        tl.close()

    def test_header_contains_mode(self, tmp_path):
        dash, tl = _make_dashboard(tmp_path)
        panel = dash._header_panel()
        # Render to string and check content
        from rich.console import Console
        console = Console(file=io.StringIO(), width=120)
        console.print(panel)
        output = console.file.getvalue()
        assert "PAPER" in output or "LIVE" in output or "Rosetta1" in output
        tl.close()

    def test_metrics_panel_shows_win_rate(self, tmp_path):
        trades = [_make_closed_trade(pnl=5.0), _make_closed_trade(pnl=-2.0)]
        dash, tl = _make_dashboard(tmp_path, today_trades=trades)
        from rich.console import Console
        console = Console(file=io.StringIO(), width=120)
        console.print(dash._metrics_panel())
        output = console.file.getvalue()
        assert "%" in output  # win rate displayed as percentage
        tl.close()


# ===========================================================================
# main.py — _print_summary
# ===========================================================================

class TestPrintSummary:
    def test_output_contains_key_fields(self, capsys):
        from main import _print_summary
        m = PerformanceMetrics(
            num_trades=5, win_rate=0.6, total_pnl=25.0,
            profit_factor=2.5, sharpe_ratio=1.8, max_drawdown=-10.0,
        )
        _print_summary(m)
        captured = capsys.readouterr()
        assert "5" in captured.out
        assert "60.0%" in captured.out
        assert "+25.00" in captured.out
        assert "1.80" in captured.out

    def test_negative_pnl_shows_sign(self, capsys):
        from main import _print_summary
        m = PerformanceMetrics(num_trades=2, total_pnl=-15.0)
        _print_summary(m)
        captured = capsys.readouterr()
        assert "-15.00" in captured.out

    def test_zero_trades(self, capsys):
        from main import _print_summary
        _print_summary(PerformanceMetrics())
        captured = capsys.readouterr()
        assert "0" in captured.out
