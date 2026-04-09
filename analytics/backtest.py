"""
analytics/backtest.py — Walk-forward strategy backtester.

Simulates the full Rosetta1 trading system on historical OHLCV data and
produces a BacktestResult that determines whether the strategy is approved
for live deployment.

Approval criteria (configurable via settings.backtest):
  - win_rate  >= settings.backtest.min_win_rate   (default 55%)
  - sharpe    >= settings.backtest.min_sharpe_ratio (default 1.2)

Simulation model:
  - Bars: 1-minute OHLCV fetched via data.history.fetch()
  - Warmup: first 80 bars discarded (indicator stabilisation)
  - Entry: next bar's open price ± simulated_slippage_pct
  - Exit (target): bar's HIGH ≥ target for long; LOW ≤ target for short
  - Exit (stop):   bar's LOW  ≤ stop  for long; HIGH ≥ stop  for short
  - Exit (EOD):    forced close at the last bar of each trading day
  - Positions: one per symbol at a time (no pyramiding)
  - No explicit capital limit — position sizing uses a fixed 100-share unit
    to produce proportional P&L; absolute dollar magnitudes are comparable

Usage:
    from analytics.backtest import BacktestEngine

    engine = BacktestEngine()
    result = engine.run(symbols=["SPY", "QQQ"], days=60)
    print(result.approved, result.metrics)
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Tuple

import pandas as pd

from analytics.performance import PerformanceMetrics, compute
from config.settings import settings
from data.history import HistoryError, fetch
from data.indicators import add_all
from execution.order_manager import ClosedTrade
from strategy.engine import SignalEngine

logger = logging.getLogger(__name__)

# Minimum bars needed before the first signal evaluation (indicator warm-up)
_WARMUP_BARS = 80

# Fixed share size used in backtest P&L calculation
_BACKTEST_SHARES = 10


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class BacktestResult:
    """
    Summary of a completed backtest run.

    Attributes:
        trades:           All simulated ClosedTrade records.
        metrics:          Aggregate performance statistics.
        approved:         True if strategy meets both approval thresholds.
        rejection_reason: Human-readable explanation when approved=False.
        symbols:          Symbols included in this run.
        start_date:       First date of the simulation window.
        end_date:         Last date of the simulation window.
    """
    trades: List[ClosedTrade]
    metrics: PerformanceMetrics
    approved: bool
    rejection_reason: str
    symbols: List[str]
    start_date: date
    end_date: date

    def summary(self) -> str:
        status = "APPROVED" if self.approved else f"REJECTED ({self.rejection_reason})"
        return (
            f"Backtest [{self.start_date} → {self.end_date}] "
            f"{len(self.symbols)} symbols | "
            f"{self.metrics.num_trades} trades | "
            f"WinRate={self.metrics.win_rate*100:.1f}% | "
            f"Sharpe={self.metrics.sharpe_ratio:.2f} | "
            f"TotalPnL=${self.metrics.total_pnl:.2f} | "
            f"{status}"
        )


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------

class BacktestEngine:
    """
    Walk-forward backtester for the Rosetta1 strategy.

    Creates a fresh SignalEngine for each run to avoid stale session state.
    """

    def __init__(self) -> None:
        pass  # stateless — engine is created fresh per run()

    def run(
        self,
        symbols: Optional[List[str]] = None,
        days: int = 60,
    ) -> BacktestResult:
        """
        Run a walk-forward backtest over *days* of history for *symbols*.

        Args:
            symbols: Tickers to include (default: settings UNIVERSE).
            days:    Number of calendar days to look back.

        Returns:
            BacktestResult with trades, metrics, and approval decision.
        """
        from config.settings import UNIVERSE
        syms = symbols or UNIVERSE

        end_dt   = datetime.utcnow().date()
        start_dt = end_dt - timedelta(days=days)

        logger.info(
            "Backtest starting | symbols=%s | %s → %s",
            syms, start_dt, end_dt,
        )

        all_trades: List[ClosedTrade] = []
        slippage = settings.backtest.simulated_slippage_pct

        for symbol in syms:
            try:
                df = fetch(symbol, days=days + 10, interval="1m")
                df = add_all(df)
            except HistoryError as exc:
                logger.warning("Backtest: skipping %s — %s", symbol, exc)
                continue

            engine = SignalEngine()  # fresh instance per symbol
            symbol_trades = _simulate_symbol(df, symbol, engine, slippage)
            logger.info(
                "Backtest [%s]: %d trades simulated", symbol, len(symbol_trades)
            )
            all_trades.extend(symbol_trades)

        metrics = compute(all_trades)
        approved, reason = _check_approval(metrics)

        result = BacktestResult(
            trades=all_trades,
            metrics=metrics,
            approved=approved,
            rejection_reason=reason,
            symbols=syms,
            start_date=start_dt,
            end_date=end_dt,
        )
        logger.info("Backtest complete | %s", result.summary())
        return result


# ---------------------------------------------------------------------------
# Per-symbol simulation
# ---------------------------------------------------------------------------

def _simulate_symbol(
    df: pd.DataFrame,
    symbol: str,
    engine: SignalEngine,
    slippage_pct: float,
) -> List[ClosedTrade]:
    """
    Walk forward through *df* bar by bar, fire signals, simulate fills.

    One position at a time — after entry, advance until exit, then look for
    the next signal.
    """
    trades: List[ClosedTrade] = []
    n = len(df)

    if n < _WARMUP_BARS + 2:
        logger.debug("_simulate_symbol [%s]: insufficient bars (%d)", symbol, n)
        return trades

    i = _WARMUP_BARS

    while i < n:
        bars = df.iloc[:i]
        current_price = float(df["close"].iloc[i - 1])

        try:
            signal = engine.evaluate(bars, symbol, current_price)
        except Exception as exc:
            logger.debug("Signal error at bar %d [%s]: %s", i, symbol, exc)
            i += 1
            continue

        if signal is None:
            i += 1
            continue

        # Entry: fill at next bar's open ± slippage
        entry_bar = i
        if entry_bar >= n:
            break

        raw_entry = float(df["open"].iloc[entry_bar])
        if signal.direction == "long":
            fill_price = raw_entry * (1 + slippage_pct)
        else:
            fill_price = raw_entry * (1 - slippage_pct)

        # Look ahead for exit
        exit_bar, exit_price, reason = _find_exit(
            df, entry_bar, signal.direction,
            signal.target_price, signal.stop_price,
        )

        if exit_bar is None:
            # No exit found — skip this signal
            i = entry_bar + 1
            continue

        # Build ClosedTrade record
        opened_at  = _bar_dt(df, entry_bar)
        closed_at  = _bar_dt(df, exit_bar)
        shares     = _BACKTEST_SHARES

        if signal.direction == "long":
            gross_pnl = (exit_price - fill_price) * shares
        else:
            gross_pnl = (fill_price - exit_price) * shares

        trade = ClosedTrade(
            trade_id=str(uuid.uuid4()),
            symbol=symbol,
            direction=signal.direction,
            shares=shares,
            entry_price=fill_price,
            exit_price=exit_price,
            gross_pnl=gross_pnl,
            signal_type=signal.signal_type,
            confidence=signal.confidence,
            entry_order_id="bt-entry",
            exit_order_id="bt-exit",
            opened_at=opened_at,
            closed_at=closed_at,
            exit_reason=reason,
        )
        trades.append(trade)

        # Advance past this trade's exit bar
        i = exit_bar + 1

    return trades


# ---------------------------------------------------------------------------
# Exit finder
# ---------------------------------------------------------------------------

def _find_exit(
    df: pd.DataFrame,
    entry_bar: int,
    direction: str,
    target_price: float,
    stop_price: float,
) -> Tuple[Optional[int], float, str]:
    """
    Scan forward from *entry_bar* + 1 for the first exit condition.

    For longs:
      - Target: bar HIGH  >= target_price  → fill at target_price
      - Stop:   bar LOW   <= stop_price    → fill at stop_price
      - EOD:    last bar of the session    → fill at bar close

    For shorts (symmetric):
      - Target: bar LOW   <= target_price  → fill at target_price
      - Stop:   bar HIGH  >= stop_price    → fill at stop_price
      - EOD:    last bar of the session    → fill at bar close

    When target and stop are both touched in the same bar, stop takes
    priority (conservative assumption — worst-case fill).

    Returns:
        (bar_index, fill_price, reason) or (None, 0.0, "") if no bars remain.
    """
    n = len(df)

    if entry_bar + 1 >= n:
        return None, 0.0, ""

    # Determine trading day of entry bar for EOD detection
    entry_date = _bar_date(df, entry_bar)

    for j in range(entry_bar + 1, n):
        high  = float(df["high"].iloc[j])
        low   = float(df["low"].iloc[j])
        close = float(df["close"].iloc[j])
        bar_date = _bar_date(df, j)

        if direction == "long":
            stop_hit   = low   <= stop_price
            target_hit = high  >= target_price
        else:
            stop_hit   = high  >= stop_price
            target_hit = low   <= target_price

        # Stop has priority over target (conservative)
        if stop_hit:
            return j, stop_price, "stop"
        if target_hit:
            return j, target_price, "target"

        # EOD: last bar of the entry day
        if bar_date != entry_date:
            # First bar of the next day → exit was the previous bar (EOD)
            return j - 1, float(df["close"].iloc[j - 1]), "eod"

    # Ran off end of data — close at last bar
    last = n - 1
    return last, float(df["close"].iloc[last]), "eod"


# ---------------------------------------------------------------------------
# Approval gate
# ---------------------------------------------------------------------------

def _check_approval(metrics: PerformanceMetrics) -> Tuple[bool, str]:
    """
    Return (approved, rejection_reason).

    Both win_rate AND sharpe thresholds must be met.
    """
    min_wr     = settings.backtest.min_win_rate
    min_sharpe = settings.backtest.min_sharpe_ratio

    if metrics.num_trades == 0:
        return False, "No trades generated — insufficient signal activity"

    failures: List[str] = []
    if metrics.win_rate < min_wr:
        failures.append(
            f"win_rate {metrics.win_rate*100:.1f}% < {min_wr*100:.0f}% required"
        )
    if metrics.sharpe_ratio < min_sharpe:
        failures.append(
            f"sharpe {metrics.sharpe_ratio:.2f} < {min_sharpe:.1f} required"
        )

    if failures:
        return False, "; ".join(failures)
    return True, ""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bar_dt(df: pd.DataFrame, i: int) -> datetime:
    """Return the bar's index as a naive UTC datetime."""
    ts = df.index[i]
    if hasattr(ts, "to_pydatetime"):
        dt = ts.to_pydatetime()
        return dt.replace(tzinfo=None)
    return datetime.utcfromtimestamp(float(ts))


def _bar_date(df: pd.DataFrame, i: int) -> date:
    return _bar_dt(df, i).date()
