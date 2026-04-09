"""
analytics/performance.py — Trade performance metrics.

Computes standard trading statistics from a list of ClosedTrade records.
Designed to be called with today's trades for the live dashboard and with
the full history for backtesting reports.

Metrics computed:
  - num_trades, num_wins, num_losses
  - win_rate          — wins / total (0.0 if no trades)
  - total_pnl         — sum of gross_pnl
  - avg_win           — mean P&L of winning trades
  - avg_loss          — mean P&L of losing trades (negative value)
  - profit_factor     — gross_profit / abs(gross_loss); inf if no losses
  - expectancy        — expected P&L per trade
  - sharpe_ratio      — annualised Sharpe of per-trade P&L (0 if < 2 trades)
  - max_drawdown      — largest peak-to-trough equity drop
  - avg_hold_seconds  — mean position duration
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from typing import List

from execution.order_manager import ClosedTrade


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class PerformanceMetrics:
    """Computed statistics for a set of trades."""
    num_trades: int      = 0
    num_wins: int        = 0
    num_losses: int      = 0
    win_rate: float      = 0.0    # 0.0 – 1.0
    total_pnl: float     = 0.0
    avg_win: float       = 0.0    # > 0 if any wins
    avg_loss: float      = 0.0    # < 0 if any losses
    profit_factor: float = 0.0    # gross_profit / abs(gross_loss)
    expectancy: float    = 0.0    # expected P&L per trade
    sharpe_ratio: float  = 0.0    # annualised, uses per-trade P&L series
    max_drawdown: float  = 0.0    # peak-to-trough equity drop (negative or 0)
    avg_hold_seconds: float = 0.0

    def __str__(self) -> str:
        return (
            f"Trades={self.num_trades} | "
            f"WinRate={self.win_rate*100:.1f}% | "
            f"TotalPnL=${self.total_pnl:.2f} | "
            f"PF={self.profit_factor:.2f} | "
            f"Sharpe={self.sharpe_ratio:.2f} | "
            f"MaxDD=${self.max_drawdown:.2f}"
        )


# ---------------------------------------------------------------------------
# Main compute function
# ---------------------------------------------------------------------------

def compute(trades: List[ClosedTrade]) -> PerformanceMetrics:
    """
    Compute performance metrics from *trades* (any order, any date range).

    Returns a zero-filled PerformanceMetrics if trades is empty.
    """
    if not trades:
        return PerformanceMetrics()

    pnls  = [t.gross_pnl for t in trades]
    wins  = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]

    num_trades = len(trades)
    num_wins   = len(wins)
    num_losses = len(losses)

    total_pnl    = sum(pnls)
    win_rate     = num_wins / num_trades
    avg_win      = statistics.mean(wins)  if wins   else 0.0
    avg_loss     = statistics.mean(losses) if losses else 0.0

    gross_profit = sum(wins)
    gross_loss   = abs(sum(losses))
    if gross_loss == 0.0:
        profit_factor = math.inf if gross_profit > 0 else 0.0
    else:
        profit_factor = gross_profit / gross_loss

    expectancy = total_pnl / num_trades

    sharpe = _sharpe(pnls)

    equity_curve = _equity_curve(pnls)
    max_dd = _max_drawdown(equity_curve)

    holds = [t.hold_seconds for t in trades]
    avg_hold = statistics.mean(holds)

    return PerformanceMetrics(
        num_trades=num_trades,
        num_wins=num_wins,
        num_losses=num_losses,
        win_rate=win_rate,
        total_pnl=total_pnl,
        avg_win=avg_win,
        avg_loss=avg_loss,
        profit_factor=profit_factor,
        expectancy=expectancy,
        sharpe_ratio=sharpe,
        max_drawdown=max_dd,
        avg_hold_seconds=avg_hold,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _equity_curve(pnls: List[float]) -> List[float]:
    """Cumulative P&L equity curve starting at 0."""
    curve: List[float] = []
    total = 0.0
    for p in pnls:
        total += p
        curve.append(total)
    return curve


def _max_drawdown(equity_curve: List[float]) -> float:
    """
    Return the maximum peak-to-trough drawdown (≤ 0).

    Iterates once through the equity curve tracking the running peak.
    """
    if not equity_curve:
        return 0.0
    peak = equity_curve[0]
    max_dd = 0.0
    for value in equity_curve:
        if value > peak:
            peak = value
        dd = value - peak
        if dd < max_dd:
            max_dd = dd
    return max_dd


def _sharpe(pnls: List[float], periods_per_year: int = 252) -> float:
    """
    Annualised Sharpe ratio treating each trade as one period.

    Uses sample std dev (ddof=1).  Returns 0.0 if fewer than 2 trades
    or std dev is zero.

    Note: this is a simplified trade-based Sharpe, not time-series Sharpe.
    It provides directional correctness for strategy comparison.
    """
    if len(pnls) < 2:
        return 0.0
    mean = statistics.mean(pnls)
    std  = statistics.stdev(pnls)
    if std == 0.0:
        return 0.0
    return (mean / std) * math.sqrt(periods_per_year)
