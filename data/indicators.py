"""
data/indicators.py — Technical indicator library.

All functions accept a clean OHLCV DataFrame (produced by data/history.py)
and return a new DataFrame with indicator columns appended.  The input is
never mutated.

Indicators implemented:
  - vwap()          — Volume-Weighted Average Price (intraday, resets at open)
  - ema()           — Exponential Moving Average (arbitrary period)
  - ema_trio()      — EMA(8), EMA(13), EMA(21) in one pass
  - rsi()           — Relative Strength Index
  - atr()           — Average True Range
  - volume_ma()     — Simple moving average of volume

Design rules:
  - Every function validates the input DataFrame has required columns.
  - Periods come from config/settings.py — no magic numbers.
  - All calculations are pure pandas/numpy — no pandas-ta dependency for
    these core indicators (keeps the library auditable and testable without
    heavy optional deps).  pandas-ta is available for Stage 5 signals if
    needed for more exotic indicators.
  - Functions raise IndicatorError, never return silently bad data.
  - Minimum bars required for a valid reading = period.  Rows before that
    will contain NaN — callers must handle or use validate_sufficient_history.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from config.settings import settings

logger = logging.getLogger(__name__)


class IndicatorError(Exception):
    """Raised when an indicator cannot be computed on the supplied DataFrame."""


# ---------------------------------------------------------------------------
# Input validation helper
# ---------------------------------------------------------------------------

def _require_cols(df: pd.DataFrame, *cols: str, fn_name: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise IndicatorError(
            f"{fn_name}: DataFrame is missing required columns: {missing}. "
            f"Available: {list(df.columns)}"
        )


def _require_min_rows(df: pd.DataFrame, n: int, fn_name: str) -> None:
    if len(df) < n:
        raise IndicatorError(
            f"{fn_name}: need at least {n} rows, got {len(df)}"
        )


# ---------------------------------------------------------------------------
# VWAP — Volume-Weighted Average Price
# ---------------------------------------------------------------------------

def vwap(df: pd.DataFrame, col: str = "vwap") -> pd.DataFrame:
    """
    Compute intraday VWAP and append as column *col*.

    Formula: cumsum(typical_price × volume) / cumsum(volume)
    Typical price = (high + low + close) / 3

    VWAP resets at the start of each trading day.  For multi-day DataFrames
    the index date is used to group sessions.

    Returns:
        New DataFrame with *col* appended.  VWAP for the first bar of each
        session equals the typical price of that bar (no prior data to average).
    """
    _require_cols(df, "high", "low", "close", "volume", fn_name="vwap")
    _require_min_rows(df, 1, "vwap")

    out = df.copy()
    tp = (out["high"] + out["low"] + out["close"]) / 3.0
    pv = tp * out["volume"]

    # Group by calendar date so VWAP resets each session
    dates = out.index.normalize()  # tz-aware date component
    cum_pv = pv.groupby(dates).cumsum()
    cum_vol = out["volume"].groupby(dates).cumsum()

    # Guard against zero volume (e.g. pre-market bars that slip through)
    with np.errstate(invalid="ignore", divide="ignore"):
        out[col] = np.where(cum_vol > 0, cum_pv / cum_vol, np.nan)

    return out


# ---------------------------------------------------------------------------
# EMA — Exponential Moving Average
# ---------------------------------------------------------------------------

def ema(df: pd.DataFrame, period: int, source: str = "close",
        col: Optional[str] = None) -> pd.DataFrame:
    """
    Compute EMA(*period*) of *source* column and append as *col*.

    Uses pandas ewm(span=period, adjust=False) which matches the standard
    EMA formula: α = 2 / (period + 1).

    Args:
        df:      OHLCV DataFrame.
        period:  Lookback window.
        source:  Column to compute EMA on (default "close").
        col:     Output column name (default "ema_{period}").

    Returns:
        New DataFrame with EMA column appended.
    """
    _require_cols(df, source, fn_name=f"ema({period})")
    _require_min_rows(df, period, f"ema({period})")

    out_col = col or f"ema_{period}"
    out = df.copy()
    out[out_col] = out[source].ewm(span=period, adjust=False).mean()
    return out


def ema_trio(df: pd.DataFrame) -> pd.DataFrame:
    """
    Append EMA(fast), EMA(mid), EMA(slow) in one call.

    Periods are read from settings.signals:
      fast = settings.signals.ema_fast  (default 8)
      mid  = settings.signals.ema_mid   (default 13)
      slow = settings.signals.ema_slow  (default 21)

    Output columns: ema_fast, ema_mid, ema_slow  (fixed names for
    downstream signal code to reference without knowing the period values).

    Returns:
        New DataFrame with all three EMA columns appended.
    """
    cfg = settings.signals
    _require_min_rows(df, cfg.ema_slow, "ema_trio")

    out = df.copy()
    for period, name in [
        (cfg.ema_fast, "ema_fast"),
        (cfg.ema_mid,  "ema_mid"),
        (cfg.ema_slow, "ema_slow"),
    ]:
        out[name] = out["close"].ewm(span=period, adjust=False).mean()
    return out


# ---------------------------------------------------------------------------
# RSI — Relative Strength Index
# ---------------------------------------------------------------------------

def rsi(df: pd.DataFrame, period: Optional[int] = None,
        source: str = "close", col: str = "rsi") -> pd.DataFrame:
    """
    Compute RSI(*period*) of *source* and append as *col*.

    Uses Wilder's smoothing (equivalent to EMA with α = 1/period):
      RS  = avg_gain / avg_loss  over the window
      RSI = 100 - 100 / (1 + RS)

    First *period* rows will be NaN (insufficient history for a reading).

    Args:
        period: Lookback (default: settings.signals.rsi_period = 14).
        source: Column to compute RSI on (default "close").
        col:    Output column name (default "rsi").
    """
    p = period or settings.signals.rsi_period
    _require_cols(df, source, fn_name=f"rsi({p})")
    _require_min_rows(df, p + 1, f"rsi({p})")

    out = df.copy()
    delta = out[source].diff()

    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)

    # Wilder smoothing = EMA with α = 1/period  (adjust=False matches Wilder)
    avg_gain = gain.ewm(alpha=1.0 / p, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / p, adjust=False).mean()

    with np.errstate(divide="ignore", invalid="ignore"):
        rs = np.where(avg_loss == 0, np.inf, avg_gain / avg_loss)
        out[col] = np.where(avg_loss == 0, 100.0, 100.0 - 100.0 / (1.0 + rs))

    # First row always NaN (no delta available)
    out.loc[out.index[0], col] = np.nan

    return out


# ---------------------------------------------------------------------------
# ATR — Average True Range
# ---------------------------------------------------------------------------

def atr(df: pd.DataFrame, period: Optional[int] = None,
        col: str = "atr") -> pd.DataFrame:
    """
    Compute ATR(*period*) and append as *col*.

    True Range = max(high - low, |high - prev_close|, |low - prev_close|)
    ATR        = EMA(TR, period) using Wilder's smoothing (α = 1/period)

    ATR is used by the risk guard and signal engine to set dynamic stop sizes.

    Args:
        period: Lookback (default: settings.signals.atr_period = 14).
    """
    p = period or settings.signals.atr_period
    _require_cols(df, "high", "low", "close", fn_name=f"atr({p})")
    _require_min_rows(df, p + 1, f"atr({p})")

    out = df.copy()
    prev_close = out["close"].shift(1)

    tr = pd.concat(
        [
            out["high"] - out["low"],
            (out["high"] - prev_close).abs(),
            (out["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    # First row has no prev_close — TR = high - low
    tr.iloc[0] = out["high"].iloc[0] - out["low"].iloc[0]

    out[col] = tr.ewm(alpha=1.0 / p, adjust=False).mean()
    return out


# ---------------------------------------------------------------------------
# Volume MA — Simple Moving Average of Volume
# ---------------------------------------------------------------------------

def volume_ma(df: pd.DataFrame, period: Optional[int] = None,
              col: str = "volume_ma") -> pd.DataFrame:
    """
    Compute a simple moving average of volume and append as *col*.

    Used for volume confirmation: a bar with volume > volume_ma indicates
    above-average participation, strengthening signal confidence.

    Args:
        period: Lookback (default: settings.signals.volume_ma_period = 20).
    """
    p = period or settings.signals.volume_ma_period
    _require_cols(df, "volume", fn_name=f"volume_ma({p})")
    _require_min_rows(df, p, f"volume_ma({p})")

    out = df.copy()
    out[col] = out["volume"].rolling(window=p, min_periods=p).mean()
    return out


# ---------------------------------------------------------------------------
# Regime score — directional efficiency ratio
# ---------------------------------------------------------------------------

def regime_score(df: pd.DataFrame, lookback: int = 20) -> float:
    """
    Classify market regime using directional efficiency ratio.

    Formula: abs(net_move) / sum(abs(bar_moves))

    Returns float 0.0 to 1.0:
        > 0.30 = strongly trending
        0.15 to 0.30 = mixed/transitional
        < 0.15 = ranging/choppy

    Requires at least lookback+1 rows.
    """
    _require_cols(df, "close", fn_name="regime_score")
    _require_min_rows(df, lookback + 1, "regime_score")

    closes = df["close"].iloc[-(lookback + 1):]
    bar_moves = closes.diff().abs().iloc[1:]
    total_movement = bar_moves.sum()
    net_movement = abs(float(closes.iloc[-1]) - float(closes.iloc[0]))

    if total_movement == 0:
        return 0.0
    return float(net_movement / total_movement)


# ---------------------------------------------------------------------------
# Convenience: compute all indicators in one call
# ---------------------------------------------------------------------------

def add_all(df: pd.DataFrame) -> pd.DataFrame:
    """
    Append all indicators to *df* in one call.

    Adds columns: vwap, ema_fast, ema_mid, ema_slow, rsi, atr, volume_ma.

    Requires at least settings.signals.ema_slow bars (21 by default).
    The signal engine calls this after seed_bars() to get a fully-populated
    DataFrame ready for strategy evaluation.

    Returns:
        New DataFrame with all indicator columns appended.
    """
    _require_min_rows(df, settings.signals.ema_slow, "add_all")

    out = vwap(df)
    out = ema_trio(out)
    out = rsi(out)
    out = atr(out)
    out = volume_ma(out)
    return out


# ---------------------------------------------------------------------------
# Helpers used by signal strategies
# ---------------------------------------------------------------------------

def latest(df: pd.DataFrame, col: str) -> float:
    """
    Return the most recent non-NaN value of *col*.

    Raises IndicatorError if the column doesn't exist or is all-NaN.
    Signal strategies call this to read the current indicator value.
    """
    if col not in df.columns:
        raise IndicatorError(f"Column '{col}' not found in DataFrame")
    series = df[col].dropna()
    if series.empty:
        raise IndicatorError(f"Column '{col}' has no valid (non-NaN) values")
    return float(series.iloc[-1])


def is_above(df: pd.DataFrame, col_a: str, col_b: str) -> bool:
    """True if the latest value of *col_a* > latest value of *col_b*."""
    return bool(latest(df, col_a) > latest(df, col_b))


def crossed_above(df: pd.DataFrame, col_a: str, col_b: str,
                  lookback: int = 2) -> bool:
    """
    True if *col_a* crossed above *col_b* on the most recent bar.

    A cross is: previous bar had col_a <= col_b, current bar has col_a > col_b.
    *lookback* rows must exist for a valid check.
    """
    if len(df) < lookback:
        return False
    tail = df[[col_a, col_b]].dropna().tail(lookback)
    if len(tail) < lookback:
        return False
    prev = tail.iloc[-2]
    curr = tail.iloc[-1]
    return bool(prev[col_a] <= prev[col_b] and curr[col_a] > curr[col_b])


def crossed_below(df: pd.DataFrame, col_a: str, col_b: str,
                  lookback: int = 2) -> bool:
    """
    True if *col_a* crossed below *col_b* on the most recent bar.
    """
    if len(df) < lookback:
        return False
    tail = df[[col_a, col_b]].dropna().tail(lookback)
    if len(tail) < lookback:
        return False
    prev = tail.iloc[-2]
    curr = tail.iloc[-1]
    return bool(prev[col_a] >= prev[col_b] and curr[col_a] < curr[col_b])
