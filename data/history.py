"""
data/history.py — Historical OHLCV data via yfinance.

Used by:
  - Indicator seeding at startup (warm up EMA/VWAP/ATR before live trading)
  - Backtesting (Stage 10)

Design rules:
  - Always returns a DataFrame with a tz-aware DatetimeIndex (US/Eastern).
  - Validates that at least settings.backtest.min_history_days of data were
    returned before the caller can rely on fully-seeded indicators.
  - Never returns partially-populated rows — drops any row missing OHLCV.
  - Raises HistoryError (not a silent empty return) on failure so callers
    know their indicators are NOT seeded.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import Optional

import pandas as pd
import yfinance as yf

from config.settings import UNIVERSE, settings

logger = logging.getLogger(__name__)

# Standard column names used everywhere in this codebase
OHLCV_COLS = ["open", "high", "low", "close", "volume"]


class HistoryError(Exception):
    """Raised when historical data cannot be fetched or is insufficient."""


def fetch(
    symbol: str,
    days: int = 0,
    interval: str = "1m",
    start: Optional[date] = None,
    end: Optional[date] = None,
) -> pd.DataFrame:
    """
    Fetch OHLCV history for *symbol* and return a clean DataFrame.

    Args:
        symbol:   Ticker — must be in UNIVERSE.
        days:     Number of calendar days of history to fetch (end = today).
                  Ignored if *start* is provided.
        interval: yfinance interval string: '1m', '2m', '5m', '15m', '1h', '1d'.
                  Note: yfinance only provides minute-level data for the last
                  30 days (1m) or 60 days (2m/5m).
        start:    Explicit start date (overrides *days*).
        end:      Explicit end date (defaults to today).

    Returns:
        DataFrame with lowercase columns [open, high, low, close, volume],
        tz-aware DatetimeIndex in US/Eastern.

    Raises:
        HistoryError: if symbol invalid, fetch fails, or result is empty.
    """
    if symbol not in UNIVERSE:
        raise HistoryError(
            f"{symbol} is not in the trading universe. Allowed: {UNIVERSE}"
        )

    today = date.today()
    end_date = end or today
    if start is not None:
        start_date = start
    elif days > 0:
        start_date = end_date - timedelta(days=days)
    else:
        # Default: enough history to seed all indicators + meet backtest minimum
        start_date = end_date - timedelta(days=settings.backtest.min_history_days + 5)

    logger.debug(
        "Fetching %s %s history: %s → %s", symbol, interval, start_date, end_date
    )

    try:
        ticker = yf.Ticker(symbol)
        raw: pd.DataFrame = ticker.history(
            start=str(start_date),
            end=str(end_date + timedelta(days=1)),  # yfinance end is exclusive
            interval=interval,
            auto_adjust=True,
            prepost=False,   # regular session only — no pre/post market data
        )
    except Exception as exc:
        raise HistoryError(
            f"yfinance fetch failed for {symbol} ({interval}): {exc}"
        ) from exc

    if raw is None or raw.empty:
        raise HistoryError(
            f"No data returned for {symbol} ({interval}) from {start_date} to {end_date}"
        )

    return _clean(raw, symbol, interval)


def fetch_multi(
    symbols: list[str] = UNIVERSE,
    days: int = 0,
    interval: str = "1m",
) -> dict[str, pd.DataFrame]:
    """
    Fetch history for multiple symbols and return a dict keyed by symbol.

    Symbols that fail are logged and excluded from the result — callers must
    check that all expected symbols are present.
    """
    result: dict[str, pd.DataFrame] = {}
    for sym in symbols:
        try:
            result[sym] = fetch(sym, days=days, interval=interval)
        except HistoryError as exc:
            logger.error("history.fetch_multi: skipping %s — %s", sym, exc)
    return result


def seed_bars(symbol: str, interval: str = "1m") -> pd.DataFrame:
    """
    Convenience wrapper: fetch exactly enough history to seed all indicators
    without requiring a full backtest window.

    Minimum needed = max(EMA slow period, ATR period, volume MA period) × bar
    count.  For 1-minute bars, 3 days of data gives ~1,170 bars — far more
    than the 21-period EMA or 20-period volume MA needs.

    Always fetches 7 calendar days to account for weekends.
    """
    return fetch(symbol, days=7, interval=interval)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _clean(raw: pd.DataFrame, symbol: str, interval: str) -> pd.DataFrame:
    """
    Normalise a raw yfinance DataFrame:
      1. Lowercase column names.
      2. Drop Dividends / Stock Splits columns if present.
      3. Drop rows with any NaN in OHLCV.
      4. Convert index to US/Eastern tz-aware.
      5. Sort ascending.
      6. Attach symbol and interval as DataFrame attrs for downstream tracing.
    """
    df = raw.copy()

    # 1. Lowercase
    df.columns = [c.lower() for c in df.columns]

    # 2. Keep only OHLCV
    present = [c for c in OHLCV_COLS if c in df.columns]
    missing = set(OHLCV_COLS) - set(present)
    if missing:
        raise HistoryError(
            f"yfinance returned data for {symbol} missing columns: {missing}"
        )
    df = df[OHLCV_COLS].copy()

    # 3. Drop incomplete rows
    before = len(df)
    df.dropna(subset=OHLCV_COLS, inplace=True)
    dropped = before - len(df)
    if dropped:
        logger.debug("%s: dropped %d rows with NaN values", symbol, dropped)

    if df.empty:
        raise HistoryError(
            f"All rows dropped after NaN filter for {symbol} ({interval})"
        )

    # 4. Ensure tz-aware index in US/Eastern
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df.index = df.index.tz_convert("America/New_York")

    # 5. Sort ascending (yfinance should already be, but be safe)
    df.sort_index(inplace=True)

    # 6. Attach metadata
    df.attrs["symbol"] = symbol
    df.attrs["interval"] = interval

    logger.debug(
        "Fetched %d bars for %s (%s): %s → %s",
        len(df), symbol, interval,
        df.index[0].isoformat(), df.index[-1].isoformat(),
    )
    return df


def validate_sufficient_history(df: pd.DataFrame, min_bars: int) -> None:
    """
    Assert that *df* contains at least *min_bars* rows.

    Raises HistoryError if the check fails.  Call this after fetch() in any
    code path that computes indicators — better to fail early with a clear
    message than to get NaN-filled indicator columns silently.
    """
    if len(df) < min_bars:
        symbol = df.attrs.get("symbol", "?")
        interval = df.attrs.get("interval", "?")
        raise HistoryError(
            f"Insufficient history for {symbol} ({interval}): "
            f"need {min_bars} bars, got {len(df)}"
        )
