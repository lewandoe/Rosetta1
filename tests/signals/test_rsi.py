"""tests/signals/test_rsi.py"""
from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from data.indicators import add_all
from signals.rsi import RsiSignal

ET = ZoneInfo("America/New_York")


def _make_df(rows=60, start=100.0, step=0.0, volume=80_000):
    idx = pd.date_range(datetime(2024, 1, 8, 9, 30, tzinfo=ET), periods=rows, freq="1min")
    closes = [start + i * step for i in range(rows)]
    df = pd.DataFrame({
        "open": closes, "high": [c + 0.5 for c in closes],
        "low": [c - 0.5 for c in closes], "close": closes,
        "volume": volume,
    }, index=idx)
    return df


def _oversold_bounce_df() -> pd.DataFrame:
    """
    Shallow decline so price stays near EMA slow (structural check passes).
    Spike volume at the bounce bar.
    Manually set RSI cross on last two bars to guarantee the signal fires.
    """
    rows = 60
    closes = [100.0 - i * 0.3 for i in range(30)] + [91.0 + i * 0.3 for i in range(30)]
    idx = pd.date_range(datetime(2024, 1, 8, 9, 30, tzinfo=ET), periods=rows, freq="1min")
    df = pd.DataFrame({
        "open": closes, "high": [c + 0.5 for c in closes],
        "low": [c - 0.5 for c in closes], "close": closes,
        "volume": 80_000,
    }, index=idx)
    df.loc[df.index[-1], "volume"] = 200_000   # volume spike on bounce bar
    df = add_all(df)
    # Force RSI cross: previous bar was below 30, current is above 30
    df.loc[df.index[-2], "rsi"] = 27.0
    df.loc[df.index[-1], "rsi"] = 32.0
    # Ensure structural condition: price >= ema_slow
    df.loc[df.index[-1], "ema_slow"] = df["close"].iloc[-1] - 1.0
    return df


def _overbought_reversal_df() -> pd.DataFrame:
    rows = 60
    closes = [100.0 + i * 0.3 for i in range(30)] + [109.0 - i * 0.3 for i in range(30)]
    idx = pd.date_range(datetime(2024, 1, 8, 9, 30, tzinfo=ET), periods=rows, freq="1min")
    df = pd.DataFrame({
        "open": closes, "high": [c + 0.5 for c in closes],
        "low": [c - 0.5 for c in closes], "close": closes,
        "volume": 80_000,
    }, index=idx)
    df.loc[df.index[-1], "volume"] = 200_000
    df = add_all(df)
    df.loc[df.index[-2], "rsi"] = 73.0
    df.loc[df.index[-1], "rsi"] = 68.0
    # Ensure structural condition: price <= ema_slow
    df.loc[df.index[-1], "ema_slow"] = df["close"].iloc[-1] + 1.0
    return df


class TestRsiSignal:
    def test_fires_long_on_oversold_cross(self):
        df = _oversold_bounce_df()
        sig = RsiSignal()
        result = sig.evaluate(df, "AMD", float(df["close"].iloc[-1]))
        assert result is not None
        assert result.direction == "long"
        assert result.signal_type == "rsi_reversal"

    def test_fires_short_on_overbought_cross(self):
        df = _overbought_reversal_df()
        sig = RsiSignal()
        result = sig.evaluate(df, "AMD", float(df["close"].iloc[-1]))
        assert result is not None
        assert result.direction == "short"

    def test_no_signal_in_neutral_rsi(self):
        df = add_all(_make_df(rows=50))
        sig = RsiSignal()
        result = sig.evaluate(df, "AMD", 100.0)
        assert result is None

    def test_missing_columns_returns_none(self):
        df = _make_df(rows=30)
        sig = RsiSignal()
        result = sig.evaluate(df, "AMD", 100.0)
        assert result is None

    def test_confidence_in_range(self):
        df = _oversold_bounce_df()
        sig = RsiSignal()
        result = sig.evaluate(df, "AMD", float(df["close"].iloc[-1]))
        if result:
            assert 0 <= result.confidence <= 100

    def test_stop_and_target_sane_long(self):
        df = _oversold_bounce_df()
        sig = RsiSignal()
        result = sig.evaluate(df, "AMD", float(df["close"].iloc[-1]))
        if result and result.direction == "long":
            assert result.stop_price < result.entry_price
            assert result.target_price > result.entry_price

    def test_blocked_when_volume_below_ma(self):
        df = _oversold_bounce_df()
        # Set volume <= volume_ma on last bar
        df.loc[df.index[-1], "volume"] = 1
        df.loc[df.index[-1], "volume_ma"] = 100_000
        sig = RsiSignal()
        result = sig.evaluate(df, "AMD", float(df["close"].iloc[-1]))
        assert result is None

    def test_metadata_has_rsi_values(self):
        df = _oversold_bounce_df()
        sig = RsiSignal()
        result = sig.evaluate(df, "AMD", float(df["close"].iloc[-1]))
        if result:
            assert "rsi_prev" in result.metadata
            assert "rsi_curr" in result.metadata
