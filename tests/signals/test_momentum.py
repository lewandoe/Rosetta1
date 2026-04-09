"""tests/signals/test_momentum.py"""
from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from data.indicators import add_all
from signals.momentum import MomentumSignal

ET = ZoneInfo("America/New_York")


def _base_df(rows: int = 80) -> pd.DataFrame:
    idx = pd.date_range(datetime(2024, 1, 8, 9, 30, tzinfo=ET), periods=rows, freq="1min")
    return pd.DataFrame(
        {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 80_000},
        index=idx,
    )


def _bullish_df() -> pd.DataFrame:
    """
    Gentle uptrend so RSI stays below overbought.
    Volume spiked on the last bar so vol > vol_ma.
    RSI overridden on last bar to stay in the 40-65 sweet spot.
    """
    df = _base_df(rows=80)
    for i in range(80):
        df.loc[df.index[i], "close"] = 100.0 + i * 0.12
        df.loc[df.index[i], "high"]  = df.loc[df.index[i], "close"] + 0.5
        df.loc[df.index[i], "low"]   = df.loc[df.index[i], "close"] - 0.5
    df["volume"] = 60_000
    df.loc[df.index[-1], "volume"] = 150_000   # spike: last bar > vol_ma
    df = add_all(df)
    df.loc[df.index[-1], "rsi"] = 55.0          # override: not overbought
    return df


def _bearish_df() -> pd.DataFrame:
    df = _base_df(rows=80)
    for i in range(80):
        df.loc[df.index[i], "close"] = 200.0 - i * 0.12
        df.loc[df.index[i], "high"]  = df.loc[df.index[i], "close"] + 0.5
        df.loc[df.index[i], "low"]   = df.loc[df.index[i], "close"] - 0.5
    df["volume"] = 60_000
    df.loc[df.index[-1], "volume"] = 150_000
    df = add_all(df)
    df.loc[df.index[-1], "rsi"] = 45.0          # override: not oversold
    return df


def _flat_df() -> pd.DataFrame:
    return add_all(_base_df(rows=60))


class TestMomentumSignal:
    def test_fires_long_in_uptrend(self):
        df = _bullish_df()
        sig = MomentumSignal()
        result = sig.evaluate(df, "SPY", float(df["close"].iloc[-1]))
        assert result is not None
        assert result.direction == "long"
        assert result.signal_type == "momentum"

    def test_fires_short_in_downtrend(self):
        df = _bearish_df()
        sig = MomentumSignal()
        result = sig.evaluate(df, "SPY", float(df["close"].iloc[-1]))
        assert result is not None
        assert result.direction == "short"

    def test_no_signal_in_flat_market(self):
        df = _flat_df()
        sig = MomentumSignal()
        result = sig.evaluate(df, "SPY", float(df["close"].iloc[-1]))
        assert result is None

    def test_long_blocked_when_rsi_overbought(self):
        df = _bullish_df()
        df.loc[df.index[-1], "rsi"] = 80.0   # force overbought
        sig = MomentumSignal()
        result = sig.evaluate(df, "SPY", float(df["close"].iloc[-1]))
        assert result is None

    def test_blocked_when_volume_below_ma(self):
        df = _bullish_df()
        # Reset volume spike so volume equals (not exceeds) MA
        df["volume"] = 60_000
        df["volume_ma"] = 60_000
        sig = MomentumSignal()
        result = sig.evaluate(df, "SPY", float(df["close"].iloc[-1]))
        assert result is None

    def test_confidence_in_valid_range(self):
        df = _bullish_df()
        sig = MomentumSignal()
        result = sig.evaluate(df, "SPY", float(df["close"].iloc[-1]))
        if result:
            assert 0 <= result.confidence <= 100

    def test_stop_below_entry_for_long(self):
        df = _bullish_df()
        sig = MomentumSignal()
        result = sig.evaluate(df, "SPY", float(df["close"].iloc[-1]))
        if result and result.direction == "long":
            assert result.stop_price < result.entry_price
            assert result.target_price > result.entry_price

    def test_stop_above_entry_for_short(self):
        df = _bearish_df()
        sig = MomentumSignal()
        result = sig.evaluate(df, "SPY", float(df["close"].iloc[-1]))
        if result and result.direction == "short":
            assert result.stop_price > result.entry_price
            assert result.target_price < result.entry_price

    def test_missing_indicator_columns_returns_none(self):
        df = _base_df(rows=30)
        sig = MomentumSignal()
        result = sig.evaluate(df, "SPY", 100.0)
        assert result is None

    def test_metadata_populated(self):
        df = _bullish_df()
        sig = MomentumSignal()
        result = sig.evaluate(df, "SPY", float(df["close"].iloc[-1]))
        if result:
            assert "ema_fast" in result.metadata
            assert "rsi" in result.metadata
