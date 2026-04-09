"""tests/signals/test_vwap.py"""
from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from data.indicators import add_all
from signals.vwap import VwapSignal

ET = ZoneInfo("America/New_York")


def _make_df(rows: int = 80, close: float = 100.0, volume: int = 60_000) -> pd.DataFrame:
    idx = pd.date_range(datetime(2024, 1, 8, 9, 30, tzinfo=ET), periods=rows, freq="1min")
    df = pd.DataFrame(
        {"open": close, "high": close + 1.0, "low": close - 1.0,
         "close": close, "volume": volume},
        index=idx,
    )
    return df


def _cross_above_vwap_df() -> pd.DataFrame:
    """
    Bullish EMA stack, then price crosses above VWAP on last bar.
    Volume spiked on last two bars so vol > vol_ma.
    """
    rows = 80
    df = _make_df(rows=rows)
    for i in range(rows):
        df.loc[df.index[i], "close"] = 100.0 + i * 0.15
        df.loc[df.index[i], "high"]  = df.loc[df.index[i], "close"] + 0.5
        df.loc[df.index[i], "low"]   = df.loc[df.index[i], "close"] - 0.5
    # Spike volume on last two bars so vol > vol_ma
    df["volume"] = 50_000
    df.loc[df.index[-2:], "volume"] = 200_000
    df = add_all(df)
    # Force cross: prev bar below vwap, current bar above
    vwap_val = float(df["vwap"].iloc[-1])
    df.loc[df.index[-2], "close"] = vwap_val - 0.30
    df.loc[df.index[-1], "close"] = vwap_val + 0.30
    return df


def _cross_below_vwap_df() -> pd.DataFrame:
    rows = 80
    df = _make_df(rows=rows)
    for i in range(rows):
        df.loc[df.index[i], "close"] = 200.0 - i * 0.15
        df.loc[df.index[i], "high"]  = df.loc[df.index[i], "close"] + 0.5
        df.loc[df.index[i], "low"]   = df.loc[df.index[i], "close"] - 0.5
    df["volume"] = 50_000
    df.loc[df.index[-2:], "volume"] = 200_000
    df = add_all(df)
    vwap_val = float(df["vwap"].iloc[-1])
    df.loc[df.index[-2], "close"] = vwap_val + 0.30
    df.loc[df.index[-1], "close"] = vwap_val - 0.30
    return df


class TestVwapSignal:
    def test_fires_long_on_cross_above(self):
        df = _cross_above_vwap_df()
        sig = VwapSignal()
        result = sig.evaluate(df, "AAPL", float(df["close"].iloc[-1]))
        assert result is not None
        assert result.direction == "long"
        assert result.signal_type == "vwap_cross"

    def test_fires_short_on_cross_below(self):
        df = _cross_below_vwap_df()
        sig = VwapSignal()
        result = sig.evaluate(df, "AAPL", float(df["close"].iloc[-1]))
        assert result is not None
        assert result.direction == "short"

    def test_no_signal_without_cross(self):
        df = add_all(_make_df(rows=60))
        sig = VwapSignal()
        result = sig.evaluate(df, "AAPL", 100.0)
        assert result is None

    def test_missing_columns_returns_none(self):
        df = _make_df(rows=30)
        sig = VwapSignal()
        result = sig.evaluate(df, "AAPL", 100.0)
        assert result is None

    def test_confidence_in_range(self):
        df = _cross_above_vwap_df()
        sig = VwapSignal()
        result = sig.evaluate(df, "AAPL", float(df["close"].iloc[-1]))
        if result:
            assert 0 <= result.confidence <= 100

    def test_stop_and_target_direction_correct(self):
        df = _cross_above_vwap_df()
        sig = VwapSignal()
        result = sig.evaluate(df, "AAPL", float(df["close"].iloc[-1]))
        if result and result.direction == "long":
            assert result.stop_price < result.entry_price
            assert result.target_price > result.entry_price
