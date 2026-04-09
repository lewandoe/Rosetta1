"""tests/signals/test_ema_cross.py"""
from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from data.indicators import add_all
from signals.ema_cross import EmaCrossSignal

ET = ZoneInfo("America/New_York")


def _make_df(rows=80, start=100.0, step=0.0, volume=90_000):
    idx = pd.date_range(datetime(2024, 1, 8, 9, 30, tzinfo=ET), periods=rows, freq="1min")
    closes = [start + i * step for i in range(rows)]
    df = pd.DataFrame(index=idx)
    df["close"] = closes
    df["open"]   = df["close"]
    df["high"]   = df["close"] + 1.0
    df["low"]    = df["close"] - 1.0
    df["volume"] = volume
    return df


def _cross_above_df():
    """Fast EMA crosses above mid: flat then sharp up on last bars."""
    rows = 80
    closes = [100.0] * 50 + [100.0 + i * 1.5 for i in range(30)]
    idx = pd.date_range(datetime(2024, 1, 8, 9, 30, tzinfo=ET), periods=rows, freq="1min")
    df = pd.DataFrame({
        "open": closes, "high": [c + 1 for c in closes],
        "low": [c - 1 for c in closes], "close": closes,
        "volume": 100_000,
    }, index=idx)
    return add_all(df)


def _cross_below_df():
    rows = 80
    closes = [100.0] * 50 + [100.0 - i * 1.5 for i in range(30)]
    idx = pd.date_range(datetime(2024, 1, 8, 9, 30, tzinfo=ET), periods=rows, freq="1min")
    df = pd.DataFrame({
        "open": closes, "high": [c + 1 for c in closes],
        "low": [c - 1 for c in closes], "close": closes,
        "volume": 100_000,
    }, index=idx)
    return add_all(df)


class TestEmaCrossSignal:
    def test_missing_columns_returns_none(self):
        df = _make_df(rows=30)
        sig = EmaCrossSignal()
        result = sig.evaluate(df, "MSFT", 100.0)
        assert result is None

    def test_flat_market_no_cross(self):
        df = add_all(_make_df(rows=60, step=0.0))
        sig = EmaCrossSignal()
        result = sig.evaluate(df, "MSFT", 100.0)
        assert result is None

    def test_confidence_in_range(self):
        df = _cross_above_df()
        sig = EmaCrossSignal()
        result = sig.evaluate(df, "MSFT", float(df["close"].iloc[-1]))
        if result:
            assert 0 <= result.confidence <= 100

    def test_signal_type_correct(self):
        df = _cross_above_df()
        sig = EmaCrossSignal()
        result = sig.evaluate(df, "MSFT", float(df["close"].iloc[-1]))
        if result:
            assert result.signal_type == "ema_cross"

    def test_stop_and_target_sane(self):
        df = _cross_above_df()
        sig = EmaCrossSignal()
        result = sig.evaluate(df, "MSFT", float(df["close"].iloc[-1]))
        if result and result.direction == "long":
            assert result.stop_price < result.entry_price
            assert result.target_price > result.entry_price
        elif result and result.direction == "short":
            assert result.stop_price > result.entry_price
            assert result.target_price < result.entry_price

    def test_result_has_required_fields(self):
        df = _cross_above_df()
        sig = EmaCrossSignal()
        result = sig.evaluate(df, "MSFT", float(df["close"].iloc[-1]))
        if result:
            assert result.symbol == "MSFT"
            assert result.direction in ("long", "short")
            assert result.entry_price > 0
            assert result.estimated_hold_seconds > 0
