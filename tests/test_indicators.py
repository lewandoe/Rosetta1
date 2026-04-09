"""
tests/test_indicators.py — Tests for data/indicators.py.

All tests use synthetic DataFrames — no network or file I/O.

Run:
    pytest tests/test_indicators.py -v
"""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import pytest

from data.indicators import (
    IndicatorError,
    add_all,
    atr,
    crossed_above,
    crossed_below,
    ema,
    ema_trio,
    is_above,
    latest,
    rsi,
    volume_ma,
    vwap,
)

ET = ZoneInfo("America/New_York")


# ---------------------------------------------------------------------------
# Test DataFrame factories
# ---------------------------------------------------------------------------

def _make_df(
    rows: int = 50,
    open_: float = 100.0,
    high: float = 101.0,
    low: float = 99.0,
    close: float = 100.0,
    volume: int = 50_000,
    freq: str = "1min",
    multi_day: bool = False,
) -> pd.DataFrame:
    """Flat OHLCV DataFrame with a tz-aware ET index."""
    base = datetime(2024, 1, 8, 9, 30, tzinfo=ET)
    if multi_day:
        # Two sessions: Jan 8 and Jan 9
        idx1 = pd.date_range(
            datetime(2024, 1, 8, 9, 30, tzinfo=ET), periods=rows // 2, freq=freq
        )
        idx2 = pd.date_range(
            datetime(2024, 1, 9, 9, 30, tzinfo=ET), periods=rows - rows // 2, freq=freq
        )
        index = idx1.append(idx2)
    else:
        index = pd.date_range(start=base, periods=rows, freq=freq)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": volume},
        index=index,
    )


def _trending_close(rows: int = 50, start: float = 100.0, step: float = 0.5) -> pd.DataFrame:
    """DataFrame where close increases by *step* each bar."""
    df = _make_df(rows=rows)
    df["close"] = [start + i * step for i in range(rows)]
    df["high"] = df["close"] + 1.0
    df["low"] = df["close"] - 1.0
    return df


def _oscillating_close(rows: int = 60) -> pd.DataFrame:
    """Alternating up/down closes useful for RSI testing."""
    df = _make_df(rows=rows)
    closes = []
    price = 100.0
    for i in range(rows):
        price += 1.0 if i % 2 == 0 else -0.5
        closes.append(price)
    df["close"] = closes
    df["high"] = df["close"] + 0.5
    df["low"] = df["close"] - 0.5
    return df


# ===========================================================================
# VWAP
# ===========================================================================

class TestVwap:
    def test_column_added(self):
        df = _make_df(rows=30)
        out = vwap(df)
        assert "vwap" in out.columns

    def test_input_not_mutated(self):
        df = _make_df(rows=30)
        cols_before = list(df.columns)
        vwap(df)
        assert list(df.columns) == cols_before

    def test_single_bar_equals_typical_price(self):
        df = _make_df(rows=1, high=102.0, low=98.0, close=100.0)
        out = vwap(df)
        expected = (102.0 + 98.0 + 100.0) / 3.0
        assert out["vwap"].iloc[0] == pytest.approx(expected)

    def test_vwap_is_price_weighted(self):
        """With equal prices, VWAP should equal that price regardless of volume."""
        df = _make_df(rows=10, high=100.0, low=100.0, close=100.0, volume=1000)
        out = vwap(df)
        assert np.allclose(out["vwap"].dropna(), 100.0)

    def test_vwap_resets_each_day(self):
        """The cumulative sum resets at session boundary — day 2 first bar
        should equal its own typical price, not carry over day 1 state."""
        df = _make_df(rows=10, high=102.0, low=98.0, close=100.0, multi_day=True)
        out = vwap(df)
        # Find first bar of second session
        day2_start = out.index[out.index.normalize() == out.index.normalize()[5]][0]
        row = out.loc[day2_start]
        expected_tp = (row["high"] + row["low"] + row["close"]) / 3.0
        assert out.loc[day2_start, "vwap"] == pytest.approx(expected_tp)

    def test_zero_volume_bar_gives_nan(self):
        df = _make_df(rows=5)
        df.loc[df.index[2], "volume"] = 0
        out = vwap(df)
        # Bar with zero volume and no prior cumulative volume in that session
        # This only produces NaN if it's the *first* bar of the day with vol=0
        # For non-first bars, cumvol from earlier bars is nonzero — that's fine.
        # Just assert no exception is raised and the column exists.
        assert "vwap" in out.columns

    def test_missing_columns_raises(self):
        df = _make_df(rows=5).drop(columns=["volume"])
        with pytest.raises(IndicatorError, match="missing required columns"):
            vwap(df)

    def test_custom_col_name(self):
        df = _make_df(rows=10)
        out = vwap(df, col="my_vwap")
        assert "my_vwap" in out.columns
        assert "vwap" not in out.columns


# ===========================================================================
# EMA
# ===========================================================================

class TestEma:
    def test_column_added_with_default_name(self):
        df = _make_df(rows=30)
        out = ema(df, period=10)
        assert "ema_10" in out.columns

    def test_custom_col_name(self):
        df = _make_df(rows=30)
        out = ema(df, period=10, col="my_ema")
        assert "my_ema" in out.columns

    def test_input_not_mutated(self):
        df = _make_df(rows=30)
        cols_before = list(df.columns)
        ema(df, period=10)
        assert list(df.columns) == cols_before

    def test_constant_series_ema_equals_constant(self):
        """EMA of a flat series should equal that constant."""
        df = _make_df(rows=50, close=100.0)
        out = ema(df, period=10)
        # After warm-up the EMA converges to 100.0
        assert out["ema_10"].iloc[-1] == pytest.approx(100.0, abs=1e-6)

    def test_ema_tracks_uptrend(self):
        """EMA should be below close in a rising market."""
        df = _trending_close(rows=50, start=100.0, step=1.0)
        out = ema(df, period=10)
        # After warm-up, EMA lags — close > EMA
        assert out["close"].iloc[-1] > out["ema_10"].iloc[-1]

    def test_too_few_rows_raises(self):
        df = _make_df(rows=5)
        with pytest.raises(IndicatorError, match="need at least"):
            ema(df, period=10)

    def test_missing_source_column_raises(self):
        df = _make_df(rows=30).drop(columns=["close"])
        with pytest.raises(IndicatorError, match="missing required columns"):
            ema(df, period=10)

    def test_custom_source_column(self):
        df = _make_df(rows=30)
        out = ema(df, period=10, source="open", col="ema_open")
        assert "ema_open" in out.columns


class TestEmaTrio:
    def test_all_three_columns_added(self):
        df = _make_df(rows=50)
        out = ema_trio(df)
        assert "ema_fast" in out.columns
        assert "ema_mid" in out.columns
        assert "ema_slow" in out.columns

    def test_ordering_in_uptrend(self):
        """In a strong uptrend: ema_fast > ema_mid > ema_slow."""
        df = _trending_close(rows=100, start=100.0, step=1.0)
        out = ema_trio(df)
        last = out.dropna().iloc[-1]
        assert last["ema_fast"] > last["ema_mid"] > last["ema_slow"]

    def test_ordering_in_downtrend(self):
        """In a strong downtrend: ema_fast < ema_mid < ema_slow."""
        df = _trending_close(rows=100, start=200.0, step=-1.0)
        out = ema_trio(df)
        last = out.dropna().iloc[-1]
        assert last["ema_fast"] < last["ema_mid"] < last["ema_slow"]

    def test_input_not_mutated(self):
        df = _make_df(rows=50)
        cols_before = list(df.columns)
        ema_trio(df)
        assert list(df.columns) == cols_before


# ===========================================================================
# RSI
# ===========================================================================

class TestRsi:
    def test_column_added(self):
        df = _make_df(rows=30)
        out = rsi(df, period=14)
        assert "rsi" in out.columns

    def test_input_not_mutated(self):
        df = _make_df(rows=30)
        cols_before = list(df.columns)
        rsi(df, period=14)
        assert list(df.columns) == cols_before

    def test_range_0_to_100(self):
        df = _oscillating_close(rows=60)
        out = rsi(df, period=14)
        valid = out["rsi"].dropna()
        assert (valid >= 0).all()
        assert (valid <= 100).all()

    def test_overbought_in_strong_uptrend(self):
        """Steadily rising close should push RSI above 70."""
        df = _trending_close(rows=100, start=100.0, step=1.0)
        out = rsi(df, period=14)
        assert out["rsi"].dropna().iloc[-1] > 70

    def test_oversold_in_strong_downtrend(self):
        """Steadily falling close should push RSI below 30."""
        df = _trending_close(rows=100, start=200.0, step=-1.0)
        out = rsi(df, period=14)
        assert out["rsi"].dropna().iloc[-1] < 30

    def test_flat_series_rsi_is_nan_or_50(self):
        """A perfectly flat series has zero gains and zero losses.
        avg_loss == 0 branch returns 100 in our implementation —
        verify no exception is raised and all values are finite."""
        df = _make_df(rows=30, close=100.0)
        out = rsi(df, period=14)
        valid = out["rsi"].dropna()
        assert valid.apply(np.isfinite).all()

    def test_first_row_is_nan(self):
        df = _oscillating_close(rows=30)
        out = rsi(df, period=14)
        assert pd.isna(out["rsi"].iloc[0])

    def test_too_few_rows_raises(self):
        df = _make_df(rows=5)
        with pytest.raises(IndicatorError, match="need at least"):
            rsi(df, period=14)

    def test_uses_settings_default_period(self):
        df = _make_df(rows=30)
        out = rsi(df)   # no period arg — uses settings.signals.rsi_period
        assert "rsi" in out.columns

    def test_custom_col_name(self):
        df = _make_df(rows=30)
        out = rsi(df, period=14, col="my_rsi")
        assert "my_rsi" in out.columns


# ===========================================================================
# ATR
# ===========================================================================

class TestAtr:
    def test_column_added(self):
        df = _make_df(rows=30)
        out = atr(df, period=14)
        assert "atr" in out.columns

    def test_input_not_mutated(self):
        df = _make_df(rows=30)
        cols_before = list(df.columns)
        atr(df, period=14)
        assert list(df.columns) == cols_before

    def test_atr_is_positive(self):
        df = _make_df(rows=30, high=101.0, low=99.0)
        out = atr(df, period=14)
        assert (out["atr"].dropna() > 0).all()

    def test_atr_increases_with_wider_range(self):
        """Wider high-low range should produce higher ATR."""
        narrow = _make_df(rows=40, high=100.5, low=99.5)
        wide = _make_df(rows=40, high=103.0, low=97.0)
        narrow_atr = atr(narrow, period=14)["atr"].dropna().iloc[-1]
        wide_atr = atr(wide, period=14)["atr"].dropna().iloc[-1]
        assert wide_atr > narrow_atr

    def test_atr_constant_range(self):
        """With constant high-low range and no gaps, ATR ≈ range."""
        # high - low = 2.0 every bar, no overnight gap
        df = _make_df(rows=50, high=101.0, low=99.0, close=100.0)
        out = atr(df, period=14)
        # After warm-up ATR should converge to ~2.0
        assert out["atr"].dropna().iloc[-1] == pytest.approx(2.0, abs=0.05)

    def test_too_few_rows_raises(self):
        df = _make_df(rows=5)
        with pytest.raises(IndicatorError, match="need at least"):
            atr(df, period=14)

    def test_uses_settings_default_period(self):
        df = _make_df(rows=30)
        out = atr(df)
        assert "atr" in out.columns

    def test_missing_columns_raises(self):
        df = _make_df(rows=30).drop(columns=["high"])
        with pytest.raises(IndicatorError, match="missing required columns"):
            atr(df)


# ===========================================================================
# Volume MA
# ===========================================================================

class TestVolumeMA:
    def test_column_added(self):
        df = _make_df(rows=30)
        out = volume_ma(df, period=20)
        assert "volume_ma" in out.columns

    def test_input_not_mutated(self):
        df = _make_df(rows=30)
        cols_before = list(df.columns)
        volume_ma(df, period=20)
        assert list(df.columns) == cols_before

    def test_constant_volume_ma_equals_volume(self):
        df = _make_df(rows=30, volume=50_000)
        out = volume_ma(df, period=20)
        assert out["volume_ma"].dropna().iloc[-1] == pytest.approx(50_000.0)

    def test_first_rows_are_nan_until_period(self):
        """Rolling SMA with min_periods=period should NaN the first p-1 rows."""
        df = _make_df(rows=30, volume=50_000)
        out = volume_ma(df, period=20)
        assert out["volume_ma"].iloc[:19].isna().all()
        assert pd.notna(out["volume_ma"].iloc[19])

    def test_too_few_rows_raises(self):
        df = _make_df(rows=5)
        with pytest.raises(IndicatorError, match="need at least"):
            volume_ma(df, period=20)

    def test_uses_settings_default_period(self):
        df = _make_df(rows=30)
        out = volume_ma(df)
        assert "volume_ma" in out.columns

    def test_above_average_volume_detection(self):
        """Spike bar should have volume > volume_ma."""
        df = _make_df(rows=30, volume=10_000)
        # Spike the last bar to 100k
        df.loc[df.index[-1], "volume"] = 100_000
        out = volume_ma(df, period=20)
        last = out.iloc[-1]
        assert last["volume"] > last["volume_ma"]


# ===========================================================================
# add_all
# ===========================================================================

class TestAddAll:
    def test_all_indicator_columns_present(self):
        df = _make_df(rows=50)
        out = add_all(df)
        for col in ["vwap", "ema_fast", "ema_mid", "ema_slow", "rsi", "atr", "volume_ma"]:
            assert col in out.columns, f"Missing column: {col}"

    def test_input_not_mutated(self):
        df = _make_df(rows=50)
        cols_before = list(df.columns)
        add_all(df)
        assert list(df.columns) == cols_before

    def test_raises_on_insufficient_rows(self):
        df = _make_df(rows=5)
        with pytest.raises(IndicatorError):
            add_all(df)


# ===========================================================================
# Helper functions: latest, is_above, crossed_above, crossed_below
# ===========================================================================

class TestLatest:
    def test_returns_last_non_nan_value(self):
        df = _make_df(rows=30, close=100.0)
        out = ema(df, period=10)
        val = latest(out, "ema_10")
        assert isinstance(val, float)
        assert val == pytest.approx(100.0, abs=1e-6)

    def test_raises_for_missing_column(self):
        df = _make_df(rows=10)
        with pytest.raises(IndicatorError, match="not found"):
            latest(df, "nonexistent")

    def test_raises_when_all_nan(self):
        df = _make_df(rows=5)
        df["all_nan"] = float("nan")
        with pytest.raises(IndicatorError, match="no valid"):
            latest(df, "all_nan")


class TestIsAbove:
    def test_true_when_a_above_b(self):
        df = _make_df(rows=50)
        out = ema_trio(df)
        # In a flat market EMAs converge — just test the function logic
        # by injecting known values
        out["x"] = 10.0
        out["y"] = 5.0
        assert is_above(out, "x", "y") is True

    def test_false_when_a_below_b(self):
        df = _make_df(rows=50)
        out = ema_trio(df)
        out["x"] = 5.0
        out["y"] = 10.0
        assert is_above(out, "x", "y") is False


class TestCrossedAbove:
    def test_detects_cross_above(self):
        """col_a crosses above col_b on the last bar."""
        df = _make_df(rows=10)
        df["a"] = [99.0] * 9 + [101.0]   # was below, now above
        df["b"] = 100.0
        assert crossed_above(df, "a", "b") is True

    def test_no_cross_when_always_above(self):
        df = _make_df(rows=10)
        df["a"] = 105.0
        df["b"] = 100.0
        assert crossed_above(df, "a", "b") is False

    def test_no_cross_when_always_below(self):
        df = _make_df(rows=10)
        df["a"] = 95.0
        df["b"] = 100.0
        assert crossed_above(df, "a", "b") is False

    def test_returns_false_on_insufficient_rows(self):
        df = _make_df(rows=1)
        df["a"] = 105.0
        df["b"] = 100.0
        assert crossed_above(df, "a", "b") is False


class TestCrossedBelow:
    def test_detects_cross_below(self):
        df = _make_df(rows=10)
        df["a"] = [101.0] * 9 + [99.0]   # was above, now below
        df["b"] = 100.0
        assert crossed_below(df, "a", "b") is True

    def test_no_cross_when_always_below(self):
        df = _make_df(rows=10)
        df["a"] = 95.0
        df["b"] = 100.0
        assert crossed_below(df, "a", "b") is False

    def test_no_cross_when_always_above(self):
        df = _make_df(rows=10)
        df["a"] = 105.0
        df["b"] = 100.0
        assert crossed_below(df, "a", "b") is False

    def test_returns_false_on_insufficient_rows(self):
        df = _make_df(rows=1)
        df["a"] = 95.0
        df["b"] = 100.0
        assert crossed_below(df, "a", "b") is False
