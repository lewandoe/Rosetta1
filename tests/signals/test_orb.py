"""tests/signals/test_orb.py"""
from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from data.indicators import add_all
from signals.orb import OrbSignal

ET = ZoneInfo("America/New_York")

ORB_MINS = 15


def _orb_df(breakout: str = "long") -> pd.DataFrame:
    """
    Two-day DataFrame so indicators are fully seeded before today's session:
      - Day 1 (Jan 7): 30 bars of flat history — seeds volume_ma(20) etc.
      - Day 2 (Jan 8): 16 bars — ORB_MINS range-forming + 1 breakout bar.

    The ORB signal filters to today_bars only for the range, but uses the
    full df for indicator lookups (volume_ma, atr, etc.).
    """
    # ── Day 1: seed bars ────────────────────────────────────────────────
    d1_idx = pd.date_range(datetime(2024, 1, 7, 9, 30, tzinfo=ET), periods=30, freq="1min")
    d1 = pd.DataFrame(
        {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 80_000},
        index=d1_idx,
    )

    # ── Day 2: opening range + breakout ─────────────────────────────────
    or_high = 101.0
    or_low  = 99.0
    today_rows = ORB_MINS + 1   # 15 range bars + 1 breakout

    closes, highs, lows, vols = [], [], [], []
    for i in range(today_rows):
        if i < ORB_MINS:
            c = 100.0 + (0.2 if i % 2 == 0 else -0.2)
            h, l, v = or_high, or_low, 80_000
        else:
            if breakout == "long":
                c = or_high + 0.50
            else:
                c = or_low  - 0.50
            h = c + 0.3
            l = c - 0.3
            v = 200_000   # high-volume breakout bar

        closes.append(c); highs.append(h); lows.append(l); vols.append(v)

    d2_idx = pd.date_range(datetime(2024, 1, 8, 9, 30, tzinfo=ET), periods=today_rows, freq="1min")
    d2 = pd.DataFrame(
        {"open": closes, "high": highs, "low": lows, "close": closes, "volume": vols},
        index=d2_idx,
    )

    df = pd.concat([d1, d2])

    # Strong trend for EMA confirmation
    if breakout == "long":
        for i in range(len(df)):
            df.loc[df.index[i], "close"] = max(df.loc[df.index[i], "close"],
                                               99.0 + i * 0.05)
    else:
        for i in range(len(df)):
            df.loc[df.index[i], "close"] = min(df.loc[df.index[i], "close"],
                                               101.0 - i * 0.05)
    return add_all(df)


class TestOrbSignal:
    def test_fires_long_on_breakout_above(self):
        df = _orb_df(breakout="long")
        sig = OrbSignal()
        breakout_price = float(df["close"].iloc[-1])
        result = sig.evaluate(df, "NVDA", breakout_price)
        assert result is not None
        assert result.direction == "long"
        assert result.signal_type == "orb"

    def test_fires_short_on_breakout_below(self):
        df = _orb_df(breakout="short")
        sig = OrbSignal()
        breakout_price = float(df["close"].iloc[-1])
        result = sig.evaluate(df, "NVDA", breakout_price)
        assert result is not None
        assert result.direction == "short"

    def test_no_signal_while_range_forming(self):
        """Pass only day-1 seed + partial day-2 range — not enough range bars."""
        df = _orb_df(breakout="long")
        sig = OrbSignal()
        # Slice to only 5 bars into today (range not complete)
        partial = df.iloc[:35]   # 30 seed + 5 today
        result = sig.evaluate(partial, "NVDA", float(partial["close"].iloc[-1]))
        assert result is None

    def test_no_signal_inside_range(self):
        df = _orb_df(breakout="long")
        sig = OrbSignal()
        # Pass all bars but supply a price inside the OR [99, 101]
        result = sig.evaluate(df, "NVDA", 100.0)
        assert result is None

    def test_fire_once_per_direction(self):
        df = _orb_df(breakout="long")
        sig = OrbSignal()
        price = float(df["close"].iloc[-1])
        r1 = sig.evaluate(df, "NVDA", price)
        r2 = sig.evaluate(df, "NVDA", price)
        assert r1 is not None
        assert r2 is None   # second call: already fired for (date, long)

    def test_reset_session_allows_re_signal(self):
        df = _orb_df(breakout="long")
        sig = OrbSignal()
        price = float(df["close"].iloc[-1])
        sig.evaluate(df, "NVDA", price)
        sig.reset_session()
        r2 = sig.evaluate(df, "NVDA", price)
        assert r2 is not None

    def test_confidence_in_range(self):
        df = _orb_df(breakout="long")
        sig = OrbSignal()
        result = sig.evaluate(df, "NVDA", float(df["close"].iloc[-1]))
        if result:
            assert 0 <= result.confidence <= 100

    def test_metadata_has_or_levels(self):
        df = _orb_df(breakout="long")
        sig = OrbSignal()
        result = sig.evaluate(df, "NVDA", float(df["close"].iloc[-1]))
        if result:
            assert "or_high" in result.metadata
            assert "or_low" in result.metadata
