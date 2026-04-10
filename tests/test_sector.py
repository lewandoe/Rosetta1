"""
tests/test_sector.py — Unit tests for cross-symbol sector confirmation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data.sector import (
    SECTOR_MAP,
    get_sector_etf,
    get_etf_trend,
    sector_confidence_adjustment,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_bars(closes: list[float]) -> pd.DataFrame:
    n = len(closes)
    arr = np.array(closes, dtype=float)
    return pd.DataFrame(
        {"open": arr, "high": arr + 0.10, "low": arr - 0.10,
         "close": arr, "volume": [1_000_000] * n},
        index=pd.date_range("2024-01-02 09:30", periods=n, freq="1min",
                            tz="America/New_York"),
    )


# ---------------------------------------------------------------------------
# get_sector_etf()
# ---------------------------------------------------------------------------

class TestGetSectorEtf:
    def test_spy_is_self_referential(self):
        assert get_sector_etf("SPY") == "SPY"

    def test_qqq_maps_to_spy(self):
        assert get_sector_etf("QQQ") == "SPY"

    def test_nvda_maps_to_qqq(self):
        assert get_sector_etf("NVDA") == "QQQ"

    def test_tsla_maps_to_qqq(self):
        assert get_sector_etf("TSLA") == "QQQ"

    def test_amd_maps_to_qqq(self):
        assert get_sector_etf("AMD") == "QQQ"

    def test_unknown_symbol_defaults_to_spy(self):
        assert get_sector_etf("XYZ") == "SPY"

    def test_all_sector_map_values_are_valid_etfs(self):
        valid_etfs = {"SPY", "QQQ"}
        for sym, etf in SECTOR_MAP.items():
            assert etf in valid_etfs, f"{sym} maps to unknown ETF {etf}"


# ---------------------------------------------------------------------------
# get_etf_trend()
# ---------------------------------------------------------------------------

class TestGetEtfTrend:
    def test_rising_bars_return_up(self):
        bars = _make_bars([100.0 + i * 0.2 for i in range(10)])
        result = get_etf_trend("QQQ", {"QQQ": bars}, lookback=5)
        assert result == "up"

    def test_falling_bars_return_down(self):
        bars = _make_bars([100.0 - i * 0.2 for i in range(10)])
        result = get_etf_trend("QQQ", {"QQQ": bars}, lookback=5)
        assert result == "down"

    def test_flat_bars_return_flat(self):
        # Change < 0.05% — must be truly flat
        bars = _make_bars([100.0] * 10)
        result = get_etf_trend("SPY", {"SPY": bars}, lookback=5)
        assert result == "flat"

    def test_missing_etf_returns_none(self):
        result = get_etf_trend("QQQ", {}, lookback=5)
        assert result is None

    def test_insufficient_bars_returns_none(self):
        bars = _make_bars([100.0, 101.0])  # only 2 bars, lookback=5 needs 6
        result = get_etf_trend("QQQ", {"QQQ": bars}, lookback=5)
        assert result is None

    def test_zero_start_price_returns_none(self):
        closes = [0.0] + [100.0] * 9
        bars = _make_bars(closes)
        result = get_etf_trend("QQQ", {"QQQ": bars}, lookback=5)
        assert result is None

    def test_uppercase_close_column_supported(self):
        """get_etf_trend handles both 'close' and 'Close' column names."""
        n = 10
        arr = np.array([100.0 + i * 0.3 for i in range(n)])
        bars = pd.DataFrame(
            {"Close": arr},
            index=pd.date_range("2024-01-02 09:30", periods=n, freq="1min",
                                tz="America/New_York"),
        )
        result = get_etf_trend("QQQ", {"QQQ": bars}, lookback=5)
        assert result == "up"


# ---------------------------------------------------------------------------
# sector_confidence_adjustment()
# ---------------------------------------------------------------------------

class TestSectorConfidenceAdjustment:
    def _qqq_up(self) -> dict[str, pd.DataFrame]:
        return {"QQQ": _make_bars([100.0 + i * 0.3 for i in range(10)])}

    def _qqq_down(self) -> dict[str, pd.DataFrame]:
        return {"QQQ": _make_bars([100.0 - i * 0.3 for i in range(10)])}

    def _qqq_flat(self) -> dict[str, pd.DataFrame]:
        return {"QQQ": _make_bars([100.0] * 10)}

    def test_long_with_etf_up_returns_bonus(self):
        adj = sector_confidence_adjustment(
            "NVDA", "long", self._qqq_up(), lookback=5, bonus=8, penalty=12
        )
        assert adj == 8

    def test_short_with_etf_down_returns_bonus(self):
        adj = sector_confidence_adjustment(
            "NVDA", "short", self._qqq_down(), lookback=5, bonus=8, penalty=12
        )
        assert adj == 8

    def test_long_with_etf_down_returns_penalty(self):
        adj = sector_confidence_adjustment(
            "NVDA", "long", self._qqq_down(), lookback=5, bonus=8, penalty=12
        )
        assert adj == -12

    def test_short_with_etf_up_returns_penalty(self):
        adj = sector_confidence_adjustment(
            "NVDA", "short", self._qqq_up(), lookback=5, bonus=8, penalty=12
        )
        assert adj == -12

    def test_flat_etf_returns_zero(self):
        adj = sector_confidence_adjustment(
            "NVDA", "long", self._qqq_flat(), lookback=5, bonus=8, penalty=12
        )
        assert adj == 0

    def test_missing_etf_bars_returns_zero(self):
        adj = sector_confidence_adjustment(
            "NVDA", "long", {}, lookback=5, bonus=8, penalty=12
        )
        assert adj == 0

    def test_spy_self_referential_returns_zero(self):
        """SPY maps to itself — no sector adjustment applied."""
        bars = {"SPY": _make_bars([100.0 + i * 0.5 for i in range(10)])}
        adj = sector_confidence_adjustment(
            "SPY", "long", bars, lookback=5, bonus=8, penalty=12
        )
        assert adj == 0

    def test_qqq_maps_to_spy(self):
        """QQQ uses SPY as its benchmark."""
        bars = {"SPY": _make_bars([100.0 + i * 0.3 for i in range(10)])}
        adj = sector_confidence_adjustment(
            "QQQ", "long", bars, lookback=5, bonus=8, penalty=12
        )
        assert adj == 8
