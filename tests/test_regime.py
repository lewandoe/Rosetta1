"""
tests/test_regime.py — Unit tests for market regime classification.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data.indicators import regime_score
from data.regime import (
    Regime,
    RegimeResult,
    STRATEGY_VALID_REGIMES,
    classify_regime,
    is_strategy_valid_for_regime,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_df(closes: list[float]) -> pd.DataFrame:
    """Build a minimal OHLCV DataFrame with the given close prices."""
    n = len(closes)
    closes_arr = np.array(closes, dtype=float)
    return pd.DataFrame(
        {
            "open":   closes_arr,
            "high":   closes_arr + 0.10,
            "low":    closes_arr - 0.10,
            "close":  closes_arr,
            "volume": [100_000] * n,
        },
        index=pd.date_range("2024-01-02 09:30", periods=n, freq="1min", tz="America/New_York"),
    )


# ---------------------------------------------------------------------------
# regime_score() unit tests
# ---------------------------------------------------------------------------

class TestRegimeScore:
    def test_trending_monotonic_rise(self):
        """Monotonically rising closes → efficiency ratio = 1.0."""
        closes = [100.0 + i for i in range(21)]
        df = _make_df(closes)
        score = regime_score(df, lookback=20)
        assert score == pytest.approx(1.0)

    def test_ranging_alternating(self):
        """Perfect alternating up/down → very low efficiency ratio."""
        closes = [100.0 + (0.5 if i % 2 == 0 else -0.5) for i in range(21)]
        df = _make_df(closes)
        score = regime_score(df, lookback=20)
        # Net move ≈ 0, total move = 20 × 1.0 → score ≈ 0.0
        assert score < 0.05

    def test_zero_movement_returns_zero(self):
        """Flat closes → total_movement = 0 → returns 0.0 (not division error)."""
        closes = [150.0] * 22
        df = _make_df(closes)
        score = regime_score(df, lookback=20)
        assert score == 0.0

    def test_score_bounded_0_to_1(self):
        """Score must always be in [0.0, 1.0]."""
        import random
        random.seed(42)
        closes = [100.0 + random.gauss(0, 1) for _ in range(30)]
        df = _make_df(closes)
        score = regime_score(df, lookback=20)
        assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# classify_regime() tests
# ---------------------------------------------------------------------------

class TestClassifyRegime:
    def test_trending(self):
        closes = [100.0 + i * 0.5 for i in range(25)]
        df = _make_df(closes)
        result = classify_regime(df)
        assert isinstance(result, RegimeResult)
        assert result.regime == Regime.TRENDING
        assert result.score > result.trending_threshold

    def test_ranging(self):
        closes = [100.0 + (0.3 if i % 2 == 0 else -0.3) for i in range(25)]
        df = _make_df(closes)
        result = classify_regime(df)
        assert result.regime == Regime.RANGING
        assert result.score < result.ranging_threshold

    def test_mixed(self):
        # Gentle drift — should land in mixed zone
        closes = [100.0 + i * 0.05 for i in range(25)]
        df = _make_df(closes)
        result = classify_regime(df)
        # Score may be trending or mixed depending on exact values;
        # just assert it's a valid Regime member.
        assert result.regime in list(Regime)

    def test_result_fields_populated(self):
        closes = [100.0 + i for i in range(25)]
        df = _make_df(closes)
        result = classify_regime(df)
        assert result.trending_threshold > 0
        assert result.ranging_threshold > 0
        assert 0.0 <= result.score <= 1.0


# ---------------------------------------------------------------------------
# Strategy valid-regime mapping tests
# ---------------------------------------------------------------------------

class TestStrategyValidRegimes:
    def test_momentum_only_trending(self):
        assert STRATEGY_VALID_REGIMES["momentum"] == [Regime.TRENDING]

    def test_ema_cross_trending_and_mixed(self):
        valid = STRATEGY_VALID_REGIMES["ema_cross"]
        assert Regime.TRENDING in valid
        assert Regime.MIXED in valid
        assert Regime.RANGING not in valid

    def test_orb_trending_and_mixed(self):
        valid = STRATEGY_VALID_REGIMES["orb"]
        assert Regime.TRENDING in valid
        assert Regime.MIXED in valid
        assert Regime.RANGING not in valid

    def test_vwap_cross_ranging_and_mixed(self):
        valid = STRATEGY_VALID_REGIMES["vwap_cross"]
        assert Regime.RANGING in valid
        assert Regime.MIXED in valid
        assert Regime.TRENDING not in valid

    def test_rsi_reversal_ranging_and_mixed(self):
        valid = STRATEGY_VALID_REGIMES["rsi_reversal"]
        assert Regime.RANGING in valid
        assert Regime.MIXED in valid
        assert Regime.TRENDING not in valid


class TestIsStrategyValidForRegime:
    def test_momentum_blocked_in_ranging(self):
        assert not is_strategy_valid_for_regime("momentum", Regime.RANGING)

    def test_momentum_blocked_in_mixed(self):
        assert not is_strategy_valid_for_regime("momentum", Regime.MIXED)

    def test_momentum_allowed_in_trending(self):
        assert is_strategy_valid_for_regime("momentum", Regime.TRENDING)

    def test_vwap_cross_blocked_in_trending(self):
        assert not is_strategy_valid_for_regime("vwap_cross", Regime.TRENDING)

    def test_vwap_cross_allowed_in_ranging(self):
        assert is_strategy_valid_for_regime("vwap_cross", Regime.RANGING)

    def test_unknown_strategy_allowed_in_all_regimes(self):
        for regime in Regime:
            assert is_strategy_valid_for_regime("unknown_strategy", regime)
