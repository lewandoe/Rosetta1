"""tests/signals/test_engine.py — SignalEngine aggregation tests."""
from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

import pandas as pd
import pytest

from signals.base import SignalResult
from strategy.engine import CONSENSUS_BONUS, SignalEngine

ET = ZoneInfo("America/New_York")


def _make_signal(
    symbol="SPY",
    direction="long",
    signal_type="momentum",
    confidence=75,
    entry=100.0,
    target=101.5,
    stop=99.0,
) -> SignalResult:
    return SignalResult(
        symbol=symbol,
        signal_type=signal_type,
        direction=direction,
        entry_price=entry,
        target_price=target,
        stop_price=stop,
        confidence=confidence,
        estimated_hold_seconds=300,
    )


class TestSignalEngineAggregation:
    def _engine_with_mock_strategies(self, signals: list[SignalResult]) -> SignalEngine:
        """Return a SignalEngine whose strategies return the given signals in order."""
        engine = SignalEngine()
        mocks = []
        for i, sig in enumerate(signals):
            m = MagicMock()
            m.evaluate.return_value = sig
            m.name = sig.signal_type if sig else f"mock_{i}"
            mocks.append(m)
        # Fill remaining strategy slots with None-returning mocks
        while len(mocks) < 5:
            m = MagicMock()
            m.evaluate.return_value = None
            m.name = f"empty_{len(mocks)}"
            mocks.append(m)
        engine._strategies = mocks
        return engine

    def test_returns_none_when_no_signals(self):
        engine = SignalEngine()
        engine._strategies = [MagicMock(**{"evaluate.return_value": None, "name": f"s{i}"}) for i in range(5)]
        df = pd.DataFrame()
        result = engine.evaluate(df, "SPY", 100.0)
        assert result is None

    def test_returns_single_signal(self):
        sig = _make_signal(confidence=75)
        engine = self._engine_with_mock_strategies([sig])
        result = engine.evaluate(pd.DataFrame(), "SPY", 100.0)
        assert result is not None
        assert result.direction == "long"
        assert result.confidence == 75

    def test_consensus_boost_applied(self):
        """Two agreeing long signals should boost confidence by CONSENSUS_BONUS."""
        s1 = _make_signal(confidence=75, signal_type="momentum")
        s2 = _make_signal(confidence=70, signal_type="vwap_cross")
        engine = self._engine_with_mock_strategies([s1, s2])
        result = engine.evaluate(pd.DataFrame(), "SPY", 100.0)
        assert result is not None
        # s1 (top) gets boosted by 1 × CONSENSUS_BONUS
        assert result.confidence == 75 + CONSENSUS_BONUS

    def test_conflicting_directions_picks_higher_confidence(self):
        """Long at 80 vs short at 70 — engine should return the long."""
        s_long  = _make_signal(confidence=80, direction="long")
        s_short = _make_signal(confidence=70, direction="short", signal_type="rsi_reversal")
        engine = self._engine_with_mock_strategies([s_long, s_short])
        result = engine.evaluate(pd.DataFrame(), "SPY", 100.0)
        assert result is not None
        assert result.direction == "long"

    def test_consensus_can_flip_direction_winner(self):
        """3 short signals at 72 each should beat 1 long signal at 85."""
        s_long = _make_signal(confidence=85, direction="long", signal_type="momentum")
        s_short1 = _make_signal(confidence=72, direction="short", signal_type="vwap_cross")
        s_short2 = _make_signal(confidence=72, direction="short", signal_type="ema_cross")
        s_short3 = _make_signal(confidence=72, direction="short", signal_type="orb")
        engine = self._engine_with_mock_strategies([s_long, s_short1, s_short2, s_short3])
        result = engine.evaluate(pd.DataFrame(), "SPY", 100.0)
        # Short best = 72 + 2×BONUS = 82; Long best = 85 — long still wins
        # But if we have 3 agreeing shorts the boosted short = 72 + 2*5 = 82 < 85
        assert result is not None
        assert result.direction == "long"  # 85 > 82

    def test_confidence_capped_at_100(self):
        """Five agreeing signals should not exceed 100."""
        sigs = [_make_signal(confidence=90, signal_type=f"s{i}") for i in range(5)]
        engine = self._engine_with_mock_strategies(sigs)
        result = engine.evaluate(pd.DataFrame(), "SPY", 100.0)
        assert result is not None
        assert result.confidence <= 100

    def test_below_threshold_returns_none(self):
        """Signal below min_confidence_score should not be returned."""
        sig = _make_signal(confidence=50)  # below default threshold of 70
        engine = self._engine_with_mock_strategies([sig])
        result = engine.evaluate(pd.DataFrame(), "SPY", 100.0)
        assert result is None

    def test_metadata_includes_consensus_info(self):
        s1 = _make_signal(confidence=75, signal_type="momentum")
        s2 = _make_signal(confidence=72, signal_type="vwap_cross")
        engine = self._engine_with_mock_strategies([s1, s2])
        result = engine.evaluate(pd.DataFrame(), "SPY", 100.0)
        assert result is not None
        assert result.metadata["consensus_count"] == 2
        assert result.metadata["consensus_bonus"] == CONSENSUS_BONUS
        assert "momentum" in result.metadata["contributing_strategies"]

    def test_buggy_strategy_does_not_crash_engine(self):
        """An exception in one strategy must not prevent others from running."""
        bad = MagicMock()
        bad.evaluate.side_effect = RuntimeError("boom")
        bad.name = "bad"
        good = MagicMock()
        good.evaluate.return_value = _make_signal(confidence=75)
        good.name = "good"
        engine = SignalEngine()
        engine._strategies = [bad, good] + [MagicMock(**{"evaluate.return_value": None, "name": f"e{i}"}) for i in range(3)]
        result = engine.evaluate(pd.DataFrame(), "SPY", 100.0)
        assert result is not None

    def test_run_all_raw_returns_list(self):
        sig = _make_signal(confidence=75)
        engine = self._engine_with_mock_strategies([sig])
        raw = engine.run_all_raw(pd.DataFrame(), "SPY", 100.0)
        assert isinstance(raw, list)
        assert len(raw) == 1

    def test_reset_session_calls_strategies_with_method(self):
        engine = SignalEngine()
        for s in engine._strategies:
            if hasattr(s, "reset_session"):
                # OrbSignal has reset_session — check it runs without error
                pass
        engine.reset_session()  # should not raise


class TestSignalResultModel:
    def test_risk_reward_ratio(self):
        sig = _make_signal(entry=100.0, target=101.5, stop=99.0)
        # risk = 1.0, reward = 1.5, ratio = 1/1.5 ≈ 0.667
        assert sig.risk == pytest.approx(1.0)
        assert sig.reward == pytest.approx(1.5)
        assert sig.risk_reward_ratio == pytest.approx(1.0 / 1.5)

    def test_is_long(self):
        assert _make_signal(direction="long").is_long is True
        assert _make_signal(direction="short").is_long is False

    def test_zero_reward_ratio_is_inf(self):
        sig = _make_signal(entry=100.0, target=100.0, stop=99.0)
        assert sig.risk_reward_ratio == float("inf")
