"""
data/regime.py — Market regime classification.

Classifies each symbol's current market as trending or ranging.
Used by the signal engine to filter inappropriate strategy types.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import pandas as pd

from config.settings import settings
from data.indicators import regime_score


class Regime(Enum):
    TRENDING = "trending"
    RANGING  = "ranging"
    MIXED    = "mixed"


@dataclass
class RegimeResult:
    regime:             Regime
    score:              float
    trending_threshold: float
    ranging_threshold:  float


def classify_regime(df: pd.DataFrame) -> RegimeResult:
    """
    Classify the current market regime for a symbol.
    Returns RegimeResult with regime type and raw score.
    """
    score = regime_score(df, lookback=settings.signals.regime_lookback)

    if score > settings.signals.regime_trending_threshold:
        regime = Regime.TRENDING
    elif score < settings.signals.regime_ranging_threshold:
        regime = Regime.RANGING
    else:
        regime = Regime.MIXED

    return RegimeResult(
        regime=regime,
        score=score,
        trending_threshold=settings.signals.regime_trending_threshold,
        ranging_threshold=settings.signals.regime_ranging_threshold,
    )


# Which regimes each strategy is valid in
STRATEGY_VALID_REGIMES: dict[str, list[Regime]] = {
    "momentum":     [Regime.TRENDING, Regime.MIXED],
    "ema_cross":    [Regime.TRENDING, Regime.MIXED, Regime.RANGING],
    "orb":          [Regime.TRENDING, Regime.MIXED],
    "vwap_cross":   [Regime.TRENDING, Regime.RANGING, Regime.MIXED],
    "rsi_reversal": [Regime.RANGING, Regime.MIXED],
}


def is_strategy_valid_for_regime(
    strategy_name: str,
    regime: Regime,
) -> bool:
    """
    Returns True if the strategy is appropriate for the current regime.
    Unknown strategies are allowed in all regimes.
    """
    valid_regimes = STRATEGY_VALID_REGIMES.get(strategy_name)
    if valid_regimes is None:
        return True
    return regime in valid_regimes
