"""
signals/base.py — Shared SignalResult dataclass and BaseSignal ABC.

All strategy modules import from here.  The execution layer imports
SignalResult only — it never touches individual strategy classes.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import pandas as pd


@dataclass
class SignalResult:
    """
    Output of every signal strategy evaluation.

    Fields match the trade log schema (Stage 8) so results can be stored
    directly without transformation.
    """
    symbol: str
    signal_type: str          # e.g. "momentum", "vwap_cross", "ema_cross", "orb", "rsi_reversal"
    direction: str            # "long" or "short"
    entry_price: float        # expected fill price (typically last/ask/bid)
    target_price: float       # take-profit level
    stop_price: float         # stop-loss level
    confidence: int           # 0–100; only signals >= min_confidence_score are acted on
    estimated_hold_seconds: int
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: dict = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Derived properties used by risk guard and order manager
    # ------------------------------------------------------------------

    @property
    def risk(self) -> float:
        """Absolute distance from entry to stop."""
        return abs(self.entry_price - self.stop_price)

    @property
    def reward(self) -> float:
        """Absolute distance from entry to target."""
        return abs(self.target_price - self.entry_price)

    @property
    def risk_reward_ratio(self) -> float:
        """risk / reward — must be ≤ settings.risk.max_loss_to_gain_ratio (2.0)."""
        if self.reward == 0:
            return float("inf")
        return self.risk / self.reward

    @property
    def is_long(self) -> bool:
        return self.direction == "long"


class BaseSignal(abc.ABC):
    """
    Abstract base for all signal strategies.

    Subclasses implement evaluate() and return a SignalResult or None.
    They must NOT import from broker/ or execution/ — dependency graph rule.
    """

    # Human-readable name used in logs and the SignalResult.signal_type field
    name: str = "base"

    @abc.abstractmethod
    def evaluate(
        self,
        df: pd.DataFrame,
        symbol: str,
        current_price: float,
    ) -> Optional[SignalResult]:
        """
        Evaluate the strategy against the latest indicator data.

        Args:
            df:            OHLCV + indicator DataFrame from data/indicators.add_all().
                           At least settings.signals.ema_slow bars must be present.
            symbol:        Ticker being evaluated.
            current_price: Latest trade price (used as entry reference).

        Returns:
            SignalResult if a signal fires with confidence >= threshold, else None.
        """

    # ------------------------------------------------------------------
    # Shared helpers available to all strategies
    # ------------------------------------------------------------------

    @staticmethod
    def _stop_and_target(
        entry: float,
        direction: str,
        atr: float,
        stop_mult: float = 1.0,
        target_mult: float = 1.5,
    ) -> tuple[float, float]:
        """
        Compute stop_price and target_price from ATR multiples.

        Default R:R = 1:1.5 (risk 1 ATR, target 1.5 ATR).
        Risk rule: loss ≤ 2× gain → 1.0 ≤ 2 × 1.5 = 3.0 ✓

        Returns:
            (stop_price, target_price)
        """
        stop_dist = atr * stop_mult
        target_dist = atr * target_mult
        if direction == "long":
            return entry - stop_dist, entry + target_dist
        else:
            return entry + stop_dist, entry - target_dist
