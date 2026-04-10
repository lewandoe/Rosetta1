"""
strategy/engine.py — Signal aggregator and confidence engine.

Wires all 5 strategy instances together and produces a single best
SignalResult per evaluation cycle, or None if no signal meets the bar.

Aggregation logic:
  1. Run all 5 strategies.  Collect results that pass their own confidence
     threshold (>= settings.signals.min_confidence_score).
  2. Group results by direction (long / short).
  3. For the dominant direction, award a consensus bonus of +CONSENSUS_BONUS
     points for each *additional* strategy that agrees (beyond the first).
     The bonus is applied to the highest-confidence signal in that group.
  4. Return the signal with the highest adjusted confidence, provided it
     still meets min_confidence_score after all adjustments.
     If both directions tie, prefer the one with more confirming strategies.

Consensus bonus: 5 points per additional confirming strategy (max 4 extras
= +20 points).  Keeps the scale meaningful without overwhelming individual
strategy scores.

Dependency rule: strategy/ may import from config/, data/, and signals/.
It must NOT import from broker/, execution/, risk/.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import pandas as pd

from config.settings import settings
from signals.base import BaseSignal, SignalResult
from signals.momentum import MomentumSignal
from signals.vwap import VwapSignal
from signals.ema_cross import EmaCrossSignal
from signals.orb import OrbSignal
from signals.rsi import RsiSignal

logger = logging.getLogger(__name__)

# Points awarded per additional strategy that agrees with the top signal
CONSENSUS_BONUS = 5


class SignalEngine:
    """
    Aggregates all 5 signal strategies for a single symbol.

    Usage (called once per poll cycle per symbol):
        engine = SignalEngine()
        result = engine.evaluate(df, symbol="SPY", current_price=512.34)
        if result:
            # send to risk guard → order manager
    """

    def __init__(self) -> None:
        self._strategies: List[BaseSignal] = [
            MomentumSignal(),
            VwapSignal(),
            EmaCrossSignal(),
            OrbSignal(),
            RsiSignal(),
        ]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(
        self,
        df: pd.DataFrame,
        symbol: str,
        current_price: float,
    ) -> Optional[SignalResult]:
        """
        Run all strategies, aggregate, return best signal or None.

        Args:
            df:            add_all() DataFrame — must have all indicator columns.
            symbol:        Ticker being evaluated.
            current_price: Latest quote price (ask for long entry reference).

        Returns:
            Best SignalResult after consensus boosting, or None.
        """
        raw_signals = self._run_all(df, symbol, current_price)

        if not raw_signals:
            return None

        best = self._aggregate(raw_signals)

        if best is None:
            return None

        logger.info(
            "SignalEngine [%s]: %s %s | confidence=%d | entry=%.2f "
            "target=%.2f stop=%.2f | R:R=1:%.2f",
            symbol, best.direction.upper(), best.signal_type,
            best.confidence, best.entry_price,
            best.target_price, best.stop_price,
            best.reward / best.risk if best.risk > 0 else 0,
        )
        return best

    def run_all_raw(
        self,
        df: pd.DataFrame,
        symbol: str,
        current_price: float,
    ) -> List[SignalResult]:
        """
        Return all individual strategy results (before aggregation).
        Used by tests and the dashboard to show scanner state per strategy.
        """
        return self._run_all(df, symbol, current_price)

    def reset_session(self) -> None:
        """
        Reset per-session state on all strategies that track it (e.g. ORB
        fire-once guard).  Call at the start of each trading day.
        """
        for s in self._strategies:
            if hasattr(s, "reset_session"):
                s.reset_session()
        logger.info("SignalEngine: session state reset for all strategies")

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _run_all(
        self,
        df: pd.DataFrame,
        symbol: str,
        current_price: float,
    ) -> List[SignalResult]:
        """Run every strategy and collect non-None results."""
        results: List[SignalResult] = []
        for strategy in self._strategies:
            try:
                result = strategy.evaluate(df, symbol, current_price)
                if result is not None:
                    results.append(result)
                    logger.debug(
                        "SignalEngine [%s]: %s fired %s confidence=%d",
                        symbol, strategy.name, result.direction, result.confidence,
                    )
            except Exception as exc:
                # A buggy strategy must never take down the engine.
                # Log loudly but continue evaluating remaining strategies.
                logger.error(
                    "SignalEngine [%s]: strategy %s raised unexpectedly: %s",
                    symbol, strategy.name, exc, exc_info=True,
                )
        return results

    def _aggregate(self, signals: List[SignalResult]) -> Optional[SignalResult]:
        """
        Apply consensus boosting and return the best signal.

        Groups by direction, finds dominant group, boosts the top signal
        in that group by CONSENSUS_BONUS per additional confirming strategy.
        """
        by_direction: Dict[str, List[SignalResult]] = {"long": [], "short": []}
        for sig in signals:
            by_direction[sig.direction].append(sig)

        # Sort each group by confidence descending
        for direction in by_direction:
            by_direction[direction].sort(key=lambda s: s.confidence, reverse=True)

        long_signals  = by_direction["long"]
        short_signals = by_direction["short"]

        # Pick dominant direction (more signals = more conviction; ties go to higher top score)
        if not long_signals and not short_signals:
            return None

        def _best_in_group(group: List[SignalResult]) -> Optional[SignalResult]:
            if not group:
                return None
            top = group[0]
            # Consensus boost: +BONUS per additional agreeing strategy
            bonus = CONSENSUS_BONUS * (len(group) - 1)
            # Return a copy with boosted confidence (cap at 100)
            boosted = SignalResult(
                symbol=top.symbol,
                signal_type=top.signal_type,
                direction=top.direction,
                entry_price=top.entry_price,
                target_price=top.target_price,
                stop_price=top.stop_price,
                confidence=min(top.confidence + bonus, 100),
                estimated_hold_seconds=top.estimated_hold_seconds,
                timestamp=top.timestamp,
                metadata={
                    **top.metadata,
                    "consensus_count": len(group),
                    "consensus_bonus": bonus,
                    "contributing_strategies": [s.signal_type for s in group],
                },
            )
            return boosted

        long_best  = _best_in_group(long_signals)
        short_best = _best_in_group(short_signals)

        # Select between long and short candidates
        candidates = [s for s in [long_best, short_best] if s is not None]
        if not candidates:
            return None

        # Primary sort: confidence desc; secondary: consensus count desc
        candidates.sort(
            key=lambda s: (s.confidence, s.metadata.get("consensus_count", 1)),
            reverse=True,
        )
        best = candidates[0]

        # Final threshold check after boosting
        if best.confidence < settings.signals.min_confidence_score:
            return None

        return best
