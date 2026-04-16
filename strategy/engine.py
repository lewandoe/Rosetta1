"""
strategy/engine.py — Regime-first signal engine.

Pipeline per evaluation cycle:
  1. Volume gate  — bar must have volume >= volume_gate_multiplier × volume_ma.
  2. Regime       — classify trending / ranging / mixed.
  3. MTF alignment — 1-min / 5-min / 15-min EMA(21) must agree with signal direction.
  4. Strategy selection — only run strategies valid for the current regime,
     excluding disabled_signals.
  5. Sector confirmation — adjust confidence ± based on sector ETF direction.
  6. Confidence gate — drop signals below min_confidence_score.

Removed from the old design:
  - Consensus bonus (_aggregate) — regime + MTF filters do a better job
    of reducing noise than voting across all strategies simultaneously.
  - Macro bias (_get_macro_bias / SPY VWAP check) — replaced by MTF
    alignment which checks the symbol's own multi-timeframe trend.
  - _run_all() / _aggregate() — folded into evaluate().

Dependency rule: strategy/ may import from config/, data/, and signals/.
It must NOT import from broker/, execution/, risk/.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import pandas as pd

from config.settings import settings
from data.indicators import latest, mtf_trend_direction
from data.regime import classify_regime, is_strategy_valid_for_regime
from data.sector import get_sector_etf, sector_confidence_adjustment
from signals.base import BaseSignal, SignalResult
from signals.ema_cross import EmaCrossSignal
from signals.momentum import MomentumSignal
from signals.orb import OrbSignal
from signals.rsi import RsiSignal
from signals.vwap import VwapSignal

logger = logging.getLogger(__name__)


class SignalEngine:
    """
    Regime-first signal engine for a single symbol.

    Usage (called once per poll cycle per symbol):
        engine = SignalEngine()
        result = engine.evaluate(df, symbol="AAPL", current_price=195.40,
                                  bars_dict=all_bars)
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
        bars_dict: dict[str, pd.DataFrame] | None = None,
        quote: object | None = None,
    ) -> Optional[SignalResult]:
        """
        Regime-first signal evaluation.

        Args:
            df:            add_all() DataFrame — must have all indicator columns.
            symbol:        Ticker being evaluated.
            current_price: Latest quote price.
            bars_dict:     All symbols' bar DataFrames keyed by symbol, used for
                           sector ETF confirmation scoring.
            quote:         Optional Quote object — if supplied, bid/ask are
                           propagated into the signal metadata so the risk guard
                           can run a spread check and the order manager can size
                           limit-order prices.

        Returns:
            Best SignalResult after regime + MTF + sector filtering, or None.
        """
        cfg = settings.signals

        # ── Gate 1: Volume ────────────────────────────────────────────────────
        try:
            vol    = latest(df, "volume")
            vol_ma = latest(df, "volume_ma")
            if vol_ma > 0 and vol < vol_ma * cfg.volume_gate_multiplier:
                logger.debug(
                    "SignalEngine [%s]: volume gate blocked (vol=%.0f, ma=%.0f, mult=%.1f)",
                    symbol, vol, vol_ma, cfg.volume_gate_multiplier,
                )
                return None
        except Exception:
            pass  # missing volume_ma is fine — gate skipped, not hard-blocked

        # ── Gate 2: Regime classification ─────────────────────────────────────
        try:
            regime_result = classify_regime(df)
            regime = regime_result.regime
        except Exception as exc:
            logger.debug("SignalEngine [%s]: regime classification failed — %s", symbol, exc)
            regime_result = None
            regime = None

        # ── Gate 3: Multi-Timeframe alignment ─────────────────────────────────
        mtf_dir = "neutral"
        if cfg.mtf_enabled:
            try:
                mtf_dir = mtf_trend_direction(df, cfg.mtf_ema_period)
            except Exception as exc:
                logger.debug("SignalEngine [%s]: MTF check failed — %s", symbol, exc)

        # ── Gate 4: Select strategies valid for this regime ───────────────────
        candidate_strategies: List[BaseSignal] = []
        for strategy in self._strategies:
            if strategy.name in cfg.disabled_signals:
                continue
            if regime is not None and not is_strategy_valid_for_regime(strategy.name, regime):
                logger.debug(
                    "SignalEngine [%s]: %s suppressed — regime=%s (score=%.2f)",
                    symbol, strategy.name,
                    regime.value, regime_result.score if regime_result else 0.0,
                )
                continue
            candidate_strategies.append(strategy)

        if not candidate_strategies:
            return None

        # ── Gate 5: Run candidates + MTF direction filter ─────────────────────
        results: List[SignalResult] = []
        for strategy in candidate_strategies:
            try:
                result = strategy.evaluate(df, symbol, current_price)
                if result is None:
                    continue
                # MTF alignment: reject signals opposing the multi-timeframe trend
                if mtf_dir != "neutral" and result.direction != mtf_dir:
                    logger.debug(
                        "SignalEngine [%s]: MTF filter rejected %s %s (mtf=%s)",
                        symbol, strategy.name, result.direction, mtf_dir,
                    )
                    continue
                results.append(result)
                logger.debug(
                    "SignalEngine [%s]: %s fired %s confidence=%d",
                    symbol, strategy.name, result.direction, result.confidence,
                )
            except Exception as exc:
                logger.error(
                    "SignalEngine [%s]: strategy %s raised unexpectedly: %s",
                    symbol, strategy.name, exc, exc_info=True,
                )

        if not results:
            return None

        # Pick the highest-confidence result
        best = max(results, key=lambda r: r.confidence)

        # ── Gate 6: Sector ETF confirmation ───────────────────────────────────
        sector_adj = 0
        if cfg.sector_confirmation_enabled and bars_dict:
            try:
                sector_adj = sector_confidence_adjustment(
                    symbol=best.symbol,
                    direction=best.direction,
                    bars_dict=bars_dict,
                    lookback=cfg.sector_trend_bars,
                    bonus=cfg.sector_confirmation_bonus,
                    penalty=cfg.sector_confirmation_penalty,
                )
                if sector_adj != 0:
                    logger.info(
                        "SignalEngine [%s]: sector adjustment %+d (ETF=%s, direction=%s)",
                        best.symbol, sector_adj,
                        get_sector_etf(best.symbol), best.direction,
                    )
            except Exception as exc:
                logger.debug("SignalEngine [%s]: sector adjustment failed — %s", symbol, exc)

        # Rebuild SignalResult with regime + MTF metadata (and sector adj if any)
        meta = {
            **best.metadata,
            "regime":        regime.value if regime else "unknown",
            "regime_score":  regime_result.score if regime_result else None,
            "mtf_direction": mtf_dir,
            "sector_adj":    sector_adj,
        }
        if quote is not None:
            bid = getattr(quote, "bid", None)
            ask = getattr(quote, "ask", None)
            if bid is not None:
                meta["bid_price"] = float(bid)
            if ask is not None:
                meta["ask_price"] = float(ask)

        best = SignalResult(
            symbol=best.symbol,
            signal_type=best.signal_type,
            direction=best.direction,
            entry_price=best.entry_price,
            target_price=best.target_price,
            stop_price=best.stop_price,
            confidence=min(max(best.confidence + sector_adj, 0), 100),
            estimated_hold_seconds=best.estimated_hold_seconds,
            timestamp=best.timestamp,
            metadata=meta,
        )

        # ── Final confidence gate ─────────────────────────────────────────────
        if best.confidence < cfg.min_confidence_score:
            logger.debug(
                "SignalEngine [%s]: %s dropped below threshold (confidence=%d, min=%d)",
                symbol, best.signal_type, best.confidence, cfg.min_confidence_score,
            )
            return None

        logger.info(
            "Signal FIRE [%s] %s %s | conf=%d | regime=%s | mtf=%s | "
            "sector_adj=%+d | entry=%.2f target=%.2f stop=%.2f | R:R=1:%.2f",
            best.symbol, best.direction.upper(), best.signal_type,
            best.confidence,
            regime.value if regime else "unknown",
            mtf_dir,
            sector_adj,
            best.entry_price, best.target_price, best.stop_price,
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
        Return all individual strategy results without regime/MTF/volume gates.
        Used by tests and the dashboard to show per-strategy scanner state.
        """
        results: List[SignalResult] = []
        for strategy in self._strategies:
            if strategy.name in settings.signals.disabled_signals:
                continue
            try:
                result = strategy.evaluate(df, symbol, current_price)
                if result is not None:
                    results.append(result)
            except Exception as exc:
                logger.error(
                    "SignalEngine [%s]: strategy %s raised unexpectedly: %s",
                    symbol, strategy.name, exc, exc_info=True,
                )
        return results

    def reset_session(self) -> None:
        """
        Reset per-session state on all strategies that track it (e.g. ORB
        fire-once guard).  Call at the start of each trading day.
        """
        for s in self._strategies:
            if hasattr(s, "reset_session"):
                s.reset_session()
        logger.info("SignalEngine: session state reset for all strategies")
