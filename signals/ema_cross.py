"""
signals/ema_cross.py — EMA fast/mid crossover signal.

Logic:
  LONG  when ema_fast crosses above ema_mid on the current bar
         AND price is above ema_slow (above long-term trend)
         AND RSI < rsi_overbought     (room to run)
         AND volume > volume_ma

  SHORT when ema_fast crosses below ema_mid
         AND price is below ema_slow
         AND RSI > rsi_oversold
         AND volume > volume_ma

Confidence scoring:
  Base:                          58
  + price above/below ema_slow:  +8    (aligned with long-term trend)
  + volume > 1.5× volume_ma:    +10
  + RSI in ideal zone:           +8    (45–65 long / 35–55 short)
  + VWAP side matches direction: +6
  Cap at 95.
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

from config.settings import settings
from data.indicators import IndicatorError, crossed_above, crossed_below, latest
from signals.base import BaseSignal, SignalResult

logger = logging.getLogger(__name__)

_HOLD_SECONDS = 600  # 10-minute target hold for EMA cross trades


class EmaCrossSignal(BaseSignal):
    name = "ema_cross"

    def evaluate(
        self,
        df: pd.DataFrame,
        symbol: str,
        current_price: float,
    ) -> Optional[SignalResult]:
        needed = ["ema_fast", "ema_mid", "ema_slow", "rsi", "volume", "volume_ma", "atr"]
        if any(c not in df.columns for c in needed):
            logger.debug("EmaCrossSignal [%s]: missing columns", symbol)
            return None
        if len(df.dropna(subset=needed)) < 2:
            return None

        try:
            es      = latest(df, "ema_slow")
            rsi_val = latest(df, "rsi")
            vol     = latest(df, "volume")
            vol_ma  = latest(df, "volume_ma")
            atr_val = latest(df, "atr")
        except IndicatorError as exc:
            logger.debug("EmaCrossSignal [%s]: indicator error — %s", symbol, exc)
            return None

        cfg = settings.signals

        # ── Detect cross ───────────────────────────────────────────────────
        is_cross_up   = crossed_above(df, "ema_fast", "ema_mid")
        is_cross_down = crossed_below(df, "ema_fast", "ema_mid")

        if not is_cross_up and not is_cross_down:
            return None

        direction = "long" if is_cross_up else "short"

        # ── Trend filter: price must be on the correct side of ema_slow ────
        if direction == "long" and current_price <= es:
            logger.debug("EmaCrossSignal [%s]: long blocked — price below ema_slow", symbol)
            return None
        if direction == "short" and current_price >= es:
            logger.debug("EmaCrossSignal [%s]: short blocked — price above ema_slow", symbol)
            return None

        # ── RSI filter ─────────────────────────────────────────────────────
        if direction == "long" and rsi_val >= cfg.rsi_overbought:
            return None
        if direction == "short" and rsi_val <= cfg.rsi_oversold:
            return None

        # ── Volume filter ──────────────────────────────────────────────────
        if vol <= vol_ma * 0.7:  # allow moderate volume (70% of avg)
            logger.debug("EmaCrossSignal [%s]: blocked — volume below MA", symbol)
            return None

        # ── Confidence ─────────────────────────────────────────────────────
        confidence = 58

        # Long-term trend alignment
        if direction == "long" and current_price > es:
            confidence += 8
        elif direction == "short" and current_price < es:
            confidence += 8

        if vol > vol_ma * 1.5:
            confidence += 10

        if direction == "long" and 45 <= rsi_val <= 65:
            confidence += 8
        elif direction == "short" and 35 <= rsi_val <= 55:
            confidence += 8

        # VWAP side confirmation (bonus only — not a hard filter)
        if "vwap" in df.columns:
            try:
                vwap_val = latest(df, "vwap")
                if direction == "long" and current_price > vwap_val:
                    confidence += 6
                elif direction == "short" and current_price < vwap_val:
                    confidence += 6
            except IndicatorError:
                pass

        confidence = min(confidence, 95)

        if confidence < cfg.min_confidence_score:
            return None

        stop_price, target_price = self._stop_and_target(
            current_price, direction, atr_val
        )

        return SignalResult(
            symbol=symbol,
            signal_type=self.name,
            direction=direction,
            entry_price=current_price,
            target_price=target_price,
            stop_price=stop_price,
            confidence=confidence,
            estimated_hold_seconds=_HOLD_SECONDS,
            metadata={
                "ema_slow": es, "rsi": rsi_val,
                "volume_ratio": vol / vol_ma if vol_ma else None,
                "atr": atr_val,
            },
        )
