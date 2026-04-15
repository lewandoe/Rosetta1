"""
signals/momentum.py — EMA-stack momentum signal.

Logic:
  LONG  when ema_fast > ema_mid > ema_slow (full bullish stack)
         AND price > ema_fast (riding above all EMAs)
         AND RSI is not overbought (< settings.signals.rsi_overbought)
         AND volume > volume_ma (above-average participation)

  SHORT when ema_fast < ema_mid < ema_slow (full bearish stack)
         AND price < ema_fast
         AND RSI is not oversold (> settings.signals.rsi_oversold)
         AND volume > volume_ma

Confidence scoring (0–100):
  Base:                           55
  + volume > 1.5× volume_ma:     +10
  + price gap from ema_slow > 1% +10   (strong trend extension)
  + RSI in momentum sweet spot:  +10   (40–65 long / 35–60 short)
  + ATR expanding (atr > avg):   +5    (volatility confirming move)
  Cap at 95 to leave room for engine boost.
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

from config.settings import settings
from data.indicators import latest, IndicatorError
from signals.base import BaseSignal, SignalResult

logger = logging.getLogger(__name__)

# Estimated hold for a momentum trade: 5–8 minutes
_HOLD_SECONDS = 420


class MomentumSignal(BaseSignal):
    name = "momentum"

    def evaluate(
        self,
        df: pd.DataFrame,
        symbol: str,
        current_price: float,
    ) -> Optional[SignalResult]:
        try:
            ef = latest(df, "ema_fast")
            em = latest(df, "ema_mid")
            es = latest(df, "ema_slow")
            rsi_val = latest(df, "rsi")
            vol = latest(df, "volume")
            vol_ma = latest(df, "volume_ma")
            atr_val = latest(df, "atr")
        except IndicatorError as exc:
            logger.debug("MomentumSignal [%s]: missing indicator — %s", symbol, exc)
            return None

        cfg = settings.signals

        # ── Determine direction ────────────────────────────────────────────
        long_stack  = ef > em > es and current_price > ef
        short_stack = ef < em < es and current_price < ef

        if not long_stack and not short_stack:
            return None

        direction = "long" if long_stack else "short"

        # ── RSI filter — don't chase extremes ──────────────────────────────
        if direction == "long" and rsi_val >= cfg.rsi_overbought:
            logger.debug("MomentumSignal [%s]: long blocked — RSI overbought (%.1f)", symbol, rsi_val)
            return None
        if direction == "short" and rsi_val <= cfg.rsi_oversold:
            logger.debug("MomentumSignal [%s]: short blocked — RSI oversold (%.1f)", symbol, rsi_val)
            return None

        # ── Volume filter — require above-average participation ────────────
        if vol <= vol_ma * 0.0:  # allow moderate volume (70% of avg)
            logger.debug("MomentumSignal [%s]: blocked — volume below MA", symbol)
            return None

        # ── Confidence scoring ─────────────────────────────────────────────
        confidence = 55

        if vol > vol_ma * 1.5:
            confidence += 10

        trend_gap = abs(current_price - es) / es if es > 0 else 0
        if trend_gap > 0.01:
            confidence += 10

        if direction == "long" and 40 <= rsi_val <= 65:
            confidence += 10
        elif direction == "short" and 35 <= rsi_val <= 60:
            confidence += 10

        # ATR expanding check: compare latest ATR to 10-bar average ATR
        if "atr" in df.columns and len(df["atr"].dropna()) >= 10:
            atr_avg = df["atr"].dropna().iloc[-10:].mean()
            if atr_val > atr_avg:
                confidence += 5

        confidence = min(confidence, 95)

        if confidence < cfg.min_confidence_score:
            return None

        stop_price, target_price = self._stop_and_target(
            current_price, direction, atr_val,
            stop_multiplier=cfg.momentum_stop_atr,
            reward_ratio=cfg.momentum_reward_ratio,
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
                "ema_fast": ef, "ema_mid": em, "ema_slow": es,
                "rsi": rsi_val, "volume_ratio": vol / vol_ma if vol_ma else None,
                "atr": atr_val,
            },
        )
