"""
signals/vwap.py — VWAP cross signal.

Logic:
  LONG  when price crosses above VWAP on the current bar
         AND EMA stack is bullish (ema_fast > ema_slow)  — trend confirmation
         AND volume > volume_ma                           — conviction

  SHORT when price crosses below VWAP
         AND EMA stack is bearish (ema_fast < ema_slow)
         AND volume > volume_ma

A "cross" means the previous close was on the opposite side of VWAP from
the current price — detected by comparing last two rows.

Confidence scoring:
  Base:                          60
  + volume > 2× volume_ma:      +10   (strong conviction on the cross)
  + EMA fast/mid/slow aligned:  +10   (full stack behind the move)
  + price deviation from VWAP:   +5   (≥ 0.1% gap = clean break)
  Cap at 95.
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

from config.settings import settings
from data.indicators import latest, IndicatorError
from signals.base import BaseSignal, SignalResult

logger = logging.getLogger(__name__)

_HOLD_SECONDS = 300  # 5-minute target hold


class VwapSignal(BaseSignal):
    name = "vwap_cross"

    def evaluate(
        self,
        df: pd.DataFrame,
        symbol: str,
        current_price: float,
    ) -> Optional[SignalResult]:
        # Need at least 2 rows to detect a cross
        needed_cols = ["close", "vwap", "ema_fast", "ema_slow", "volume", "volume_ma", "atr"]
        if any(c not in df.columns for c in needed_cols):
            logger.debug("VwapSignal [%s]: missing indicator columns", symbol)
            return None
        if len(df.dropna(subset=needed_cols)) < 2:
            return None

        try:
            clean = df.dropna(subset=needed_cols)
            prev_close = float(clean["close"].iloc[-2])
            prev_vwap  = float(clean["vwap"].iloc[-2])
            curr_vwap  = latest(df, "vwap")
            ef         = latest(df, "ema_fast")
            es         = latest(df, "ema_slow")
            em         = latest(df, "ema_mid")
            vol        = latest(df, "volume")
            vol_ma     = latest(df, "volume_ma")
            atr_val    = latest(df, "atr")
        except IndicatorError as exc:
            logger.debug("VwapSignal [%s]: indicator error — %s", symbol, exc)
            return None

        cfg = settings.signals

        # ── Detect cross ───────────────────────────────────────────────────
        crossed_above_vwap = prev_close < prev_vwap and current_price > curr_vwap
        crossed_below_vwap = prev_close > prev_vwap and current_price < curr_vwap

        if not crossed_above_vwap and not crossed_below_vwap:
            return None

        direction = "long" if crossed_above_vwap else "short"

        # ── Trend filter — EMA must agree with direction ───────────────────
        if direction == "long" and ef <= es:
            logger.debug("VwapSignal [%s]: long blocked — EMA bearish", symbol)
            return None
        if direction == "short" and ef >= es:
            logger.debug("VwapSignal [%s]: short blocked — EMA bullish", symbol)
            return None

        # ── Volume filter ──────────────────────────────────────────────────
        if vol <= vol_ma:
            logger.debug("VwapSignal [%s]: blocked — volume below MA", symbol)
            return None

        # ── Confidence ─────────────────────────────────────────────────────
        confidence = 60

        if vol > vol_ma * 2.0:
            confidence += 10

        # Full EMA stack aligned with direction
        full_bull = ef > em > es
        full_bear = ef < em < es
        if (direction == "long" and full_bull) or (direction == "short" and full_bear):
            confidence += 10

        vwap_gap = abs(current_price - curr_vwap) / curr_vwap if curr_vwap > 0 else 0
        if vwap_gap >= 0.001:
            confidence += 5

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
                "vwap": curr_vwap, "vwap_gap_pct": vwap_gap,
                "volume_ratio": vol / vol_ma if vol_ma else None,
                "ema_fast": ef, "ema_slow": es,
            },
        )
