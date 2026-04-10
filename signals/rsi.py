"""
signals/rsi.py — RSI mean-reversion signal.

Logic:
  LONG  when RSI crosses above rsi_oversold (30) from below
         — oversold bounce, price likely to recover
         AND price is above VWAP or EMA slow (some structural support)
         AND volume > volume_ma (confirming the bounce)

  SHORT when RSI crosses below rsi_overbought (70) from above
         — overbought reversal
         AND price is below VWAP or EMA slow
         AND volume > volume_ma

  A cross is detected by comparing the last two RSI values: previous was
  on the far side of the threshold, current has crossed through.

Confidence scoring:
  Base:                                 60
  + RSI depth (deeper = stronger):     +8   (< 25 long / > 75 short)
  + volume > 1.5× volume_ma:           +8
  + VWAP side aligned:                 +8
  + EMA slope aligned with bounce:     +6   (ema_fast turning)
  Cap at 95.
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

from config.settings import settings
from data.indicators import IndicatorError, latest
from signals.base import BaseSignal, SignalResult

logger = logging.getLogger(__name__)

_HOLD_SECONDS = 300  # Mean-reversion trades are short — target 5 min


class RsiSignal(BaseSignal):
    name = "rsi_reversal"

    def evaluate(
        self,
        df: pd.DataFrame,
        symbol: str,
        current_price: float,
    ) -> Optional[SignalResult]:
        needed = ["rsi", "volume", "volume_ma", "atr"]
        if any(c not in df.columns for c in needed):
            logger.debug("RsiSignal [%s]: missing columns", symbol)
            return None

        clean = df.dropna(subset=needed)
        if len(clean) < 2:
            return None

        try:
            curr_rsi = float(clean["rsi"].iloc[-1])
            prev_rsi = float(clean["rsi"].iloc[-2])
            vol      = latest(df, "volume")
            vol_ma   = latest(df, "volume_ma")
            atr_val  = latest(df, "atr")
        except (IndicatorError, IndexError) as exc:
            logger.debug("RsiSignal [%s]: data error — %s", symbol, exc)
            return None

        cfg = settings.signals

        # ── Detect RSI cross ────────────────────────────────────────────────
        crossed_up   = prev_rsi < cfg.rsi_oversold  and curr_rsi >= cfg.rsi_oversold
        crossed_down = prev_rsi > cfg.rsi_overbought and curr_rsi <= cfg.rsi_overbought

        if not crossed_up and not crossed_down:
            return None

        direction = "long" if crossed_up else "short"

        # ── Volume filter ───────────────────────────────────────────────────
        if vol <= vol_ma:
            logger.debug("RsiSignal [%s]: blocked — volume below MA", symbol)
            return None

        # ── Structural support / resistance filter ──────────────────────────
        structural_ok = False
        if "vwap" in df.columns:
            try:
                vwap_val = latest(df, "vwap")
                if direction == "long" and current_price >= vwap_val:
                    structural_ok = True
                elif direction == "short" and current_price <= vwap_val:
                    structural_ok = True
            except IndicatorError:
                pass

        if not structural_ok and "ema_slow" in df.columns:
            try:
                es = latest(df, "ema_slow")
                if direction == "long" and current_price >= es:
                    structural_ok = True
                elif direction == "short" and current_price <= es:
                    structural_ok = True
            except IndicatorError:
                pass

        if not structural_ok:
            logger.debug("RsiSignal [%s]: %s blocked — no structural support", symbol, direction)
            return None

        # ── Confidence ──────────────────────────────────────────────────────
        confidence = 60

        # Depth bonus — deeper extremes signal stronger reversals
        if direction == "long" and prev_rsi < 25:
            confidence += 8
        elif direction == "short" and prev_rsi > 75:
            confidence += 8

        if vol > vol_ma * 1.5:
            confidence += 8

        if "vwap" in df.columns:
            try:
                vwap_val = latest(df, "vwap")
                if (direction == "long" and current_price >= vwap_val) or \
                   (direction == "short" and current_price <= vwap_val):
                    confidence += 8
            except IndicatorError:
                pass

        # EMA fast slope — is it turning in the right direction?
        if "ema_fast" in df.columns:
            try:
                ef_now  = float(df["ema_fast"].dropna().iloc[-1])
                ef_prev = float(df["ema_fast"].dropna().iloc[-2])
                if direction == "long" and ef_now > ef_prev:
                    confidence += 6
                elif direction == "short" and ef_now < ef_prev:
                    confidence += 6
            except (IndicatorError, IndexError):
                pass

        confidence = min(confidence, 95)

        if confidence < cfg.min_confidence_score:
            return None

        stop_price, target_price = self._stop_and_target(
            current_price, direction, atr_val,
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
                "rsi_prev": prev_rsi, "rsi_curr": curr_rsi,
                "volume_ratio": vol / vol_ma if vol_ma else None,
                "atr": atr_val,
            },
        )
