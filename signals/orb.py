"""
signals/orb.py — Opening Range Breakout signal.

Logic:
  Define the Opening Range (OR) as the high and low of the first
  settings.signals.orb_minutes bars after market open (9:30 ET).

  LONG  when current price breaks above OR high
         AND volume on breakout bar > volume_ma
         AND price is above VWAP (institutional bias)

  SHORT when current price breaks below OR low
         AND volume > volume_ma
         AND price is below VWAP

  The signal only fires ONCE per session per direction — once the
  breakout is confirmed there is no value in re-signalling the same level.
  A simple set tracks already-fired directions per session date.

  No signal is generated while still inside the opening range window
  (the range is still being formed).

Confidence scoring:
  Base:                              62
  + breakout size > 0.2% of OR:    +8    (decisive break)
  + volume > 2× volume_ma:         +10
  + EMA stack aligned:             +10
  + VWAP aligned:                  +5
  Cap at 95.
"""

from __future__ import annotations

import logging
from datetime import time as dtime
from typing import Optional, Set, Tuple

import pandas as pd
import pytz

from config.settings import settings
from data.indicators import IndicatorError, latest
from signals.base import BaseSignal, SignalResult

logger = logging.getLogger(__name__)

ET = pytz.timezone("America/New_York")
_MARKET_OPEN = dtime(9, 30)

_HOLD_SECONDS = 1200  # 20-minute target hold — ORB trades run longer


class OrbSignal(BaseSignal):
    """
    Opening Range Breakout.

    State: tracks (session_date, direction) pairs already fired so the
    signal only triggers once per direction per day.
    """

    name = "orb"

    def __init__(self) -> None:
        # Tracks (date_str, direction) combos already signalled today
        self._fired: Set[Tuple[str, str]] = set()

    def evaluate(
        self,
        df: pd.DataFrame,
        symbol: str,
        current_price: float,
    ) -> Optional[SignalResult]:
        needed = ["open", "high", "low", "close", "volume", "volume_ma", "atr"]
        if any(c not in df.columns for c in needed):
            logger.debug("OrbSignal [%s]: missing columns", symbol)
            return None

        try:
            vol    = latest(df, "volume")
            vol_ma = latest(df, "volume_ma")
            atr_val = latest(df, "atr")
        except IndicatorError as exc:
            logger.debug("OrbSignal [%s]: indicator error — %s", symbol, exc)
            return None

        cfg = settings.signals

        # ── Isolate today's bars ────────────────────────────────────────────
        now_et = df.index[-1].tz_convert(ET)
        today_str = now_et.date().isoformat()
        today_bars = df[df.index.normalize() == df.index[-1].normalize()]

        if today_bars.empty:
            return None

        # ── Build the opening range from the first orb_minutes bars ────────
        orb_end = today_bars.index[0] + pd.Timedelta(minutes=cfg.orb_minutes)
        orb_bars = today_bars[today_bars.index < orb_end]

        if len(orb_bars) < cfg.orb_minutes:
            # Range still forming — do not signal
            logger.debug("OrbSignal [%s]: opening range still forming (%d/%d bars)",
                         symbol, len(orb_bars), cfg.orb_minutes)
            return None

        or_high = float(orb_bars["high"].max())
        or_low  = float(orb_bars["low"].min())
        or_range = or_high - or_low

        if or_range <= 0:
            return None

        # ── Detect breakout ─────────────────────────────────────────────────
        if current_price > or_high:
            direction = "long"
        elif current_price < or_low:
            direction = "short"
        else:
            return None  # still inside range

        # ── Fire-once guard ─────────────────────────────────────────────────
        fire_key = (today_str, direction)
        if fire_key in self._fired:
            return None
        self._fired.add(fire_key)

        # ── Volume filter ───────────────────────────────────────────────────
        if vol <= vol_ma:
            logger.debug("OrbSignal [%s]: blocked — volume below MA", symbol)
            self._fired.discard(fire_key)  # allow retry if volume improves
            return None

        # ── VWAP filter ─────────────────────────────────────────────────────
        vwap_ok = True
        if "vwap" in df.columns:
            try:
                vwap_val = latest(df, "vwap")
                if direction == "long" and current_price < vwap_val:
                    vwap_ok = False
                elif direction == "short" and current_price > vwap_val:
                    vwap_ok = False
            except IndicatorError:
                pass

        if not vwap_ok:
            logger.debug("OrbSignal [%s]: %s blocked — VWAP opposing", symbol, direction)
            self._fired.discard(fire_key)
            return None

        # ── Confidence ──────────────────────────────────────────────────────
        confidence = 62

        breakout_pct = abs(current_price - (or_high if direction == "long" else or_low)) / or_range
        if breakout_pct > 0.002:
            confidence += 8

        if vol > vol_ma * 2.0:
            confidence += 10

        if "ema_fast" in df.columns and "ema_slow" in df.columns:
            try:
                ef = latest(df, "ema_fast")
                es = latest(df, "ema_slow")
                if (direction == "long" and ef > es) or (direction == "short" and ef < es):
                    confidence += 10
            except IndicatorError:
                pass

        if "vwap" in df.columns:
            try:
                vwap_val = latest(df, "vwap")
                if (direction == "long" and current_price > vwap_val) or \
                   (direction == "short" and current_price < vwap_val):
                    confidence += 5
            except IndicatorError:
                pass

        confidence = min(confidence, 95)

        if confidence < cfg.min_confidence_score:
            self._fired.discard(fire_key)
            return None

        stop_price, target_price = self._stop_and_target(
            current_price, direction, atr_val,
            stop_mult=1.0, target_mult=2.0,   # ORB targets run further
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
                "or_high": or_high, "or_low": or_low, "or_range": or_range,
                "breakout_pct": breakout_pct,
                "volume_ratio": vol / vol_ma if vol_ma else None,
                "atr": atr_val,
            },
        )

    def reset_session(self) -> None:
        """Clear fired-signal state.  Call at start of each trading day."""
        self._fired.clear()
