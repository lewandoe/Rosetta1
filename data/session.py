"""
data/session.py — Trading session classification.

Classifies the current time into trading sessions and determines
whether new entries are allowed based on time-of-day rules.
"""
from __future__ import annotations

import logging
from datetime import datetime, time
from enum import Enum
from zoneinfo import ZoneInfo

from config.settings import settings

logger = logging.getLogger(__name__)

ET = ZoneInfo("America/New_York")


class Session(Enum):
    PRE_MARKET    = "pre_market"
    POWER_OPEN    = "power_open"      # 9:30 - 11:30 ET — best edge
    LUNCH_DEAD    = "lunch_dead"      # 11:30 - 1:00 ET — low vol, choppy
    POWER_CLOSE   = "power_close"     # 1:00 - 3:45 ET — volume returns
    EOD_WIND_DOWN = "eod_wind_down"   # 3:45 - 4:00 ET — closing only
    AFTER_HOURS   = "after_hours"


def current_session() -> Session:
    """Classify the current ET time into a trading session."""
    now_et = datetime.now(ET).time()
    cfg = settings.session

    market_open    = time(cfg.market_open_hour, cfg.market_open_minute)
    power_open_end = time(cfg.power_open_end_hour, cfg.power_open_end_minute)
    lunch_end      = time(cfg.lunch_end_hour, cfg.lunch_end_minute)
    eod_start      = time(
        cfg.market_close_hour,
        max(0, cfg.market_close_minute - cfg.eod_close_minutes_before),
    )
    market_close   = time(cfg.market_close_hour, cfg.market_close_minute)

    if now_et < market_open:
        return Session.PRE_MARKET
    elif now_et < power_open_end:
        return Session.POWER_OPEN
    elif now_et < lunch_end:
        return Session.LUNCH_DEAD
    elif now_et < eod_start:
        return Session.POWER_CLOSE
    elif now_et < market_close:
        return Session.EOD_WIND_DOWN
    else:
        return Session.AFTER_HOURS


def is_entry_allowed() -> tuple[bool, str]:
    """
    Check if new trade entries are allowed right now.

    Returns (allowed: bool, reason: str).
    """
    session = current_session()

    if session == Session.PRE_MARKET:
        return False, "pre-market"
    elif session == Session.LUNCH_DEAD:
        if not settings.session.lunch_trading_enabled:
            return False, "lunch dead zone (11:30-1:00 ET)"
        return True, "lunch (enabled)"
    elif session == Session.EOD_WIND_DOWN:
        return False, "EOD wind-down — closing positions only"
    elif session == Session.AFTER_HOURS:
        return False, "after hours"

    return True, session.value
