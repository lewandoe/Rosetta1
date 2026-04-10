"""
data/sector.py — Cross-symbol sector confirmation.

Maps each symbol to its benchmark ETF and provides
trend direction for confirmation scoring.
"""
from __future__ import annotations

import pandas as pd

# Maps each stock to its sector benchmark
SECTOR_MAP: dict[str, str] = {
    "SPY":   "SPY",    # self-referential — always confirms itself
    "QQQ":   "SPY",    # QQQ uses SPY as macro filter
    "TSLA":  "QQQ",    # mega-cap tech
    "NVDA":  "QQQ",    # semiconductor / AI
    "AAPL":  "QQQ",    # mega-cap tech
    "MSFT":  "QQQ",    # mega-cap tech
    "GOOGL": "QQQ",    # mega-cap tech
    "AMZN":  "QQQ",    # mega-cap tech
    "META":  "QQQ",    # mega-cap tech
    "AMD":   "QQQ",    # semiconductor
}


def get_sector_etf(symbol: str) -> str:
    """Return the benchmark ETF for a given symbol. Defaults to SPY."""
    return SECTOR_MAP.get(symbol, "SPY")


def get_etf_trend(
    etf_symbol: str,
    bars_dict: dict[str, pd.DataFrame],
    lookback: int = 5,
) -> str | None:
    """
    Classify the ETF's recent trend as 'up', 'down', or 'flat'.

    Uses the last `lookback` bars of close prices.
    Returns None if ETF bars are not available.

    'up'   = close[-1] > close[-lookback] by more than 0.05%
    'down' = close[-1] < close[-lookback] by more than 0.05%
    'flat' = within 0.05% — no clear direction
    """
    bars = bars_dict.get(etf_symbol)
    if bars is None or len(bars) < lookback + 1:
        return None

    col = "close" if "close" in bars.columns else "Close"
    closes = bars[col]
    start_price = float(closes.iloc[-(lookback + 1)])
    end_price   = float(closes.iloc[-1])

    if start_price == 0:
        return None

    change_pct = (end_price - start_price) / start_price

    if change_pct > 0.0005:     # +0.05%
        return "up"
    elif change_pct < -0.0005:  # -0.05%
        return "down"
    else:
        return "flat"


def sector_confidence_adjustment(
    symbol: str,
    direction: str,
    bars_dict: dict[str, pd.DataFrame],
    lookback: int = 5,
    bonus: int = 8,
    penalty: int = 12,
) -> int:
    """
    Calculate confidence adjustment based on sector ETF alignment.

    Returns:
        +bonus   if ETF trend agrees with signal direction
        -penalty if ETF trend disagrees with signal direction
        0        if ETF is flat, unavailable, or self-referential
    """
    etf = get_sector_etf(symbol)
    if etf == symbol:
        return 0  # self-referential — no adjustment

    trend = get_etf_trend(etf, bars_dict, lookback)

    if trend is None or trend == "flat":
        return 0

    agrees = (
        (direction == "long"  and trend == "up") or
        (direction == "short" and trend == "down")
    )

    return bonus if agrees else -penalty
