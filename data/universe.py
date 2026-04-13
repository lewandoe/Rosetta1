"""
data/universe.py — Dynamic trading universe builder.

Fetches today's highest volume stocks at startup and combines
with a fixed core universe. Filters out stocks below $500M
market cap and penny stocks below $2.
"""
from __future__ import annotations

import logging

import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

# Fixed core symbols — always traded regardless of volume
CORE_UNIVERSE: list[str] = [
    "SPY", "QQQ",
    "TSLA", "NVDA", "AAPL", "MSFT", "GOOGL", "AMZN", "META", "AMD",
    "INTC", "MU", "MRVL", "AVGO", "PLTR",
]

# Candidate pool to scan for high volume additions
VOLUME_CANDIDATES: list[str] = [
    "SOFI", "HOOD", "BABA", "AAL", "T", "SMCI", "COIN", "UBER",
    "NFLX", "CRM", "MSTR", "RIVN", "NIO", "SNAP", "BB", "GME",
    "LCID", "F", "BAC", "C", "WFC", "JPM", "GS", "MS",
    "XOM", "CVX", "DIS", "WMT", "COST", "TGT",
    "ARM", "DELL", "HPE", "CRWD", "PANW", "CYBR",
    "RBLX", "CLSK", "HUT", "RIOT", "MARA", "BITF",
    "CCL", "RCL", "UAL", "DAL", "LUV",
]

MIN_MARKET_CAP    = 500_000_000  # $500M
MIN_PRICE         = 2.0          # No penny stocks
MAX_VOLUME_ADDITIONS = 15        # Top 15 by volume


def build_universe() -> list[str]:
    """
    Build today's trading universe.

    Returns CORE_UNIVERSE + up to 15 highest volume stocks from
    VOLUME_CANDIDATES that meet market cap and price filters.
    Deduplicates automatically.
    """
    logger.info(
        "Building dynamic universe — scanning %d candidates...",
        len(VOLUME_CANDIDATES),
    )

    additions: list[tuple[str, int]] = []  # (symbol, volume)

    try:
        candidates_not_in_core = [
            s for s in VOLUME_CANDIDATES if s not in CORE_UNIVERSE
        ]

        data = yf.download(
            candidates_not_in_core,
            period="1d",
            interval="1d",
            progress=False,
            threads=True,
        )

        if data.empty:
            logger.warning(
                "universe.py: yfinance returned no data — using core only"
            )
            return list(CORE_UNIVERSE)

        volumes = data["Volume"].iloc[-1] if len(data) > 0 else pd.Series()
        prices  = data["Close"].iloc[-1]  if len(data) > 0 else pd.Series()

        for sym in candidates_not_in_core:
            try:
                vol   = int(volumes[sym])   if sym in volumes.index else 0
                price = float(prices[sym].item()) if sym in prices.index else 0.0

                if price < MIN_PRICE or vol == 0:
                    continue

                # Check market cap via fast_info
                try:
                    market_cap = getattr(yf.Ticker(sym).fast_info, "market_cap", None) or 0
                    if market_cap < MIN_MARKET_CAP:
                        logger.debug(
                            "universe.py: %s skipped — market cap $%.0fM < $500M",
                            sym, market_cap / 1_000_000,
                        )
                        continue
                except Exception:
                    pass  # Cannot get market cap — include anyway

                additions.append((sym, vol))

            except Exception as exc:
                logger.debug("universe.py: %s error — %s", sym, exc)
                continue

        # Sort by volume descending, take top N
        additions.sort(key=lambda x: x[1], reverse=True)
        top_additions = [sym for sym, _ in additions[:MAX_VOLUME_ADDITIONS]]

        universe = list(CORE_UNIVERSE) + top_additions

        logger.info(
            "Dynamic universe built: %d core + %d volume additions = %d total | additions: %s",
            len(CORE_UNIVERSE), len(top_additions), len(universe),
            ", ".join(top_additions),
        )
        return universe

    except Exception as exc:
        logger.error(
            "universe.py: failed to build dynamic universe — %s", exc
        )
        logger.warning("Falling back to core universe only")
        return list(CORE_UNIVERSE)


def get_universe() -> list[str]:
    """Public entry point — returns today's trading universe."""
    return build_universe()
