"""
main.py — Rosetta1 autonomous intraday scalping system.

Entry point.  Orchestrates all subsystems:

  FeedManager     → live quotes for each symbol in UNIVERSE
  SignalEngine    → evaluates all 5 strategies on each quote update
  RiskGuard       → pre-order risk checks
  OrderManager    → entry/exit execution + position monitor loop
  TradeLogger     → durable SQLite trade log
  Dashboard       → Rich terminal UI

Usage:
    python main.py                   # paper mode, all symbols
    python main.py --live            # live trading (Robinhood)
    python main.py --symbols SPY QQQ # subset of universe
    python main.py --no-dashboard    # suppress terminal UI
    python main.py --log-level DEBUG

Safety:
  Live trading requires ROBINHOOD_USERNAME, ROBINHOOD_PASSWORD, and
  ROBINHOOD_TOTP_SECRET environment variables (or .env file).
  The TradingModeError guard in RobinhoodBroker prevents accidental live
  orders — it must be removed only after full paper-mode validation.

Bar seeding:
  OHLCV bars are fetched once at startup (7-day window, 1-min interval) and
  used for indicator calculation throughout the session.  In a future version
  the bars will be updated in real-time via a streaming API.
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional

import pandas as pd
import pytz

from analytics.dashboard import Dashboard
from analytics.logger import TradeLogger
from analytics.performance import compute
from broker.base import BrokerInterface, Quote
from broker.paper import PaperBroker
from config.settings import UNIVERSE, settings
from data.feed import FeedManager
from data.history import seed_bars, HistoryError
from data.indicators import add_all
from execution.order_manager import OrderManager
from risk.guard import RiskGuard
from signals.base import SignalResult
from strategy.engine import SignalEngine

ET = pytz.timezone("America/New_York")

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Rosetta1 autonomous scalping bot")
    p.add_argument(
        "--live", action="store_true",
        help="Use live Robinhood broker (default: paper)",
    )
    p.add_argument(
        "--symbols", nargs="+", default=None, metavar="SYM",
        help="Symbols to trade (default: full UNIVERSE)",
    )
    p.add_argument(
        "--no-dashboard", action="store_true",
        help="Disable the Rich terminal dashboard",
    )
    p.add_argument(
        "--log-level", default=settings.monitoring.log_level,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Broker factory
# ---------------------------------------------------------------------------

def _make_broker(live: bool) -> BrokerInterface:
    if live:
        from broker.robinhood import RobinhoodBroker
        return RobinhoodBroker()
    return PaperBroker()


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

class Rosetta1:
    """
    Top-level orchestrator.

    Lifecycle:
        r = Rosetta1(args)
        r.run()          # blocks until EOD or SIGINT
    """

    def __init__(self, args: argparse.Namespace) -> None:
        self._args = args
        self._symbols: List[str] = args.symbols or UNIVERSE
        self._started_at = datetime.utcnow()
        self._shutdown_requested = False

        # ── Core components ─────────────────────────────────────────────────
        self._broker = _make_broker(args.live)
        self._risk   = RiskGuard()
        self._om     = OrderManager(self._broker, self._risk)
        self._logger = TradeLogger()
        self._engine = SignalEngine()
        self._feed   = FeedManager(self._broker, symbols=self._symbols)

        # OHLCV bars seeded at startup, keyed by symbol
        self._bars: Dict[str, Optional[pd.DataFrame]] = {
            sym: None for sym in self._symbols
        }

        # ── Wire trade-closed callbacks ──────────────────────────────────────
        self._om.on_trade_closed(self._logger.log_trade)

        # ── Per-symbol quote callbacks ───────────────────────────────────────
        for sym in self._symbols:
            # Default-arg capture avoids the late-binding closure problem
            self._feed.subscribe(sym, lambda q, s=sym: self._on_quote(s, q))

        # ── Dashboard ────────────────────────────────────────────────────────
        self._dashboard: Optional[Dashboard] = None
        if not args.no_dashboard:
            self._dashboard = Dashboard(
                self._om, self._risk, self._logger,
                started_at=self._started_at,
            )

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Seed bars, start all subsystems, and block until shutdown."""
        mode = "LIVE" if self._args.live else "PAPER"
        logger.info("Rosetta1 starting | mode=%s | symbols=%s", mode, self._symbols)

        # ── Seed historical bars ─────────────────────────────────────────────
        logger.info("Seeding OHLCV bars…")
        for sym in self._symbols:
            try:
                raw = seed_bars(sym)
                raw.columns = [c.lower() for c in raw.columns]
                if len(raw) > 0 and (raw["volume"].iloc[-1] == 0 or pd.isna(raw["volume"].iloc[-1])):
                    raw = raw.iloc[:-1]
                self._bars[sym] = add_all(raw)
                logger.info("Seeded bars for %s (%d rows)", sym, len(self._bars[sym]))
            except HistoryError as exc:
                logger.warning("Could not seed bars for %s: %s — skipping", sym, exc)

        # ── Start subsystems ─────────────────────────────────────────────────
        self._om.start()
        self._feed.start()
        if self._dashboard:
            self._dashboard.start()

        logger.info("All subsystems running — press Ctrl+C to stop")

        # ── Block until shutdown ─────────────────────────────────────────────
        try:
            while not self._shutdown_requested:
                time.sleep(0.5)
                self._check_eod()
        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt received — shutting down")
        finally:
            self._shutdown()

    # ------------------------------------------------------------------
    # Quote callback (invoked from FeedManager poll threads)
    # ------------------------------------------------------------------

    def _on_quote(self, symbol: str, quote: Quote) -> None:
        """
        Invoked by FeedManager whenever a new quote arrives for *symbol*.

        Runs signal evaluation → risk check → order execution.
        Exceptions are caught so a bad symbol never kills the feed thread.
        """
        if self._shutdown_requested or self._risk.is_halted:
            return

        bars = self._bars.get(symbol)
        if bars is None or len(bars) < 30:
            return

        if bars["volume"].iloc[-1] == 0 or pd.isna(bars["volume"].iloc[-1]):
            bars = bars.iloc[:-1]

        if len(bars) < 30:
            return

        if not quote.last or quote.last <= 0:
            return

        try:
            # ── Refresh indicators with latest quote price ────────────────
            # Update the last bar's close to the current quote, then
            # recompute all indicator columns.  add_all() is fast enough
            # here because the feed only delivers quotes every 10 s.
            bars = bars.copy()
            bars.loc[bars.index[-1], "close"] = quote.last
            bars = add_all(bars)
            self._bars[symbol] = bars

            # ── Signal evaluation ─────────────────────────────────────────
            signal: Optional[SignalResult] = self._engine.evaluate(
                bars, symbol, quote.last
            )
            if signal is None:
                return

            # ── Risk check ────────────────────────────────────────────────
            account        = self._broker.get_account()
            open_positions = self._broker.get_open_positions()
            is_open        = self._broker.is_market_open()
            decision       = self._risk.check(
                signal, quote, account, open_positions, is_open
            )

            if not decision.approved:
                logger.debug(
                    "RiskGuard rejected %s %s: %s",
                    symbol, signal.direction, decision.reason,
                )
                return

            # ── Execute ───────────────────────────────────────────────────
            self._om.execute_signal(signal, decision)

        except Exception as exc:
            logger.error(
                "Error processing quote for %s: %s", symbol, exc, exc_info=True
            )

    # ------------------------------------------------------------------
    # EOD check
    # ------------------------------------------------------------------

    def _check_eod(self) -> None:
        """
        Trigger graceful shutdown once EOD liquidation + buffer has passed
        and OrderManager has closed all positions.
        """
        now_et = datetime.now(ET)
        eod_h = settings.risk.eod_liquidation_hour
        eod_m = settings.risk.eod_liquidation_minute
        # Wait 5 minutes past EOD cutoff before shutting down main loop
        eod = now_et.replace(
            hour=eod_h, minute=eod_m + 5, second=0, microsecond=0
        )
        if now_et >= eod and self._om.open_trade_count() == 0:
            logger.info("EOD + all positions closed — initiating shutdown")
            self._shutdown_requested = True

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    def _shutdown(self) -> None:
        logger.info("Rosetta1 shutting down…")

        # Force-close any remaining positions before stopping the monitor
        if self._om.open_trade_count() > 0:
            logger.warning(
                "Forcing close of %d open position(s)", self._om.open_trade_count()
            )
            self._om.force_close_all(reason="manual")

        # Stop subsystems in reverse dependency order
        if self._dashboard:
            self._dashboard.stop()
        self._feed.stop()
        self._om.stop()
        self._logger.close()

        # Print session summary
        today_trades = self._logger.get_today_trades()
        metrics = compute(today_trades)
        logger.info("Session complete | %s", metrics)
        _print_summary(metrics)


# ---------------------------------------------------------------------------
# Session summary (plain-text, shown after dashboard stops)
# ---------------------------------------------------------------------------

def _print_summary(metrics) -> None:
    print("\n" + "═" * 60)
    print("  Rosetta1 Session Summary")
    print("═" * 60)
    print(f"  Trades:        {metrics.num_trades}")
    print(f"  Win rate:      {metrics.win_rate*100:.1f}%")
    print(f"  Total P&L:     ${metrics.total_pnl:+.2f}")
    print(f"  Profit factor: {metrics.profit_factor:.2f}")
    print(f"  Sharpe ratio:  {metrics.sharpe_ratio:.2f}")
    print(f"  Max drawdown:  ${metrics.max_drawdown:.2f}")
    print("═" * 60 + "\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    args = _parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    bot = Rosetta1(args)

    def _handle_signal(signum, frame):
        logger.warning("Signal %d received — requesting shutdown", signum)
        bot._shutdown_requested = True

    signal.signal(signal.SIGINT,  _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    bot.run()


if __name__ == "__main__":
    main()
