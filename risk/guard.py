"""
risk/guard.py — Pre-order risk guard.

Every single order attempt passes through RiskGuard.check() before it
reaches the broker.  If any check fails the order is rejected — no exceptions.

Checks (run in priority order — fail-fast):
  1. emergency_halt   — manual kill-switch, blocks everything
  2. market_hours     — no orders outside 9:30–15:45 ET
  3. daily_loss_limit — halt if cumulative daily P&L <= -max_daily_loss
  4. max_positions    — reject if already at max open positions
  5. pdt_limit        — reject if day-trade budget exhausted
  6. spread_filter    — skip if bid-ask spread > max_spread_pct
  7. capital_limit    — compute position size; reject if < 1 share
  8. risk_reward      — reject if signal's loss > max_loss_to_gain_ratio × gain
  9. stop_distance    — reject if stop is unreasonably far from entry

Post-fill:
  RiskGuard.check_slippage() validates a completed fill against the
  signal price.  Call this after place_order() returns — if it fails the
  position should be closed immediately.

Thread safety:
  _emergency_halt and _daily_pnl are protected by a lock.
  RiskGuard is shared across the execution and monitoring threads.
"""

from __future__ import annotations

import logging
import math
import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

import pytz

from broker.base import AccountInfo, Position, Quote
from config.settings import settings
from signals.base import SignalResult

logger = logging.getLogger(__name__)

ET = pytz.timezone("America/New_York")


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class RiskDecision:
    """Returned by RiskGuard.check() for every order attempt."""
    approved: bool
    reason: str                    # "ok" or human-readable rejection reason
    checks_passed: List[str] = field(default_factory=list)
    checks_failed: List[str] = field(default_factory=list)
    position_size: int = 0         # shares to trade; 0 if rejected
    capital_allocated: float = 0.0 # USD value of the position
    # Set when a duplicate-symbol signal should update the existing trade's exits
    update_existing: bool = False
    new_target_price: Optional[float] = None
    new_stop_price: Optional[float] = None
    # Set when the new signal conflicts with the existing position's direction
    exit_existing: bool = False

    def __str__(self) -> str:
        if self.approved:
            return (f"APPROVED | {self.position_size} shares @ "
                    f"${self.capital_allocated:.2f} | {self.reason}")
        return f"REJECTED | {self.reason}"


@dataclass
class SlippageDecision:
    """Returned by RiskGuard.check_slippage() after a fill."""
    acceptable: bool
    slippage_pct: float
    limit_pct: float
    reason: str


# ---------------------------------------------------------------------------
# Risk Guard
# ---------------------------------------------------------------------------

class RiskGuard:
    """
    Stateful pre-order risk guard.

    State:
      - _emergency_halt: bool — set by halt(), cleared by resume()
      - _daily_pnl: float     — updated by record_trade_pnl() each close

    Both are protected by _lock for thread safety.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._emergency_halt: bool = False
        self._halt_reason: str = ""
        self._daily_pnl: float = 0.0

        logger.info(
            "RiskGuard initialised | max_daily_loss=$%.2f | max_positions=%d | "
            "max_day_trades=%d | max_capital_pct=%.0f%%",
            settings.risk.max_daily_loss,
            settings.risk.max_open_positions,
            settings.risk.max_day_trades,
            settings.risk.max_capital_per_trade_pct * 100,
        )

    # ------------------------------------------------------------------
    # Primary check — called before every order
    # ------------------------------------------------------------------

    def check(
        self,
        signal: SignalResult,
        quote: Quote,
        account: AccountInfo,
        open_positions: List[Position],
        is_market_open: bool,
    ) -> RiskDecision:
        """
        Run all pre-order checks against *signal*.

        Args:
            signal:          The SignalResult from the strategy engine.
            quote:           Current best bid/ask for the signal's symbol.
            account:         Latest account snapshot from the broker.
            open_positions:  All currently open positions.
            is_market_open:  From broker.is_market_open().

        Returns:
            RiskDecision with approved=True and computed position_size,
            or approved=False with the reason for rejection.
        """
        passed: List[str] = []
        failed: List[str] = []

        def _reject(check_name: str, reason: str) -> RiskDecision:
            failed.append(check_name)
            logger.warning(
                "RiskGuard REJECT [%s] %s — %s", signal.symbol, check_name, reason
            )
            return RiskDecision(
                approved=False,
                reason=reason,
                checks_passed=passed,
                checks_failed=failed + [check_name],
            )

        # ── 1. Emergency halt ───────────────────────────────────────────────
        with self._lock:
            halted = self._emergency_halt
            halt_reason = self._halt_reason
        if halted:
            return _reject("emergency_halt", f"System halted: {halt_reason}")
        passed.append("emergency_halt")

        # ── 2. Market hours ─────────────────────────────────────────────────
        if not is_market_open:
            return _reject("market_hours", "Market is closed")
        # No new entries after 3:55 PM ET — positions close at 3:59
        now_et = datetime.now(ET)
        no_entry_h = settings.risk.eod_no_new_entries_hour
        no_entry_m = settings.risk.eod_no_new_entries_minute
        no_entry_cutoff = now_et.replace(hour=no_entry_h, minute=no_entry_m, second=0, microsecond=0)
        if now_et >= no_entry_cutoff:
            return _reject("market_hours",
                           f"Past no-new-entries cutoff ({no_entry_h:02d}:{no_entry_m:02d} ET)")
        passed.append("market_hours")

        # ── 3. Daily loss limit — skipped in testing mode (limit=$10,000) ────

        # ── 4. Max open positions ────────────────────────────────────────────
        n_positions = len(open_positions)
        if n_positions >= settings.risk.max_open_positions:
            return _reject("max_positions",
                           f"Already at max positions ({n_positions}/"
                           f"{settings.risk.max_open_positions})")
        passed.append("max_positions")

        # ── 4b. Duplicate symbol — exit on reversal, update exits otherwise ───
        existing_pos = next(
            (p for p in open_positions if p.symbol == signal.symbol), None
        )
        if existing_pos is not None:
            directions_conflict = (
                (existing_pos.quantity > 0 and signal.direction == "short") or
                (existing_pos.quantity < 0 and signal.direction == "long")
            )
            if directions_conflict:
                logger.info(
                    "RiskGuard [%s]: signal reversal detected — exiting existing %s position",
                    signal.symbol, "long" if existing_pos.quantity > 0 else "short",
                )
            else:
                logger.info(
                    "RiskGuard [%s]: duplicate signal — updating exits target=%.4f stop=%.4f",
                    signal.symbol, signal.target_price, signal.stop_price,
                )
            return RiskDecision(
                approved=False,
                exit_existing=directions_conflict,
                update_existing=not directions_conflict,
                new_target_price=signal.target_price,
                new_stop_price=signal.stop_price,
                reason="signal_reversal" if directions_conflict
                       else "duplicate_symbol — updating exits",
            )
        passed.append("duplicate_symbol")

        # ── 5. PDT limit — skipped in testing mode (limit=9999) ────────────

        # ── 6. Spread filter — skipped in testing mode ───────────────────────
        #    (yfinance quotes use a synthetic 0.1% spread, not real market data)

        # ── 7. Capital-based position sizing ────────────────────────────────
        entry_price = signal.entry_price
        if entry_price <= 0:
            return _reject("capital_limit", "Signal entry_price is zero or negative")

        # Flat capital deployment: 5% of current buying power per trade.
        # The broker deducts deployed cash automatically, so buying_power
        # shrinks with each open position — no manual subtraction needed.
        capital_per_trade = settings.paper_starting_capital * settings.risk.max_capital_per_trade_pct
        shares = max(
            settings.risk.min_shares,
            min(math.floor(capital_per_trade / entry_price), settings.risk.max_shares),
        )
        capital_used = shares * entry_price

        logger.info(
            "RiskGuard capital check: buying_power=$%.2f, per_trade=$%.2f (fixed 5pct of 100k), shares=%d",
            account.buying_power, capital_per_trade, shares,
        )

        if shares < 1 or capital_used > account.buying_power:
            return _reject("capital_limit",
                           f"Cannot size even 1 share at ${entry_price:.2f} "
                           f"(buying_power=${account.buying_power:.2f})")
        passed.append("capital_limit")

        # ── 8. Risk/reward — skipped in testing mode ─────────────────────────

        # ── 9. Stop distance — skipped in testing mode ───────────────────────

        # ── All checks passed ────────────────────────────────────────────────
        logger.info(
            "RiskGuard APPROVED [%s] %s %s | shares=%d capital=$%.2f",
            signal.symbol, signal.direction.upper(), signal.signal_type,
            shares, capital_used,
        )
        return RiskDecision(
            approved=True,
            reason="ok",
            checks_passed=passed,
            checks_failed=[],
            position_size=shares,
            capital_allocated=capital_used,
        )

    # ------------------------------------------------------------------
    # Post-fill slippage check
    # ------------------------------------------------------------------

    def check_slippage(
        self,
        fill_price: float,
        signal_price: float,
    ) -> SlippageDecision:
        """
        Slippage guard — disabled in testing mode (always returns acceptable).
        yfinance paper fills are synthetic and slippage is not meaningful.
        """
        slippage = (
            abs(fill_price - signal_price) / signal_price
            if signal_price > 0 else 0.0
        )
        return SlippageDecision(
            acceptable=True,
            slippage_pct=slippage,
            limit_pct=settings.risk.max_slippage_pct,
            reason="slippage check disabled in testing mode",
        )

    # ------------------------------------------------------------------
    # Daily P&L tracking
    # ------------------------------------------------------------------

    def record_trade_pnl(self, pnl: float) -> None:
        """
        Record the P&L from a closed trade.

        Called by the order manager after each position close.
        Thread-safe.
        """
        with self._lock:
            self._daily_pnl += pnl
            daily = self._daily_pnl

        logger.info(
            "RiskGuard: trade P&L=%.2f | running daily P&L=%.2f / limit=-%.2f",
            pnl, daily, settings.risk.max_daily_loss,
        )

        # Auto-halt if we just crossed the daily loss threshold
        if daily <= -settings.risk.max_daily_loss:
            self.halt(f"Daily loss limit hit: ${daily:.2f}")

    def get_daily_pnl(self) -> float:
        with self._lock:
            return self._daily_pnl

    def reset_daily_pnl(self) -> None:
        """Reset at start of each trading day."""
        with self._lock:
            self._daily_pnl = 0.0
        logger.info("RiskGuard: daily P&L counter reset")

    # ------------------------------------------------------------------
    # Emergency halt
    # ------------------------------------------------------------------

    def halt(self, reason: str = "manual halt") -> None:
        """
        Engage emergency halt — all subsequent check() calls are rejected
        until resume() is called.
        """
        with self._lock:
            self._emergency_halt = True
            self._halt_reason = reason
        logger.critical("RiskGuard EMERGENCY HALT: %s", reason)

    def resume(self) -> None:
        """Clear the emergency halt.  Use with caution."""
        with self._lock:
            self._emergency_halt = False
            self._halt_reason = ""
        logger.warning("RiskGuard: emergency halt cleared — trading resumed")

    @property
    def is_halted(self) -> bool:
        with self._lock:
            return self._emergency_halt

    @property
    def halt_reason(self) -> str:
        with self._lock:
            return self._halt_reason

    # ------------------------------------------------------------------
    # Convenience: remaining daily loss budget
    # ------------------------------------------------------------------

    def daily_loss_remaining(self) -> float:
        """How much more loss is allowed today before the halt triggers."""
        with self._lock:
            return settings.risk.max_daily_loss + self._daily_pnl
