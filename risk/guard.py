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
        # Also enforce EOD cutoff — no new entries after 3:45 PM ET
        now_et = datetime.now(ET)
        eod_h = settings.risk.eod_liquidation_hour
        eod_m = settings.risk.eod_liquidation_minute
        eod_cutoff = now_et.replace(hour=eod_h, minute=eod_m, second=0, microsecond=0)
        if now_et >= eod_cutoff:
            return _reject("market_hours",
                           f"Past EOD cutoff ({eod_h:02d}:{eod_m:02d} ET) — no new entries")
        passed.append("market_hours")

        # ── 3. Daily loss limit ─────────────────────────────────────────────
        with self._lock:
            daily_pnl = self._daily_pnl
        if daily_pnl <= -settings.risk.max_daily_loss:
            # Trigger emergency halt so subsequent calls are blocked instantly
            self.halt(f"Daily loss limit hit: ${daily_pnl:.2f}")
            return _reject("daily_loss_limit",
                           f"Daily loss ${abs(daily_pnl):.2f} exceeds limit "
                           f"${settings.risk.max_daily_loss:.2f}")
        passed.append("daily_loss_limit")

        # ── 4. Max open positions ────────────────────────────────────────────
        n_positions = len(open_positions)
        if n_positions >= settings.risk.max_open_positions:
            return _reject("max_positions",
                           f"Already at max positions ({n_positions}/"
                           f"{settings.risk.max_open_positions})")
        passed.append("max_positions")

        # ── 4b. Duplicate symbol ─────────────────────────────────────────────
        open_symbols = [p.symbol for p in open_positions]
        if signal.symbol in open_symbols:
            return _reject("duplicate_symbol",
                           f"Already have open position in {signal.symbol}")
        passed.append("duplicate_symbol")

        # ── 5. PDT limit ────────────────────────────────────────────────────
        # AccountInfo.day_trades_used counts trades in the rolling 5-day window.
        # We conservatively block at max_day_trades (not max_day_trades - 1) so
        # the final day trade is still usable for emergency exits.
        if account.day_trades_used >= settings.risk.max_day_trades:
            return _reject("pdt_limit",
                           f"PDT limit reached: {account.day_trades_used}/"
                           f"{settings.risk.max_day_trades} day trades used")
        passed.append("pdt_limit")

        # ── 6. Spread filter ────────────────────────────────────────────────
        spread = quote.spread_pct
        if spread > settings.risk.max_spread_pct:
            return _reject("spread_filter",
                           f"Spread {spread*100:.3f}% exceeds limit "
                           f"{settings.risk.max_spread_pct*100:.2f}%")
        passed.append("spread_filter")

        # ── 7. Capital limit + position sizing ──────────────────────────────
        max_capital = account.buying_power * settings.risk.max_capital_per_trade_pct
        entry_price = signal.entry_price
        if entry_price <= 0:
            return _reject("capital_limit", "Signal entry_price is zero or negative")

        shares = math.floor(max_capital / entry_price)
        if shares < 1:
            return _reject("capital_limit",
                           f"Cannot afford even 1 share at ${entry_price:.2f} "
                           f"with ${max_capital:.2f} allocated capital")

        capital_used = shares * entry_price
        if capital_used > account.buying_power:
            # Reduce by one share to stay within buying power
            shares -= 1
            capital_used = shares * entry_price
            if shares < 1:
                return _reject("capital_limit",
                               f"Insufficient buying power: ${account.buying_power:.2f}")
        passed.append("capital_limit")

        # ── 8. Risk/reward validation ────────────────────────────────────────
        rr = signal.risk_reward_ratio
        if rr > settings.risk.max_loss_to_gain_ratio:
            return _reject("risk_reward",
                           f"R:R {rr:.2f} exceeds max {settings.risk.max_loss_to_gain_ratio:.1f} "
                           f"(risk=${signal.risk:.2f}, reward=${signal.reward:.2f})")
        passed.append("risk_reward")

        # ── 9. Stop distance ────────────────────────────────────────────────
        stop_dist_pct = abs(signal.entry_price - signal.stop_price) / signal.entry_price
        if stop_dist_pct > settings.risk.max_stop_distance_pct:
            return _reject("stop_distance",
                           f"Stop distance {stop_dist_pct*100:.2f}% exceeds limit "
                           f"{settings.risk.max_stop_distance_pct*100:.2f}%")
        passed.append("stop_distance")

        # ── All checks passed ────────────────────────────────────────────────
        logger.info(
            "RiskGuard APPROVED [%s] %s %s | shares=%d capital=$%.2f | "
            "spread=%.3f%% rr=%.2f stop_dist=%.2f%%",
            signal.symbol, signal.direction.upper(), signal.signal_type,
            shares, capital_used,
            spread * 100, rr, stop_dist_pct * 100,
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
        Compare actual fill price against the signal's entry price.

        Call this immediately after place_order() returns a fill.
        If slippage is unacceptable, the caller should close the position.

        Args:
            fill_price:   avg_fill_price from OrderResult.
            signal_price: signal.entry_price used when the order was placed.
        """
        if signal_price <= 0:
            return SlippageDecision(
                acceptable=False,
                slippage_pct=0.0,
                limit_pct=settings.risk.max_slippage_pct,
                reason="Signal price is zero — cannot validate slippage",
            )

        slippage = abs(fill_price - signal_price) / signal_price
        limit = settings.risk.max_slippage_pct
        acceptable = slippage <= limit

        if not acceptable:
            logger.warning(
                "RiskGuard SLIPPAGE: fill=%.4f signal=%.4f slippage=%.4f%% limit=%.4f%%",
                fill_price, signal_price, slippage * 100, limit * 100,
            )

        return SlippageDecision(
            acceptable=acceptable,
            slippage_pct=slippage,
            limit_pct=limit,
            reason=(
                "ok" if acceptable else
                f"Slippage {slippage*100:.4f}% exceeds limit {limit*100:.4f}%"
            ),
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
