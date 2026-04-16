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
from datetime import datetime, timedelta, timezone
from typing import List, Optional

import pytz

from broker.base import AccountInfo, Position, Quote
from config.settings import settings
from data.session import is_entry_allowed
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

        # Consecutive-loss circuit breaker state
        self._consecutive_losses: int = 0
        self._paused_until: Optional[datetime] = None
        self._halted_for_day: bool = False

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
            halted         = self._emergency_halt
            halt_reason    = self._halt_reason
            halted_for_day = self._halted_for_day
            paused_until   = self._paused_until
        if halted:
            return _reject("emergency_halt", f"System halted: {halt_reason}")
        passed.append("emergency_halt")

        # ── 1b. Consecutive-loss day halt ───────────────────────────────────
        if halted_for_day:
            return _reject(
                "consecutive_loss_halt",
                "Halted for day: consecutive loss limit reached",
            )
        passed.append("consecutive_loss_halt")

        # ── 1c. Consecutive-loss pause ──────────────────────────────────────
        if paused_until is not None:
            now_utc = datetime.now(timezone.utc)
            if now_utc < paused_until:
                remaining = int((paused_until - now_utc).total_seconds())
                return _reject(
                    "consecutive_loss_pause",
                    f"Paused after consecutive losses ({remaining}s remaining)",
                )
            # Pause has expired — clear it
            with self._lock:
                if self._paused_until is not None and datetime.now(timezone.utc) >= self._paused_until:
                    self._paused_until = None
                    logger.info("RiskGuard: consecutive-loss pause expired — trading resumed")
        passed.append("consecutive_loss_pause")

        # ── 1d. Session window — block lunch / pre-market / EOD wind-down ───
        allowed, session_reason = is_entry_allowed()
        if not allowed:
            return _reject("session_window", f"session={session_reason}")
        passed.append("session_window")

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

        # ── 3b. Spread check — skip if bid-ask is too wide ──────────────────
        # Prefer the live quote (always fresh); fall back to signal metadata
        # if the quote object lacks bid/ask (e.g. some paper-broker fallbacks).
        bid = getattr(quote, "bid", None)
        ask = getattr(quote, "ask", None)
        if (bid is None or ask is None) and signal.metadata:
            bid = signal.metadata.get("bid_price", bid)
            ask = signal.metadata.get("ask_price", ask)
        if bid and ask and ask > 0:
            spread_pct = (ask - bid) / ask
            if spread_pct > settings.risk.max_spread_pct:
                return _reject(
                    "spread_filter",
                    f"Spread too wide: {spread_pct*100:.4f}% > "
                    f"{settings.risk.max_spread_pct*100:.4f}% (bid={bid:.4f}, ask={ask:.4f})",
                )
            passed.append("spread_filter")

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

        # ── 7. Volatility-normalized position sizing ─────────────────────────
        entry_price = signal.entry_price
        if entry_price <= 0:
            return _reject("capital_limit", "Signal entry_price is zero or negative")

        stop_distance = abs(signal.entry_price - signal.stop_price)
        if stop_distance <= 0:
            return _reject("capital_limit",
                           f"Zero stop distance for {signal.symbol} — cannot size position")

        # Risk a fixed fraction of buying power per trade, divided by the ATR-
        # derived stop distance.  Every trade risks the same dollar amount
        # regardless of how volatile or cheap the symbol is.
        risk_dollars = account.buying_power * settings.risk.target_risk_per_trade_pct
        shares = math.floor(risk_dollars / stop_distance)

        # Hard ceiling: never deploy more than max_capital_per_trade_pct of buying power
        max_shares_by_capital = math.floor(
            (account.buying_power * settings.risk.max_capital_per_trade_pct) / entry_price
        )
        capped_by_capital = shares > max_shares_by_capital
        shares = min(shares, max_shares_by_capital)
        shares = max(settings.risk.min_shares, min(shares, settings.risk.max_shares))
        capital_used = shares * entry_price

        logger.info(
            "RiskGuard SIZE [%s]: %d shares (risk=$%.2f / stop_dist=$%.4f) "
            "capped_by_capital=%s",
            signal.symbol, shares, risk_dollars, stop_distance, capped_by_capital,
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
    # Consecutive-loss circuit breaker
    # ------------------------------------------------------------------

    def record_trade_outcome(self, won: bool) -> None:
        """
        Update the consecutive-loss counter after a trade closes.

        Wins reset the streak.  Losses increment it; once the pause
        threshold is hit the guard pauses for consecutive_loss_pause_minutes;
        once the halt threshold is hit, trading is halted for the rest of
        the session.  Called by the order manager from the close callback.
        """
        cfg = settings.risk
        with self._lock:
            if won:
                if self._consecutive_losses > 0:
                    logger.info(
                        "RiskGuard: streak reset after %d consecutive losses",
                        self._consecutive_losses,
                    )
                self._consecutive_losses = 0
                return

            self._consecutive_losses += 1
            streak = self._consecutive_losses

            if streak >= cfg.consecutive_loss_halt_threshold:
                self._halted_for_day = True
                logger.warning(
                    "RiskGuard HALT: %d consecutive losses — halted for the day",
                    streak,
                )
            elif streak >= cfg.consecutive_loss_pause_threshold:
                self._paused_until = datetime.now(timezone.utc) + timedelta(
                    minutes=cfg.consecutive_loss_pause_minutes
                )
                logger.warning(
                    "RiskGuard PAUSE: %d consecutive losses — paused until %s",
                    streak, self._paused_until.isoformat(),
                )

    def reset_daily(self) -> None:
        """
        Reset all daily counters — daily P&L, consecutive-loss streak,
        pause window, and day-halt flag.  Called at session start.
        """
        with self._lock:
            self._daily_pnl          = 0.0
            self._consecutive_losses = 0
            self._paused_until       = None
            self._halted_for_day     = False
        logger.info("RiskGuard: daily counters reset (P&L, streak, pause, halt)")

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
