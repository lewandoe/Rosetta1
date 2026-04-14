"""
broker/robinhood.py — Robinhood broker implementation via robin_stocks.

SAFETY CONTRACT:
  - All order-placement paths check settings.broker.trading_mode == "live".
  - If mode is not "live", place_order raises TradingModeError immediately.
  - This check is the last line of defence before real money moves.

Authentication:
  - Calls robin_stocks.robinhood.login() once at construction.
  - Supports TOTP 2-FA via pyotp if mfa_secret is set in .env.
  - Stores the session token in robin_stocks' internal session object —
    we do not persist it to disk ourselves.

Rate limiting / retries:
  - Transient HTTP errors are retried up to settings.execution.max_order_retries
    times with exponential back-off.
  - After max retries, BrokerError is raised — caller halts the trade.
"""

from __future__ import annotations

import logging
import time
import uuid
from datetime import datetime
from typing import Optional

import pytz
import robin_stocks.robinhood as rh

from config.settings import settings
from broker.base import (
    AccountInfo,
    BrokerError,
    BrokerInterface,
    Order,
    OrderResult,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
    Quote,
    SymbolNotInUniverseError,
    TradingModeError,
)

logger = logging.getLogger(__name__)

ET = pytz.timezone("America/New_York")

# robin_stocks status strings → our OrderStatus enum
_RH_STATUS_MAP: dict[str, OrderStatus] = {
    "filled": OrderStatus.FILLED,
    "partially_filled": OrderStatus.PARTIALLY_FILLED,
    "queued": OrderStatus.PENDING,
    "unconfirmed": OrderStatus.PENDING,
    "confirmed": OrderStatus.PENDING,
    "cancelled": OrderStatus.CANCELLED,
    "rejected": OrderStatus.REJECTED,
    "failed": OrderStatus.FAILED,
}


def _map_status(rh_status: str) -> OrderStatus:
    return _RH_STATUS_MAP.get(rh_status.lower(), OrderStatus.FAILED)


class RobinhoodBroker(BrokerInterface):
    """
    Live/paper Robinhood broker backed by robin_stocks.

    Instantiation triggers authentication.  Raise BrokerError if login fails
    so the system never starts in an unauthenticated state.
    """

    def __init__(self) -> None:
        self._authenticated = False
        self._authenticate()

    # ------------------------------------------------------------------
    # Authentication
    # ------------------------------------------------------------------

    def _authenticate(self) -> None:
        cfg = settings.broker
        if not cfg.username or not cfg.password:
            raise BrokerError(
                "Robinhood credentials not set — check ROBINHOOD_USERNAME and "
                "ROBINHOOD_PASSWORD in .env"
            )

        mfa_code: Optional[str] = None
        if cfg.mfa_secret:
            try:
                import pyotp
                mfa_code = pyotp.TOTP(cfg.mfa_secret).now()
            except ImportError:
                raise BrokerError(
                    "pyotp is required for MFA login — run: pip install pyotp"
                )

        try:
            login_kwargs: dict = {
                "username": cfg.username,
                "password": cfg.password,
                "store_session": False,  # never write tokens to disk
            }
            if mfa_code:
                login_kwargs["mfa_code"] = mfa_code

            rh.login(**login_kwargs)
            self._authenticated = True
            logger.info("RobinhoodBroker: authenticated as %s", cfg.username)
        except Exception as exc:
            raise BrokerError(f"Robinhood login failed: {exc}") from exc

    def _require_auth(self) -> None:
        if not self._authenticated:
            raise BrokerError("RobinhoodBroker is not authenticated")

    # ------------------------------------------------------------------
    # Retry wrappers
    # ------------------------------------------------------------------

    def _with_retry(self, fn, *args, **kwargs):
        """
        Call fn(*args, **kwargs) up to max_order_retries times.
        Exponential back-off: 1s, 2s, 4s, …
        """
        max_retries = settings.execution.max_order_retries
        for attempt in range(1, max_retries + 1):
            try:
                return fn(*args, **kwargs)
            except BrokerError:
                raise  # our own errors are not retried
            except Exception as exc:
                if attempt == max_retries:
                    raise BrokerError(
                        f"API call failed after {max_retries} attempts: {exc}"
                    ) from exc
                wait = 2 ** (attempt - 1)
                logger.warning(
                    "RobinhoodBroker: attempt %d/%d failed (%s), retrying in %ds",
                    attempt, max_retries, exc, wait,
                )
                time.sleep(wait)

    @staticmethod
    def _is_rate_limited(exc: Exception) -> bool:
        """Return True if *exc* looks like an HTTP 429 rate-limit response."""
        msg = str(exc).lower()
        return "429" in msg or "too many requests" in msg or "rate limit" in msg

    def _with_retry_rate_aware(self, fn, *args, **kwargs):
        """
        Like _with_retry but applies a longer pause on HTTP 429.

        Retry schedule:
          - 429 (rate limited): 1 s, 2 s, 4 s  (exponential, no further retries)
          - Other transient error: standard 1s/2s/4s back-off
        """
        max_retries = settings.execution.max_order_retries
        for attempt in range(1, max_retries + 1):
            try:
                return fn(*args, **kwargs)
            except BrokerError:
                raise
            except Exception as exc:
                if attempt == max_retries:
                    raise BrokerError(
                        f"API call failed after {max_retries} attempts: {exc}"
                    ) from exc
                if self._is_rate_limited(exc):
                    # Rate-limited: back off more aggressively
                    wait = 2 ** attempt  # 2s, 4s, 8s
                    logger.warning(
                        "RobinhoodBroker: rate-limited (429) on attempt %d/%d — "
                        "backing off %ds before retry",
                        attempt, max_retries, wait,
                    )
                else:
                    wait = 2 ** (attempt - 1)  # 1s, 2s, 4s
                    logger.warning(
                        "RobinhoodBroker: attempt %d/%d failed (%s), retrying in %ds",
                        attempt, max_retries, exc, wait,
                    )
                time.sleep(wait)

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------

    def _validate_symbol(self, symbol: str) -> None:
        # Universe is dynamic — validation is best-effort; skip if not loaded yet
        try:
            from data.universe import get_universe
            universe = get_universe()
            if symbol not in universe:
                raise SymbolNotInUniverseError(
                    f"{symbol} is not in the trading universe"
                )
        except SymbolNotInUniverseError:
            raise
        except Exception:
            pass  # unable to load universe — allow the quote request through

    # ------------------------------------------------------------------
    # Market data
    # ------------------------------------------------------------------

    def get_quote(self, symbol: str) -> Quote:
        self._require_auth()
        self._validate_symbol(symbol)

        def _fetch():
            data = rh.stocks.get_quotes(symbol)
            if not data or data[0] is None:
                raise BrokerError(f"Empty quote response for {symbol}")
            q = data[0]
            bid = float(q.get("bid_price") or 0)
            ask = float(q.get("ask_price") or 0)
            last = float(q.get("last_trade_price") or q.get("last_extended_hours_trade_price") or 0)
            volume = int(float(q.get("trading_halted") or 0))  # not in quotes; default 0
            if bid == 0 and ask == 0:
                raise BrokerError(f"Zero bid/ask for {symbol} — market may be closed")
            return Quote(
                symbol=symbol,
                bid=bid,
                ask=ask,
                last=last,
                volume=volume,
                timestamp=datetime.utcnow(),
            )

        # Use rate-aware retry so HTTP 429 responses get a longer back-off
        return self._with_retry_rate_aware(_fetch)

    # ------------------------------------------------------------------
    # Account
    # ------------------------------------------------------------------

    def get_account(self) -> AccountInfo:
        self._require_auth()

        def _fetch():
            profile = rh.profiles.load_account_profile()
            buying_power = float(profile.get("buying_power") or 0)
            portfolio_cash = float(profile.get("portfolio_cash") or buying_power)

            # Day trades used — from the PDT endpoint
            try:
                pdt_data = rh.account.get_day_trades()
                day_trades_used = len(pdt_data) if pdt_data else 0
            except Exception:
                # PDT data is best-effort; don't halt on unavailability
                day_trades_used = 0
                logger.warning("RobinhoodBroker: could not fetch PDT data")

            # Portfolio value requires a separate call
            try:
                port = rh.profiles.load_portfolio_profile()
                portfolio_value = float(port.get("extended_hours_equity") or
                                        port.get("equity") or buying_power)
            except Exception:
                portfolio_value = buying_power

            return AccountInfo(
                buying_power=buying_power,
                portfolio_value=portfolio_value,
                cash=portfolio_cash,
                day_trades_used=day_trades_used,
                timestamp=datetime.utcnow(),
            )

        return self._with_retry(_fetch)

    # ------------------------------------------------------------------
    # Orders
    # ------------------------------------------------------------------

    def place_order(self, order: Order) -> OrderResult:
        self._require_auth()
        self._validate_symbol(order.symbol)

        # ── SAFETY CHECK ─────────────────────────────────────────────────────
        if settings.broker.trading_mode != "live":
            raise TradingModeError(
                f"place_order called on RobinhoodBroker but trading_mode="
                f"'{settings.broker.trading_mode}'.  Set TRADING_MODE=live in "
                f".env to send real orders."
            )
        # ─────────────────────────────────────────────────────────────────────

        if order.quantity <= 0:
            raise BrokerError(f"Order quantity must be > 0, got {order.quantity}")

        logger.info(
            "RobinhoodBroker: placing %s %s x%d (type=%s)",
            order.side.value.upper(),
            order.symbol,
            order.quantity,
            order.order_type.value,
        )

        def _submit():
            tif = order.time_in_force.value
            qty = str(order.quantity)

            if order.order_type == OrderType.MARKET:
                if order.side == OrderSide.BUY:
                    raw = rh.orders.order_buy_market(
                        order.symbol, qty, timeInForce=tif
                    )
                else:
                    raw = rh.orders.order_sell_market(
                        order.symbol, qty, timeInForce=tif
                    )

            elif order.order_type == OrderType.LIMIT:
                if order.limit_price is None:
                    raise BrokerError("Limit order requires limit_price")
                price = str(round(order.limit_price, 2))
                if order.side == OrderSide.BUY:
                    raw = rh.orders.order_buy_limit(
                        order.symbol, qty, price, timeInForce=tif
                    )
                else:
                    raw = rh.orders.order_sell_limit(
                        order.symbol, qty, price, timeInForce=tif
                    )

            elif order.order_type == OrderType.STOP:
                if order.stop_price is None:
                    raise BrokerError("Stop order requires stop_price")
                stop = str(round(order.stop_price, 2))
                if order.side == OrderSide.BUY:
                    raw = rh.orders.order_buy_stop_loss(
                        order.symbol, qty, stop, timeInForce=tif
                    )
                else:
                    raw = rh.orders.order_sell_stop_loss(
                        order.symbol, qty, stop, timeInForce=tif
                    )

            elif order.order_type == OrderType.STOP_LIMIT:
                if order.stop_price is None or order.limit_price is None:
                    raise BrokerError("Stop-limit order requires both stop_price and limit_price")
                stop = str(round(order.stop_price, 2))
                price = str(round(order.limit_price, 2))
                if order.side == OrderSide.BUY:
                    raw = rh.orders.order_buy_stop_limit(
                        order.symbol, qty, price, stop, timeInForce=tif
                    )
                else:
                    raw = rh.orders.order_sell_stop_limit(
                        order.symbol, qty, price, stop, timeInForce=tif
                    )
            else:
                raise BrokerError(f"Unsupported order type: {order.order_type}")

            if raw is None or "id" not in raw:
                raise BrokerError(
                    f"Robinhood returned unexpected response: {raw}"
                )
            return raw

        raw = self._with_retry(_submit)
        return self._parse_order_result(raw)

    def cancel_order(self, order_id: str) -> bool:
        self._require_auth()

        def _cancel():
            result = rh.orders.cancel_stock_order(order_id)
            # robin_stocks returns {} on success, None or error on failure
            return result is not None

        try:
            cancelled = self._with_retry(_cancel)
            logger.info("RobinhoodBroker: cancelled order %s", order_id)
            return cancelled
        except BrokerError:
            raise
        except Exception as exc:
            raise BrokerError(f"Failed to cancel order {order_id}: {exc}") from exc

    # ------------------------------------------------------------------
    # Positions
    # ------------------------------------------------------------------

    def get_open_positions(self) -> list[Position]:
        self._require_auth()

        def _fetch():
            raw_positions = rh.account.get_open_stock_positions()
            if raw_positions is None:
                raise BrokerError("Robinhood returned None for open positions")

            positions: list[Position] = []
            for p in raw_positions:
                symbol = self._instrument_url_to_symbol(p.get("instrument", ""))
                if not symbol:
                    continue
                quantity = int(float(p.get("quantity") or 0))
                avg_cost = float(p.get("average_buy_price") or 0)

                # Get current price for this position
                try:
                    quote = self.get_quote(symbol)
                    current_price = quote.last
                except Exception:
                    current_price = avg_cost  # fallback to cost if quote fails

                positions.append(Position(
                    symbol=symbol,
                    quantity=quantity,
                    avg_cost=avg_cost,
                    current_price=current_price,
                    timestamp=datetime.utcnow(),
                ))
            return positions

        return self._with_retry(_fetch)

    # ------------------------------------------------------------------
    # Order status
    # ------------------------------------------------------------------

    def get_order_status(self, order_id: str) -> OrderResult:
        self._require_auth()

        def _fetch():
            raw = rh.orders.get_stock_order_info(order_id)
            if raw is None:
                raise BrokerError(f"No order found with id {order_id}")
            return self._parse_order_result(raw)

        return self._with_retry(_fetch)

    # ------------------------------------------------------------------
    # Market hours
    # ------------------------------------------------------------------

    def is_market_open(self) -> bool:
        """
        Checks Robinhood's market hours endpoint for today's trading hours.
        Falls back to a clock-only check if the API call fails.
        """
        try:
            market_data = rh.markets.get_market_hours("XNAS", str(datetime.now(ET).date()))
            if market_data and market_data.get("is_open"):
                opens_at = market_data.get("opens_at")
                closes_at = market_data.get("closes_at")
                if opens_at and closes_at:
                    now = datetime.now(ET)
                    # Parse ISO-8601 timestamps from Robinhood
                    open_dt = datetime.fromisoformat(opens_at.replace("Z", "+00:00")).astimezone(ET)
                    close_dt = datetime.fromisoformat(closes_at.replace("Z", "+00:00")).astimezone(ET)
                    return open_dt <= now < close_dt
            return False
        except Exception as exc:
            logger.warning(
                "RobinhoodBroker: market hours API failed (%s), using clock fallback", exc
            )
            return self._clock_market_open()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _clock_market_open() -> bool:
        now_et = datetime.now(ET)
        if now_et.weekday() >= 5:
            return False
        market_open = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
        market_close = now_et.replace(hour=16, minute=0, second=0, microsecond=0)
        return market_open <= now_et < market_close

    @staticmethod
    def _instrument_url_to_symbol(url: str) -> Optional[str]:
        """Resolve a Robinhood instrument URL to a ticker symbol."""
        if not url:
            return None
        try:
            data = rh.stocks.get_instrument_by_url(url)
            return data.get("symbol") if data else None
        except Exception:
            return None

    @staticmethod
    def _parse_order_result(raw: dict) -> OrderResult:
        """Convert a raw robin_stocks order dict to an OrderResult."""
        status_str = raw.get("state", "failed")
        status = _map_status(status_str)

        executions = raw.get("executions") or []
        filled_qty = int(float(raw.get("cumulative_quantity") or 0))

        avg_fill: Optional[float] = None
        if executions:
            # Weighted average of all partial fills
            total_shares = sum(float(e.get("quantity", 0)) for e in executions)
            if total_shares > 0:
                avg_fill = sum(
                    float(e.get("price", 0)) * float(e.get("quantity", 0))
                    for e in executions
                ) / total_shares
        elif raw.get("average_price"):
            avg_fill = float(raw["average_price"])

        side_str = raw.get("side", "buy")
        side = OrderSide.BUY if side_str == "buy" else OrderSide.SELL

        def _parse_dt(s: Optional[str]) -> datetime:
            if not s:
                return datetime.utcnow()
            try:
                return datetime.fromisoformat(s.replace("Z", "+00:00"))
            except Exception:
                return datetime.utcnow()

        return OrderResult(
            order_id=raw.get("id", str(uuid.uuid4())),
            symbol=raw.get("instrument_id", "UNKNOWN"),  # resolved upstream
            side=side,
            quantity=int(float(raw.get("quantity") or 0)),
            filled_quantity=filled_qty,
            avg_fill_price=avg_fill,
            status=status,
            created_at=_parse_dt(raw.get("created_at")),
            updated_at=_parse_dt(raw.get("updated_at")),
            raw=raw,
        )
