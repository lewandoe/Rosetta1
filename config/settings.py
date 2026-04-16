"""
config/settings.py — Single source of truth for every tunable parameter.

All modules import from here. No magic numbers anywhere else in the codebase.
"""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


# ---------------------------------------------------------------------------
# Target universe — built dynamically at startup via data/universe.py
# Core symbols are always included; top 15 volume stocks are added daily.
# Import: from data.universe import get_universe
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Sub-setting groups
# ---------------------------------------------------------------------------

class BrokerSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="ROBINHOOD_", env_file=".env", extra="ignore")

    username: str = Field(default="", description="Robinhood account email")
    password: str = Field(default="", description="Robinhood account password")
    mfa_secret: str = Field(default="", description="TOTP secret for 2-FA (base-32)")

    # "paper" is the safe default — must be explicitly changed to "live"
    trading_mode: str = Field(default="paper", description="'paper' or 'live'")


class RiskSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Hard dollar ceiling on daily losses.  Trading halts when hit.
    max_daily_loss: float = Field(default=10000.0, description="Max daily loss in USD before halt")

    # Capital allocation per trade as fraction of buying power (0–1)
    max_capital_per_trade_pct: float = Field(default=0.05, description="Max fraction of buying power per trade")

    # Maximum number of simultaneously open positions
    max_open_positions: int = Field(default=20, description="Max concurrent open positions")

    # Pattern-Day-Trader: max day trades per rolling 5-business-day window
    max_day_trades: int = Field(default=9999, description="PDT limit — day trades per rolling 5-day window")

    # EOD forced liquidation time (Eastern, 24-h)
    eod_liquidation_hour: int = Field(default=15, description="Force-close hour (ET)")
    eod_liquidation_minute: int = Field(default=59, description="Force-close minute (ET) — close all positions")
    # Last time to open a new position (must be before liquidation)
    eod_no_new_entries_hour: int = Field(default=15, description="No new entries hour (ET)")
    eod_no_new_entries_minute: int = Field(default=55, description="No new entries minute (ET) — no new positions after this")

    # Slippage guard: cancel/flag if actual fill deviates more than this fraction
    max_slippage_pct: float = Field(default=0.0015, description="Max acceptable fill slippage (0.15%)")

    # Spread filter: skip trade if (ask - bid) / mid > threshold
    max_spread_pct: float = Field(default=0.0020, description="Max bid-ask spread allowed (0.20%)")

    # Stop-loss must be placed within this fraction of entry price
    # (used as a sanity check — real stop price comes from signal)
    max_stop_distance_pct: float = Field(default=0.02, description="Max stop distance from entry (2%)")

    # Max loss per trade must not exceed this multiple of the target gain
    max_loss_to_gain_ratio: float = Field(default=2.0, description="Max risk/reward ratio (loss ≤ 2× gain)")

    # ATR-normalized position sizing
    target_risk_per_trade_pct: float = Field(
        default=0.005,
        ge=0.001,
        le=0.02,
        description="Target dollar risk per trade as fraction of account value. "
                    "0.005 = risk 0.5% per trade. On $35k account = $175 risk per trade. "
                    "Stop distance determines share count.",
    )
    min_shares: int = Field(
        default=1,
        description="Minimum shares per trade regardless of sizing.",
    )
    max_shares: int = Field(
        default=500,
        description="Hard ceiling on shares per trade. "
                    "Prevents runaway sizing on very low ATR symbols.",
    )

    # Consecutive loss circuit breaker
    consecutive_loss_pause_threshold: int = Field(
        default=3,
        description="Pause trading for consecutive_loss_pause_minutes after this many consecutive losses.",
    )
    consecutive_loss_pause_minutes: int = Field(
        default=10,
        description="Minutes to pause after hitting consecutive_loss_pause_threshold.",
    )
    consecutive_loss_halt_threshold: int = Field(
        default=5,
        description="Halt trading for the session after this many consecutive losses. "
                    "Reset at start of next trading day.",
    )


class SignalSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Only fire signals above this confidence threshold
    min_confidence_score: int = Field(default=55, description="Minimum signal confidence (0–100)")
    macro_bias_enabled: bool = Field(
        default=True,
        description="Only take longs when SPY is above VWAP and trending up. "
                    "Only take shorts when SPY is below VWAP and trending down. "
                    "Most impactful filter for avoiding trades against the market."
    )
    macro_bias_symbol: str = Field(
        default="SPY",
        description="Symbol to use as macro bias indicator."
    )
    macro_bias_bars: int = Field(
        default=10,
        description="Number of bars to measure SPY trend direction."
    )
    disabled_signals: list[str] = Field(
        default=["momentum"],
        description="Signal types to skip. Valid: momentum, vwap_cross, ema_cross, orb, rsi"
    )

    # EMA periods used by ema_cross strategy
    ema_fast: int = Field(default=8)
    ema_mid: int = Field(default=13)
    ema_slow: int = Field(default=21)

    # RSI parameters
    rsi_period: int = Field(default=14)
    rsi_oversold: int = Field(default=30)
    rsi_overbought: int = Field(default=70)

    # ATR period (used for dynamic stop sizing)
    atr_period: int = Field(default=7)

    # Volume MA period (used for volume confirmation)
    volume_ma_period: int = Field(default=20)

    # Opening Range Breakout: minutes after open that define the range
    orb_minutes: int = Field(default=15, description="ORB range window in minutes after open")

    # ATR-based stop sizing — global fallback (used only if per-signal override is missing)
    stop_atr_multiplier: float = Field(
        default=1.0,
        description="Fallback ATR multiplier for stop distance.",
    )
    reward_risk_ratio: float = Field(
        default=1.5,
        description="Fallback target as multiple of stop distance.",
    )

    # Per-signal stop multipliers and reward/risk ratios
    # Trend signals (momentum, ema_cross, orb) run wider stops and larger targets.
    # Mean-reversion signals (vwap_cross, rsi_reversal) bank gains quickly.
    momentum_stop_atr: float = Field(default=2.5, description="Momentum stop: 2.5× ATR")
    momentum_reward_ratio: float = Field(default=1.3, description="Momentum R:R 1.3")
    ema_cross_stop_atr: float = Field(default=2.0, description="EMA cross stop: 2.0× ATR")
    ema_cross_reward_ratio: float = Field(default=1.2, description="EMA cross R:R 1.2")
    vwap_cross_stop_atr: float = Field(default=1.5, description="VWAP cross stop: 1.5× ATR")
    vwap_cross_reward_ratio: float = Field(default=1.0, description="VWAP cross R:R 1.0")
    rsi_reversal_stop_atr: float = Field(default=1.5, description="RSI reversal stop: 1.5× ATR")
    rsi_reversal_reward_ratio: float = Field(default=1.0, description="RSI reversal R:R 1.0")
    orb_stop_atr: float = Field(default=2.0, description="ORB stop: 2.0× ATR")
    orb_reward_ratio: float = Field(default=2.0, description="ORB R:R 2.0")

    # Trailing stop breakeven activation (in R multiples).
    # Mean-reversion signals move to breakeven at 0.5R; trend signals at 1.0R.
    momentum_breakeven_r: float = Field(default=0.75)
    ema_cross_breakeven_r: float = Field(default=0.75)
    vwap_cross_breakeven_r: float = Field(default=0.5)
    rsi_reversal_breakeven_r: float = Field(default=0.5)
    orb_breakeven_r: float = Field(default=0.75)

    # Cross-symbol sector confirmation
    sector_confirmation_enabled: bool = Field(
        default=True,
        description="Adjust confidence based on sector ETF alignment.",
    )
    sector_confirmation_bonus: int = Field(
        default=8,
        description="Confidence points added when sector ETF agrees with signal direction.",
    )
    sector_confirmation_penalty: int = Field(
        default=12,
        description="Confidence points deducted when sector ETF disagrees with signal direction. "
                    "Higher than bonus because fighting the sector is riskier than missing a move.",
    )
    sector_trend_bars: int = Field(
        default=5,
        description="Number of recent bars to measure ETF trend direction.",
    )

    # Volume gate — bar must exceed this multiple of volume_ma to be tradeable
    volume_gate_multiplier: float = Field(
        default=1.5,
        description="Minimum volume required as multiple of volume_ma. "
                    "Filters out low-conviction bars during dead zones.",
    )

    # Multi-timeframe EMA alignment
    mtf_enabled: bool = Field(
        default=True,
        description="Require 5-min EMA trend to align with signal direction before entry.",
    )
    mtf_ema_period: int = Field(
        default=21,
        description="EMA period used for multi-timeframe trend check on the 5-min chart.",
    )

    # Market regime filter
    regime_trending_threshold: float = Field(
        default=0.30,
        description="Regime score above this = trending market. "
                    "Momentum and EMA cross signals preferred.",
    )
    regime_ranging_threshold: float = Field(
        default=0.15,
        description="Regime score below this = ranging market. "
                    "VWAP reversion signals preferred. "
                    "Momentum signals suppressed.",
    )
    regime_lookback: int = Field(
        default=20,
        description="Number of bars for regime classification.",
    )


class FeedSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Maximum concurrent quote fetch threads
    max_workers: int = Field(default=8, description="ThreadPoolExecutor size for quote fetching")

    # Full-universe scan cadence
    poll_interval_seconds: float = Field(
        default=10.0,
        description="Seconds between full-symbol-universe quote scans",
    )

    # Per-request courtesy delay (helps stay under Robinhood's rate limit)
    request_delay_seconds: float = Field(
        default=0.2,
        description="Seconds each worker sleeps before issuing a quote request. "
                    "With 8 workers and 0.2s delay: ~40 req/s ceiling, well under "
                    "Robinhood's ~100 req/min limit for a 25-symbol universe.",
    )


class ExecutionSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # How often (seconds) the position monitor loop ticks
    position_monitor_interval_seconds: int = Field(default=5)

    # How often (seconds) the signal scanner polls each symbol
    signal_scan_interval_seconds: int = Field(default=10)

    # Default order time-in-force for intraday entries
    default_time_in_force: str = Field(default="gfd", description="'gfd' = good-for-day")

    # Maximum retries for transient broker API errors before halting
    max_order_retries: int = Field(default=3)

    # Maximum hold time per signal type before a time-based exit fires.
    # Trades that haven't moved have decaying edge — exit and redeploy capital.
    momentum_max_hold_seconds: int = Field(default=480,  description="8 min")
    ema_cross_max_hold_seconds: int = Field(default=720,  description="12 min")
    vwap_cross_max_hold_seconds: int = Field(default=360,  description="6 min")
    rsi_reversal_max_hold_seconds: int = Field(default=480,  description="8 min")
    orb_max_hold_seconds: int = Field(default=1500, description="25 min")

    # Limit order entry — caps adverse fill price vs. accepting market slippage
    use_limit_orders: bool = Field(
        default=True,
        description="Use limit orders for entries instead of market orders. "
                    "Caps the worst-case fill price.",
    )
    limit_order_offset_pct: float = Field(
        default=0.0005,
        description="How far past the touch to set the limit price (0.0005 = 5 bps). "
                    "BUY: ask × (1 + offset). SELL: bid × (1 - offset). "
                    "Higher = more likely to fill, less price protection.",
    )
    limit_order_timeout_seconds: int = Field(
        default=5,
        description="Cancel an unfilled limit entry after this many seconds. "
                    "Stale signals lose edge fast — better to skip than chase.",
    )


class BacktestSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Minimum historical days required before approving a strategy for live
    min_history_days: int = Field(default=60)

    # Simulated round-trip slippage per trade
    simulated_slippage_pct: float = Field(default=0.0007, description="0.07% slippage model")

    # Approval thresholds — strategy must beat both to go live
    min_win_rate: float = Field(default=0.55, description="Minimum win rate for live approval")
    min_sharpe_ratio: float = Field(default=1.2, description="Minimum Sharpe ratio for live approval")


class MonitoringSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    log_level: str = Field(default="INFO")

    # Dashboard refresh rate (seconds)
    dashboard_refresh_seconds: float = Field(default=1.0)

    # SQLite database path for trade log
    db_path: str = Field(default="/Users/eric/Rosetta1/db/trades.db")


class TradingSessionSettings(BaseSettings):
    """Controls which parts of the trading day are active."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Market open: 9:30 ET
    market_open_hour: int = Field(
        default=9,
        description="Hour (ET, 24h) of regular market open.",
    )
    market_open_minute: int = Field(
        default=30,
        description="Minute (ET) of regular market open.",
    )

    # Market close: 15:59 ET (last actionable minute — aligns with EOD liquidation)
    market_close_hour: int = Field(
        default=15,
        description="Hour (ET, 24h) of regular market close.",
    )
    market_close_minute: int = Field(
        default=59,
        description="Minute (ET) of regular market close.",
    )

    # EOD wind-down window — close-only zone before market_close
    eod_close_minutes_before: int = Field(
        default=14,
        description="Minutes before market_close when wind-down starts. "
                    "With close 15:59 and offset 14 → wind-down starts 15:45 ET.",
    )

    # Power open window: 9:30–11:30 ET (highest volume, cleanest moves)
    power_open_end_hour: int = Field(
        default=11,
        description="Hour (ET, 24h) when power open window ends.",
    )
    power_open_end_minute: int = Field(
        default=30,
        description="Minute (ET) when power open window ends.",
    )

    # Lunch dead zone: 11:30–13:00 ET (low volume, choppy, avoid)
    lunch_end_hour: int = Field(
        default=13,
        description="Hour (ET, 24h) when lunch dead zone ends and afternoon trading resumes.",
    )
    lunch_end_minute: int = Field(
        default=0,
        description="Minute (ET) when lunch dead zone ends.",
    )

    # Set to True to allow entries during lunch (not recommended)
    lunch_trading_enabled: bool = Field(
        default=False,
        description="Allow new entries during lunch dead zone (11:30–13:00 ET). "
                    "Default False — lunch is low-volume and historically unprofitable.",
    )


# ---------------------------------------------------------------------------
# Root settings object — import this everywhere
# ---------------------------------------------------------------------------

class Settings(BaseSettings):
    """Aggregate settings loaded once at startup."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    paper_starting_capital: float = Field(
        default=100000.0, description="Paper trading starting cash"
    )

    broker: BrokerSettings = Field(default_factory=BrokerSettings)
    risk: RiskSettings = Field(default_factory=RiskSettings)
    signals: SignalSettings = Field(default_factory=SignalSettings)
    execution: ExecutionSettings = Field(default_factory=ExecutionSettings)
    feed: FeedSettings = Field(default_factory=FeedSettings)
    backtest: BacktestSettings = Field(default_factory=BacktestSettings)
    monitoring: MonitoringSettings = Field(default_factory=MonitoringSettings)
    session: TradingSessionSettings = Field(default_factory=TradingSessionSettings)


# Module-level singleton — import `settings` directly in all other modules
settings = Settings()
