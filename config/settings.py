"""
config/settings.py — Single source of truth for every tunable parameter.

All modules import from here. No magic numbers anywhere else in the codebase.
"""

from __future__ import annotations

from typing import List
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


# ---------------------------------------------------------------------------
# Target universe — trade ONLY these symbols
# ---------------------------------------------------------------------------
UNIVERSE: List[str] = ["SPY", "TSLA", "NVDA", "AAPL", "MSFT", "GOOGL", "AMZN", "META", "AMD", "QQQ"]


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
    max_daily_loss: float = Field(default=200.0, description="Max daily loss in USD before halt")

    # Capital allocation per trade as fraction of buying power (0–1)
    max_capital_per_trade_pct: float = Field(default=0.10, description="Max fraction of buying power per trade")

    # Maximum number of simultaneously open positions
    max_open_positions: int = Field(default=5, description="Max concurrent open positions")

    # Pattern-Day-Trader: max day trades per rolling 5-business-day window
    max_day_trades: int = Field(default=500, description="PDT limit — day trades per rolling 5-day window")

    # EOD forced liquidation time (Eastern, 24-h)
    eod_liquidation_hour: int = Field(default=15, description="Force-close hour (ET)")
    eod_liquidation_minute: int = Field(default=45, description="Force-close minute (ET)")

    # Slippage guard: cancel/flag if actual fill deviates more than this fraction
    max_slippage_pct: float = Field(default=0.0015, description="Max acceptable fill slippage (0.15%)")

    # Spread filter: skip trade if (ask - bid) / mid > threshold
    max_spread_pct: float = Field(default=0.0020, description="Max bid-ask spread allowed (0.20%)")

    # Stop-loss must be placed within this fraction of entry price
    # (used as a sanity check — real stop price comes from signal)
    max_stop_distance_pct: float = Field(default=0.02, description="Max stop distance from entry (2%)")

    # Max loss per trade must not exceed this multiple of the target gain
    max_loss_to_gain_ratio: float = Field(default=2.0, description="Max risk/reward ratio (loss ≤ 2× gain)")


class SignalSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # Only fire signals above this confidence threshold
    min_confidence_score: int = Field(default=70, description="Minimum signal confidence (0–100)")

    # EMA periods used by ema_cross strategy
    ema_fast: int = Field(default=8)
    ema_mid: int = Field(default=13)
    ema_slow: int = Field(default=21)

    # RSI parameters
    rsi_period: int = Field(default=14)
    rsi_oversold: int = Field(default=30)
    rsi_overbought: int = Field(default=70)

    # ATR period (used for dynamic stop sizing)
    atr_period: int = Field(default=14)

    # Volume MA period (used for volume confirmation)
    volume_ma_period: int = Field(default=20)

    # Opening Range Breakout: minutes after open that define the range
    orb_minutes: int = Field(default=15, description="ORB range window in minutes after open")

    # ATR-based stop sizing
    stop_atr_multiplier: float = Field(
        default=3.0,
        description="ATR multiplier for stop distance. "
                    "3.0 = stop placed 3× ATR from entry. "
                    "Higher = more room to breathe, larger losses when stopped.",
    )
    reward_risk_ratio: float = Field(
        default=1.5,
        description="Target distance as multiple of stop distance. "
                    "1.5 = target is 1.5× further than stop. "
                    "Higher = larger wins but lower win rate.",
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


# ---------------------------------------------------------------------------
# Root settings object — import this everywhere
# ---------------------------------------------------------------------------

class Settings(BaseSettings):
    """Aggregate settings loaded once at startup."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    paper_starting_capital: float = Field(
        default=35000.0, description="Paper trading starting cash"
    )

    broker: BrokerSettings = Field(default_factory=BrokerSettings)
    risk: RiskSettings = Field(default_factory=RiskSettings)
    signals: SignalSettings = Field(default_factory=SignalSettings)
    execution: ExecutionSettings = Field(default_factory=ExecutionSettings)
    backtest: BacktestSettings = Field(default_factory=BacktestSettings)
    monitoring: MonitoringSettings = Field(default_factory=MonitoringSettings)


# Module-level singleton — import `settings` directly in all other modules
settings = Settings()
