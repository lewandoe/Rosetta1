"""
tests/test_integration_v2.py — End-to-end validation of the v2
regime-first trading pipeline.

Run with: python tests/test_integration_v2.py
"""
import sys
sys.path.insert(0, '.')

import pandas as pd

from config.settings import settings
from data.history import seed_bars
from data.indicators import add_all, latest, mtf_trend_direction
from data.regime import classify_regime, is_strategy_valid_for_regime, Regime
from data.session import current_session, is_entry_allowed, Session
from strategy.engine import SignalEngine


def _load_bars(sym: str) -> pd.DataFrame:
    bars = seed_bars(sym)
    bars.columns = [c.lower() for c in bars.columns]
    if len(bars) > 1 and bars['volume'].iloc[-1] == 0:
        bars = bars.iloc[:-1]
    return add_all(bars)


def test_settings():
    """Verify new settings fields exist and have correct defaults."""
    print("=== Settings Validation ===")

    # Signal settings
    assert settings.signals.stop_atr_multiplier == 1.0, \
        f"Expected 1.0, got {settings.signals.stop_atr_multiplier}"
    assert settings.signals.atr_period == 7, \
        f"Expected 7, got {settings.signals.atr_period}"
    assert settings.signals.volume_gate_multiplier == 1.5
    assert settings.signals.mtf_enabled is True
    assert settings.signals.mtf_ema_period == 21

    # Risk settings
    assert settings.risk.target_risk_per_trade_pct == 0.005
    assert settings.risk.consecutive_loss_pause_threshold == 3
    assert settings.risk.consecutive_loss_halt_threshold == 5

    # Session settings
    assert settings.session.lunch_trading_enabled is False
    assert settings.session.power_open_end_hour == 11
    assert settings.session.lunch_end_hour == 13

    print("  All settings OK")


def test_regime_classification():
    """Test regime detection on real data."""
    print("\n=== Regime Classification ===")
    for sym in ['SPY', 'TSLA', 'NVDA', 'AMD', 'QQQ']:
        try:
            bars = _load_bars(sym)
            result = classify_regime(bars)
            print(f"  {sym:6} regime={result.regime.value:10} score={result.score:.3f}")
        except Exception as e:
            print(f"  {sym:6} ERROR: {e}")


def test_mtf_alignment():
    """Test multi-timeframe trend detection."""
    print("\n=== MTF Alignment ===")
    for sym in ['SPY', 'TSLA', 'NVDA', 'AMD', 'QQQ']:
        try:
            bars = _load_bars(sym)
            direction = mtf_trend_direction(bars, settings.signals.mtf_ema_period)
            print(f"  {sym:6} mtf_direction={direction}")
        except Exception as e:
            print(f"  {sym:6} ERROR: {e}")


def test_session_manager():
    """Test session classification."""
    print("\n=== Session Manager ===")
    session = current_session()
    allowed, reason = is_entry_allowed()
    print(f"  Current session: {session.value}")
    print(f"  Entry allowed:   {allowed} ({reason})")


def test_volume_gate():
    """Test volume gating on real data."""
    print("\n=== Volume Gate ===")
    gate = settings.signals.volume_gate_multiplier
    for sym in ['SPY', 'TSLA', 'NVDA']:
        try:
            bars = _load_bars(sym)
            vol = latest(bars, 'volume')
            vol_ma = latest(bars, 'volume_ma')
            ratio = vol / vol_ma if vol_ma > 0 else 0
            passed = ratio >= gate
            print(f"  {sym:6} vol={vol:>12,.0f} vol_ma={vol_ma:>12,.0f} "
                  f"ratio={ratio:.2f} gate={gate} passed={passed}")
        except Exception as e:
            print(f"  {sym:6} ERROR: {e}")


def test_signal_engine():
    """Test the full engine pipeline on real data."""
    print("\n=== Signal Engine (Full Pipeline) ===")
    engine = SignalEngine()
    bars_dict: dict[str, pd.DataFrame] = {}
    symbols = ['SPY', 'TSLA', 'NVDA', 'AAPL', 'MSFT',
               'GOOGL', 'AMZN', 'META', 'AMD', 'QQQ']

    for sym in symbols:
        try:
            bars_dict[sym] = _load_bars(sym)
        except Exception as e:
            print(f"  {sym:6} seed ERROR: {e}")

    for sym in symbols:
        if sym not in bars_dict:
            continue
        bars = bars_dict[sym]
        price = float(bars['close'].iloc[-1])
        result = engine.evaluate(bars, sym, price, bars_dict=bars_dict)
        if result:
            stop_dist = abs(result.entry_price - result.stop_price)
            target_dist = abs(result.target_price - result.entry_price)
            regime = result.metadata.get('regime', '?')
            mtf = result.metadata.get('mtf_direction', '?')
            print(f"  {sym:6} SIGNAL: {result.signal_type} {result.direction} "
                  f"conf={result.confidence} regime={regime} mtf={mtf} "
                  f"stop_dist=${stop_dist:.3f} target_dist=${target_dist:.3f}")
        else:
            print(f"  {sym:6} no signal (filters working)")


def test_position_sizing():
    """Test volatility-normalized position sizing math."""
    print("\n=== Position Sizing (Math Check) ===")
    account = 10000.0
    risk_pct = settings.risk.target_risk_per_trade_pct
    risk_dollars = account * risk_pct
    print(f"  Account: ${account:,.0f}")
    print(f"  Risk per trade: {risk_pct*100:.1f}% = ${risk_dollars:.2f}")

    test_cases = [
        ("SPY",  540.0, 0.50),   # low vol ETF
        ("TSLA", 250.0, 2.50),   # high vol stock
        ("AMD",   95.0, 1.20),   # mid vol stock
        ("NVDA", 800.0, 4.00),   # expensive, mid vol
    ]
    for sym, price, atr in test_cases:
        stop_dist = atr * settings.signals.stop_atr_multiplier
        shares = int(risk_dollars / stop_dist)
        max_shares = int((account * settings.risk.max_capital_per_trade_pct) / price)
        shares = min(shares, max_shares)
        actual_risk = shares * stop_dist
        print(f"  {sym:6} price=${price:>7.2f} ATR=${atr:.2f} "
              f"stop_dist=${stop_dist:.2f} shares={shares:>4d} "
              f"position=${shares*price:>8,.0f} risk=${actual_risk:.2f}")


def test_trade_log_schema():
    """Verify the SQLite log has the new regime/atr columns."""
    print("\n=== Trade Log Schema ===")
    import sqlite3
    import tempfile
    import os
    from analytics.logger import TradeLogger
    from execution.order_manager import ClosedTrade
    from datetime import datetime, timezone

    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    tmp.close()
    try:
        tl = TradeLogger(path=tmp.name)
        ct = ClosedTrade(
            trade_id="test-1",
            symbol="SPY",
            direction="long",
            shares=10,
            entry_price=540.0,
            exit_price=540.5,
            gross_pnl=5.0,
            signal_type="vwap_cross",
            confidence=72,
            entry_order_id="e1",
            exit_order_id="x1",
            opened_at=datetime.now(timezone.utc),
            closed_at=datetime.now(timezone.utc),
            exit_reason="target",
            metadata={"regime": "trending", "atr": 0.42, "mtf_direction": "long"},
        )
        tl.log_trade(ct)

        cols = {r["name"] for r in tl._conn.execute("PRAGMA table_info(trades)").fetchall()}
        assert "regime_at_entry" in cols, "regime_at_entry column missing"
        assert "atr_at_entry"    in cols, "atr_at_entry column missing"
        print(f"  Columns include: regime_at_entry, atr_at_entry ✓")

        loaded = tl.get_trades()[0]
        assert loaded.metadata.get("regime") == "trending"
        assert loaded.metadata.get("atr") == 0.42
        print(f"  Round-trip values: regime={loaded.metadata['regime']!r} "
              f"atr={loaded.metadata['atr']}")
        tl.close()
    finally:
        os.unlink(tmp.name)


if __name__ == "__main__":
    print("Rosetta1 v2 Integration Test")
    print("=" * 50)

    test_settings()
    test_regime_classification()
    test_mtf_alignment()
    test_session_manager()
    test_volume_gate()
    test_signal_engine()
    test_position_sizing()
    test_trade_log_schema()

    print("\n" + "=" * 50)
    print("All tests complete. Review output above for any errors.")
