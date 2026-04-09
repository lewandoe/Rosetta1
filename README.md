# Rosetta1 — Autonomous Intraday Scalping System

Rosetta1 is a fully automated intraday scalping system targeting 10 liquid equities via Robinhood.  
Default mode is **paper trading** — live orders require explicit configuration.

---

## Target Universe
`SPY, TSLA, NVDA, AAPL, MSFT, GOOGL, AMZN, META, AMD, QQQ`

---

## Architecture

```
config/         — pydantic-settings (single source of truth for all parameters)
broker/         — BrokerInterface ABC, Robinhood impl, Paper impl
data/           — Real-time feed, historical OHLCV, indicator library
signals/        — Momentum, VWAP, EMA-cross, ORB, RSI strategies
strategy/       — Signal aggregator + confidence engine
risk/           — Pre-order guard: loss limits, PDT, spread, slippage
execution/      — Order manager, position monitor, EOD liquidation
analytics/      — SQLite trade log, performance metrics, Rich dashboard
tests/          — Mirrors every module above
main.py         — Event loop, wires everything together
```

### Dependency Graph (enforced — no exceptions)
```
config      → nothing
data        → config
signals     → config, data
strategy    → config, data, signals
risk        → config, data
execution   → config, broker, risk, strategy
analytics   → config, execution
main        → everything
```

---

## Risk Rules (hard-coded, never bypassed)
| Rule | Value |
|---|---|
| Stop-loss | Every trade, placed immediately after entry |
| Max loss per trade | ≤ 2× target gain |
| Max open positions | 3 |
| Max daily loss | $200 (halt all trading if hit) |
| Max capital per trade | 10% of buying power |
| PDT limit | 3 day trades per rolling 5 business days |
| EOD force-close | 3:45 PM ET |
| Slippage guard | Cancel if fill deviates > 0.15% from signal price |
| Spread filter | Skip if bid-ask > 0.20% of price |

---

## Development Sequence

| Stage | Description | Status |
|---|---|---|
| 1 | Foundation — config, requirements, folder structure | ✅ |
| 2 | Broker Layer — BrokerInterface, Robinhood, Paper | ✅ |
| 3 | Data Feed — real-time polling + historical OHLCV | 🔜 |
| 4 | Indicator Library — VWAP, EMA, RSI, ATR, volume MA | 🔜 |
| 5 | Signal Engine — 5 strategies + confidence aggregator | 🔜 |
| 6 | Risk Guard — pre-order checks | 🔜 |
| 7 | Order Manager — entry/exit/stop, position loop | 🔜 |
| 8 | Trade Logger & Analytics | 🔜 |
| 9 | Main Orchestrator + Rich Dashboard | 🔜 |
| 10 | Backtesting — 60-day sim, Sharpe > 1.2, win rate > 55% | 🔜 |

---

## Quick Start

```bash
# 1. Clone and install
git clone https://github.com/lewandoe/Rosetta1.git
cd Rosetta1
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Configure
cp .env.example .env
# Edit .env with your Robinhood credentials
# Leave TRADING_MODE=paper until fully tested

# 3. Run tests
pytest tests/ -v

# 4. Start (paper mode)
python main.py
```

---

## Branch Strategy
- `main` — stable releases only
- `dev` — integration branch
- `feature/[stage-name]` — one branch per stage
