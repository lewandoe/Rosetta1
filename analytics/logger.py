"""
analytics/logger.py — Durable SQLite trade log.

Every ClosedTrade emitted by OrderManager is written here.  The database is
the canonical record of all trading activity; in-memory state in OrderManager
is intentionally ephemeral.

Schema (table: trades):
  trade_id        TEXT PRIMARY KEY
  symbol          TEXT
  direction       TEXT
  shares          INTEGER
  entry_price     REAL
  exit_price      REAL
  gross_pnl       REAL
  signal_type     TEXT
  confidence      INTEGER
  entry_order_id  TEXT
  exit_order_id   TEXT
  opened_at       TEXT   (ISO-8601 UTC)
  closed_at       TEXT   (ISO-8601 UTC)
  exit_reason     TEXT
  hold_seconds    REAL

Thread safety:
  SQLite connections are not thread-safe by default.  TradeLogger creates one
  connection per instance and serialises all writes via a threading.Lock.
  Do NOT share a TradeLogger across processes.
"""

from __future__ import annotations

import logging
import sqlite3
import threading
from datetime import datetime, date
from pathlib import Path
from typing import List, Optional

from config.settings import settings
from execution.order_manager import ClosedTrade

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# TradeLogger
# ---------------------------------------------------------------------------

class TradeLogger:
    """
    Writes ClosedTrade records to a SQLite database.

    Usage:
        tl = TradeLogger()                    # uses settings.db_path
        tl = TradeLogger(path="trades.db")    # explicit path (useful in tests)

        om.on_trade_closed(tl.log_trade)      # wire into OrderManager

        today = tl.get_today_trades()
        all_  = tl.get_trades()
    """

    def __init__(self, path: Optional[str] = None) -> None:
        db_path = Path(path) if path else Path(settings.monitoring.db_path)
        db_path.parent.mkdir(parents=True, exist_ok=True)

        self._path = str(db_path)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self._path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._create_table()

        logger.info("TradeLogger initialised | db=%s", self._path)

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _create_table(self) -> None:
        with self._lock:
            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    trade_id        TEXT PRIMARY KEY,
                    symbol          TEXT    NOT NULL,
                    direction       TEXT    NOT NULL,
                    shares          INTEGER NOT NULL,
                    entry_price     REAL    NOT NULL,
                    exit_price      REAL    NOT NULL,
                    gross_pnl       REAL    NOT NULL,
                    signal_type     TEXT    NOT NULL,
                    confidence      INTEGER NOT NULL,
                    entry_order_id  TEXT    NOT NULL,
                    exit_order_id   TEXT    NOT NULL,
                    opened_at       TEXT    NOT NULL,
                    closed_at       TEXT    NOT NULL,
                    exit_reason     TEXT    NOT NULL,
                    hold_seconds    REAL    NOT NULL,
                    regime_at_entry TEXT,
                    atr_at_entry    REAL
                )
            """)
            # Backwards-compatible migration for pre-existing DBs missing these columns
            existing_cols = {row["name"] for row in self._conn.execute(
                "PRAGMA table_info(trades)"
            ).fetchall()}
            if "regime_at_entry" not in existing_cols:
                self._conn.execute("ALTER TABLE trades ADD COLUMN regime_at_entry TEXT")
            if "atr_at_entry" not in existing_cols:
                self._conn.execute("ALTER TABLE trades ADD COLUMN atr_at_entry REAL")
            self._conn.commit()

    # ------------------------------------------------------------------
    # Write
    # ------------------------------------------------------------------

    def log_trade(self, trade: ClosedTrade) -> None:
        """
        Persist a ClosedTrade.  Called by OrderManager via on_trade_closed().
        Thread-safe.  Silently ignores duplicate trade_ids (INSERT OR IGNORE).
        """
        meta = trade.metadata or {}
        regime_at_entry = meta.get("regime")
        atr_at_entry = meta.get("atr")
        if atr_at_entry is not None:
            try:
                atr_at_entry = float(atr_at_entry)
            except (TypeError, ValueError):
                atr_at_entry = None

        row = (
            trade.trade_id,
            trade.symbol,
            trade.direction,
            trade.shares,
            trade.entry_price,
            trade.exit_price,
            trade.gross_pnl,
            trade.signal_type,
            trade.confidence,
            trade.entry_order_id,
            trade.exit_order_id,
            _iso(trade.opened_at),
            _iso(trade.closed_at),
            trade.exit_reason,
            trade.hold_seconds,
            regime_at_entry,
            atr_at_entry,
        )
        with self._lock:
            self._conn.execute(
                """
                INSERT OR IGNORE INTO trades
                  (trade_id, symbol, direction, shares,
                   entry_price, exit_price, gross_pnl,
                   signal_type, confidence,
                   entry_order_id, exit_order_id,
                   opened_at, closed_at, exit_reason, hold_seconds,
                   regime_at_entry, atr_at_entry)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                row,
            )
            self._conn.commit()
        logger.debug(
            "TradeLogger: logged %s %s P&L=%.2f reason=%s",
            trade.symbol, trade.direction, trade.gross_pnl, trade.exit_reason,
        )

    # ------------------------------------------------------------------
    # Read
    # ------------------------------------------------------------------

    def get_trades(
        self,
        symbol: Optional[str] = None,
        date_: Optional[date] = None,
    ) -> List[ClosedTrade]:
        """
        Return all trades, optionally filtered by symbol and/or date.

        Args:
            symbol: Filter by ticker (e.g. "SPY").
            date_:  Filter by closed_at date (UTC).  Defaults to no filter.
        """
        clauses: list[str] = []
        params: list = []

        if symbol:
            clauses.append("symbol = ?")
            params.append(symbol)
        if date_:
            clauses.append("DATE(closed_at) = ?")
            params.append(date_.isoformat())

        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        sql = f"SELECT * FROM trades {where} ORDER BY closed_at ASC"

        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()

        return [_row_to_trade(r) for r in rows]

    def get_today_trades(self) -> List[ClosedTrade]:
        """Shorthand: trades closed today (UTC date)."""
        return self.get_trades(date_=datetime.utcnow().date())

    def trade_count(self) -> int:
        with self._lock:
            return self._conn.execute("SELECT COUNT(*) FROM trades").fetchone()[0]

    # ------------------------------------------------------------------
    # Maintenance
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the database connection."""
        with self._lock:
            self._conn.close()

    def clear(self) -> None:
        """Delete all rows.  Intended for tests only."""
        with self._lock:
            self._conn.execute("DELETE FROM trades")
            self._conn.commit()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _iso(dt: datetime) -> str:
    """Return ISO-8601 string for storage.  Strips tzinfo for consistency."""
    return dt.replace(tzinfo=None).isoformat()


def _row_to_trade(row: sqlite3.Row) -> ClosedTrade:
    cols = row.keys()
    meta: dict = {}
    if "regime_at_entry" in cols and row["regime_at_entry"] is not None:
        meta["regime"] = row["regime_at_entry"]
    if "atr_at_entry" in cols and row["atr_at_entry"] is not None:
        meta["atr"] = row["atr_at_entry"]
    return ClosedTrade(
        trade_id=row["trade_id"],
        symbol=row["symbol"],
        direction=row["direction"],
        shares=row["shares"],
        entry_price=row["entry_price"],
        exit_price=row["exit_price"],
        gross_pnl=row["gross_pnl"],
        signal_type=row["signal_type"],
        confidence=row["confidence"],
        entry_order_id=row["entry_order_id"],
        exit_order_id=row["exit_order_id"],
        opened_at=datetime.fromisoformat(row["opened_at"]),
        closed_at=datetime.fromisoformat(row["closed_at"]),
        exit_reason=row["exit_reason"],
        metadata=meta,
    )
