"""
analytics/dashboard.py — Rich terminal dashboard for live monitoring.

Renders a live-updating terminal UI showing:
  - System status (mode, uptime, halt state)
  - Open positions with unrealised P&L
  - Today's closed trades summary
  - Performance metrics (win rate, total P&L, profit factor)
  - Recent closed trades log (last 10)

Usage:
    dashboard = Dashboard(order_manager, risk_guard, trade_logger)
    dashboard.start()   # begins rendering in a background thread
    ...
    dashboard.stop()    # stops the live display

The dashboard runs in its own thread and never blocks the trading loop.
"""

from __future__ import annotations

import threading
import time
from datetime import datetime
from typing import Optional

import pytz
from rich.columns import Columns
from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from analytics.logger import TradeLogger
from analytics.performance import compute
from config.settings import settings
from execution.order_manager import ClosedTrade, OpenTrade, OrderManager
from risk.guard import RiskGuard

ET = pytz.timezone("America/New_York")


class Dashboard:
    """
    Terminal dashboard rendered via Rich Live.

    Starts a background daemon thread that refreshes the display every
    `settings.monitoring.dashboard_refresh_seconds` seconds.
    """

    def __init__(
        self,
        order_manager: OrderManager,
        risk_guard: RiskGuard,
        trade_logger: TradeLogger,
        started_at: Optional[datetime] = None,
    ) -> None:
        self._om = order_manager
        self._risk = risk_guard
        self._logger = trade_logger
        self._started_at = started_at or datetime.utcnow()

        self._console = Console()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Launch the dashboard render thread."""
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            name="dashboard",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        """Stop the dashboard."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=3.0)

    # ------------------------------------------------------------------
    # Render loop
    # ------------------------------------------------------------------

    def _run(self) -> None:
        refresh = settings.monitoring.dashboard_refresh_seconds
        with Live(
            self._build(),
            console=self._console,
            refresh_per_second=max(1, int(1 / refresh)),
            screen=False,
        ) as live:
            while not self._stop_event.is_set():
                live.update(self._build())
                self._stop_event.wait(timeout=refresh)

    # ------------------------------------------------------------------
    # Layout builder
    # ------------------------------------------------------------------

    def _build(self) -> Layout:
        layout = Layout()
        layout.split_column(
            Layout(name="header",  size=3),
            Layout(name="middle",  size=14),
            Layout(name="trades",  size=14),
            Layout(name="footer",  size=3),
        )
        layout["middle"].split_row(
            Layout(name="positions", ratio=3),
            Layout(name="metrics",   ratio=2),
        )

        layout["header"].update(self._header_panel())
        layout["positions"].update(self._positions_panel())
        layout["metrics"].update(self._metrics_panel())
        layout["trades"].update(self._closed_trades_panel())
        layout["footer"].update(self._footer_panel())

        return layout

    # ------------------------------------------------------------------
    # Panels
    # ------------------------------------------------------------------

    def _header_panel(self) -> Panel:
        now_et = datetime.now(ET).strftime("%Y-%m-%d %H:%M:%S ET")
        uptime = _fmt_duration(
            (datetime.utcnow() - self._started_at).total_seconds()
        )
        mode = settings.broker.trading_mode.upper()
        halt_text = (
            Text("  HALTED", style="bold red")
            if self._risk.is_halted
            else Text("  LIVE", style="bold green")
        )
        header = Text(f"Rosetta1  |  {mode}  |  {now_et}  |  Uptime: {uptime}")
        header.append_text(halt_text)
        return Panel(header, style="bold blue")

    def _positions_panel(self) -> Panel:
        open_trades: list[OpenTrade] = self._om.get_open_trades()

        table = Table(show_header=True, header_style="bold cyan", expand=True)
        table.add_column("Symbol",    width=6)
        table.add_column("Dir",       width=5)
        table.add_column("Shares",    justify="right", width=6)
        table.add_column("Entry",     justify="right", width=8)
        table.add_column("Target",    justify="right", width=8)
        table.add_column("Stop",      justify="right", width=8)
        table.add_column("Signal",    width=10)
        table.add_column("Conf",      justify="right", width=5)
        table.add_column("Age",       justify="right", width=7)

        if not open_trades:
            table.add_row("—", "", "", "", "", "", "", "", "")
        else:
            for t in open_trades:
                age = _fmt_duration(
                    (datetime.utcnow() - t.opened_at).total_seconds()
                )
                dir_style = "green" if t.direction == "long" else "red"
                table.add_row(
                    t.symbol,
                    Text(t.direction.upper(), style=dir_style),
                    str(t.shares),
                    f"{t.entry_price:.2f}",
                    f"{t.target_price:.2f}",
                    f"{t.stop_price:.2f}",
                    t.signal_type,
                    str(t.confidence),
                    age,
                )

        return Panel(table, title=f"[bold]Open Positions ({len(open_trades)})[/bold]")

    def _metrics_panel(self) -> Panel:
        today_trades = self._logger.get_today_trades()
        m = compute(today_trades)

        daily_pnl = self._risk.get_daily_pnl()
        pnl_style = "green" if daily_pnl >= 0 else "red"
        remaining = self._risk.daily_loss_remaining()

        table = Table(show_header=False, expand=True, box=None)
        table.add_column("Metric", style="bold", width=18)
        table.add_column("Value",  justify="right")

        def row(label: str, value: str, style: str = "") -> None:
            table.add_row(label, Text(value, style=style))

        row("Trades today",   str(m.num_trades))
        row("Win rate",       f"{m.win_rate*100:.1f}%",
            "green" if m.win_rate >= 0.5 else "yellow")
        row("Total P&L",      f"${m.total_pnl:+.2f}",
            "green" if m.total_pnl >= 0 else "red")
        row("Daily P&L",      f"${daily_pnl:+.2f}", pnl_style)
        row("Loss budget",    f"${remaining:.2f}",
            "green" if remaining > 50 else "yellow" if remaining > 0 else "red")
        row("Profit factor",  f"{m.profit_factor:.2f}" if m.profit_factor != float("inf") else "∞")
        row("Avg win",        f"${m.avg_win:.2f}", "green")
        row("Avg loss",       f"${m.avg_loss:.2f}", "red")

        return Panel(table, title="[bold]Today's Metrics[/bold]")

    def _closed_trades_panel(self) -> Panel:
        all_today = self._logger.get_today_trades()
        recent = all_today[-10:]  # last 10

        table = Table(show_header=True, header_style="bold cyan", expand=True)
        table.add_column("Symbol",  width=6)
        table.add_column("Dir",     width=5)
        table.add_column("Shares",  justify="right", width=6)
        table.add_column("Entry",   justify="right", width=8)
        table.add_column("Exit",    justify="right", width=8)
        table.add_column("P&L",     justify="right", width=9)
        table.add_column("Reason",  width=8)
        table.add_column("Hold",    justify="right", width=7)
        table.add_column("Closed",  width=8)

        if not recent:
            table.add_row("—", "", "", "", "", "", "", "", "")
        else:
            for t in reversed(recent):
                pnl_style = "green" if t.gross_pnl >= 0 else "red"
                dir_style  = "green" if t.direction == "long" else "red"
                table.add_row(
                    t.symbol,
                    Text(t.direction.upper(), style=dir_style),
                    str(t.shares),
                    f"{t.entry_price:.2f}",
                    f"{t.exit_price:.2f}",
                    Text(f"${t.gross_pnl:+.2f}", style=pnl_style),
                    t.exit_reason,
                    _fmt_duration(t.hold_seconds),
                    t.closed_at.strftime("%H:%M:%S"),
                )

        return Panel(
            table,
            title=f"[bold]Recent Closed Trades (today: {len(all_today)})[/bold]",
        )

    def _footer_panel(self) -> Panel:
        halt_msg = (
            f"  HALTED: {self._risk.halt_reason}" if self._risk.is_halted
            else "  System nominal"
        )
        style = "red" if self._risk.is_halted else "dim"
        return Panel(Text(halt_msg, style=style), style="dim")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt_duration(seconds: float) -> str:
    """Format seconds as Xm Ys or Xs."""
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    m, s = divmod(seconds, 60)
    if m < 60:
        return f"{m}m{s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m"
