"""
examples/live_dry_run.py
=========================
Run the iron condor live runner for one market session.

Defaults to dry-run mode (``--dry-run``).  Pass ``--no-dry-run`` to submit
real orders to the Alpaca **paper** account — you will be asked to confirm
a second time before the loop starts.

In dry-run mode
---------------
- No orders are submitted.
- Every entry/exit decision is logged with timestamps.
- Position monitoring uses simulated theta decay.
- Press Ctrl+C to exit cleanly.

Logging
-------
Console: INFO level, human-readable format.
File:    logs/dry_run_YYYYMMDD.log  (same format, RotatingFileHandler —
         rotates at 10 MB, keeps 30 backups).

Run with::

    uv run python examples/live_dry_run.py           # dry-run (safe)
    uv run python examples/live_dry_run.py --no-dry-run  # paper orders
"""

from __future__ import annotations

import logging
import os
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path

import typer
from rich import box
from rich import print as rprint
from rich.panel import Panel

from alpaca_options.live.runner import LiveRunner
from alpaca_options.risk.manager import RiskManager
from alpaca_options.strategies.iron_condor_0dte import IronCondor0DTE, IronCondorConfig


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def _setup_logging(log_dir: Path = Path("logs")) -> None:
    """Configure root logger: console (INFO) + rotating file (INFO).

    File name: ``{log_dir}/dry_run_YYYYMMDD.log``
    Rotation:  10 MB per file, 30 backups retained.
    """
    log_dir.mkdir(parents=True, exist_ok=True)

    date_str  = datetime.now().strftime("%Y%m%d")
    log_path  = log_dir / f"dry_run_{date_str}.log"
    fmt       = "%(asctime)s  %(levelname)-8s  %(name)s: %(message)s"
    datefmt   = "%H:%M:%S"
    formatter = logging.Formatter(fmt, datefmt=datefmt)

    root = logging.getLogger()
    root.setLevel(logging.INFO)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    root.addHandler(console_handler)

    # Rotating file handler — 10 MB × 30 backups
    file_handler = RotatingFileHandler(
        log_path,
        maxBytes=10 * 1024 * 1024,   # 10 MB
        backupCount=30,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    logging.getLogger(__name__).info(
        "Logging initialised — file: %s", log_path.resolve()
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

app = typer.Typer(add_completion=False)


@app.command()
def main(
    dry_run: bool = typer.Option(
        True,
        "--dry-run/--no-dry-run",
        help=(
            "DRY-RUN (default): decisions are logged, no orders submitted. "
            "NO-DRY-RUN: orders go to your Alpaca paper account."
        ),
    ),
) -> None:
    _setup_logging()

    # ── Banner ─────────────────────────────────────────────────────────────
    mode_label = "DRY-RUN (no orders)" if dry_run else "⚠  PAPER LIVE (orders WILL be submitted)"
    rprint(Panel.fit(
        f"[bold cyan]🦙 Iron Condor 0DTE — {mode_label}[/bold cyan]",
        box=box.DOUBLE,
    ))

    if dry_run:
        rprint(Panel(
            "[bold yellow]DRY-RUN MODE[/bold yellow]\n\n"
            "• No orders will be submitted to Alpaca.\n"
            "• Entry/exit decisions are logged every 60 s (no position) / 30 s (open position).\n"
            "• Position value is simulated via theta decay (0.5 %/min).\n"
            "• Log file: logs/dry_run_YYYYMMDD.log  (10 MB rotate, 30 backups).\n"
            "• Press [bold]Ctrl+C[/bold] to stop — any simulated position is logged on exit.\n\n"
            "Strategy gates: entry window 10:00–14:00 ET  │  VIX 16–25  │  min_credit=8 %",
            border_style="yellow",
        ))
    else:
        rprint(Panel(
            "[bold red]⚠  PAPER LIVE MODE[/bold red]\n\n"
            "• Orders WILL be submitted to your Alpaca [bold]paper[/bold] account.\n"
            "• Entry/exit decisions are logged every 60 s / 30 s.\n"
            "• Log file: logs/dry_run_YYYYMMDD.log  (10 MB rotate, 30 backups).\n"
            "• Press [bold]Ctrl+C[/bold] to stop — any open position is closed before exit.\n\n"
            "Strategy gates: entry window 10:00–14:00 ET  │  VIX 16–25  │  min_credit=8 %",
            border_style="red",
        ))

    # ── Primary confirmation ────────────────────────────────────────────────
    confirm = typer.confirm(
        "\nStart loop?" if dry_run else "\n[DRY-RUN=False] Start loop?",
        default=False,
    )
    if not confirm:
        rprint("[yellow]Cancelled.[/yellow]")
        raise typer.Exit(0)

    # ── Extra confirmation for live (paper) mode ────────────────────────────
    if not dry_run:
        rprint(
            "\n[bold red]WARNING[/bold red]: This will submit REAL orders to your "
            "Alpaca paper account. Confirm you understand this is NOT simulated."
        )
        confirm2 = typer.confirm("I understand — proceed with paper trading?", default=False)
        if not confirm2:
            rprint("[yellow]Cancelled.[/yellow]")
            raise typer.Exit(0)

    # ── Build strategy and risk objects (explicit min_vix=16.0 on both) ────
    strategy = IronCondor0DTE(config=IronCondorConfig(
        short_delta=0.10,
        wing_width=5.0,
        min_credit_pct=0.08,
        profit_target_pct=0.50,
        stop_loss_multiplier=2.0,
        max_short_delta_breach=0.25,
        min_vix=16.0,          # explicit — do not rely on default
    ))

    risk = RiskManager(
        vix_threshold=25.0,
        max_loss_per_trade=500.0,
        max_daily_loss_multiplier=2.0,
        max_drawdown_pct=0.10,
        max_concurrent_positions=1,
        min_vix=16.0,          # explicit — must match IronCondorConfig.min_vix
    )

    runner = LiveRunner(
        dry_run=dry_run,
        strategy=strategy,
        risk=risk,
        poll_interval_entry=60,     # market open, no position
        poll_interval_monitor=30,   # market open, position open
        poll_interval_closed=300,   # market closed (pre/after/weekend)
    )

    mode_str = "Dry-run" if dry_run else "Paper-live"
    rprint(f"[bold green]{mode_str} loop started. Press Ctrl+C to stop.[/bold green]\n")
    runner.run()


if __name__ == "__main__":
    app()
