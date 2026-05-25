"""
examples/backtest_90_days.py
=============================
Run the iron condor 0DTE backtest over the last 90 calendar days.
Prints a stats table and saves equity_curve.png.

⚠  IMPORTANT DISCLAIMER:
   Backtest uses bar midpoints for fills, which is optimistic.
   Expect live results to be 10-20% worse due to:
     • Bid/ask spread slippage
     • Partial fills and queue priority at the limit price
     • Model error (Black-Scholes assumes constant IV)
     • Data gaps for 0DTE historical bars (BS fallback used where unavailable)

Run with:
    uv run python examples/backtest_90_days.py
"""

from __future__ import annotations

import logging
import sys
from datetime import date, timedelta
from pathlib import Path

from rich import print as rprint
from rich.panel import Panel
from rich.table import Table
from rich import box

from alpaca_options.backtest.replay import BacktestEngine
from alpaca_options.risk.manager import RiskManager
from alpaca_options.strategies.iron_condor_0dte import IronCondor0DTE, IronCondorConfig

# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)s %(name)s: %(message)s",
)
logging.getLogger("alpaca_options").setLevel(logging.INFO)
# ---------------------------------------------------------------------------


DISCLAIMER = (
    "⚠  [bold yellow]Backtest uses bar midpoints for fills, which is optimistic.[/bold yellow]\n"
    "   [yellow]Expect live results to be 10-20% worse due to slippage, "
    "spread costs, and IV model error.[/yellow]"
)

PLOT_PATH = Path("equity_curve.png")


def main() -> None:
    rprint(Panel.fit(
        "[bold cyan]🦙 Iron Condor 0DTE — 90-Day Backtest[/bold cyan]",
        box=box.DOUBLE,
    ))
    rprint(Panel(DISCLAIMER, border_style="yellow"))

    end   = date.today() - timedelta(days=1)   # yesterday (last complete day)
    start = end - timedelta(days=90)

    rprint(f"\n  Date range: [bold]{start}[/bold] → [bold]{end}[/bold]")
    rprint("  Fetching market data and simulating…\n")

    # Strategy configuration
    config = IronCondorConfig(
        short_delta=0.10,
        wing_width=5.0,
        min_credit_pct=0.08,
        profit_target_pct=0.50,
        stop_loss_multiplier=2.0,
        max_short_delta_breach=0.25,
    )
    strategy = IronCondor0DTE(config=config)
    risk     = RiskManager(
        vix_threshold=25.0,
        max_loss_per_trade=500.0,
        max_daily_loss_multiplier=2.0,
        max_drawdown_pct=0.10,
        max_concurrent_positions=1,
    )

    engine = BacktestEngine(
        initial_equity=100_000.0,
        vix_override=18.0,   # constant IV; see disclaimer above
    )

    try:
        results = engine.run(start, end, strategy, risk)
    except Exception as exc:
        rprint(f"[red]Backtest failed: {exc}[/red]")
        raise

    # ── Stats table ───────────────────────────────────────────────────────────
    table = Table(
        title="Backtest Results",
        box=box.ROUNDED,
        border_style="cyan",
        show_header=True,
        header_style="bold cyan",
    )
    table.add_column("Metric", style="bold", min_width=24)
    table.add_column("Value", justify="right", min_width=16)

    pnl_color = "green" if results.total_pnl >= 0 else "red"
    sharpe_color = "green" if results.sharpe >= 1.0 else ("yellow" if results.sharpe >= 0 else "red")
    wr_color = "green" if results.win_rate >= 0.65 else ("yellow" if results.win_rate >= 0.50 else "red")

    table.add_row("Period", f"{start} → {end}")
    table.add_row("Trading days scanned", str(results.num_trades + results.num_filtered_out))
    table.add_row("Trades entered", str(results.num_trades))
    table.add_row("Days filtered out", str(results.num_filtered_out))
    table.add_row(
        "Total P&L",
        f"[{pnl_color}]${results.total_pnl:+,.2f}[/{pnl_color}]",
    )
    table.add_row(
        "Return on capital",
        f"[{pnl_color}]{results.total_pnl / results.initial_equity:+.2%}[/{pnl_color}]",
    )
    table.add_row(
        "Annualised Sharpe",
        f"[{sharpe_color}]{results.sharpe:.2f}[/{sharpe_color}]",
    )
    table.add_row(
        "Win rate",
        f"[{wr_color}]{results.win_rate:.1%}[/{wr_color}]",
    )
    table.add_row(
        "Max drawdown",
        f"[{'red' if results.max_drawdown > 0.05 else 'green'}]"
        f"{results.max_drawdown:.2%}[/]",
    )
    if results.best_day[1] != 0:
        table.add_row(
            "Best day",
            f"[green]${results.best_day[1]:+,.2f}[/green] ({results.best_day[0]})",
        )
    if results.worst_day[1] != 0:
        table.add_row(
            "Worst day",
            f"[red]${results.worst_day[1]:+,.2f}[/red] ({results.worst_day[0]})",
        )
    table.add_row(
        "Final equity",
        f"${results.initial_equity + results.total_pnl:,.2f}",
    )

    rprint(table)

    # ── EV explanation ────────────────────────────────────────────────────────
    rprint(Panel(
        "[bold]Why we backtest first — the EV math:[/bold]\n\n"
        "  A raw 0DTE iron condor wins ~75% of the time (SPY stays inside shorts).\n"
        "  But on the 25% losing days the loss ≈ 2–5× the credit collected.\n\n"
        "  Unfiltered EV estimate:\n"
        "    0.75 × $100 credit  −  0.25 × $300 avg loss  =  [red]−$0[/red] (break-even before costs)\n\n"
        "  With risk filters (VIX < 25, skip FOMC/CPI/NFP, kill-switch):\n"
        "    • We remove the highest-vol losing days.\n"
        "    • Empirically improves win rate to ~80%+ on filtered subset.\n"
        "    • Still negative-EV without enough credit — that's why min_credit_pct matters.\n\n"
        "  [dim]The backtest above shows which filter combination actually produced edge.[/dim]",
        title="[bold]Why Backtest First?[/bold]",
        border_style="blue",
    ))

    # ── Equity curve plot ─────────────────────────────────────────────────────
    try:
        results.save_equity_curve_plot(str(PLOT_PATH))
        rprint(f"\n[bold green]✅ Equity curve saved:[/bold green] {PLOT_PATH.resolve()}")
    except Exception as exc:
        rprint(f"\n[yellow]⚠ Could not save plot: {exc}[/yellow]")

    rprint("\n[dim]Backtest complete.[/dim]")


if __name__ == "__main__":
    main()
