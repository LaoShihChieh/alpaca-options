"""
examples/risk_manager_demo.py
==============================
Pure-logic demonstration of the RiskManager without any API calls.

Shows how each rule blocks or allows a trade entry.

Run with:
    uv run python examples/risk_manager_demo.py
"""

from __future__ import annotations

import logging
from datetime import date

from rich import print as rprint
from rich.table import Table
from rich.panel import Panel
from rich import box

from alpaca_options.risk.manager import RiskManager

logging.basicConfig(level=logging.WARNING)

ACCOUNT = 100_000.0


def run_scenario(
    name: str,
    rm: RiskManager,
    account: float,
    vix: float,
    today: date,
    events: set[date],
    description: str,
) -> None:
    allowed, reason = rm.check_entry_allowed(
        account_value=account,
        vix=vix,
        today=today,
        calendar_events=events,
    )
    status = "[green]✅ ALLOWED[/green]" if allowed else "[red]❌ BLOCKED[/red]"
    rprint(f"\n  [bold]{name}[/bold]")
    rprint(f"    {description}")
    rprint(f"    Result: {status}")
    if not allowed:
        rprint(f"    Reason: [yellow]{reason}[/yellow]")


def main() -> None:
    rprint(Panel.fit(
        "[bold cyan]🦙 RiskManager Demo — All Guard Rails[/bold cyan]",
        box=box.DOUBLE,
    ))

    # ── 1. Baseline — everything passes ──────────────────────────────────────
    rprint("\n[bold underline]Section 1: Baseline (all rules pass)[/bold underline]")
    rm = RiskManager(vix_threshold=25.0, max_loss_per_trade=500.0)
    rm.peak_equity = ACCOUNT
    rm.reset_day(ACCOUNT)

    run_scenario(
        name="Normal conditions",
        rm=rm, account=ACCOUNT, vix=18.0,
        today=date(2025, 3, 3),   # not an event day
        events=set(),
        description="VIX=18, no events, no losses, no positions.",
    )

    # ── 2. VIX filter ─────────────────────────────────────────────────────────
    rprint("\n[bold underline]Section 2: VIX regime filter[/bold underline]")
    for vix_level in [24.9, 25.0, 30.0]:
        rm2 = RiskManager(vix_threshold=25.0)
        rm2.peak_equity = ACCOUNT
        rm2.reset_day(ACCOUNT)
        run_scenario(
            name=f"VIX={vix_level}",
            rm=rm2, account=ACCOUNT, vix=vix_level,
            today=date(2025, 3, 3), events=set(),
            description=f"VIX threshold is 25.0.",
        )

    # ── 3. Calendar event filter ──────────────────────────────────────────────
    rprint("\n[bold underline]Section 3: Economic calendar filter[/bold underline]")
    fomc_day = date(2025, 3, 19)
    cpi_day  = date(2025, 3, 12)
    safe_day = date(2025, 3, 5)

    for today, events, label in [
        (safe_day, {fomc_day, cpi_day}, "Non-event day"),
        (fomc_day, {fomc_day, cpi_day}, "FOMC day"),
        (cpi_day,  {fomc_day, cpi_day}, "CPI day"),
    ]:
        rm3 = RiskManager()
        rm3.peak_equity = ACCOUNT
        rm3.reset_day(ACCOUNT)
        run_scenario(
            name=f"{label} ({today})",
            rm=rm3, account=ACCOUNT, vix=18.0,
            today=today, events=events,
            description=f"Calendar has {len(events)} event day(s).",
        )

    # ── 4. Daily loss limit ───────────────────────────────────────────────────
    rprint("\n[bold underline]Section 4: Daily loss limit (2× $500 = $1,000)[/bold underline]")
    for prior_losses in [0, 499, 500, 999, 1000]:
        rm4 = RiskManager(max_loss_per_trade=500.0, max_daily_loss_multiplier=2.0)
        rm4.peak_equity = ACCOUNT
        rm4.reset_day(ACCOUNT)
        if prior_losses > 0:
            # Simulate losses across one or more trades
            step = min(prior_losses, 500)
            remaining = prior_losses
            while remaining > 0:
                loss = min(remaining, step)
                rm4.record_trade_result(-loss)
                remaining -= loss
        run_scenario(
            name=f"Losses today: ${prior_losses}",
            rm=rm4, account=ACCOUNT - prior_losses, vix=18.0,
            today=date(2025, 3, 3), events=set(),
            description=f"Limit = $1,000. Accumulated = ${prior_losses}.",
        )

    # ── 5. Drawdown kill-switch ───────────────────────────────────────────────
    rprint("\n[bold underline]Section 5: Drawdown kill-switch (10% from peak)[/bold underline]")
    for dd_pct, equity in [(5, 95_000), (9, 91_000), (10, 90_000), (11, 89_000)]:
        rm5 = RiskManager(max_drawdown_pct=0.10)
        rm5.peak_equity = ACCOUNT
        rm5.reset_day(equity)
        run_scenario(
            name=f"Drawdown {dd_pct}%  (equity=${equity:,})",
            rm=rm5, account=equity, vix=18.0,
            today=date(2025, 3, 3), events=set(),
            description=f"Peak=${ACCOUNT:,}, current=${equity:,}.",
        )

    # ── 6. Concurrent position cap ────────────────────────────────────────────
    rprint("\n[bold underline]Section 6: Max concurrent positions (limit=1)[/bold underline]")
    for n_open in [0, 1]:
        rm6 = RiskManager(max_concurrent_positions=1)
        rm6.peak_equity = ACCOUNT
        rm6.reset_day(ACCOUNT)
        for _ in range(n_open):
            rm6.open_position()
        run_scenario(
            name=f"{n_open} position(s) open",
            rm=rm6, account=ACCOUNT, vix=18.0,
            today=date(2025, 3, 3), events=set(),
            description=f"Cap = 1 concurrent condor.",
        )

    # ── Summary ───────────────────────────────────────────────────────────────
    rprint("\n")
    rprint(Panel(
        "[bold]Risk manager summary:[/bold]\n"
        "  Rule a — VIX ≥ 25         → skip (elevated vol, wider markets)\n"
        "  Rule b — Economic event   → skip (FOMC, CPI, NFP cause spike risk)\n"
        "  Rule c — Daily loss limit → stop (daily P&L > 2× max-loss-per-trade)\n"
        "  Rule d — Drawdown kill    → stop (portfolio > 10% below peak)\n"
        "  Rule e — Position cap     → wait (only 1 condor open at a time)\n\n"
        "[dim]No API calls were made in this demo.[/dim]",
        title="[bold green]Risk Controls[/bold green]",
        border_style="green",
    ))


if __name__ == "__main__":
    main()
