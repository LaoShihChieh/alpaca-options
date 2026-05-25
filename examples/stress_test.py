"""
examples/stress_test.py
=======================
Adversarial stress test for the IronCondor0DTE strategy.

Runs each synthetic scenario 100 times with ±1 strike noise, reports
per-scenario stats in a rich table, and saves raw results to
stress_test_results.json.

Scenario 8 (RegimeShift2022) is a 60-consecutive-day multi-day regime
simulation; its equity curve is saved to regime_shift_2022.png.

Run with:
    uv run python examples/stress_test.py
"""

from __future__ import annotations

import json
import logging
import random
from collections import defaultdict
from datetime import date

import matplotlib
matplotlib.use("Agg")   # non-interactive; must come before pyplot import
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from alpaca_options.backtest.stress import (
    ALL_SCENARIOS,
    BASE_SPOT,
    RegimeShift2022,
    SyntheticBacktest,
)
from alpaca_options.strategies.iron_condor_0dte import IronCondorConfig

# Suppress strategy/HTTP noise during the tight simulation loop
logging.getLogger("alpaca_options").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

RUNS_PER_SCENARIO = 100
RANDOM_SEED       = 42
OUTPUT_JSON       = "stress_test_results.json"
TEST_DATE         = date(2026, 1, 15)   # Thursday, no known US holiday
REGIME_PNG        = "regime_shift_2022.png"


def main() -> None:
    random.seed(RANDOM_SEED)
    console = Console()

    # ── Warning banner ────────────────────────────────────────────────────────
    console.print()
    console.print(Panel(
        "[bold yellow]⚠  Stress tests use synthetic scenarios. "
        "Real markets are weirder. "
        "This is a lower bound on bad days, not an upper bound.[/bold yellow]",
        border_style="yellow",
        padding=(0, 2),
    ))

    # ── Set up engine ─────────────────────────────────────────────────────────
    config = IronCondorConfig(
        short_delta=0.10,
        wing_width=5.0,
        min_credit_pct=0.08,
        profit_target_pct=0.50,
        stop_loss_multiplier=2.0,
        max_short_delta_breach=0.25,
    )
    engine = SyntheticBacktest(config=config)

    # ── Run per-scenario ensemble (100 runs each) ─────────────────────────────
    all_results: dict[str, list] = {}
    with console.status("[bold cyan]Running stress scenarios…[/bold cyan]"):
        for scenario in ALL_SCENARIOS:
            results = engine.run_scenario(
                scenario,
                n=RUNS_PER_SCENARIO,
                d=TEST_DATE,
                prior_close=BASE_SPOT,
            )
            all_results[scenario.name] = results

    # ── Run RegimeShift2022 (60 consecutive days) ─────────────────────────────
    regime_scenario = RegimeShift2022()
    with console.status(
        "[bold cyan]Running RegimeShift2022 (60 synthetic trading days)…[/bold cyan]"
    ):
        regime_result = regime_scenario.run_regime(
            engine,
            initial_spot=BASE_SPOT,
            seed=RANDOM_SEED,
        )

    # ── Compute per-scenario stats ────────────────────────────────────────────
    stats = []
    for scenario in ALL_SCENARIOS:
        results  = all_results[scenario.name]
        entered  = [r for r in results if r.entered]
        skipped  = len(results) - len(entered)

        if entered:
            pnls     = [r.pnl for r in entered]
            mean_pnl = sum(pnls) / len(pnls)
            worst    = min(pnls)
            best     = max(pnls)
            wins     = sum(1 for p in pnls if p > 0)
            win_rate = wins / len(entered) * 100.0

            exits: dict[str, int] = defaultdict(int)
            for r in entered:
                exits[r.exit_reason] += 1
        else:
            mean_pnl = worst = best = win_rate = 0.0
            exits = defaultdict(int)

        stats.append({
            "name":     scenario.name,
            "total":    len(results),
            "entered":  len(entered),
            "skipped":  skipped,
            "win_rate": win_rate,
            "mean_pnl": mean_pnl,
            "worst":    worst,
            "best":     best,
            "exits":    dict(exits),
        })

    # Append RegimeShift2022 as a pseudo-row
    rr = regime_result
    regime_pnls = [p for p in rr.pnls if p != 0.0]
    regime_exits = rr.exit_counts
    stats.append({
        "name":     "RegimeShift2022",
        "total":    rr.days_total,
        "entered":  rr.days_traded,
        "skipped":  rr.days_vix_blocked + rr.days_no_credit,
        "win_rate": rr.win_rate * 100.0,
        "mean_pnl": sum(regime_pnls) / len(regime_pnls) if regime_pnls else 0.0,
        "worst":    min(regime_pnls) if regime_pnls else 0.0,
        "best":     max(regime_pnls) if regime_pnls else 0.0,
        "exits":    regime_exits,
        "_regime":  True,   # sentinel for footer note
    })

    # ── Rich table ────────────────────────────────────────────────────────────
    table = Table(
        title="[bold]IronCondor 0DTE — Synthetic Stress Test[/bold]  "
              f"[dim](n={RUNS_PER_SCENARIO} per scenario, ±1 strike noise)[/dim]",
        box=box.ROUNDED,
        header_style="bold cyan",
        border_style="dim",
        min_width=112,
        show_lines=False,
    )

    table.add_column("Scenario",   style="bold white",  width=20)
    table.add_column("Entered",    justify="right",     width=9)
    table.add_column("Win %",      justify="right",     width=7)
    table.add_column("Mean P&L",   justify="right",     width=10)
    table.add_column("Worst P&L",  justify="right",     width=10)
    table.add_column("Best P&L",   justify="right",     width=10)
    table.add_column("Profit ✓",   justify="right",     width=10)
    table.add_column("Stop ✗",     justify="right",     width=10)
    table.add_column("Delta ✗",    justify="right",     width=10)
    table.add_column("Time →",     justify="right",     width=9)

    def _pnl(v: float) -> str:
        if v > 0:
            return f"[green]${v:+.0f}[/green]"
        if v < 0:
            return f"[red]${v:+.0f}[/red]"
        return "[dim]$0[/dim]"

    def _pct(v: float) -> str:
        if v >= 65:
            return f"[green]{v:.0f}%[/green]"
        if v >= 40:
            return f"[yellow]{v:.0f}%[/yellow]"
        return f"[red]{v:.0f}%[/red]"

    def _exit_cell(n_exit: int, n_entered: int) -> str:
        if n_entered == 0:
            return "[dim]—[/dim]"
        pct = n_exit / n_entered * 100
        return f"{n_exit} [dim]({pct:.0f}%)[/dim]"

    for i, row in enumerate(stats):
        n  = row["entered"]
        ex = row["exits"]
        is_regime = row.get("_regime", False)

        # Exit columns reference the standard exit-reason keys
        n_profit = ex.get("close_profit", 0)
        n_stop   = ex.get("close_stop",   0)
        n_delta  = ex.get("close_delta_breach", 0)
        n_time   = ex.get("close_time",   0)

        # For RegimeShift2022, "Entered" shows "N/60d" to flag that the
        # denominator is trading days, not independent runs.
        entered_cell = (
            f"{n}/{row['total']}d"
            if is_regime
            else f"{n}/{row['total']}"
        )

        table.add_row(
            row["name"],
            entered_cell,
            _pct(row["win_rate"])  if n else "[dim]—[/dim]",
            _pnl(row["mean_pnl"]) if n else "[dim]—[/dim]",
            _pnl(row["worst"])     if n else "[dim]—[/dim]",
            _pnl(row["best"])      if n else "[dim]—[/dim]",
            _exit_cell(n_profit, n),
            _exit_cell(n_stop,   n),
            _exit_cell(n_delta,  n),
            _exit_cell(n_time,   n),
            style="on grey7" if i % 2 == 0 else "",
        )

    console.print()
    console.print(table)

    # ── Footer legend ─────────────────────────────────────────────────────────
    console.print(
        "\n[dim]"
        "  Profit ✓ = closed at 50% profit target  │"
        "  Stop ✗   = closed at 2× credit stop      │"
        "  Delta ✗  = short-leg delta ≥ 0.25        │"
        "  Time →   = 15:30 ET force-close"
        "[/dim]"
    )
    console.print(
        "[dim]  P&L is per 1-contract position (×100 shares).  "
        "Does not include commissions or bid/ask slippage.[/dim]"
    )
    console.print(
        "[dim]  RegimeShift2022: 60-day simulation; "
        f"VIX-blocked={rr.days_vix_blocked}d, "
        f"no-credit={rr.days_no_credit}d, "
        f"shock days={len(rr.shock_days)}.  "
        f"Total P&L {_pnl(rr.total_pnl)}  │  "
        f"Max drawdown [red]${rr.max_drawdown:.0f}[/red][/dim]\n"
    )

    # ── Per-scenario summary notes ────────────────────────────────────────────
    fomc_row = next(r for r in stats if r["name"] == "FOMCSurprise")
    n_entered = fomc_row["entered"]
    if n_entered > 0:
        mean = fomc_row["mean_pnl"]
        n_stop  = fomc_row["exits"].get("close_stop", 0)
        n_delta = fomc_row["exits"].get("close_delta_breach", 0)
        n_profit= fomc_row["exits"].get("close_profit", 0)
        caught  = n_stop + n_delta

        if caught > 0:
            console.print(
                f"  [red]⚠  FOMCSurprise:[/red] "
                f"{caught}/{n_entered} entered trades were open at the shock "
                f"({n_stop} stop, {n_delta} delta breach). Mean P&L {_pnl(mean)}.  "
                f"This is the cost of a broken or stale calendar filter."
            )
        else:
            console.print(
                f"  [yellow]⚠  FOMCSurprise:[/yellow] "
                f"All {n_profit} entered trades hit the 50% profit target before "
                f"the shock — the strategy escaped.  "
                f"Theta decay exits 0DTE positions by noon on calm mornings; "
                f"earlier or larger shocks would catch the strategy in the trade."
            )

    low_vix_row = next(r for r in stats if r["name"] == "LowVIXShock")
    lv_n = low_vix_row["entered"]
    if lv_n > 0:
        lv_stop   = low_vix_row["exits"].get("close_stop", 0)
        lv_delta  = low_vix_row["exits"].get("close_delta_breach", 0)
        lv_profit = low_vix_row["exits"].get("close_profit", 0)
        caught    = lv_stop + lv_delta
        console.print(
            f"  [red]⚠  LowVIXShock:[/red] "
            f"{caught}/{lv_n} trades caught by the shock "
            f"({lv_stop} stop, {lv_delta} delta breach), "
            f"{lv_profit} escaped via profit target.  "
            f"Mean P&L {_pnl(low_vix_row['mean_pnl'])}.  "
            f"Calendar filter blind spot — no event on any calendar."
        )

    console.print(
        f"\n  [dim]RegimeShift2022:[/dim] "
        f"{rr.days_traded}/{rr.days_total} days traded across 60-day regime.  "
        f"Total {_pnl(rr.total_pnl)},  "
        f"max drawdown [red]${rr.max_drawdown:.0f}[/red].  "
        f"Shock days: {len(rr.shock_days)} "
        f"({', '.join(str(s+1) for s in rr.shock_days) or '—'})."
    )

    # ── Equity curve plot ─────────────────────────────────────────────────────
    _save_equity_curve(rr, REGIME_PNG)
    console.print(f"\n  [dim]Equity curve saved to [bold]{REGIME_PNG}[/bold][/dim]")

    # ── Save JSON ─────────────────────────────────────────────────────────────
    payload: dict = {}
    for name, results in all_results.items():
        payload[name] = [
            {
                "entered":       r.entered,
                "pnl":           r.pnl,
                "exit_reason":   r.exit_reason,
                "entry_credit":  r.entry_credit,
                "exit_value":    r.exit_value,
                "strike_offset": r.strike_offset,
            }
            for r in results
        ]

    payload["RegimeShift2022"] = {
        "days_total":      rr.days_total,
        "days_traded":     rr.days_traded,
        "days_vix_blocked": rr.days_vix_blocked,
        "days_no_credit":  rr.days_no_credit,
        "total_pnl":       rr.total_pnl,
        "max_drawdown":    rr.max_drawdown,
        "win_rate":        round(rr.win_rate, 4),
        "shock_days":      rr.shock_days,
        "exit_counts":     rr.exit_counts,
        "equity_curve":    [round(v, 2) for v in rr.equity_curve],
        "pnls":            [round(v, 2) for v in rr.pnls],
    }

    with open(OUTPUT_JSON, "w") as fh:
        json.dump(payload, fh, indent=2)

    console.print(f"  [dim]Raw results saved to [bold]{OUTPUT_JSON}[/bold][/dim]\n")


# ---------------------------------------------------------------------------
# Equity curve plot
# ---------------------------------------------------------------------------

def _save_equity_curve(rr: "RegimeResult", path: str) -> None:  # noqa: F821
    """Save a labelled equity-curve PNG for the RegimeShift2022 simulation."""
    days = list(range(1, rr.days_total + 1))
    curve = rr.equity_curve

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_facecolor("#0d1117")
    fig.patch.set_facecolor("#0d1117")

    # Zero line
    ax.axhline(0, color="#555", linewidth=0.8, linestyle="--", zorder=1)

    # Colour-coded fill
    ax.fill_between(
        days, curve, 0,
        where=[v >= 0 for v in curve],
        alpha=0.25, color="#4caf50", zorder=2,
    )
    ax.fill_between(
        days, curve, 0,
        where=[v < 0 for v in curve],
        alpha=0.25, color="#f44336", zorder=2,
    )

    # Equity line
    ax.plot(days, curve, linewidth=1.6, color="#64b5f6", zorder=3, label="Cumulative P&L")

    # Mark shock days
    if rr.shock_days:
        sx = [s + 1 for s in rr.shock_days]
        sy = [curve[s] for s in rr.shock_days]
        ax.scatter(sx, sy, color="#ff7043", s=45, zorder=4,
                   label=f"Shock day ({len(rr.shock_days)})")

    # Mark max drawdown trough
    peak = 0.0
    trough_idx = 0
    peak_idx   = 0
    running_peak = 0.0
    for i, v in enumerate(curve):
        running_peak = max(running_peak, v)
        if (running_peak - v) > (peak - curve[trough_idx]):
            peak = running_peak
            trough_idx = i
            peak_idx = next(
                (j for j in range(i, -1, -1) if curve[j] == running_peak), i
            )

    if rr.max_drawdown > 0:
        ax.annotate(
            f"  MaxDD: −${rr.max_drawdown:.0f}",
            xy=(trough_idx + 1, curve[trough_idx]),
            fontsize=8, color="#ef9a9a",
            va="top" if curve[trough_idx] < 0 else "bottom",
        )

    # EOD final
    ax.annotate(
        f"  EOD: ${curve[-1]:+.0f}",
        xy=(days[-1], curve[-1]),
        fontsize=8, color="#a5d6a7" if curve[-1] >= 0 else "#ef9a9a",
        va="bottom" if curve[-1] >= 0 else "top",
    )

    # Axes styling
    for spine in ax.spines.values():
        spine.set_color("#333")
    ax.tick_params(colors="#aaa", labelsize=8)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"${v:+.0f}"))
    ax.set_xlabel("Trading Day", color="#aaa", fontsize=9)
    ax.set_ylabel("Cumulative P&L  (per contract, ×$100)", color="#aaa", fontsize=9)
    ax.set_title(
        f"RegimeShift2022 — 60-Day Equity Curve\n"
        f"Traded {rr.days_traded}/{rr.days_total} days  │  "
        f"Win rate {rr.win_rate:.0%}  │  "
        f"Total {'+' if rr.total_pnl >= 0 else ''}${rr.total_pnl:.0f}  │  "
        f"Max drawdown ${rr.max_drawdown:.0f}  │  "
        f"Shock days: {len(rr.shock_days)}",
        color="#ccc", fontsize=10, pad=8,
    )
    ax.legend(fontsize=8, facecolor="#1a1a2e", labelcolor="#ccc",
              framealpha=0.7, loc="upper left")
    ax.grid(True, alpha=0.12, color="#aaa")
    plt.tight_layout()
    plt.savefig(path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)


if __name__ == "__main__":
    main()
