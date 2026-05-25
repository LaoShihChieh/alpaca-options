"""
examples/real_backtest.py
=========================
Clean real-data backtest: 2024-01-19 → today.

Uses Alpaca's OptionHistoricalDataClient for all P&L simulation.
NEVER falls back to Black-Scholes when option bars are unavailable.
If a day lacks all four legs with bars starting at or before 10:10 ET,
the day is skipped with reason "incomplete_data".

Entry credit is computed from the real bar price of each leg at the
first bar at-or-after 10:00 ET — not from a BS model.

VIX-adaptive strike selection: per-day ^VIX close is fetched via yfinance
at startup (get_historical_vix_range).  target_delta(vix) and
adaptive_wing_width(vix) select strikes closer to ATM on low-vol days so
that real bar prices exceed the minimum credit threshold.

Run with:
    uv run python examples/real_backtest.py
"""

from __future__ import annotations

import logging
import math
import os
from collections import defaultdict
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mticker

from dotenv import load_dotenv
load_dotenv()

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from alpaca_options.backtest._occ import occ_symbol
from alpaca_options.backtest.replay import BacktestEngine, _BACKTEST_VIX_DEFAULT, _RISK_FREE_RATE
from alpaca_options.data.calendar import get_event_days
from alpaca_options.data.vix import get_historical_vix_range
from alpaca_options.risk.manager import RiskManager
from alpaca_options.strategies.iron_condor_0dte import (
    IronCondorConfig,
    adaptive_wing_width,
    target_delta,
)
from alpaca_options.utils.black_scholes import strike_for_delta

logging.basicConfig(level=logging.WARNING)

ET = ZoneInfo("America/New_York")

# ── Configuration ─────────────────────────────────────────────────────────────
START_DATE            = date(2024, 1, 19)   # first date with full-session bars
END_DATE              = date.today()
INITIAL_EQUITY        = 100_000.0
VIX_CONSTANT          = _BACKTEST_VIX_DEFAULT   # 18.0
SIGMA                 = VIX_CONSTANT / 100.0
MORNING_CUTOFF_HOUR   = 10
MORNING_CUTOFF_MIN    = 10    # option bars must start at or before 10:10 ET
EQUITY_CURVE_PATH     = "real_backtest_equity_curve.png"

CONFIG = IronCondorConfig(
    short_delta=0.10,
    wing_width=5.0,
    min_credit_pct=0.08,
    profit_target_pct=0.50,
    stop_loss_multiplier=2.0,
    max_short_delta_breach=0.25,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _first_bar_at_or_after(bars: list, day: date, hour: int, minute: int):
    """Return the first bar whose ET timestamp is ≥ hour:minute on day."""
    target = datetime(day.year, day.month, day.day, hour, minute, tzinfo=ET)
    for b in sorted(bars, key=lambda b: b.timestamp):
        if b.timestamp.astimezone(ET) >= target:
            return b
    return None


def _is_morning_complete(bars: list, day: date) -> bool:
    """True if bars contains a record at or before MORNING_CUTOFF on day."""
    if not bars:
        return False
    cutoff = datetime(
        day.year, day.month, day.day,
        MORNING_CUTOFF_HOUR, MORNING_CUTOFF_MIN, tzinfo=ET,
    )
    return any(b.timestamp.astimezone(ET) <= cutoff for b in bars)


def _simulate_real(
    option_bars: dict[str, list],
    legs_info: dict,
    real_entry_credit: float,
    day: date,
    cfg: IronCondorConfig,
) -> tuple[float, str]:
    """Simulate P&L using real option bar midpoints (close prices).

    Returns (pnl_per_share, exit_reason).
    """
    entry_time = datetime(day.year, day.month, day.day, 10, 0, tzinfo=ET)
    close_time = datetime(day.year, day.month, day.day, 15, 30, tzinfo=ET)

    call_short_sym = legs_info["call_short_sym"]
    call_long_sym  = legs_info["call_long_sym"]
    put_short_sym  = legs_info["put_short_sym"]
    put_long_sym   = legs_info["put_long_sym"]

    # Build unified timeline within trading window
    all_ts: set[datetime] = set()
    for sym in [call_short_sym, call_long_sym, put_short_sym, put_long_sym]:
        for b in option_bars.get(sym, []):
            t = b.timestamp.astimezone(ET)
            if entry_time <= t <= close_time:
                all_ts.add(t)

    if not all_ts:
        return 0.0, "no_timestamps"

    def last_close_at(sym: str, t: datetime) -> float:
        """Close price of most recent bar at or before t."""
        best = None
        for b in option_bars.get(sym, []):
            bt = b.timestamp.astimezone(ET)
            if bt <= t:
                best = b
        return float(best.close) if best else 0.0

    profit_target = real_entry_credit * (1 - cfg.profit_target_pct)
    stop_level    = real_entry_credit * cfg.stop_loss_multiplier

    for t in sorted(all_ts):
        cs = last_close_at(call_short_sym, t)
        cl = last_close_at(call_long_sym,  t)
        ps = last_close_at(put_short_sym,  t)
        pl = last_close_at(put_long_sym,   t)
        cost = (cs + ps) - (cl + pl)

        if cost <= profit_target:
            return real_entry_credit - cost, "profit_target"
        if cost >= stop_level:
            return real_entry_credit - cost, "stop_loss"

    # Force-close at last bar
    last_t = max(all_ts)
    cs = last_close_at(call_short_sym, last_t)
    cl = last_close_at(call_long_sym,  last_t)
    ps = last_close_at(put_short_sym,  last_t)
    pl = last_close_at(put_long_sym,   last_t)
    cost = (cs + ps) - (cl + pl)
    return real_entry_credit - cost, "time_close"


# ── Main backtest loop ────────────────────────────────────────────────────────

def run_backtest(console: Console) -> dict:
    engine = BacktestEngine(initial_equity=INITIAL_EQUITY, vix_override=VIX_CONSTANT)
    risk   = RiskManager(max_loss_per_trade=500.0)

    trading_days    = engine._get_trading_days(START_DATE, END_DATE)
    calendar_events = get_event_days(START_DATE, END_DATE)

    equity        = INITIAL_EQUITY
    equity_curve: list[tuple[date, float]] = []
    trades:       list[dict]  = []   # {date, pnl, exit_reason, credit, real_bars}
    skip_counts   = defaultdict(int)
    total_days    = len(trading_days)

    # Fetch per-day ^VIX closes via yfinance (one bulk download for the window)
    console.print("[dim]Fetching ^VIX history from yfinance…[/dim]")
    vix_by_date = get_historical_vix_range(START_DATE, END_DATE)
    coverage    = len(vix_by_date)
    console.print(
        f"[dim]  VIX data: {coverage} days fetched "
        f"({'yfinance' if coverage else 'fallback constant %.1f' % VIX_CONSTANT})[/dim]\n"
    )

    console.print(
        f"[bold cyan]Running real-data backtest[/bold cyan]  "
        f"[dim]{START_DATE} → {END_DATE}  ({total_days} trading days)[/dim]\n"
    )

    with console.status("[cyan]Fetching bars day by day…[/cyan]"):
        for day_idx, day in enumerate(trading_days):
            risk.reset_day(equity)

            # ── 0. Per-day VIX lookup (used for risk gate AND strike selection)
            vix_day = vix_by_date.get(day, VIX_CONSTANT)

            # ── 1. Calendar / risk filter ──────────────────────────────────
            allowed, reason = risk.check_entry_allowed(
                account_value=equity,
                vix=vix_day,
                today=day,
                calendar_events=calendar_events,
            )
            if not allowed:
                if "event" in reason.lower():
                    skip_counts["calendar"] += 1
                elif reason == "vix_too_low":
                    skip_counts["vix_too_low"] += 1
                elif "VIX" in reason:
                    skip_counts["vix"] += 1
                else:
                    skip_counts["risk_other"] += 1
                equity_curve.append((day, equity))
                continue

            # ── 2. SPY bars ────────────────────────────────────────────────
            spy_bars = engine._get_spy_bars(day)
            entry_spy = _first_bar_at_or_after(spy_bars, day, 10, 0)
            if entry_spy is None:
                skip_counts["no_spy"] += 1
                equity_curve.append((day, equity))
                continue

            spot = float(entry_spy.close)
            hours_rem = 6.0
            T = hours_rem / (6.5 * 252)

            # ── 3. VIX-adaptive strike computation ────────────────────────────
            sigma_day = vix_day / 100.0
            eff_delta = target_delta(vix_day)
            eff_wing  = adaptive_wing_width(vix_day)

            try:
                cs_raw = strike_for_delta(spot, T, _RISK_FREE_RATE, sigma_day, eff_delta, True)
                ps_raw = strike_for_delta(spot, T, _RISK_FREE_RATE, sigma_day, -eff_delta, False)
            except Exception:
                skip_counts["strike_error"] += 1
                equity_curve.append((day, equity))
                continue

            call_short = float(round(cs_raw))
            put_short  = float(round(ps_raw))
            call_long  = call_short + eff_wing
            put_long   = put_short  - eff_wing

            call_short_sym = occ_symbol("SPY", day, True,  call_short)
            call_long_sym  = occ_symbol("SPY", day, True,  call_long)
            put_short_sym  = occ_symbol("SPY", day, False, put_short)
            put_long_sym   = occ_symbol("SPY", day, False, put_long)

            # ── 4. Fetch all 4 option bar series ──────────────────────────
            option_bars: dict[str, list] = {}
            for sym in [call_short_sym, call_long_sym, put_short_sym, put_long_sym]:
                bars = engine._get_option_bars_1min(sym, day)
                if bars:
                    option_bars[sym] = bars

            # ── 5. Completeness check (NO BS FALLBACK) ─────────────────────
            if len(option_bars) < 4:
                skip_counts["incomplete_data"] += 1
                equity_curve.append((day, equity))
                continue

            # All 4 legs must have a bar at or before 10:10 ET
            morning_ok = all(
                _is_morning_complete(option_bars[sym], day)
                for sym in [call_short_sym, call_long_sym, put_short_sym, put_long_sym]
            )
            if not morning_ok:
                skip_counts["morning_gap"] += 1
                equity_curve.append((day, equity))
                continue

            # ── 6. Real entry credit ───────────────────────────────────────
            def _entry_price(sym: str) -> float | None:
                b = _first_bar_at_or_after(option_bars[sym], day, 10, 0)
                return float(b.close) if b else None

            cs_price = _entry_price(call_short_sym)
            cl_price = _entry_price(call_long_sym)
            ps_price = _entry_price(put_short_sym)
            pl_price = _entry_price(put_long_sym)

            if None in (cs_price, cl_price, ps_price, pl_price):
                skip_counts["incomplete_data"] += 1
                equity_curve.append((day, equity))
                continue

            real_credit = (cs_price + ps_price) - (cl_price + pl_price)   # type: ignore[operator]
            min_credit  = eff_wing * CONFIG.min_credit_pct

            if real_credit < min_credit:
                skip_counts["low_credit"] += 1
                equity_curve.append((day, equity))
                continue

            # ── 7. Simulate with real bars ─────────────────────────────────
            legs_info = {
                "call_short_sym": call_short_sym,
                "call_long_sym":  call_long_sym,
                "put_short_sym":  put_short_sym,
                "put_long_sym":   put_long_sym,
            }
            pnl_per_share, exit_reason = _simulate_real(
                option_bars, legs_info, real_credit, day, CONFIG
            )
            dollar_pnl = pnl_per_share * 100.0  # 1 contract = 100 shares
            equity += dollar_pnl
            risk.record_trade_result(dollar_pnl)

            trades.append({
                "date":        day,
                "pnl":         dollar_pnl,
                "exit_reason": exit_reason,
                "credit":      real_credit,
                "real_bars":   True,
                "spot":        spot,
                "call_short":  call_short,
                "put_short":   put_short,
                "vix_day":     vix_day,
                "eff_delta":   eff_delta,
                "eff_wing":    eff_wing,
            })
            equity_curve.append((day, equity))

    return {
        "trades":       trades,
        "skip_counts":  dict(skip_counts),
        "equity_curve": equity_curve,
        "total_days":   total_days,
        "initial_equity": INITIAL_EQUITY,
    }


# ── Aggregation ───────────────────────────────────────────────────────────────

def aggregate(results: dict) -> dict:
    trades       = results["trades"]
    equity_curve = results["equity_curve"]
    skip_counts  = results["skip_counts"]
    total_days   = results["total_days"]
    initial_eq   = results["initial_equity"]

    pnls      = [t["pnl"] for t in trades]
    num_trades = len(pnls)
    total_pnl  = sum(pnls)
    wins       = sum(1 for p in pnls if p > 0)
    win_rate   = wins / num_trades if num_trades > 0 else 0.0

    # Sharpe (annualised, using trade P&L series; zeros for non-trade days
    # would unfairly deflate — use trade-days only, then annualise)
    if num_trades > 1:
        arr    = np.array(pnls)
        sharpe = float(arr.mean() / (arr.std(ddof=1) + 1e-9) * math.sqrt(252))
    else:
        sharpe = 0.0

    # Max drawdown
    equities = [e for _, e in equity_curve]
    peak, max_dd = initial_eq, 0.0
    for e in equities:
        peak  = max(peak, e)
        dd    = (peak - e) / peak if peak > 0 else 0.0
        max_dd = max(max_dd, dd)

    worst = min(trades, key=lambda t: t["pnl"]) if trades else None
    best  = max(trades, key=lambda t: t["pnl"]) if trades else None

    # Year-by-year breakdown
    by_year: dict[int, dict] = defaultdict(lambda: {"trades": [], "equity_start": None})
    for t in trades:
        by_year[t["date"].year]["trades"].append(t)

    # Exit reason distribution
    exit_dist: dict[str, int] = defaultdict(int)
    for t in trades:
        exit_dist[t["exit_reason"]] += 1

    # Real vs synthetic count
    real_bar_count = sum(1 for t in trades if t["real_bars"])
    bs_count       = num_trades - real_bar_count

    # Per-VIX-bucket breakdown
    # Buckets align with target_delta() thresholds: ≤12, 12-16, 16-20, 20-25
    vix_buckets: dict[str, list[dict]] = {
        "≤12 (delta=0.20)":   [],
        "12–16 (delta=0.15)": [],
        "16–20 (delta=0.12)": [],
        "20–25 (delta=0.10)": [],
        ">25 (blocked)":      [],
    }
    for t in trades:
        v = t.get("vix_day", VIX_CONSTANT)
        if v <= 12:
            vix_buckets["≤12 (delta=0.20)"].append(t)
        elif v <= 16:
            vix_buckets["12–16 (delta=0.15)"].append(t)
        elif v <= 20:
            vix_buckets["16–20 (delta=0.12)"].append(t)
        elif v <= 25:
            vix_buckets["20–25 (delta=0.10)"].append(t)
        else:
            vix_buckets[">25 (blocked)"].append(t)

    return {
        "num_trades":   num_trades,
        "total_pnl":    total_pnl,
        "win_rate":     win_rate,
        "sharpe":       sharpe,
        "max_drawdown": max_dd,
        "worst":        worst,
        "best":         best,
        "skip_counts":  skip_counts,
        "total_days":   total_days,
        "by_year":      dict(by_year),
        "exit_dist":    dict(exit_dist),
        "real_bar_count": real_bar_count,
        "bs_count":     bs_count,
        "equity_curve": equity_curve,
        "initial_equity": initial_eq,
        "vix_buckets":  vix_buckets,
    }


# ── Equity curve plot ─────────────────────────────────────────────────────────

def save_equity_curve(agg: dict, trades: list[dict], path: str) -> None:
    equity_curve = agg["equity_curve"]
    initial_eq   = agg["initial_equity"]
    if not equity_curve:
        return

    dates    = [e[0] for e in equity_curve]
    equities = [e[1] for e in equity_curve]
    trade_dates = [t["date"] for t in trades]
    trade_pnls  = [t["pnl"]  for t in trades]

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(15, 8),
        gridspec_kw={"height_ratios": [3, 1]},
        sharex=True,
    )
    fig.patch.set_facecolor("#0d1117")
    ax1.set_facecolor("#0d1117")
    ax2.set_facecolor("#0d1117")

    # Equity curve
    ax1.plot(dates, equities, color="#64b5f6", linewidth=1.5, zorder=3)
    ax1.fill_between(dates, initial_eq, equities,
                     where=[e >= initial_eq for e in equities],
                     alpha=0.2, color="#4caf50", zorder=2)
    ax1.fill_between(dates, initial_eq, equities,
                     where=[e < initial_eq for e in equities],
                     alpha=0.2, color="#f44336", zorder=2)
    ax1.axhline(initial_eq, color="#555", linestyle="--", linewidth=0.7, zorder=1)

    # Year boundary lines
    years_seen = set()
    for d, _ in equity_curve:
        if d.year not in years_seen and d.month == 1:
            ax1.axvline(d, color="#888", linestyle=":", linewidth=0.8, alpha=0.6)
            ax1.text(d, ax1.get_ylim()[1] if ax1.get_ylim()[1] != ax1.get_ylim()[0] else initial_eq * 1.001,
                     f" {d.year}", color="#aaa", fontsize=7, va="top")
            years_seen.add(d.year)

    ax1.set_ylabel("Portfolio Value ($)", color="#aaa", fontsize=9)
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax1.tick_params(colors="#aaa")
    for sp in ax1.spines.values():
        sp.set_color("#333")
    ax1.grid(True, alpha=0.12, color="#aaa")

    # Daily P&L bars (trade days only)
    colors = ["#4caf50" if p >= 0 else "#f44336" for p in trade_pnls]
    ax2.bar(trade_dates, trade_pnls, color=colors, alpha=0.8, width=0.8)
    ax2.axhline(0, color="#555", linestyle="--", linewidth=0.7)
    ax2.set_ylabel("Trade P&L ($)", color="#aaa", fontsize=9)
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:+,.0f}"))
    ax2.tick_params(colors="#aaa")
    for sp in ax2.spines.values():
        sp.set_color("#333")
    ax2.grid(True, alpha=0.12, color="#aaa")
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b '%y"))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=35, ha="right", fontsize=7, color="#aaa")

    fig.suptitle(
        f"Iron Condor 0DTE — Real-Data Backtest  {START_DATE} → {END_DATE}\n"
        f"Total P&L: ${agg['total_pnl']:+,.0f}  │  "
        f"Sharpe: {agg['sharpe']:.2f}  │  "
        f"Win Rate: {agg['win_rate']:.0%}  │  "
        f"Max DD: {agg['max_drawdown']:.1%}  │  "
        f"{agg['num_trades']} trades / {agg['total_days']} days",
        color="#ccc", fontsize=10,
    )

    plt.tight_layout()
    plt.savefig(path, dpi=150, facecolor=fig.get_facecolor())
    plt.close(fig)


# ── Rich report ───────────────────────────────────────────────────────────────

def print_report(agg: dict, console: Console) -> None:
    skip = agg["skip_counts"]
    total_skipped = sum(skip.values())

    console.print()
    console.print(Panel(
        "[bold yellow]⚠  Real-data backtest: per-day ^VIX from yfinance drives "
        "VIX-adaptive strike selection (target_delta / adaptive_wing_width). "
        "Entry credit from actual bar prices; "
        "P&L from real 1-min option bars only — no BS fallback.[/bold yellow]",
        border_style="yellow", padding=(0, 2),
    ))
    console.print()

    # ── Overall metrics ───────────────────────────────────────────────────────
    t = Table(title="[bold]Overall Results[/bold]",
              box=box.ROUNDED, header_style="bold cyan",
              border_style="dim", min_width=60)
    t.add_column("Metric",  style="bold white", width=30)
    t.add_column("Value",   justify="right",    width=28)

    worst = agg["worst"]
    best  = agg["best"]
    t.add_row("Window",
              f"{START_DATE} → {END_DATE}")
    t.add_row("Total trading days",  str(agg["total_days"]))
    t.add_row("Days actually traded",
              f"[bold]{agg['num_trades']}[/bold]")
    t.add_row("Days skipped (all reasons)", str(total_skipped))
    t.add_row("Total P&L",
              f"[{'green' if agg['total_pnl'] >= 0 else 'red'}]"
              f"${agg['total_pnl']:+,.2f}[/]")
    t.add_row("Win rate",
              f"{agg['win_rate']:.1%}  ({sum(1 for tr in agg.get('exit_dist',{}).items())} exit types)")
    t.add_row("Max drawdown",  f"{agg['max_drawdown']:.2%}")
    t.add_row("Sharpe (trade-day annualised)", f"{agg['sharpe']:.2f}")
    t.add_row("Worst day",
              f"[red]{worst['date']} ${worst['pnl']:+,.2f}[/red]" if worst else "—")
    t.add_row("Best day",
              f"[green]{best['date']} ${best['pnl']:+,.2f}[/green]" if best else "—")
    t.add_row("Real bar trades",
              f"[green]{agg['real_bar_count']} (100%)[/green]" if agg['bs_count'] == 0
              else f"[red]{agg['real_bar_count']} / {agg['num_trades']}[/red]")
    t.add_row("BS fallback trades",
              f"[green]0[/green]" if agg['bs_count'] == 0
              else f"[red]{agg['bs_count']}[/red]")
    console.print(t)
    console.print()

    # ── Skip breakdown ────────────────────────────────────────────────────────
    s = Table(title="[bold]Skip Breakdown[/bold]",
              box=box.ROUNDED, header_style="bold cyan",
              border_style="dim", min_width=60)
    s.add_column("Reason",  style="bold white", width=28)
    s.add_column("Days",    justify="right",    width=8)
    s.add_column("% of total", justify="right", width=12)

    skip_labels = {
        "calendar":        "Calendar (FOMC/CPI/NFP)",
        "vix_too_low":     "VIX < 16 (floor filter)",
        "vix":             "VIX ≥ 25 (ceiling filter)",
        "risk_other":      "Other risk rule",
        "no_spy":          "No SPY bars",
        "strike_error":    "Strike computation failed",
        "incomplete_data": "Incomplete option data",
        "morning_gap":     "Morning gap (bars after 10:10 ET)",
        "low_credit":      "Low credit (real bars)",
    }
    for key, label in skip_labels.items():
        n = skip.get(key, 0)
        if n:
            pct = n / agg["total_days"] * 100
            s.add_row(label, str(n), f"{pct:.1f}%")

    traded_pct = agg["num_trades"] / agg["total_days"] * 100
    s.add_row("[bold]Traded[/bold]",
              f"[bold]{agg['num_trades']}[/bold]",
              f"[bold]{traded_pct:.1f}%[/bold]")
    console.print(s)
    console.print()

    # ── Exit distribution ─────────────────────────────────────────────────────
    if agg.get("exit_dist"):
        e = Table(title="[bold]Exit Distribution (traded days)[/bold]",
                  box=box.ROUNDED, header_style="bold cyan",
                  border_style="dim", min_width=50)
        e.add_column("Exit reason", style="bold white", width=20)
        e.add_column("Count",       justify="right",   width=8)
        e.add_column("% of trades", justify="right",   width=12)
        n_trades = agg["num_trades"]
        exit_labels = {
            "profit_target": "Profit target (50%)",
            "stop_loss":     "Stop loss (2× credit)",
            "time_close":    "Time close (15:30 ET)",
            "no_timestamps": "No bar timestamps",
        }
        for key, label in exit_labels.items():
            n = agg["exit_dist"].get(key, 0)
            if n:
                pct = n / n_trades * 100 if n_trades else 0
                colour = "green" if key == "profit_target" else ("red" if key == "stop_loss" else "")
                s_n   = f"[{colour}]{n}[/]" if colour else str(n)
                s_pct = f"[{colour}]{pct:.1f}%[/]" if colour else f"{pct:.1f}%"
                e.add_row(label, s_n, s_pct)
        console.print(e)
        console.print()

    # ── Year-by-year breakdown ────────────────────────────────────────────────
    y = Table(title="[bold]Year-by-Year Breakdown[/bold]",
              box=box.ROUNDED, header_style="bold cyan",
              border_style="dim", min_width=80)
    y.add_column("Year",       style="bold white", width=8)
    y.add_column("Trades",     justify="right",    width=8)
    y.add_column("Win %",      justify="right",    width=7)
    y.add_column("Total P&L",  justify="right",    width=12)
    y.add_column("Mean P&L",   justify="right",    width=10)
    y.add_column("Worst day",  justify="right",    width=10)
    y.add_column("Best day",   justify="right",    width=10)

    for yr in sorted(agg["by_year"].keys()):
        yr_trades = agg["by_year"][yr]["trades"]
        if not yr_trades:
            continue
        yr_pnls  = [t["pnl"] for t in yr_trades]
        yr_wins  = sum(1 for p in yr_pnls if p > 0)
        yr_wr    = yr_wins / len(yr_pnls) if yr_pnls else 0.0
        yr_total = sum(yr_pnls)
        yr_mean  = yr_total / len(yr_pnls)
        yr_worst = min(yr_pnls)
        yr_best  = max(yr_pnls)

        label = f"{yr}" + (" [dim](YTD)[/dim]" if yr == date.today().year else "")
        colour = "green" if yr_total >= 0 else "red"
        y.add_row(
            label,
            str(len(yr_pnls)),
            f"{yr_wr:.0%}",
            f"[{colour}]${yr_total:+,.0f}[/]",
            f"${yr_mean:+.0f}",
            f"[red]${yr_worst:+.0f}[/red]",
            f"[green]${yr_best:+.0f}[/green]",
        )
    console.print(y)
    console.print()

    # ── Per-VIX-bucket breakdown ──────────────────────────────────────────────
    bkt_table = Table(
        title="[bold]Per-VIX-Bucket Breakdown[/bold]",
        box=box.ROUNDED, header_style="bold cyan",
        border_style="dim", min_width=90,
    )
    bkt_table.add_column("VIX Bucket",    style="bold white", width=22)
    bkt_table.add_column("Trades",        justify="right",    width=8)
    bkt_table.add_column("Win %",         justify="right",    width=8)
    bkt_table.add_column("Total P&L",     justify="right",    width=12)
    bkt_table.add_column("Mean P&L",      justify="right",    width=10)
    bkt_table.add_column("Worst day",     justify="right",    width=10)
    bkt_table.add_column("Best day",      justify="right",    width=10)
    bkt_table.add_column("Avg Credit",    justify="right",    width=10)

    for bkt_label, bkt_trades in agg.get("vix_buckets", {}).items():
        if not bkt_trades:
            continue
        bp = [t["pnl"] for t in bkt_trades]
        bw = sum(1 for p in bp if p > 0)
        bwr = bw / len(bp) if bp else 0.0
        btot = sum(bp)
        bmean = btot / len(bp)
        bworst = min(bp)
        bbest = max(bp)
        avg_credit = sum(t["credit"] for t in bkt_trades) / len(bkt_trades)
        colour = "green" if btot >= 0 else "red"
        bkt_table.add_row(
            bkt_label,
            str(len(bp)),
            f"{bwr:.0%}",
            f"[{colour}]${btot:+,.0f}[/]",
            f"${bmean:+.0f}",
            f"[red]${bworst:+.0f}[/red]",
            f"[green]${bbest:+.0f}[/green]",
            f"${avg_credit:.2f}",
        )
    console.print(bkt_table)


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    console = Console()
    console.print()
    console.print(Panel(
        "[bold]Iron Condor 0DTE — Clean Real-Data Backtest[/bold]\n"
        "[dim]No BS fallback. Skip-on-incomplete. Real entry credit from bars.[/dim]",
        border_style="cyan", padding=(0, 2),
    ))

    raw     = run_backtest(console)
    agg     = aggregate(raw)
    trades  = raw["trades"]

    print_report(agg, console)

    save_equity_curve(agg, trades, EQUITY_CURVE_PATH)
    console.print(f"  [dim]Equity curve saved to [bold]{EQUITY_CURVE_PATH}[/bold][/dim]\n")


if __name__ == "__main__":
    main()
