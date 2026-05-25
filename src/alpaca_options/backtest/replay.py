"""
backtest/replay.py — Historical iron-condor simulation engine.

Simulation approach
-------------------
1. Fetch SPY 1-minute bars for each trading day via Alpaca's
   ``StockHistoricalDataClient``.
2. At 10:00 ET compute 0.10-delta strikes using Black-Scholes.
3. **Primary path**: try to fetch real 1-minute option bars for those OCC
   symbols (``OptionHistoricalDataClient``).  This works for recent contracts
   still in Alpaca's data store.
4. **Fallback path**: if bars are unavailable, price the condor via
   Black-Scholes with a constant IV derived from the VIX estimate.  A warning
   is logged when this path is taken.

Limitations
-----------
- Fills are computed at bar midpoints — live fills will be worse by 10-20 %.
- The VIX used for pricing is a constant 18 unless overridden via ``vix_override``.
- No early-assignment, dividend, or pin-risk modelling.

Usage::

    from datetime import date
    from alpaca_options.backtest.replay import BacktestEngine
    from alpaca_options.strategies.iron_condor_0dte import IronCondor0DTE
    from alpaca_options.risk.manager import RiskManager

    engine = BacktestEngine(initial_equity=100_000)
    results = engine.run(
        start=date(2025, 1, 2),
        end=date(2025, 3, 31),
        strategy=IronCondor0DTE(),
        risk=RiskManager(),
    )
    print(f"Total P&L: ${results.total_pnl:,.2f}, Sharpe: {results.sharpe:.2f}")
    results.save_equity_curve_plot("equity_curve.png")
"""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Optional
from zoneinfo import ZoneInfo

import numpy as np

from alpaca_options.backtest._occ import occ_symbol
from alpaca_options.client import get_clients
from alpaca_options.data.calendar import get_event_days
from alpaca_options.risk.manager import RiskManager
from alpaca_options.strategies.iron_condor_0dte import (
    CondorLegs,
    IronCondor0DTE,
    IronCondorConfig,
)
from alpaca_options.utils.black_scholes import bs_delta, bs_price, strike_for_delta

logger = logging.getLogger(__name__)

ET = ZoneInfo("America/New_York")
_BACKTEST_VIX_DEFAULT = 18.0     # historical VIX average
_RISK_FREE_RATE = 0.05
_CONTRACTS_PER_SPREAD = 100      # 1 contract = 100 shares


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


@dataclass
class BacktestResults:
    """Aggregated results from a :class:`BacktestEngine` run.

    Attributes
    ----------
    total_pnl:
        Sum of all trade P&Ls in USD.
    sharpe:
        Annualised Sharpe ratio (daily P&L / std(daily P&L) × √252).
    max_drawdown:
        Largest peak-to-trough equity decline (as a positive fraction, e.g. 0.08 = 8 %).
    win_rate:
        Fraction of trades that were profitable.
    num_trades:
        Number of condors entered.
    num_filtered_out:
        Number of trading days skipped by risk/event filters.
    worst_day:
        ``(date, pnl)`` tuple for the worst single-day dollar loss.
    best_day:
        ``(date, pnl)`` tuple for the best single-day dollar gain.
    equity_curve:
        ``[(date, equity)]`` list, starting at *initial_equity*.
    initial_equity:
        Starting portfolio value.
    """

    total_pnl: float
    sharpe: float
    max_drawdown: float
    win_rate: float
    num_trades: int
    num_filtered_out: int
    worst_day: tuple[date, float]
    best_day: tuple[date, float]
    equity_curve: list[tuple[date, float]]
    initial_equity: float

    def save_equity_curve_plot(self, path: str) -> None:
        """Save an equity-curve PNG (with daily P&L bars below).

        Parameters
        ----------
        path:
            File path for the output PNG (e.g. ``"equity_curve.png"``).
        """
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates
            import matplotlib.ticker as ticker
        except ImportError as exc:
            raise ImportError("matplotlib is required for plots.") from exc

        if not self.equity_curve:
            logger.warning("No equity curve data to plot.")
            return

        dates = [e[0] for e in self.equity_curve]
        equities = [e[1] for e in self.equity_curve]
        daily_pnl = [0.0] + [equities[i] - equities[i - 1] for i in range(1, len(equities))]

        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(14, 8),
            gridspec_kw={"height_ratios": [3, 1]},
            sharex=True,
        )
        fig.suptitle(
            f"Iron Condor 0DTE — Backtest Equity Curve\n"
            f"Total P&L: ${self.total_pnl:+,.0f}  |  "
            f"Sharpe: {self.sharpe:.2f}  |  "
            f"Win Rate: {self.win_rate:.0%}  |  "
            f"Max DD: {self.max_drawdown:.1%}",
            fontsize=11,
        )

        # Equity curve
        ax1.plot(dates, equities, color="steelblue", linewidth=1.5, label="Equity")
        ax1.fill_between(
            dates, self.initial_equity, equities,
            where=[e >= self.initial_equity for e in equities],
            alpha=0.25, color="green",
        )
        ax1.fill_between(
            dates, self.initial_equity, equities,
            where=[e < self.initial_equity for e in equities],
            alpha=0.25, color="red",
        )
        ax1.axhline(self.initial_equity, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
        ax1.set_ylabel("Portfolio Value ($)", fontsize=9)
        ax1.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"${x:,.0f}"))
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)

        # Daily P&L bars
        colors = ["#2ecc71" if p >= 0 else "#e74c3c" for p in daily_pnl]
        ax2.bar(dates, daily_pnl, color=colors, alpha=0.75, width=0.8)
        ax2.axhline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
        ax2.set_ylabel("Daily P&L ($)", fontsize=9)
        ax2.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"${x:+,.0f}"))
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        ax2.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0, interval=2))
        ax2.grid(True, alpha=0.3)

        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=35, ha="right", fontsize=8)
        plt.tight_layout()
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info("Equity curve saved: %s", path)


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class BacktestEngine:
    """Simulate the iron condor strategy over a historical date range.

    Parameters
    ----------
    initial_equity:
        Starting portfolio value in USD (default $100 000).
    vix_override:
        Fix IV at this VIX level for all days instead of fetching live data.
        Useful for deterministic tests.  Set to ``None`` to use the default
        constant of 18.
    """

    def __init__(
        self,
        initial_equity: float = 100_000.0,
        vix_override: Optional[float] = None,
    ) -> None:
        self.initial_equity = initial_equity
        self._vix = vix_override if vix_override is not None else _BACKTEST_VIX_DEFAULT
        self._trading, self._option_data = get_clients()
        self._stock_data = self._make_stock_client()

    @staticmethod
    def _make_stock_client():
        from alpaca.data.historical.stock import StockHistoricalDataClient
        api_key = os.environ.get("ALPACA_API_KEY", "")
        secret_key = os.environ.get("ALPACA_SECRET_KEY", "")
        return StockHistoricalDataClient(api_key=api_key, secret_key=secret_key)

    # ------------------------------------------------------------------
    # Calendar utilities
    # ------------------------------------------------------------------

    def _get_trading_days(self, start: date, end: date) -> list[date]:
        """Return market-open days via Alpaca calendar API."""
        try:
            from alpaca.trading.requests import GetCalendarRequest
            req = GetCalendarRequest(start=str(start), end=str(end))
            calendar = self._trading.get_calendar(req)
            days = [c.date for c in calendar]
            logger.info("Calendar: %d trading days from %s to %s.", len(days), start, end)
            return days
        except Exception as exc:
            logger.warning("Calendar API failed (%s). Falling back to weekdays.", exc)
            import pandas as pd
            bdays = pd.bdate_range(str(start), str(end))
            return [d.date() for d in bdays]

    # ------------------------------------------------------------------
    # Market data fetching
    # ------------------------------------------------------------------

    def _get_spy_bars(self, day: date) -> list:
        """Fetch SPY 1-minute bars for *day* (9:30–16:00 ET)."""
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame

        start_dt = datetime(day.year, day.month, day.day, 9, 30, tzinfo=ET)
        end_dt = datetime(day.year, day.month, day.day, 16, 0, tzinfo=ET)

        req = StockBarsRequest(
            symbol_or_symbols="SPY",
            timeframe=TimeFrame.Minute,
            start=start_dt,
            end=end_dt,
        )
        try:
            result = self._stock_data.get_stock_bars(req)
            bars = result.data.get("SPY", [])
            logger.debug("SPY bars for %s: %d bars.", day, len(bars))
            return bars
        except Exception as exc:
            logger.warning("Could not fetch SPY bars for %s: %s", day, exc)
            return []

    def _get_option_bars_1min(self, symbol: str, day: date) -> list:
        """Try to fetch 1-minute bars for *symbol* from Alpaca.

        Falls back to 5-minute bars and logs a limitation note.
        Returns an empty list if no data is available.
        """
        from alpaca.data.requests import OptionBarsRequest
        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

        start_dt = datetime(day.year, day.month, day.day, 9, 30, tzinfo=ET)
        end_dt = datetime(day.year, day.month, day.day, 16, 0, tzinfo=ET)

        for tf_label, tf in [("1-min", TimeFrame.Minute), ("5-min", TimeFrame(5, TimeFrameUnit.Minute))]:
            try:
                req = OptionBarsRequest(
                    symbol_or_symbols=symbol,
                    timeframe=tf,
                    start=start_dt,
                    end=end_dt,
                )
                result = self._option_data.get_option_bars(req)
                bars = result.data.get(symbol, [])
                if bars:
                    if tf_label != "1-min":
                        logger.info(
                            "LIMITATION: using %s bars for %s (1-min unavailable). "
                            "P&L simulation is less precise.",
                            tf_label, symbol,
                        )
                    return bars
            except Exception:
                pass

        return []  # no data available

    # ------------------------------------------------------------------
    # Black-Scholes simulation utilities
    # ------------------------------------------------------------------

    def _T_remaining(self, bar_dt: datetime, day: date) -> float:
        """Fraction-of-year remaining from *bar_dt* to 4:00 PM ET on *day*."""
        close_dt = datetime(day.year, day.month, day.day, 16, 0, tzinfo=ET)
        bar_et = bar_dt.astimezone(ET)
        seconds_left = (close_dt - bar_et).total_seconds()
        return max(seconds_left / (6.5 * 3600 * 252), 1e-6)

    def _condor_spread_value(
        self,
        spot: float,
        legs: CondorLegs,
        T: float,
        sigma: float,
    ) -> float:
        """Current theoretical cost-to-close the condor (per share).

        A declining value is good — it means the spread has decayed.
        """
        cs = bs_price(spot, legs.call_short_strike, T, _RISK_FREE_RATE, sigma, is_call=True)
        cl = bs_price(spot, legs.call_long_strike, T, _RISK_FREE_RATE, sigma, is_call=True)
        ps = bs_price(spot, legs.put_short_strike, T, _RISK_FREE_RATE, sigma, is_call=False)
        pl = bs_price(spot, legs.put_long_strike, T, _RISK_FREE_RATE, sigma, is_call=False)
        # Cost to close = buy-back shorts − sell longs
        return (cs + ps) - (cl + pl)

    def _condor_credit_bs(
        self,
        spot: float,
        call_short: float,
        call_long: float,
        put_short: float,
        put_long: float,
        T: float,
        sigma: float,
    ) -> float:
        """Net credit when we open the condor via Black-Scholes pricing."""
        fake_legs = CondorLegs(
            put_long_symbol="", put_short_symbol="",
            call_short_symbol="", call_long_symbol="",
            put_long_strike=put_long, put_short_strike=put_short,
            call_short_strike=call_short, call_long_strike=call_long,
            net_credit=0.0,
        )
        return self._condor_spread_value(spot, fake_legs, T, sigma)

    # ------------------------------------------------------------------
    # Per-day simulation
    # ------------------------------------------------------------------

    def _find_bar_at_or_after(self, bars: list, day: date, hour: int, minute: int):
        """Return the first bar at or after *hour:minute* ET on *day*."""
        target = datetime(day.year, day.month, day.day, hour, minute, tzinfo=ET)
        for bar in bars:
            if bar.timestamp.astimezone(ET) >= target:
                return bar
        return None

    def _simulate_day_with_real_bars(
        self,
        option_bars: dict[str, list],
        legs: CondorLegs,
        entry_credit: float,
        day: date,
        config: IronCondorConfig,
    ) -> float:
        """Simulate P&L using real option bar midpoints (primary path).

        Returns per-share P&L (multiply by 100 for dollar value).
        """
        entry_time = datetime(day.year, day.month, day.day, 10, 0, tzinfo=ET)
        close_time = datetime(day.year, day.month, day.day, 15, 30, tzinfo=ET)

        # Build a unified timeline from all four legs' bars
        all_timestamps = set()
        for sym_bars in option_bars.values():
            for b in sym_bars:
                t = b.timestamp.astimezone(ET)
                if entry_time <= t <= close_time:
                    all_timestamps.add(t)

        if not all_timestamps:
            return 0.0

        def mid_at(sym: str, t: datetime) -> float:
            """Nearest bar close price for symbol at time t."""
            bars_sym = option_bars.get(sym, [])
            best = None
            for b in bars_sym:
                bt = b.timestamp.astimezone(ET)
                if bt <= t:
                    best = b
            return float(best.close) if best else 0.0

        for t in sorted(all_timestamps):
            cs = mid_at(legs.call_short_symbol, t)
            cl = mid_at(legs.call_long_symbol, t)
            ps = mid_at(legs.put_short_symbol, t)
            pl = mid_at(legs.put_long_symbol, t)
            current_cost = (cs + ps) - (cl + pl)

            if entry_credit > 0:
                if current_cost <= entry_credit * (1 - config.profit_target_pct):
                    return entry_credit - current_cost
                if current_cost >= entry_credit * config.stop_loss_multiplier:
                    return entry_credit - current_cost

        # Force-close at last bar
        last_t = max(all_timestamps)
        cs = mid_at(legs.call_short_symbol, last_t)
        cl = mid_at(legs.call_long_symbol, last_t)
        ps = mid_at(legs.put_short_symbol, last_t)
        pl = mid_at(legs.put_long_symbol, last_t)
        final_cost = (cs + ps) - (cl + pl)
        return entry_credit - final_cost

    def _simulate_day_bs(
        self,
        spy_bars: list,
        legs: CondorLegs,
        entry_credit: float,
        day: date,
        sigma: float,
        config: IronCondorConfig,
    ) -> float:
        """Simulate P&L using Black-Scholes pricing on SPY bars (fallback).

        Returns per-share P&L.
        """
        entry_time = datetime(day.year, day.month, day.day, 10, 0, tzinfo=ET)
        close_time = datetime(day.year, day.month, day.day, 15, 30, tzinfo=ET)

        active_bars = [
            b for b in spy_bars
            if entry_time <= b.timestamp.astimezone(ET) <= close_time
        ]
        if not active_bars:
            return 0.0

        for bar in active_bars:
            spot = float(bar.close)
            T = self._T_remaining(bar.timestamp, day)
            current_cost = self._condor_spread_value(spot, legs, T, sigma)

            if entry_credit > 0:
                if current_cost <= entry_credit * (1 - config.profit_target_pct):
                    return entry_credit - current_cost
                if current_cost >= entry_credit * config.stop_loss_multiplier:
                    return entry_credit - current_cost

        # Force-close at 15:30 (last active bar)
        last_bar = active_bars[-1]
        spot = float(last_bar.close)
        T = self._T_remaining(last_bar.timestamp, day)
        final_cost = self._condor_spread_value(spot, legs, T, sigma)
        return entry_credit - final_cost

    # ------------------------------------------------------------------
    # Main run method
    # ------------------------------------------------------------------

    def run(
        self,
        start: date,
        end: date,
        strategy: IronCondor0DTE,
        risk: RiskManager,
    ) -> BacktestResults:
        """Simulate the iron condor strategy over [start, end].

        Parameters
        ----------
        start, end:
            Date range (inclusive).  Uses Alpaca's market calendar.
        strategy:
            :class:`IronCondor0DTE` instance (config is read from it).
        risk:
            :class:`RiskManager` instance (state is mutated during the run).

        Returns
        -------
        BacktestResults
        """
        sigma = self._vix / 100.0

        trading_days = self._get_trading_days(start, end)
        calendar_events = get_event_days(start, end)

        equity = self.initial_equity
        equity_curve: list[tuple[date, float]] = []
        day_pnls: list[tuple[date, float]] = []   # (date, dollar_pnl) for each trade
        num_filtered = 0

        logger.info(
            "Backtest: %d trading days, IV=%.1f%%, initial_equity=$%.0f",
            len(trading_days), self._vix, equity,
        )

        for day in trading_days:
            risk.reset_day(equity)

            # Risk / event filter
            allowed, reason = risk.check_entry_allowed(
                account_value=equity,
                vix=self._vix,
                today=day,
                calendar_events=calendar_events,
            )
            if not allowed:
                logger.info("Day %s skipped: %s", day, reason)
                num_filtered += 1
                equity_curve.append((day, equity))
                continue

            # Fetch SPY 1-minute bars
            spy_bars = self._get_spy_bars(day)
            entry_bar = self._find_bar_at_or_after(spy_bars, day, hour=10, minute=0)
            if entry_bar is None:
                logger.warning("No SPY bars at 10:00 for %s — skipping.", day)
                equity_curve.append((day, equity))
                continue

            spot = float(entry_bar.close)
            hours_rem = 6.0  # approximately 10:00 → 16:00
            T = hours_rem / (6.5 * 252)

            # Compute strikes
            try:
                call_short_raw = strike_for_delta(spot, T, _RISK_FREE_RATE, sigma, strategy.config.short_delta, is_call=True)
                put_short_raw  = strike_for_delta(spot, T, _RISK_FREE_RATE, sigma, -strategy.config.short_delta, is_call=False)
            except Exception as exc:
                logger.warning("Strike computation failed for %s: %s", day, exc)
                equity_curve.append((day, equity))
                continue

            call_short = round(call_short_raw)
            put_short  = round(put_short_raw)
            call_long  = call_short + strategy.config.wing_width
            put_long   = put_short  - strategy.config.wing_width

            # Compute opening credit via BS
            entry_credit = self._condor_credit_bs(
                spot, call_short, call_long, put_short, put_long, T, sigma
            )
            min_credit = strategy.config.wing_width * strategy.config.min_credit_pct

            if entry_credit < min_credit:
                logger.info(
                    "Day %s: BS credit $%.4f < min $%.4f — skipping.",
                    day, entry_credit, min_credit,
                )
                num_filtered += 1
                equity_curve.append((day, equity))
                continue

            # Build synthetic legs structure for simulation
            exp_str = day.strftime("%y%m%d")
            legs = CondorLegs(
                put_long_symbol=occ_symbol("SPY", day, False, put_long),
                put_short_symbol=occ_symbol("SPY", day, False, put_short),
                call_short_symbol=occ_symbol("SPY", day, True, call_short),
                call_long_symbol=occ_symbol("SPY", day, True, call_long),
                put_long_strike=put_long,
                put_short_strike=put_short,
                call_short_strike=call_short,
                call_long_strike=call_long,
                net_credit=entry_credit,
            )

            logger.info(
                "Day %s: spot=%.2f strikes=[%.0f/%.0f/%.0f/%.0f] credit=$%.4f",
                day, spot, put_long, put_short, call_short, call_long, entry_credit,
            )

            # Try real option bars (primary) — fall back to BS (secondary)
            option_bars: dict[str, list] = {}
            use_real_bars = False
            for sym in [legs.put_long_symbol, legs.put_short_symbol, legs.call_short_symbol, legs.call_long_symbol]:
                bars = self._get_option_bars_1min(sym, day)
                if bars:
                    option_bars[sym] = bars
            if len(option_bars) == 4:
                use_real_bars = True
                logger.debug("Day %s: using real option bars.", day)
            else:
                if option_bars:
                    logger.info(
                        "Day %s: only %d/4 option bar series available — using BS fallback.",
                        day, len(option_bars),
                    )
                else:
                    logger.debug("Day %s: no option bars — using BS fallback.", day)

            if use_real_bars:
                pnl_per_share = self._simulate_day_with_real_bars(
                    option_bars, legs, entry_credit, day, strategy.config
                )
            else:
                pnl_per_share = self._simulate_day_bs(
                    spy_bars, legs, entry_credit, day, sigma, strategy.config
                )

            dollar_pnl = pnl_per_share * _CONTRACTS_PER_SPREAD
            equity += dollar_pnl
            risk.record_trade_result(dollar_pnl)

            day_pnls.append((day, dollar_pnl))
            equity_curve.append((day, equity))

            logger.info(
                "Day %s done: P&L=$%.2f, equity=$%.2f", day, dollar_pnl, equity
            )

        return self._aggregate(equity_curve, day_pnls, num_filtered)

    # ------------------------------------------------------------------
    # Result aggregation
    # ------------------------------------------------------------------

    def _aggregate(
        self,
        equity_curve: list[tuple[date, float]],
        day_pnls: list[tuple[date, float]],
        num_filtered: int,
    ) -> BacktestResults:
        total_pnl = sum(p for _, p in day_pnls)
        num_trades = len(day_pnls)

        # Sharpe
        if num_trades > 1:
            pnl_arr = np.array([p for _, p in day_pnls])
            sharpe = float(pnl_arr.mean() / (pnl_arr.std(ddof=1) + 1e-9) * math.sqrt(252))
        else:
            sharpe = 0.0

        # Max drawdown
        equities = [e for _, e in equity_curve]
        if equities:
            peak = equities[0]
            max_dd = 0.0
            for e in equities:
                peak = max(peak, e)
                dd = (peak - e) / peak if peak > 0 else 0.0
                max_dd = max(max_dd, dd)
        else:
            max_dd = 0.0

        # Win rate
        wins = sum(1 for _, p in day_pnls if p > 0)
        win_rate = wins / num_trades if num_trades > 0 else 0.0

        # Best / worst day
        if day_pnls:
            worst = min(day_pnls, key=lambda x: x[1])
            best  = max(day_pnls, key=lambda x: x[1])
        else:
            worst = best = (date.today(), 0.0)

        return BacktestResults(
            total_pnl=total_pnl,
            sharpe=sharpe,
            max_drawdown=max_dd,
            win_rate=win_rate,
            num_trades=num_trades,
            num_filtered_out=num_filtered,
            worst_day=worst,
            best_day=best,
            equity_curve=equity_curve,
            initial_equity=self.initial_equity,
        )
