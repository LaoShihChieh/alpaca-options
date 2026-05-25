"""
backtest/stress.py — Adversarial scenario injection for IronCondor0DTE.

Each scenario class implements ``generate_day(d, prior_close) -> SyntheticDayData``.
``SyntheticBacktest`` runs the *real* IronCondor0DTE strategy (unchanged) against
synthetic spot/IV paths, pricing options at every monitoring tick with
Black-Scholes so the strategy sees realistic price changes.

No API calls are made.  The strategy's entry/exit logic (should_enter, monitor)
runs exactly as in production; only market data is synthetic.

Design invariants
-----------------
- Only ``should_enter`` and ``monitor`` from IronCondor0DTE are exercised.
  ``build_condor``, ``enter``, and ``exit`` touch real APIs and are replaced
  by direct BS computation inside ``SyntheticBacktest``.
- The VIX value visible to ``monitor()``'s delta-breach check is patched per
  tick to match the scenario's current IV, then restored after each run.
- Option prices use the same bs_price / strike_for_delta functions as the
  live strategy, so BS model error is consistent between entry and monitoring.
"""

from __future__ import annotations

import logging
import random
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Optional
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

import alpaca_options.strategies.iron_condor_0dte as _strat_module
from alpaca_options.strategies.iron_condor_0dte import (
    CondorLegs,
    CondorPosition,
    ExitDecision,
    IronCondor0DTE,
    IronCondorConfig,
    adaptive_wing_width,
    target_delta,
)
from alpaca_options.utils.black_scholes import bs_price, strike_for_delta

ET = ZoneInfo("America/New_York")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

BASE_SPOT: float = 580.0    # representative SPY price
BASE_IV:   float = 0.18     # annualised IV on a calm day (~18% for SPY 0DTE)
RISK_FREE: float = 0.05     # matches IronCondor0DTE default

_ENTRY_H,  _ENTRY_M  = 10,  5   # first tick after the 10:00 window opens
_FORCE_H,  _FORCE_M  = 15, 30   # same as strategy's force-close
_EXPIRY_H, _EXPIRY_M = 16,  0   # SPY option expiry (T denominator anchor)
_POLL_MIN             = 5        # minutes between monitor() calls


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class SyntheticDayData:
    """Minute-resolution synthetic market data for one trading day.

    Attributes
    ----------
    date:
        Calendar date of the simulated session.
    scenario_name:
        Human-readable label for results reporting.
    prior_close:
        SPY closing price of the preceding session (used to compute gap size).
    spot_path:
        List of ``(ET datetime, SPY price)`` anchor points.  Values between
        anchors are linearly interpolated by ``spot_at()``.
    iv_path:
        List of ``(ET datetime, annualised IV)`` anchor points.
    """

    date: date
    scenario_name: str
    prior_close: float
    spot_path: list[tuple[datetime, float]]
    iv_path:   list[tuple[datetime, float]]

    def spot_at(self, t: datetime) -> float:
        """SPY price at time *t* (linear interpolation)."""
        return _lerp(t, self.spot_path)

    def iv_at(self, t: datetime) -> float:
        """Annualised implied volatility at time *t* (linear interpolation)."""
        return _lerp(t, self.iv_path)


@dataclass
class TradeResult:
    """Outcome of one synthetic condor trade."""

    scenario_name: str
    entered: bool           # False → skipped (time window, no credit, …)
    pnl: float              # dollars per 1-contract (×100 shares); 0.0 if not entered
    exit_reason: str        # ExitDecision.value, "NO_ENTRY", or "NO_CREDIT"
    entry_credit: float     # per share ($); 0.0 if not entered
    exit_value: float       # cost-to-close at exit per share ($)
    strike_offset: int      # applied noise: +1 = wider condor, -1 = tighter


# ---------------------------------------------------------------------------
# Path-building helpers
# ---------------------------------------------------------------------------

def _lerp(t: datetime, path: list[tuple[datetime, float]]) -> float:
    """Linear interpolation along a sorted (datetime, value) path."""
    if not path:
        return 0.0
    if t <= path[0][0]:
        return path[0][1]
    if t >= path[-1][0]:
        return path[-1][1]
    for i in range(len(path) - 1):
        t0, v0 = path[i]
        t1, v1 = path[i + 1]
        if t0 <= t <= t1:
            frac = (t - t0).total_seconds() / (t1 - t0).total_seconds()
            return v0 + frac * (v1 - v0)
    return path[-1][1]


def _path(d: date, anchors: list[tuple[int, int, float]]) -> list[tuple[datetime, float]]:
    """Build a path from ``(hour, minute, value)`` anchors via linear interpolation.

    Returns a list of ``(ET datetime, value)`` tuples at minute resolution.
    """
    pts = [
        (datetime(d.year, d.month, d.day, h, m, tzinfo=ET), v)
        for h, m, v in anchors
    ]
    out: list[tuple[datetime, float]] = []
    for i in range(len(pts) - 1):
        t0, v0 = pts[i]
        t1, v1 = pts[i + 1]
        n_min = int((t1 - t0).total_seconds() / 60)
        for j in range(n_min):
            out.append((t0 + timedelta(minutes=j), v0 + (v1 - v0) * j / n_min))
    out.append(pts[-1])
    return out


# ---------------------------------------------------------------------------
# Scenario classes
# ---------------------------------------------------------------------------

class GapDownDay:
    """SPY opens -3% from prior close, drifts another -1% by 10:00 ET,
    then fully recovers to prior close by 15:30.  IV spikes 2× at open.

    Adversarial pattern: the strategy enters with strikes based on the gapped-
    down spot (~556).  The call short is set at ~572.  As SPY recovers through
    the short strike, the call spread expands and triggers a stop or delta breach.
    """
    name = "GapDownDay"

    def generate_day(self, d: date, prior_close: float = BASE_SPOT) -> SyntheticDayData:
        gap_open = prior_close * 0.97          # -3% gap at open
        low      = gap_open   * 0.99          # drift another -1% by 10:00
        recovery = prior_close                  # full recovery by 15:30

        return SyntheticDayData(
            date=d, scenario_name=self.name, prior_close=prior_close,
            spot_path=_path(d, [
                (9,  30, gap_open),
                (10,  0, low),
                (15, 30, recovery),
            ]),
            iv_path=_path(d, [
                (9,  30, BASE_IV * 2.0),
                (10,  0, BASE_IV * 2.0),
                (12,  0, BASE_IV * 1.6),
                (15, 30, BASE_IV * 1.2),
            ]),
        )


class GapUpDay:
    """Mirror of GapDownDay on the upside.

    SPY opens +3%, grinds higher to +4% by 10:00, then gives back the entire
    gap by 15:30.  The put short is set far above the closing spot, threatening
    the put spread as the reversal unfolds.
    """
    name = "GapUpDay"

    def generate_day(self, d: date, prior_close: float = BASE_SPOT) -> SyntheticDayData:
        gap_open = prior_close * 1.03
        high     = gap_open   * 1.01
        giveback = prior_close                  # fully reverts

        return SyntheticDayData(
            date=d, scenario_name=self.name, prior_close=prior_close,
            spot_path=_path(d, [
                (9,  30, gap_open),
                (10,  0, high),
                (15, 30, giveback),
            ]),
            iv_path=_path(d, [
                (9,  30, BASE_IV * 2.0),
                (10,  0, BASE_IV * 2.0),
                (12,  0, BASE_IV * 1.6),
                (15, 30, BASE_IV * 1.2),
            ]),
        )


class VolSpikeIntraday:
    """SPY flat at open, then drops 2% between 13:00 and 14:00 — after the
    typical entry window but before the 15:30 force-close.  IV spikes mid-day.

    Tests whether the stop-loss / delta-breach fires when the market moves
    against an already-entered position mid-session.
    """
    name = "VolSpikeIntraday"

    def generate_day(self, d: date, prior_close: float = BASE_SPOT) -> SyntheticDayData:
        drop = prior_close * 0.98

        return SyntheticDayData(
            date=d, scenario_name=self.name, prior_close=prior_close,
            spot_path=_path(d, [
                (9,  30, prior_close),
                (13,  0, prior_close),          # flat all morning
                (14,  0, drop),                  # -2% in 60 minutes
                (15, 30, prior_close * 0.985),   # partial recovery
            ]),
            iv_path=_path(d, [
                (9,  30, BASE_IV),
                (12, 30, BASE_IV),
                (13,  0, BASE_IV * 1.8),         # IV spikes as drop begins
                (14,  0, BASE_IV * 2.4),
                (15, 30, BASE_IV * 1.8),
            ]),
        )


class WhipsawDay:
    """SPY oscillates ±1.5% three times across the session.

    Designed to repeatedly approach both short strikes and test whether the
    delta-breach exit fires correctly.  The elevated IV means option prices
    move significantly even when spot is not yet at the short strike.
    """
    name = "WhipsawDay"

    def generate_day(self, d: date, prior_close: float = BASE_SPOT) -> SyntheticDayData:
        hi = prior_close * 1.015
        lo = prior_close * 0.985

        return SyntheticDayData(
            date=d, scenario_name=self.name, prior_close=prior_close,
            spot_path=_path(d, [
                (9,  30, prior_close),
                (10, 30, hi),
                (11, 30, lo),
                (12, 30, hi),
                (13, 30, lo),
                (14, 30, hi),
                (15, 30, prior_close),
            ]),
            iv_path=_path(d, [
                (9,  30, BASE_IV * 1.5),
                (10, 30, BASE_IV * 1.8),
                (11, 30, BASE_IV * 1.6),
                (12, 30, BASE_IV * 1.8),
                (13, 30, BASE_IV * 1.6),
                (14, 30, BASE_IV * 1.8),
                (15, 30, BASE_IV * 1.4),
            ]),
        )


class FOMCSurprise:
    """SPY flat then a 2% shock at 11:30 ET — inside the strategy's holding window.

    Models a surprise inter-meeting Fed statement, an unscheduled data release,
    or any macro event whose date was missing from the calendar filter (FRED
    outage, hardcoded dates stale, or a genuinely unscheduled announcement).

    The strategy enters at 10:05 with normal IV and is fully exposed when the
    11:30 shock arrives — the profit target has not yet had time to fire (0DTE
    theta decay reaches 50% around noon, so the trade is still live at 11:30).

    Direction is random per call; 100 runs produce a mix of up and down shocks.
    Results show the real cost of a missed or broken calendar filter.
    """
    name = "FOMCSurprise"

    def generate_day(self, d: date, prior_close: float = BASE_SPOT) -> SyntheticDayData:
        direction  = random.choice([-1, 1])
        post_shock = prior_close * (1.0 + direction * 0.02)
        tail       = post_shock  * (1.0 + direction * 0.005)

        return SyntheticDayData(
            date=d, scenario_name=self.name, prior_close=prior_close,
            spot_path=_path(d, [
                (9,  30, prior_close),
                (11, 30, prior_close),           # flat until the surprise
                (11, 45, post_shock),            # 2% shock in 15 minutes
                (15, 30, tail),
            ]),
            iv_path=_path(d, [
                (9,  30, BASE_IV),               # calm open; no one saw it coming
                (11, 30, BASE_IV),
                (11, 35, BASE_IV * 2.8),         # vol explosion at announcement
                (12, 30, BASE_IV * 2.2),
                (15, 30, BASE_IV * 1.8),
            ]),
        )


class SteadyTrendUp:
    """SPY grinds up 1% linearly across the full session.  No vol spike.

    The 'easy' control scenario.  The call side is approached slowly;
    IV falls slightly (typical on up days).  Provides a baseline for how
    often each exit fires on a boring winning day.
    """
    name = "SteadyTrendUp"

    def generate_day(self, d: date, prior_close: float = BASE_SPOT) -> SyntheticDayData:
        return SyntheticDayData(
            date=d, scenario_name=self.name, prior_close=prior_close,
            spot_path=_path(d, [
                (9,  30, prior_close),
                (15, 30, prior_close * 1.01),
            ]),
            iv_path=_path(d, [
                (9,  30, BASE_IV * 0.90),
                (15, 30, BASE_IV * 0.80),       # vol typically compresses on up day
            ]),
        )


class SteadyTrendDown:
    """SPY grinds down 1% linearly across the full session.  No vol spike.

    Mirror of SteadyTrendUp.  IV rises slightly (typical on down days).
    The put side is approached slowly.
    """
    name = "SteadyTrendDown"

    def generate_day(self, d: date, prior_close: float = BASE_SPOT) -> SyntheticDayData:
        return SyntheticDayData(
            date=d, scenario_name=self.name, prior_close=prior_close,
            spot_path=_path(d, [
                (9,  30, prior_close),
                (15, 30, prior_close * 0.99),
            ]),
            iv_path=_path(d, [
                (9,  30, BASE_IV * 1.10),
                (15, 30, BASE_IV * 1.20),
            ]),
        )


class LowVIXShock:
    """SPY opens flat with VIX ~16 — filters pass and the strategy enters normally.

    At a random time between 11:00 and 14:00 ET, SPY drops 2.0–2.5% over a
    30–60 minute window.  IV spikes from the calm ~16% baseline to ~26%.

    This is the "all filters green, bad news hits anyway" case — models
    geopolitical shocks, surprise data releases, or any event that bypasses
    the usual pre-market telegraphing (c.f. the July 2024 vol unwind or a
    sudden headline during a calm session).  The calendar filter is irrelevant
    because the event was not on any economic calendar.

    Each run randomises drop timing (11:00–14:00), duration (30–60 min),
    and magnitude (2.0–2.5%) so the 100-run ensemble covers a range of
    intraday positions when the shock arrives.
    """
    name = "LowVIXShock"
    _OPEN_IV = 0.16   # VIX ~16: calm open, well below the 25 cutoff
    _PEAK_IV = 0.26   # IV at the bottom of the shock move

    def generate_day(self, d: date, prior_close: float = BASE_SPOT) -> SyntheticDayData:
        # Shock start: random offset (0–180 min) after 11:00 → covers 11:00–14:00
        sm   = random.randint(0, 180)
        s_h, s_m = 11 + sm // 60, sm % 60

        # Shock duration: 30–60 minutes
        sd   = random.randint(30, 60)
        em   = sm + sd                          # max = 180 + 60 = 240 → 15:00; before 15:30
        e_h, e_m = 11 + em // 60, em % 60

        drop_pct  = random.uniform(0.020, 0.025)
        shock_low = prior_close * (1.0 - drop_pct)
        end_spot  = shock_low  * 1.002          # tiny stabilisation after the drop

        return SyntheticDayData(
            date=d, scenario_name=self.name, prior_close=prior_close,
            spot_path=_path(d, [
                (9,   30, prior_close),
                (s_h, s_m, prior_close),        # flat until the shock
                (e_h, e_m, shock_low),           # drops 2.0–2.5% over 30–60 min
                (15,  30,  end_spot),            # stays near lows into close
            ]),
            iv_path=_path(d, [
                (9,   30, self._OPEN_IV),
                (s_h, s_m, self._OPEN_IV),       # calm until the drop begins
                (e_h, e_m, self._PEAK_IV),       # spikes with the move
                (15,  30,  self._PEAK_IV * 0.88),
            ]),
        )


# ---------------------------------------------------------------------------
# RegimeShift2022 — multi-day regime simulation
# ---------------------------------------------------------------------------

@dataclass
class RegimeResult:
    """Cumulative outcome of the 60-day RegimeShift2022 simulation.

    Attributes
    ----------
    equity_curve:
        Cumulative P&L after each simulated trading day (including zero for
        days the strategy did not enter).
    pnls:
        Per-day P&L.  0.0 for skipped / no-credit days.
    days_total:
        Total number of trading days simulated.
    days_traded:
        Days where the condor was actually opened (entered=True).
    days_vix_blocked:
        Days where VIX ≥ 25 prevented the RiskManager from entering.
    days_no_credit:
        Days where VIX was acceptable but the condor credit was too low.
    total_pnl:
        Sum of all per-day P&L.
    max_drawdown:
        Largest peak-to-trough equity decline (positive = loss in dollars).
    win_rate:
        Fraction of entered days with positive P&L.
    exit_counts:
        Raw exit-reason counts from the monitoring loop.
    shock_days:
        Indices (0-based) of days that had an intraday shock injected.
    """
    equity_curve:    list[float]
    pnls:            list[float]
    days_total:      int
    days_traded:     int
    days_vix_blocked: int
    days_no_credit:  int
    total_pnl:       float
    max_drawdown:    float
    win_rate:        float
    exit_counts:     dict[str, int]
    shock_days:      list[int] = field(default_factory=list)


class RegimeShift2022:
    """60-day simulation of a sustained high-volatility regime (2022 style).

    Each day is classified at open:

    * **High-VIX day** (60% probability): VIX drawn from [25.5, 35].  The
      RiskManager would block entry (VIX ≥ 25).  Simulated here by skipping
      entry explicitly — the strategy does not run.
    * **Entry day** (40% probability): VIX drawn from [22, 24.9].  The
      condor is opened if credit is sufficient.  Daily SPY move is drawn
      from N(0, 1.4%) — roughly 2.8× calmer-regime sigma, reflecting the
      sustained choppiness of 2022.
    * **Shock day** (1-in-8 entry days): a hawkish Fed/CPI surprise with no
      calendar warning.  SPY drops 2–3% over 30–60 minutes between 11:00
      and 14:00 ET; IV spikes to 2× the entry level.

    Run via ``run_regime(engine)`` rather than the standard ``run_scenario``
    interface.  Results include the full equity curve and max drawdown so
    the simulation can be plotted externally.
    """
    name          = "RegimeShift2022"
    N_DAYS        = 60
    ENTRY_VIX_LOW  = 22.0
    ENTRY_VIX_HIGH = 24.9    # strictly below the 25.0 RiskManager cutoff
    ENTRY_PROB     = 0.40
    SHOCK_PROB     = 1.0 / 8
    DAILY_SIGMA    = 0.014   # σ of daily SPY moves on entry days

    # ------------------------------------------------------------------
    # Day generators
    # ------------------------------------------------------------------

    def _entry_day(
        self, d: date, prior_close: float, rng: random.Random, iv: float,
    ) -> SyntheticDayData:
        """Normal entry day: monotonic drift drawn from N(0, DAILY_SIGMA)."""
        daily_ret = rng.gauss(0.0, self.DAILY_SIGMA)
        end_spot  = prior_close * (1.0 + daily_ret)
        iv_end    = iv * (0.95 if daily_ret >= 0 else 1.06)  # vol/price anti-corr.

        return SyntheticDayData(
            date=d, scenario_name=self.name, prior_close=prior_close,
            spot_path=_path(d, [(9, 30, prior_close), (15, 30, end_spot)]),
            iv_path=_path(d,   [(9, 30, iv),          (15, 30, iv_end)]),
        )

    def _shock_day(
        self, d: date, prior_close: float, rng: random.Random, iv: float,
    ) -> SyntheticDayData:
        """Entry day with an unannounced –2% to –3% intraday shock."""
        drop_pct  = rng.uniform(0.020, 0.030)
        shock_low = prior_close * (1.0 - drop_pct)
        iv_spike  = min(0.50, iv * 2.0)

        sm   = rng.randint(0, 180)              # shock start: 0–180 min after 11:00
        sd   = rng.randint(30, 60)              # shock duration
        s_h, s_m = 11 + sm // 60, sm % 60
        e_h, e_m = 11 + (sm + sd) // 60, (sm + sd) % 60

        return SyntheticDayData(
            date=d, scenario_name=self.name, prior_close=prior_close,
            spot_path=_path(d, [
                (9,   30, prior_close),
                (s_h, s_m, prior_close),        # flat until shock
                (e_h, e_m, shock_low),           # –2% to –3% over 30–60 min
                (15,  30,  shock_low * 1.003),
            ]),
            iv_path=_path(d, [
                (9,   30, iv),
                (s_h, s_m, iv),
                (e_h, e_m, iv_spike),
                (15,  30,  iv_spike * 0.85),
            ]),
        )

    def _high_vix_day(
        self, d: date, prior_close: float, rng: random.Random, iv: float,
    ) -> SyntheticDayData:
        """High-VIX day: generate a spot path for EOD price tracking only.
        The strategy does not enter; this is used solely to evolve ``spot``.
        """
        daily_ret = rng.gauss(0.0, self.DAILY_SIGMA * 1.3)   # choppier on high-VIX days
        end_spot  = prior_close * (1.0 + daily_ret)

        return SyntheticDayData(
            date=d, scenario_name=self.name, prior_close=prior_close,
            spot_path=_path(d, [(9, 30, prior_close), (15, 30, end_spot)]),
            iv_path=_path(d,   [(9, 30, iv),          (15, 30, iv * 1.02)]),
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run_regime(
        self,
        engine: "SyntheticBacktest",
        start_date: Optional[date] = None,
        n_days: int = N_DAYS,
        initial_spot: float = BASE_SPOT,
        seed: int = 0,
    ) -> RegimeResult:
        """Simulate *n_days* consecutive trading days in a 2022-style regime.

        VIX ≥ 25 days are blocked without calling ``engine.run_one`` —
        mirroring what the RiskManager does in production.

        Parameters
        ----------
        engine:
            A configured ``SyntheticBacktest`` instance.
        start_date:
            First trading day.  Defaults to 2025-01-06 (within the
            hardcoded FOMC/CPI/NFP calendar range).
        n_days:
            Number of trading days to simulate.
        initial_spot:
            SPY price on the eve of day 1.
        seed:
            RNG seed for reproducibility.

        Returns
        -------
        RegimeResult
        """
        if start_date is None:
            start_date = date(2025, 1, 6)   # first Monday of 2025

        rng          = random.Random(seed)
        cum_pnl      = 0.0
        peak_pnl     = 0.0
        max_dd       = 0.0
        equity_curve: list[float] = []
        pnls:         list[float] = []
        exit_counts                = defaultdict(int)
        shock_days:  list[int]    = []

        days_vix_blocked = 0
        days_no_credit   = 0
        days_traded      = 0

        spot = initial_spot
        d    = start_date

        for day_idx in range(n_days):
            # Skip weekends
            while d.weekday() >= 5:
                d += timedelta(days=1)

            if rng.random() < self.ENTRY_PROB:
                # --- Entry-candidate day (VIX 22–24.9) ---
                iv  = rng.uniform(self.ENTRY_VIX_LOW, self.ENTRY_VIX_HIGH) / 100.0
                is_shock = rng.random() < self.SHOCK_PROB
                if is_shock:
                    day = self._shock_day(d, spot, rng, iv)
                    shock_days.append(day_idx)
                else:
                    day = self._entry_day(d, spot, rng, iv)

                offset = rng.choices([-1, 0, 1], weights=[1, 2, 1])[0]
                result = engine.run_one(day, strike_offset=offset)

                if result.entered:
                    days_traded += 1
                    exit_counts[result.exit_reason] += 1
                    pnl = result.pnl
                else:
                    days_no_credit += 1
                    exit_counts[result.exit_reason] += 1
                    pnl = 0.0

            else:
                # --- High-VIX day: RiskManager blocks entry ---
                iv  = rng.uniform(25.5, 35.0) / 100.0
                day = self._high_vix_day(d, spot, rng, iv)
                days_vix_blocked += 1
                exit_counts["NO_ENTRY_HIGH_VIX"] += 1
                pnl = 0.0

            cum_pnl += pnl
            pnls.append(pnl)
            equity_curve.append(cum_pnl)

            peak_pnl = max(peak_pnl, cum_pnl)
            max_dd   = max(max_dd, peak_pnl - cum_pnl)

            # Advance spot to EOD for the next day's prior_close
            eod_dt = datetime(d.year, d.month, d.day, 15, 30, tzinfo=ET)
            spot   = day.spot_at(eod_dt)
            d     += timedelta(days=1)

        wins     = sum(1 for p in pnls if p > 0)
        win_rate = wins / days_traded if days_traded > 0 else 0.0

        return RegimeResult(
            equity_curve=equity_curve,
            pnls=pnls,
            days_total=n_days,
            days_traded=days_traded,
            days_vix_blocked=days_vix_blocked,
            days_no_credit=days_no_credit,
            total_pnl=round(cum_pnl, 2),
            max_drawdown=round(max_dd, 2),
            win_rate=win_rate,
            exit_counts=dict(exit_counts),
            shock_days=shock_days,
        )


#: All scenarios in presentation order.
ALL_SCENARIOS: list = [
    GapDownDay(),
    GapUpDay(),
    VolSpikeIntraday(),
    WhipsawDay(),
    FOMCSurprise(),
    SteadyTrendUp(),
    SteadyTrendDown(),
    LowVIXShock(),
]


# ---------------------------------------------------------------------------
# Synthetic backtester
# ---------------------------------------------------------------------------

class SyntheticBacktest:
    """Run IronCondor0DTE strategy logic against synthetic market data.

    The strategy object is used unchanged: ``should_enter`` gates entry,
    ``monitor`` drives the exit loop.  Option prices at entry and at each
    tick are computed with Black-Scholes using the scenario's spot/IV path.

    The VIX value seen by ``monitor``'s delta-breach calculation is patched
    per tick (``get_current_vix`` in the strategy module is temporarily
    replaced with a lambda that returns the scenario IV × 100), then restored
    after each run.

    Parameters
    ----------
    config:
        IronCondorConfig to use.  Defaults to standard production settings.
    """

    def __init__(self, config: Optional[IronCondorConfig] = None) -> None:
        self.config = config or IronCondorConfig()
        with patch(
            "alpaca_options.strategies.iron_condor_0dte.get_clients",
            return_value=(MagicMock(), MagicMock()),
        ):
            self.strategy = IronCondor0DTE(config=self.config)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _T(self, current: datetime, expiry: datetime) -> float:
        """Fraction of a trading year remaining from *current* to *expiry*."""
        secs = max((expiry - current).total_seconds(), 60.0)
        return secs / (3600.0 * 6.5 * 252.0)

    def _condor_value(
        self,
        spot: float,
        T: float,
        iv: float,
        csk: float,   # call short strike
        psk: float,   # put  short strike
        clk: float,   # call long  strike
        plk: float,   # put  long  strike
    ) -> float:
        """Cost-to-close the condor (buy back short legs, sell long legs).

        Always ≥ 0: each spread component is non-negative by construction
        (short strike closer to ATM ⟹ more expensive than long strike).
        """
        return max(
            0.0,
            bs_price(spot, csk, T, RISK_FREE, iv, True)   # buy call short
            + bs_price(spot, psk, T, RISK_FREE, iv, False) # buy put  short
            - bs_price(spot, clk, T, RISK_FREE, iv, True)  # sell call long
            - bs_price(spot, plk, T, RISK_FREE, iv, False), # sell put  long
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run_one(
        self,
        day: SyntheticDayData,
        strike_offset: int = 0,
    ) -> TradeResult:
        """Simulate one condor trade on *day*.

        Parameters
        ----------
        day:
            Synthetic market data from any scenario's ``generate_day``.
        strike_offset:
            Integer added to the call short strike and subtracted from the
            put short strike.  ``+1`` → wider condor (safer but less credit),
            ``-1`` → tighter condor (more credit but closer to spot).

        Returns
        -------
        TradeResult
            Outcome including P&L and exit reason.
        """
        d          = day.date
        entry_dt   = datetime(d.year, d.month, d.day, _ENTRY_H,  _ENTRY_M,  tzinfo=ET)
        expiry_dt  = datetime(d.year, d.month, d.day, _EXPIRY_H, _EXPIRY_M, tzinfo=ET)
        force_dt   = datetime(d.year, d.month, d.day, _FORCE_H,  _FORCE_M,  tzinfo=ET)

        # ── Entry window check ──────────────────────────────────────────────
        iv_e = day.iv_at(entry_dt)
        if not self.strategy.should_enter(entry_dt, vix=iv_e * 100.0):
            return TradeResult(
                scenario_name=day.scenario_name, entered=False, pnl=0.0,
                exit_reason="NO_ENTRY", entry_credit=0.0, exit_value=0.0,
                strike_offset=strike_offset,
            )

        spot_e = day.spot_at(entry_dt)
        T_e    = self._T(entry_dt, expiry_dt)

        # ── VIX-adaptive strike selection ────────────────────────────────────
        vix_e     = iv_e * 100.0
        eff_delta = target_delta(vix_e)
        eff_wing  = adaptive_wing_width(vix_e)

        try:
            cs_raw = strike_for_delta(spot_e, T_e, RISK_FREE, iv_e,
                                      eff_delta, True)
            ps_raw = strike_for_delta(spot_e, T_e, RISK_FREE, iv_e,
                                      -eff_delta, False)
        except Exception as exc:
            logger.warning("strike_for_delta failed: %s", exc)
            return TradeResult(
                scenario_name=day.scenario_name, entered=False, pnl=0.0,
                exit_reason="STRIKE_ERROR", entry_credit=0.0, exit_value=0.0,
                strike_offset=strike_offset,
            )

        # strike_offset shifts the short strikes outward (+) or inward (-)
        csk = float(round(cs_raw)) + strike_offset
        psk = float(round(ps_raw)) - strike_offset
        clk = csk + eff_wing
        plk = psk - eff_wing

        # ── Entry credit check ──────────────────────────────────────────────
        entry_credit = self._condor_value(spot_e, T_e, iv_e, csk, psk, clk, plk)
        min_credit   = eff_wing * self.config.min_credit_pct

        if entry_credit < min_credit:
            return TradeResult(
                scenario_name=day.scenario_name, entered=False, pnl=0.0,
                exit_reason="NO_CREDIT",
                entry_credit=round(entry_credit, 4), exit_value=0.0,
                strike_offset=strike_offset,
            )

        # ── Build CondorPosition (OCC symbols are synthetic stubs) ──────────
        def _occ(call: bool, K: float) -> str:
            return f"SPY{d.strftime('%y%m%d')}{'C' if call else 'P'}{round(K * 1000):08d}"

        legs = CondorLegs(
            put_long_symbol=_occ(False, plk), put_short_symbol=_occ(False, psk),
            call_short_symbol=_occ(True, csk), call_long_symbol=_occ(True, clk),
            put_long_strike=plk, put_short_strike=psk,
            call_short_strike=csk, call_long_strike=clk,
            net_credit=entry_credit,
        )
        position = CondorPosition(
            legs=legs, order_id="synthetic",
            entry_time=entry_dt,
            entry_credit=entry_credit,
            current_value=entry_credit,
            underlying_price=spot_e,
        )

        # ── Monitoring loop ─────────────────────────────────────────────────
        # Save and restore the real get_current_vix so other code isn't affected
        _orig_vix = _strat_module.get_current_vix
        exit_reason = ExitDecision.CLOSE_TIME
        exit_value  = entry_credit

        try:
            t = entry_dt + timedelta(minutes=_POLL_MIN)
            # Run until one tick past force_close so monitor() sees 15:30+ and
            # returns CLOSE_TIME cleanly if no other trigger fires first.
            while t <= force_dt + timedelta(minutes=_POLL_MIN):
                spot = day.spot_at(t)
                iv   = day.iv_at(t)
                T    = self._T(t, expiry_dt)
                cv   = self._condor_value(spot, T, iv, csk, psk, clk, plk)

                position.current_value   = cv
                position.underlying_price = spot

                # Inject scenario IV so monitor's delta-breach uses the right vol
                _strat_module.get_current_vix = lambda _iv=iv: _iv * 100.0

                decision = self.strategy.monitor(position, now=t)
                if decision != ExitDecision.HOLD:
                    exit_reason = decision
                    exit_value  = cv
                    break

                t += timedelta(minutes=_POLL_MIN)

        finally:
            _strat_module.get_current_vix = _orig_vix

        pnl = round((entry_credit - exit_value) * 100.0, 2)  # per contract

        return TradeResult(
            scenario_name=day.scenario_name,
            entered=True,
            pnl=pnl,
            exit_reason=exit_reason.value,
            entry_credit=round(entry_credit, 4),
            exit_value=round(exit_value, 4),
            strike_offset=strike_offset,
        )

    def run_scenario(
        self,
        scenario: object,
        n: int = 100,
        d: Optional[date] = None,
        prior_close: float = BASE_SPOT,
    ) -> list[TradeResult]:
        """Run *scenario* *n* times with random ±1 strike offsets.

        Each call to ``scenario.generate_day`` produces a fresh path, allowing
        stochastic scenarios (e.g. FOMCSurprise's random direction) to vary
        across runs.  Deterministic scenarios get the same path each time, but
        strike offsets vary.

        Parameters
        ----------
        scenario:
            Any object with a ``generate_day(d, prior_close)`` method.
        n:
            Number of simulation runs.
        d:
            Date to use.  Defaults to a representative Thursday in 2026.
        prior_close:
            SPY prior-close price fed into the scenario.
        """
        if d is None:
            d = date(2026, 1, 15)   # Thursday, no known holiday
        results = []
        for _ in range(n):
            # Weight toward 0 offset: 25% chance of -1, 50% chance of 0, 25% of +1
            offset = random.choices([-1, 0, 1], weights=[1, 2, 1])[0]
            day    = scenario.generate_day(d, prior_close)
            results.append(self.run_one(day, strike_offset=offset))
        return results
