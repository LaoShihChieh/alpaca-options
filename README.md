# alpaca-options 🦙

A **paper-trading-only** Python toolkit for options strategies on [Alpaca Markets](https://alpaca.markets/),
built on [`alpaca-py`](https://github.com/alpacahq/alpaca-py).

---

## Prerequisites

| Tool | Version |
|------|---------|
| Python | ≥ 3.12 |
| [uv](https://docs.astral.sh/uv/) | ≥ 0.4 |
| Alpaca paper account | — |

---

## Setup

```bash
# 1. Clone / enter the repo
cd alpaca-options

# 2. Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 3. Copy your credentials into .env (already done if you cloned with it)
cat .env
# ALPACA_API_KEY=...
# ALPACA_SECRET_KEY=...
# ALPACA_PAPER=true

# 4. Install dependencies
uv sync
```

> **Paper-only guard**: the library asserts `ALPACA_PAPER=true` at client init time.
> The code will never run against a live account.

---

## Running the examples

### Buy ATM Call (single leg)

```bash
uv run python examples/buy_atm_call.py
```

### Bull Call Spread

```bash
uv run python examples/vertical_spread.py
```

### Risk Manager Demo (no API calls — pure logic)

```bash
uv run python examples/risk_manager_demo.py
```

### 90-Day Iron Condor Backtest

```bash
uv run python examples/backtest_90_days.py
# Saves equity_curve.png in the current directory
```

### Live Dry-Run (will ask for confirmation, no orders submitted)

```bash
uv run python examples/live_dry_run.py
```

### Tests

```bash
uv run pytest -v
```

---

## Iron Condor 0DTE Strategy

### Design overview

The strategy sells a put-spread below spot and a call-spread above spot on SPY,
both expiring **today** (0DTE).  Because SPY options expire every weekday, there
is always a 0DTE chain available.

```
                put_long  put_short   spot   call_short  call_long
Premium:           $0.40     $1.00    580      $1.00       $0.40
                   ←←← long wing ←←←       →→→ long wing →→→
Net credit = ($1.00 + $1.00) − ($0.40 + $0.40) = $1.20 per share
```

#### Entry rules (all must pass)

| # | Rule | Value |
|---|------|-------|
| 1 | Time window | 10:00–14:00 ET |
| 2 | VIX | < 25 |
| 3 | Not an event day | FOMC, CPI, NFP skipped |
| 4 | Min credit | ≥ 8% of wing width |
| 5 | Short-strike delta | ≈ 0.10 (Black-Scholes) |
| 6 | Wing width | $5 |

#### Exit rules (first triggered wins)

| Trigger | Condition |
|---------|-----------|
| Profit target | Spread cost ≤ 50% of opening credit |
| Stop loss | Spread cost ≥ 2× opening credit |
| Delta breach | Short-leg delta ≥ 0.25 |
| Time | 15:30 ET force-close |

---

## Why we backtest first — the EV math

A raw 0DTE iron condor wins roughly **75% of the time** (SPY stays inside the
short strikes).  But the losing 25% produce losses that are 2–5× the credit
collected:

```
Unfiltered EV ≈ 0.75 × $100 credit  −  0.25 × $300 avg loss  ≈  $0
```

That's break-even *before* commissions and slippage, which means negative EV in
practice.  The filters create edge:

1. **VIX filter** — skip the highest-vol days when options are expensive *and*
   moves are biggest (the regime where condors blow up most often).
2. **Event filter** — FOMC/CPI/NFP days have fat-tailed distributions that BS
   doesn't model.  Skipping them removes the worst outliers.
3. **Min-credit filter** — only trade when the market is pricing enough premium
   for the risk.  Low-credit condors have tiny rewards and full losses.
4. **Kill-switches** — daily-loss limit and 10% drawdown stops prevent a bad
   week from becoming a catastrophic month.

The backtest measures which combination of filter parameters actually produced
positive risk-adjusted returns over the test period.

> ⚠️  **Backtest optimism**: fills are computed at bar midpoints.
> Expect live results to be **10–20% worse** due to bid/ask slippage, queue
> priority at limit prices, and IV-model error (constant-IV BS ≠ real smile).

---

## Where the Edge Comes From

This strategy is short gamma and short vega. The base rate for that profile in retail options is negative: most people who sell premium blow up within a year. So the question isn't "does selling premium work" (it doesn't, on average). The question is "what specifically am I doing that flips the EV positive?"

Four things, in order of importance:

### 0. The VIX regime gate (the precondition that makes everything else possible)

The strategy only deploys when VIX is between 16 and 25. Backtest evidence shows that below 16 the available premium is insufficient to cover the directional risk of selling strikes close enough to ATM to even meet the minimum credit threshold. Above 25, the regime is too unstable to trade with confidence. The strategy is structurally a "rich vol harvester" and requires vol to be rich to function.

Per-bucket analysis (2024-01-19 → present): the VIX 16–20 bucket shows 100% win rate with $+150 total P&L; the VIX 20–25 bucket shows 100% win rate with $+71 total P&L. The VIX 12–16 bucket shows 45% win rate with $-193 total P&L across 11 trades — negative expected value, not rescued by higher delta.

Frequency: ~5–10 trades per year in current regimes. This is by design. Lower frequency is the cost of only trading when conditions actually favor the strategy.

### 1. The event calendar filter (largest source of edge)

On a non-event day, SPY 0DTE moves are roughly normal with a tight standard deviation — typically 0.5–0.8% intraday range. The strategy's 10-delta strikes are well outside this distribution, so the 75–85% theoretical win rate is realistic.

On an event day (FOMC, CPI, NFP), the distribution gets fat tails. A 2% move that's a 4-sigma event on a normal day is a 1-sigma event on an FOMC day. Selling 10-delta strangles into that distribution is a different game with a different — and worse — expected value.

Filtering ~13% of trading days (the event days) doesn't reduce returns by 13%. It reduces tail risk by far more, because event days carry disproportionate tail weight. This is the single biggest reason the strategy can be profitable.

### 2. The minimum credit threshold (second largest source)

On low-vol days, a 10-delta condor might pay $0.15 against a $5 wing width. That's a 32:1 risk/reward, requiring a ~97% win rate to break even. Not viable.

The 8% credit-to-width minimum filters these days out automatically. It self-adjusts to volatility regime: when VIX is low, premium is thin, and more days get filtered out. When VIX is high (but below the 25 regime cutoff), premium is fatter and more days qualify. The filter converts a static strategy into a regime-aware one without explicit regime logic.

### 3. Defined risk via the iron condor structure (the survival layer)

The first two filters create positive expected value. The condor structure ensures that EV gets a chance to play out across enough trades to converge.

A naked short strangle has unlimited risk on each side. One tail day where SPY moves 4% — rare but real: Aug 2024 yen carry unwind, March 2020 Covid drop, Feb 2018 Volmageddon — can wipe out months of accumulated credits. Even with positive EV, the strategy doesn't survive long enough to realize it.

The condor caps max loss at wing width minus credit. On a $5 wing collecting $0.40 credit, the worst possible loss is $4.60 per contract. Bounded. The strategy can be wrong on a tail day and still be standing the next morning to keep trading.

This is why iron condors typically have higher Sharpe ratios than naked strangles even though their per-trade expected return is lower. Survival compounds.

### What does NOT create edge

- **Strike selection within the 10-delta range.** Going from 0.08-delta to 0.12-delta shifts the win rate by a few percentage points but doesn't fundamentally change the EV math.
- **Exit timing tweaks** (50% profit target vs 40% vs 60%). These optimize within a regime but don't make a losing strategy winning.
- **Backtesting on benign periods.** A backtest showing a 100% win rate during a low-vol trending market tells you nothing about live performance.

### How the edge could disappear

- **Calendar source breaks** (FRED outage, hardcoded dates go stale): events slip through the filter, and the strategy trades into known volatility shocks.
- **VIX regime cutoff drifts.** 25 was right historically; it may not be right forever.
- **Market microstructure shifts.** If option market makers tighten spreads, the gap between mid-price and actual fill narrows, eroding the credit assumption.
- **Behavioral override.** Trader skips a filter on a "feels safe" day and defeats the whole point.

**Quarterly review checklist:**

- Verify FRED calendar matches BLS/Fed published schedule
- Re-check VIX regime cutoff against recent realized vol
- Compare backtest fills to live fills; recalibrate if slippage exceeds 20%

---

## Backtest and Stress Test Results

This strategy has not been tested on real options data covering tail events.
Everything below is either **(a)** a synthetic backtest using Black-Scholes pricing
or **(b)** constructed stress scenarios.
Both are systematically optimistic compared to live trading.
Treat results as **ranking tools** (which configurations are better than others)
rather than **predictions** (this is what live returns will be).

### Synthetic Backtest (2026-02-21 to 2026-05-22)

64 trading days, low-vol trending regime, no real tail events in window.

| Metric | Value | Interpretation |
|---|---|---|
| Trades entered | 57 (7 filtered) | 11% filter rate — low because no FOMC clustering in window |
| Total P&L | +$2,418 on $100k | +2.4% over 3 months |
| Sharpe | 71.1 | Not real. Constant IV + benign window = no losing days. |
| Win rate | 100% | Same caveat |
| Max drawdown | 0% | Same caveat |

**What this tells us:** the modules wire together, filters reject the right days,
exits trigger on the right conditions.
Nothing about whether the strategy makes money in real markets.

### Stress Test Scenarios (100 iterations each, except RegimeShift)

All numbers from `stress_test_results.json` (seed=42, SPY≈$580, BASE\_IV=18%).
P&L is per 1-contract position (×$100), excluding commissions and slippage.

| Scenario | Entered | Win % | Mean P&L | Worst P&L | Primary exit |
|---|---|---|---|---|---|
| GapDownDay | 100/100 | 0% | −$16 | −$26 | Delta breach (100%) |
| GapUpDay | 100/100 | 0% | −$16 | −$25 | Delta breach (100%) |
| VolSpikeIntraday | 74/100 | 100% | +$30 | +$27 (worst win) | Profit target (100%) |
| WhipsawDay | 100/100 | 0% | −$43 | −$55 | Delta breach (100%) |
| FOMCSurprise (filter bypass) | 73/100 | 0% | −$201 | −$203 | Stop loss (100%) |
| SteadyTrendUp | 82/100 | 100% | +$29 | +$26 (worst win) | Profit target (100%) |
| SteadyTrendDown | 71/100 | 100% | +$29 | +$27 (worst win) | Profit target (100%) |
| LowVIXShock | 72/100 | 72% | +$9 | −$72 | Mixed (72% profit, 18% delta, 10% stop) |
| RegimeShift2022 (60 days) | 24/60 days | 67% | +$9/trade | −$61 | Mixed; total +$226, max DD −$92 |

### What the stress tests confirmed

**Defined risk works.** Worst observed loss across all gap and whipsaw scenarios is −$55,
against a theoretical maximum loss of −$460 per contract (full wing width × 100 shares).
The delta-breach exit fires early enough to prevent full-wing losses on routine bad days.

**Filters are the primary defense.** The FOMCSurprise scenario simulates what happens
when the calendar filter is bypassed: −$201 average loss, roughly 7× a winning day's
credit collected. Every one of these losses represents a calendar failure — FRED outage,
hardcoded date drift, or a genuinely unscheduled announcement.
The math of the strategy assumes these days are skipped.

**Unfiltered low-vol shocks cost roughly 2.5× a winning day's credit on average.**
LowVIXShock worst case is −$72. This is the residual tail risk you cannot filter out:
surprise geopolitical headlines, unannounced policy moves, flash crashes during calm VIX
regimes. Expect 5–10 of these per year.

**High-vol regimes get filtered, not survived.**
RegimeShift2022 traded only 24 of 60 days (33 VIX-blocked, 3 no-credit).
The VIX filter removed the strategy from the market for most of a 2022-style regime,
leaving modest gains (+$226 total, −$92 max drawdown) on the limited days that did trade.
The strategy doesn't survive 2022 by being clever during 2022;
it survives by mostly not trading.

### What the stress tests did NOT confirm

- **Pre-2024 events.** Alpaca's historical options bars start January 2024.
  The strategy cannot be tested against the COVID crash (March 2020),
  Volmageddon (Feb 2018), the 2022 hiking cycle, the Aug 2024 yen unwind
  (SPY 0DTE options did not exist yet in 2018), or any other event before
  Jan 2024. The stress test scenarios are the only available proxy for these.
- **Real fill quality.** Synthetic backtests assume mid-price fills. Live multi-leg
  fills are worse, especially during fast moves.
- **Real IV behaviour under stress.** Black-Scholes uses static IV inputs.
  Real options markets see IV smile changes during shocks that affect both legs of the
  condor asymmetrically.
- **Statistical significance.** 100 synthetic iterations of an idealized scenario is
  not the same as 100 real instances of that scenario. Real instances differ from each
  other in ways the simulator does not capture.
- **Behavioural robustness.** The simulator follows every rule perfectly. Live trading
  involves judgment calls, hesitation, overrides, and emotional pressure that the
  tests do not model.

### Rough annual expectation (not a forecast)

Combining base rates with scenario outcomes:

| Day type | Estimated frequency | EV per trade |
|---|---|---|
| Calm clean day | ~70% | +$22 |
| Filtered (FOMC / CPI / NFP / VIX) | ~13% | $0 |
| Filtered (low credit) | ~10% | $0 |
| Low-vol shock (residual) | ~5% | +$1 |
| Filter slip-through | ~2% | −$150 |

Synthetic annual estimate: **+$3,100 per contract** on a maximum loss of $460.
Realistic live range — accounting for slippage, missed fills, and behavioural drag —
is probably **30–50% lower**.
Volatile years (2022-style) could be **flat to mildly negative**.

### Required next steps before any live capital

1. Paper dry-run for at least 3 months. Log every decision. Compare actual fills to
   synthetic fills to estimate slippage.
2. Re-validate the calendar quarterly against FRED, BLS, and Fed sources.
3. Recompute the VIX regime cutoff against the trailing 6 months of realized vol.
4. Review this section after every drawdown > 10%. If a real event behaves
   differently than the corresponding stress scenario, update the scenario.

---

## Risk controls

> These controls protect against the strategy's failure modes (see [Where the Edge Comes From](#where-the-edge-comes-from) above for what those failure modes are).

| Control | Default | What it does |
|---------|---------|--------------|
| `vix_threshold` | 25 | No entry when VIX ≥ 25 |
| `max_drawdown_pct` | 10% | Kill-switch: stop trading if portfolio drops > 10% from peak |
| `max_daily_loss_multiplier` | 2× | Stop after losing 2 × `max_loss_per_trade` in one day |
| `max_loss_per_trade` | $500 | Sets the daily loss limit scale |
| `max_concurrent_positions` | 1 | Only 1 condor open at a time |
| Calendar filter | FOMC/CPI/NFP | Hardcoded 2025-2026 fallback; FRED API if key provided |

**`ALPACA_PAPER=true` is asserted at client init** — the library will refuse to
run against a live account.  Both example scripts ask for confirmation before
submitting any order.

---

## Module reference

### `alpaca_options.client`

| Function | Description |
|----------|-------------|
| `get_clients() → (TradingClient, OptionHistoricalDataClient)` | Returns a cached pair of paper-mode Alpaca clients. Asserts `ALPACA_PAPER=true`. |

---

### `alpaca_options.contracts`

| Function | Description |
|----------|-------------|
| `get_option_contracts(underlying_symbol, expiration_gte, expiration_lte, strike_gte, strike_lte, contract_type, limit) → List[OptionContract]` | Filter and return option contracts. |
| `find_atm_call(underlying_symbol, expiration, underlying_price, strike_window) → OptionContract` | Return the call nearest to ATM. |
| `find_atm_put(underlying_symbol, expiration, underlying_price, strike_window) → OptionContract` | Return the put nearest to ATM. |

---

### `alpaca_options.quotes`

| Function | Description |
|----------|-------------|
| `get_latest_quote(symbol) → Quote` | Fetch current NBBO for an OCC option symbol. |
| `get_option_bars(symbol, timeframe, start, end, limit) → List[Bar]` | Fetch OHLCV bars. |
| `midpoint(quote) → float` | Bid/ask midpoint, handles zero-side quotes. |

---

### `alpaca_options.orders`

| Function | Description |
|----------|-------------|
| `buy_to_open_market(symbol, qty) → Order` | Market BTO. |
| `buy_to_open_limit(symbol, qty, limit_price) → Order` | Limit BTO. |
| `sell_to_close_market(symbol, qty) → Order` | Market STC. |
| `sell_to_close_limit(symbol, qty, limit_price) → Order` | Limit STC. |
| `bull_call_spread(long_symbol, short_symbol, qty, net_debit) → Order` | MLEG bull call spread. |
| `straddle(call_symbol, put_symbol, qty, net_debit) → Order` | MLEG long straddle. |

---

### `alpaca_options.positions`

| Function | Description |
|----------|-------------|
| `list_option_positions() → pd.DataFrame` | Open option positions with P&L columns. |
| `close_all_options(cancel_orders) → List[ClosePositionResponse]` | Close all positions. |

---

### `alpaca_options.data.calendar`

| Function | Description |
|----------|-------------|
| `get_event_days(start, end) → set[date]` | FOMC + CPI + NFP dates in range. Uses FRED API if `FRED_API_KEY` env var set; hardcoded fallback otherwise. |

---

### `alpaca_options.data.vix`

| Function | Description |
|----------|-------------|
| `get_current_vix() → float` | Current VIX via Alpaca VIXY proxy or yfinance fallback. |
| `is_high_vol_regime(vix, threshold) → bool` | Returns True if VIX ≥ threshold (default 25). |

---

### `alpaca_options.risk.manager`

| Method | Description |
|--------|-------------|
| `RiskManager(vix_threshold, max_daily_loss_multiplier, max_drawdown_pct, max_concurrent_positions, max_loss_per_trade)` | Constructor — all params have sensible defaults. |
| `reset_day(account_value)` | Reset intraday counters at session start. |
| `check_entry_allowed(account_value, vix, today, calendar_events) → (bool, str)` | Check all 5 guard rails. Returns `(True, "OK")` or `(False, reason)`. |
| `open_position()` | Increment open position counter. |
| `record_trade_result(pnl)` | Update state after trade closes. |
| `summary() → dict` | Snapshot of risk state. |

---

### `alpaca_options.strategies.iron_condor_0dte`

| Item | Description |
|------|-------------|
| `IronCondorConfig` | Dataclass with all strategy parameters. |
| `CondorLegs` | Frozen dataclass with 4 OCC symbols and strikes. |
| `CondorPosition` | Mutable live position (current_value updated each tick). |
| `ExitDecision` | Enum: HOLD / CLOSE_PROFIT / CLOSE_STOP / CLOSE_DELTA_BREACH / CLOSE_TIME. |
| `IronCondor0DTE.should_enter(now, vix) → bool` | Time-window check (10:00–14:00 ET). |
| `IronCondor0DTE.build_condor(underlying) → CondorLegs \| None` | Fetch chain, compute delta strikes, check credit, return legs or None. |
| `IronCondor0DTE.enter(legs) → order_id` | Submit MLEG limit order. |
| `IronCondor0DTE.monitor(position, now) → ExitDecision` | Check profit/stop/delta/time conditions. |
| `IronCondor0DTE.exit(position, reason) → order_id` | Submit MLEG market close. |

---

### `alpaca_options.backtest.replay`

| Item | Description |
|------|-------------|
| `BacktestEngine(initial_equity, vix_override)` | Constructor. |
| `BacktestEngine.run(start, end, strategy, risk) → BacktestResults` | Simulate strategy over date range. Fetches SPY 1-min bars; uses real option bars when available, Black-Scholes otherwise. |
| `BacktestResults.save_equity_curve_plot(path)` | Save PNG with equity curve + daily P&L. |

---

### `alpaca_options.live.runner`

| Item | Description |
|------|-------------|
| `LiveRunner(dry_run, strategy, risk, poll_interval)` | Constructor. `dry_run=True` by default. |
| `LiveRunner.run()` | Main loop (blocks). Polls every 60 s during market hours. Graceful SIGINT. |

---

### `alpaca_options.utils.black_scholes`

| Function | Description |
|----------|-------------|
| `bs_price(S, K, T, r, sigma, is_call) → float` | Black-Scholes theoretical option price. |
| `bs_delta(S, K, T, r, sigma, is_call) → float` | Black-Scholes delta. |
| `strike_for_delta(S, T, r, sigma, delta, is_call) → float` | Closed-form strike for a target delta. |
| `norm_cdf(x) → float` | Standard normal CDF. |
| `norm_ppf(p) → float` | Standard normal inverse CDF (no scipy needed). |

---

## Project layout

```
alpaca-options/
├── .env                             # API credentials (never commit!)
├── pyproject.toml
├── src/
│   └── alpaca_options/
│       ├── __init__.py
│       ├── client.py                # Client factory (paper=True guard)
│       ├── contracts.py             # Contract discovery + ATM helpers
│       ├── quotes.py                # Latest quotes + historical bars
│       ├── orders.py                # Single-leg + MLEG order submission
│       ├── positions.py             # Position listing + P&L DataFrame
│       ├── utils/
│       │   └── black_scholes.py     # BS pricing, delta, strike-for-delta
│       ├── data/
│       │   ├── calendar.py          # FOMC/CPI/NFP event calendar
│       │   └── vix.py               # VIX fetching + regime detection
│       ├── risk/
│       │   └── manager.py           # RiskManager — 5-rule entry gate
│       ├── strategies/
│       │   └── iron_condor_0dte.py  # IronCondor0DTE strategy class
│       ├── backtest/
│       │   ├── replay.py            # BacktestEngine + BacktestResults
│       │   └── _occ.py              # OCC symbol builder
│       └── live/
│           └── runner.py            # LiveRunner (dry-run + live)
├── examples/
│   ├── buy_atm_call.py
│   ├── vertical_spread.py
│   ├── risk_manager_demo.py
│   ├── backtest_90_days.py
│   └── live_dry_run.py
└── tests/
    ├── conftest.py
    ├── test_risk_manager.py
    ├── test_iron_condor.py
    └── test_backtest.py
```

---

## Logging

```python
import logging
logging.getLogger("alpaca_options").setLevel(logging.INFO)
```

---

## FRED API key (optional, improves calendar accuracy)

```bash
# Get a free key at https://fred.stlouisfed.org/docs/api/api_key.html
# Add to .env:
FRED_API_KEY=your_key_here
```

---

## Safety reminders

- `ALPACA_PAPER=true` is **asserted** at client init.
- Both example scripts **ask for confirmation** before submitting any order.
- The backtest uses bar midpoints — live results will be worse.
- Options trading involves significant risk, even in simulation.
