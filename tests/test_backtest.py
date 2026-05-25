"""
tests/test_backtest.py — Deterministic backtest replay tests.

Uses fully-synthetic SPY bar data and mocked Alpaca clients.
No network calls — results are deterministic given the synthetic bars.
"""

from __future__ import annotations

import math
from datetime import date, datetime, timedelta
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

import pytest

from alpaca_options.backtest.replay import BacktestEngine, BacktestResults
from alpaca_options.backtest._occ import occ_symbol
from alpaca_options.risk.manager import RiskManager
from alpaca_options.strategies.iron_condor_0dte import IronCondor0DTE, IronCondorConfig
from alpaca_options.utils.black_scholes import bs_price, strike_for_delta

ET = ZoneInfo("America/New_York")

# ---------------------------------------------------------------------------
# OCC symbol helper test
# ---------------------------------------------------------------------------

def test_occ_symbol_format():
    sym = occ_symbol("SPY", date(2025, 6, 2), True, 580.0)
    assert sym == "SPY250602C00580000"

def test_occ_symbol_put():
    sym = occ_symbol("SPY", date(2025, 6, 2), False, 560.0)
    assert sym == "SPY250602P00560000"

def test_occ_symbol_fractional_strike():
    sym = occ_symbol("SPY", date(2025, 12, 31), True, 123.456)
    # 123.456 × 1000 = 123456, padded to 8 digits
    assert sym == "SPY251231C00123456"


# ---------------------------------------------------------------------------
# Black-Scholes sanity checks (used by the engine)
# ---------------------------------------------------------------------------

def test_bs_price_call_positive():
    price = bs_price(S=580, K=585, T=1/252, r=0.05, sigma=0.20, is_call=True)
    assert price > 0

def test_bs_price_deep_itm_call():
    price = bs_price(S=600, K=500, T=1/252, r=0.05, sigma=0.20, is_call=True)
    assert price > 99.0

def test_bs_price_zero_time():
    price = bs_price(S=580, K=585, T=0, r=0.05, sigma=0.20, is_call=True)
    assert price == 0.0  # OTM at expiry

def test_strike_for_delta_call():
    K = strike_for_delta(S=580, T=6/252/6.5, r=0.05, sigma=0.20, delta=0.10, is_call=True)
    assert K > 580  # call with delta=0.10 must be OTM

def test_strike_for_delta_put():
    K = strike_for_delta(S=580, T=6/252/6.5, r=0.05, sigma=0.20, delta=-0.10, is_call=False)
    assert K < 580  # put with delta=-0.10 must be OTM


# ---------------------------------------------------------------------------
# Synthetic bar factory
# ---------------------------------------------------------------------------

def make_spy_bar(dt: datetime, price: float):
    """Create a minimal mock bar object."""
    bar = MagicMock()
    bar.timestamp = dt
    bar.close = price
    bar.open = price
    bar.high = price + 0.50
    bar.low = price - 0.50
    return bar


def make_spy_bars_flat(day: date, price: float, start_hour: int = 9, end_hour: int = 16):
    """Generate 1-minute SPY bars at constant *price* for a full trading day."""
    bars = []
    t = datetime(day.year, day.month, day.day, start_hour, 31, tzinfo=ET)
    end = datetime(day.year, day.month, day.day, end_hour, 0, tzinfo=ET)
    while t <= end:
        bars.append(make_spy_bar(t, price))
        t += timedelta(minutes=1)
    return bars


def make_spy_bars_trending(day: date, start_price: float, end_price: float):
    """Generate 1-minute bars with a linear drift from start to end price."""
    bars = []
    t = datetime(day.year, day.month, day.day, 9, 31, tzinfo=ET)
    end = datetime(day.year, day.month, day.day, 16, 0, tzinfo=ET)
    steps = int((end - t).total_seconds() / 60)
    for i in range(steps + 1):
        price = start_price + (end_price - start_price) * i / max(steps, 1)
        bars.append(make_spy_bar(t + timedelta(minutes=i), price))
    return bars


# ---------------------------------------------------------------------------
# Backtest engine fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_engine(monkeypatch):
    """BacktestEngine with all Alpaca calls mocked."""
    monkeypatch.setenv("ALPACA_API_KEY", "FAKE")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "FAKE")
    monkeypatch.setenv("ALPACA_PAPER", "true")

    trading = MagicMock()
    data    = MagicMock()

    # Calendar: return [day] for any get_calendar call
    def fake_calendar(req):
        cal = MagicMock()
        # Return dates between req.start and req.end
        from datetime import datetime as dt
        s = date.fromisoformat(str(req.start))
        e = date.fromisoformat(str(req.end))
        result = []
        d = s
        while d <= e:
            if d.weekday() < 5:
                c = MagicMock()
                c.date = d
                result.append(c)
            d += timedelta(days=1)
        return result

    trading.get_calendar.side_effect = fake_calendar

    # Option bars: always return empty (force BS fallback)
    data.get_option_bars.side_effect = Exception("No option data")

    with patch("alpaca_options.backtest.replay.get_clients", return_value=(trading, data)), \
         patch("alpaca_options.backtest.replay.BacktestEngine._make_stock_client", return_value=MagicMock()):
        engine = BacktestEngine(initial_equity=100_000, vix_override=18.0)
        engine._trading = trading
        engine._option_data = data
    return engine


# ---------------------------------------------------------------------------
# Flat day — condor should decay profitably
# ---------------------------------------------------------------------------

def test_flat_day_generates_profit(mock_engine, monkeypatch):
    """A flat SPY day should let the condor decay → profit."""
    day = date(2025, 6, 2)  # Monday
    spy_bars = make_spy_bars_flat(day, price=580.0)

    mock_engine._get_spy_bars = MagicMock(return_value=spy_bars)
    mock_engine._get_option_bars_1min = MagicMock(return_value=[])

    strategy = IronCondor0DTE(config=IronCondorConfig(
        short_delta=0.10,
        wing_width=5.0,
        min_credit_pct=0.08,
        profit_target_pct=0.50,
        stop_loss_multiplier=2.0,
    ))
    risk = RiskManager()

    results = mock_engine.run(day, day, strategy, risk)

    assert results.num_trades == 1, f"Expected 1 trade, got {results.num_trades}"
    assert results.total_pnl > 0, f"Expected profit on flat day, got {results.total_pnl:.2f}"
    assert results.win_rate == 1.0


# ---------------------------------------------------------------------------
# Large move — stop loss should trigger
# ---------------------------------------------------------------------------

def test_large_move_triggers_stop_loss(mock_engine, monkeypatch):
    """A big SPY move should blow through the short strike → stop loss."""
    day = date(2025, 6, 2)
    # SPY drops 3% intraday (well past the 0.10-delta short strike)
    spy_bars = make_spy_bars_trending(day, start_price=580.0, end_price=563.0)

    mock_engine._get_spy_bars = MagicMock(return_value=spy_bars)
    mock_engine._get_option_bars_1min = MagicMock(return_value=[])

    strategy = IronCondor0DTE(config=IronCondorConfig(
        short_delta=0.10,
        wing_width=5.0,
        min_credit_pct=0.08,
        profit_target_pct=0.50,
        stop_loss_multiplier=2.0,
    ))
    risk = RiskManager()

    results = mock_engine.run(day, day, strategy, risk)

    assert results.num_trades == 1
    # Stop loss hit → P&L negative
    assert results.total_pnl < 0, f"Expected loss on big move, got {results.total_pnl:.2f}"
    assert results.win_rate == 0.0


# ---------------------------------------------------------------------------
# Event day — filtered out
# ---------------------------------------------------------------------------

def test_event_day_is_filtered_out(mock_engine):
    """An FOMC day should be skipped by the risk manager."""
    fomc_day = date(2025, 3, 19)  # in our hardcoded list

    spy_bars = make_spy_bars_flat(fomc_day, price=580.0)
    mock_engine._get_spy_bars = MagicMock(return_value=spy_bars)
    mock_engine._get_option_bars_1min = MagicMock(return_value=[])

    strategy = IronCondor0DTE()
    risk     = RiskManager()

    results = mock_engine.run(fomc_day, fomc_day, strategy, risk)

    assert results.num_trades == 0
    assert results.num_filtered_out >= 1


# ---------------------------------------------------------------------------
# Multi-day run — equity curve length
# ---------------------------------------------------------------------------

def test_multi_day_equity_curve_length(mock_engine):
    """Equity curve should have one entry per trading day (incl. filtered)."""
    start = date(2025, 6, 2)   # Monday
    end   = date(2025, 6, 6)   # Friday → 5 trading days

    def _spy_bars(d):
        return make_spy_bars_flat(d, 580.0)

    mock_engine._get_spy_bars = MagicMock(side_effect=_spy_bars)
    mock_engine._get_option_bars_1min = MagicMock(return_value=[])

    results = mock_engine.run(start, end, IronCondor0DTE(), RiskManager())

    assert len(results.equity_curve) == 5


# ---------------------------------------------------------------------------
# BacktestResults structure
# ---------------------------------------------------------------------------

def test_backtest_results_fields(mock_engine):
    day = date(2025, 6, 2)
    mock_engine._get_spy_bars = MagicMock(return_value=make_spy_bars_flat(day, 580.0))
    mock_engine._get_option_bars_1min = MagicMock(return_value=[])

    results = mock_engine.run(day, day, IronCondor0DTE(), RiskManager())

    assert isinstance(results, BacktestResults)
    assert isinstance(results.total_pnl, float)
    assert isinstance(results.sharpe, float)
    assert 0.0 <= results.max_drawdown <= 1.0
    assert 0.0 <= results.win_rate <= 1.0
    assert isinstance(results.num_trades, int)
    assert isinstance(results.num_filtered_out, int)
    assert isinstance(results.equity_curve, list)
    assert results.initial_equity == 100_000.0


# ---------------------------------------------------------------------------
# Equity curve plot (no display, just file write)
# ---------------------------------------------------------------------------

def test_save_equity_curve_plot(tmp_path, mock_engine):
    day = date(2025, 6, 2)
    mock_engine._get_spy_bars = MagicMock(return_value=make_spy_bars_flat(day, 580.0))
    mock_engine._get_option_bars_1min = MagicMock(return_value=[])

    results = mock_engine.run(day, day, IronCondor0DTE(), RiskManager())
    plot_path = str(tmp_path / "test_curve.png")
    results.save_equity_curve_plot(plot_path)

    import os
    assert os.path.exists(plot_path), "Plot file not created."
    assert os.path.getsize(plot_path) > 1000, "Plot file seems empty."
