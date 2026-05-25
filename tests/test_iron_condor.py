"""
tests/test_iron_condor.py — Unit tests for IronCondor0DTE strategy.

Tests cover:
- should_enter() time-window logic
- monitor() exit trigger conditions
- build_condor() returns None when credit is too low
- ExitDecision enum values
"""

from __future__ import annotations

from datetime import date, datetime
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

import pytest

from alpaca_options.strategies.iron_condor_0dte import (
    CondorLegs,
    CondorPosition,
    ExitDecision,
    IronCondor0DTE,
    IronCondorConfig,
    adaptive_wing_width,
    target_delta,
)

ET = ZoneInfo("America/New_York")

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def make_legs(credit: float = 1.00) -> CondorLegs:
    return CondorLegs(
        put_long_symbol="SPY250602P00555000",
        put_short_symbol="SPY250602P00560000",
        call_short_symbol="SPY250602C00580000",
        call_long_symbol="SPY250602C00585000",
        put_long_strike=555.0,
        put_short_strike=560.0,
        call_short_strike=580.0,
        call_long_strike=585.0,
        net_credit=credit,
    )


def make_position(entry_credit: float = 1.00, current_value: float = 1.00,
                  underlying: float = 570.0, entry_hour: int = 10) -> CondorPosition:
    return CondorPosition(
        legs=make_legs(entry_credit),
        order_id="test-order",
        entry_time=datetime(2025, 6, 2, entry_hour, 0, tzinfo=ET),
        entry_credit=entry_credit,
        current_value=current_value,
        underlying_price=underlying,
    )


@pytest.fixture
def strategy(monkeypatch):
    """IronCondor0DTE with get_clients mocked."""
    trading = MagicMock()
    data    = MagicMock()
    monkeypatch.setenv("ALPACA_API_KEY", "FAKE")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "FAKE")
    monkeypatch.setenv("ALPACA_PAPER", "true")
    with patch("alpaca_options.strategies.iron_condor_0dte.get_clients", return_value=(trading, data)):
        s = IronCondor0DTE(config=IronCondorConfig(
            short_delta=0.10,
            wing_width=5.0,
            min_credit_pct=0.08,
            profit_target_pct=0.50,
            stop_loss_multiplier=2.0,
            max_short_delta_breach=0.25,
        ))
    return s


# ---------------------------------------------------------------------------
# should_enter — time window
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("hour,minute,expected", [
    (9,  59, False),   # before window
    (10, 0,  True),    # window opens exactly at 10:00
    (12, 0,  True),    # mid-window
    (14, 0,  True),    # window closes at 14:00
    (14, 1,  False),   # past window
    (15, 30, False),   # force-close time
])
def test_should_enter_time_window(strategy, hour, minute, expected):
    now = datetime(2025, 6, 2, hour, minute, tzinfo=ET)
    result = strategy.should_enter(now, vix=18.0)  # VIX=18 clears the floor
    assert result == expected, f"should_enter({hour:02d}:{minute:02d}) = {result}, want {expected}"


# ---------------------------------------------------------------------------
# should_enter — VIX floor (min_vix=16.0 default)
# ---------------------------------------------------------------------------

def test_should_enter_blocked_below_min_vix(monkeypatch):
    """VIX below min_vix must block regardless of time."""
    monkeypatch.setenv("ALPACA_API_KEY", "FAKE")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "FAKE")
    monkeypatch.setenv("ALPACA_PAPER", "true")
    with patch("alpaca_options.strategies.iron_condor_0dte.get_clients",
               return_value=(MagicMock(), MagicMock())):
        s = IronCondor0DTE(config=IronCondorConfig(min_vix=16.0))
    now = datetime(2025, 6, 2, 11, 0, tzinfo=ET)   # mid-window
    assert s.should_enter(now, vix=15.9) is False
    assert s.should_enter(now, vix=12.0) is False


def test_should_enter_allowed_at_min_vix(monkeypatch):
    """VIX exactly at min_vix is allowed (strict < check)."""
    monkeypatch.setenv("ALPACA_API_KEY", "FAKE")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "FAKE")
    monkeypatch.setenv("ALPACA_PAPER", "true")
    with patch("alpaca_options.strategies.iron_condor_0dte.get_clients",
               return_value=(MagicMock(), MagicMock())):
        s = IronCondor0DTE(config=IronCondorConfig(min_vix=16.0))
    now = datetime(2025, 6, 2, 11, 0, tzinfo=ET)
    assert s.should_enter(now, vix=16.0) is True


def test_should_enter_vix_floor_zero_disables_check(monkeypatch):
    """min_vix=0 means the floor check never fires."""
    monkeypatch.setenv("ALPACA_API_KEY", "FAKE")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "FAKE")
    monkeypatch.setenv("ALPACA_PAPER", "true")
    with patch("alpaca_options.strategies.iron_condor_0dte.get_clients",
               return_value=(MagicMock(), MagicMock())):
        s = IronCondor0DTE(config=IronCondorConfig(min_vix=0.0))
    now = datetime(2025, 6, 2, 11, 0, tzinfo=ET)
    # Even VIX=5 passes when floor is 0
    assert s.should_enter(now, vix=5.0) is True


# ---------------------------------------------------------------------------
# monitor — profit target
# ---------------------------------------------------------------------------

def test_close_profit_at_50_pct(strategy):
    """Close when spread cost drops to ≤50% of initial credit."""
    pos = make_position(entry_credit=1.00, current_value=0.50)
    now = datetime(2025, 6, 2, 12, 0, tzinfo=ET)
    with patch("alpaca_options.strategies.iron_condor_0dte.get_current_vix", return_value=18.0), \
         patch("alpaca_options.data.vix.get_current_vix", return_value=18.0):
        decision = strategy.monitor(pos, now=now)
    assert decision == ExitDecision.CLOSE_PROFIT


def test_close_profit_just_below_threshold(strategy):
    """current_value = 0.49 < 0.50 → CLOSE_PROFIT."""
    pos = make_position(entry_credit=1.00, current_value=0.49)
    now = datetime(2025, 6, 2, 12, 0, tzinfo=ET)
    with patch("alpaca_options.strategies.iron_condor_0dte.get_current_vix", return_value=18.0), \
         patch("alpaca_options.data.vix.get_current_vix", return_value=18.0):
        decision = strategy.monitor(pos, now=now)
    assert decision == ExitDecision.CLOSE_PROFIT


def test_hold_when_cost_above_profit_target(strategy):
    """current_value = 0.60 > 0.50 → should not trigger profit target."""
    pos = make_position(entry_credit=1.00, current_value=0.60)
    now = datetime(2025, 6, 2, 12, 0, tzinfo=ET)
    with patch("alpaca_options.strategies.iron_condor_0dte.get_current_vix", return_value=18.0), \
         patch("alpaca_options.data.vix.get_current_vix", return_value=18.0):
        decision = strategy.monitor(pos, now=now)
    # Should be HOLD (assuming underlying is safely between strikes, delta < 0.25)
    assert decision in (ExitDecision.HOLD, ExitDecision.CLOSE_DELTA_BREACH)


# ---------------------------------------------------------------------------
# monitor — stop loss
# ---------------------------------------------------------------------------

def test_close_stop_at_2x_credit(strategy):
    """Spread cost = 2× initial credit → CLOSE_STOP."""
    pos = make_position(entry_credit=1.00, current_value=2.00)
    now = datetime(2025, 6, 2, 12, 0, tzinfo=ET)
    with patch("alpaca_options.strategies.iron_condor_0dte.get_current_vix", return_value=18.0), \
         patch("alpaca_options.data.vix.get_current_vix", return_value=18.0):
        decision = strategy.monitor(pos, now=now)
    assert decision == ExitDecision.CLOSE_STOP


def test_close_stop_exceeds_multiplier(strategy):
    """cost = 2.5× → also CLOSE_STOP."""
    pos = make_position(entry_credit=1.00, current_value=2.50)
    now = datetime(2025, 6, 2, 12, 0, tzinfo=ET)
    with patch("alpaca_options.strategies.iron_condor_0dte.get_current_vix", return_value=18.0), \
         patch("alpaca_options.data.vix.get_current_vix", return_value=18.0):
        decision = strategy.monitor(pos, now=now)
    assert decision == ExitDecision.CLOSE_STOP


def test_stop_not_triggered_below_multiplier(strategy):
    """cost = 1.99 < 2× → should not trigger stop."""
    pos = make_position(entry_credit=1.00, current_value=1.99, underlying=570.0)
    now = datetime(2025, 6, 2, 12, 0, tzinfo=ET)
    with patch("alpaca_options.strategies.iron_condor_0dte.get_current_vix", return_value=18.0), \
         patch("alpaca_options.data.vix.get_current_vix", return_value=18.0):
        decision = strategy.monitor(pos, now=now)
    assert decision in (ExitDecision.HOLD, ExitDecision.CLOSE_DELTA_BREACH)


# ---------------------------------------------------------------------------
# monitor — time-based close
# ---------------------------------------------------------------------------

def test_close_time_at_1530(strategy):
    """Past 15:30 ET → CLOSE_TIME regardless of P&L."""
    pos = make_position(entry_credit=1.00, current_value=0.80)  # not at profit/stop
    now = datetime(2025, 6, 2, 15, 30, tzinfo=ET)
    with patch("alpaca_options.strategies.iron_condor_0dte.get_current_vix", return_value=18.0), \
         patch("alpaca_options.data.vix.get_current_vix", return_value=18.0):
        decision = strategy.monitor(pos, now=now)
    assert decision == ExitDecision.CLOSE_TIME


def test_close_time_after_1530(strategy):
    pos = make_position(entry_credit=1.00, current_value=0.80)
    now = datetime(2025, 6, 2, 15, 45, tzinfo=ET)
    with patch("alpaca_options.strategies.iron_condor_0dte.get_current_vix", return_value=18.0), \
         patch("alpaca_options.data.vix.get_current_vix", return_value=18.0):
        decision = strategy.monitor(pos, now=now)
    assert decision == ExitDecision.CLOSE_TIME


# ---------------------------------------------------------------------------
# monitor — priority order (time > profit > stop > delta)
# ---------------------------------------------------------------------------

def test_time_beats_profit_target(strategy):
    """At 15:30 with profitable position → CLOSE_TIME (not CLOSE_PROFIT)."""
    pos = make_position(entry_credit=1.00, current_value=0.10)  # very profitable
    now = datetime(2025, 6, 2, 15, 30, tzinfo=ET)
    with patch("alpaca_options.strategies.iron_condor_0dte.get_current_vix", return_value=18.0), \
         patch("alpaca_options.data.vix.get_current_vix", return_value=18.0):
        decision = strategy.monitor(pos, now=now)
    assert decision == ExitDecision.CLOSE_TIME


# ---------------------------------------------------------------------------
# build_condor — returns None when credit too low (mocked quotes)
# ---------------------------------------------------------------------------

def test_build_condor_returns_none_when_credit_too_low(monkeypatch):
    """
    Mock the chain, quote, and underlying-price lookups so that
    the computed net credit is $0.00 < min_credit ($0.40).
    """
    monkeypatch.setenv("ALPACA_API_KEY", "FAKE")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "FAKE")
    monkeypatch.setenv("ALPACA_PAPER", "true")

    trading = MagicMock()
    data    = MagicMock()

    # Fake contract object
    def make_contract(sym, strike):
        c = MagicMock()
        c.symbol = sym
        c.strike_price = str(strike)
        return c

    fake_contracts = [
        make_contract("SPY250602P00555000", 555),
        make_contract("SPY250602P00560000", 560),
        make_contract("SPY250602C00580000", 580),
        make_contract("SPY250602C00585000", 585),
    ]

    # Fake quote returning ~$0 bid and ask → net credit ≈ 0
    fake_quote = MagicMock()
    fake_quote.bid_price = 0.01
    fake_quote.ask_price = 0.01

    with patch("alpaca_options.strategies.iron_condor_0dte.get_clients", return_value=(trading, data)), \
         patch("alpaca_options.strategies.iron_condor_0dte.get_option_contracts", return_value=fake_contracts), \
         patch("alpaca_options.strategies.iron_condor_0dte.get_latest_quote", return_value=fake_quote), \
         patch("alpaca_options.strategies.iron_condor_0dte.get_current_vix", return_value=18.0), \
         patch.object(IronCondor0DTE, "_get_underlying_price", return_value=570.0):

        strategy = IronCondor0DTE(config=IronCondorConfig(
            wing_width=5.0,
            min_credit_pct=0.08,   # min credit = 5 × 0.08 = $0.40
        ))
        result = strategy.build_condor("SPY")

    # net credit ≈ 0.01 + 0.01 - 0.01 - 0.01 = 0 < $0.40
    assert result is None


def test_build_condor_returns_legs_when_credit_sufficient(monkeypatch):
    """With generous quotes and _fetch_contract stubbed, build_condor returns CondorLegs."""
    monkeypatch.setenv("ALPACA_API_KEY", "FAKE")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "FAKE")
    monkeypatch.setenv("ALPACA_PAPER", "true")

    trading = MagicMock()
    data    = MagicMock()

    # _fetch_contract called with (underlying, exp, strike, is_call).
    # Return a contract with the exact requested strike so downstream logic works.
    def fake_fetch_contract(self_unused, underlying, exp, strike, is_call):
        c = MagicMock()
        suffix = "C" if is_call else "P"
        c.symbol = f"SPY250602{suffix}{round(strike * 1000):08d}"
        c.strike_price = str(strike)
        return c

    # Short legs pay $1.00 each mid, long legs $0.40 mid → net credit ≈ $1.20
    def fake_quote_factory(sym):
        q = MagicMock()
        # Symbols for short legs contain the rounded short strikes; everything
        # else is a long leg. We check for "C" vs "P" and wing membership
        # by counting that short strikes have smaller OTM distance.
        # Simplest: assign by position — if mid strike, use short price.
        q.bid_price = 0.95
        q.ask_price = 1.05
        # long legs (call_long and put_long) will be fetched last; override at call-site
        return q

    # Track calls so we can distinguish short vs long legs
    call_count = [0]
    def fake_quote_with_index(sym):
        q = MagicMock()
        call_count[0] += 1
        if call_count[0] in (1, 2):  # first two: short legs (sell)
            q.bid_price = 0.95
            q.ask_price = 1.05
        else:                         # last two: long legs (buy)
            q.bid_price = 0.35
            q.ask_price = 0.45
        return q

    with patch("alpaca_options.strategies.iron_condor_0dte.get_clients", return_value=(trading, data)), \
         patch("alpaca_options.strategies.iron_condor_0dte.get_current_vix", return_value=18.0), \
         patch("alpaca_options.strategies.iron_condor_0dte.get_latest_quote", side_effect=fake_quote_with_index), \
         patch.object(IronCondor0DTE, "_get_underlying_price", return_value=570.0), \
         patch.object(IronCondor0DTE, "_hours_remaining", return_value=6.0), \
         patch.object(IronCondor0DTE, "_fetch_contract", fake_fetch_contract):

        strategy = IronCondor0DTE(config=IronCondorConfig(wing_width=5.0, min_credit_pct=0.08))
        result = strategy.build_condor("SPY")

    assert result is not None, "Expected CondorLegs but got None (credit check may be failing)"
    assert isinstance(result, CondorLegs)
    # short pair pays ~$1.00 each, long pair costs ~$0.40 each → net ≈ $1.20 > $0.40 min
    assert result.net_credit > 0.40


# ---------------------------------------------------------------------------
# target_delta — VIX-adaptive short-leg delta
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("vix,expected", [
    # Boundary: vix exactly 12 → lowest bucket
    (12.0,  0.20),
    # Interior of lowest bucket
    (10.0,  0.20),
    # Boundary: vix exactly 16 → second bucket
    (16.0,  0.15),
    # Interior: between 12 and 16
    (14.0,  0.15),
    # Boundary: vix exactly 20 → third bucket
    (20.0,  0.12),
    # Interior: between 16 and 20
    (18.0,  0.12),
    # Above 20 → fallthrough bucket (strategy gate allows up to VIX=25)
    (25.0,  0.10),
    (22.5,  0.10),
])
def test_target_delta(vix, expected):
    result = target_delta(vix)
    assert result == expected, (
        f"target_delta({vix}) = {result}, want {expected}"
    )


# ---------------------------------------------------------------------------
# adaptive_wing_width — VIX-adaptive wing width
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("vix,expected", [
    # Boundary: vix exactly 14 → narrow wing
    (14.0,  3.0),
    # Interior: below 14
    (10.0,  3.0),
    # Boundary: vix exactly 20 → medium wing
    (20.0,  5.0),
    # Interior: between 14 and 20
    (17.0,  5.0),
    (18.0,  5.0),
    # Above 20 → wide wing
    (21.0,  7.0),
    (25.0,  7.0),
])
def test_adaptive_wing_width(vix, expected):
    result = adaptive_wing_width(vix)
    assert result == expected, (
        f"adaptive_wing_width({vix}) = {result}, want {expected}"
    )


# ---------------------------------------------------------------------------
# ExitDecision enum
# ---------------------------------------------------------------------------

def test_exit_decision_enum_values():
    assert ExitDecision.HOLD.value == "hold"
    assert ExitDecision.CLOSE_PROFIT.value == "close_profit"
    assert ExitDecision.CLOSE_STOP.value == "close_stop"
    assert ExitDecision.CLOSE_DELTA_BREACH.value == "close_delta_breach"
    assert ExitDecision.CLOSE_TIME.value == "close_time"
