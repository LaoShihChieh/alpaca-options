"""
tests/test_risk_manager.py — Unit tests for RiskManager.

Each test exercises one specific gate rule in isolation.
All tests are pure in-memory — no network calls.
"""

from __future__ import annotations

from datetime import date

import pytest

from alpaca_options.risk.manager import RiskManager


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def rm():
    """Fresh RiskManager with predictable defaults."""
    mgr = RiskManager(
        vix_threshold=25.0,
        max_daily_loss_multiplier=2.0,
        max_drawdown_pct=0.10,
        max_concurrent_positions=1,
        max_loss_per_trade=500.0,
    )
    mgr.peak_equity = 100_000.0
    mgr.reset_day(100_000.0)
    return mgr


SAFE_DATE = date(2025, 3, 1)   # Not an event day
SAFE_VIX  = 18.0
SAFE_EQUITY = 100_000.0


# ---------------------------------------------------------------------------
# Rule a: VIX regime — floor (min_vix) and ceiling (vix_threshold)
# ---------------------------------------------------------------------------

def test_vix_floor_blocks_entry_below_min(rm):
    """VIX below min_vix (16) must be blocked."""
    allowed, reason = rm.check_entry_allowed(
        SAFE_EQUITY, vix=15.9, today=SAFE_DATE, calendar_events=set()
    )
    assert not allowed
    assert reason == "vix_too_low"


def test_vix_at_floor_is_blocked(rm):
    """VIX exactly at min_vix (< not ≤) — boundary is exclusive, so 15.999 blocks."""
    allowed, reason = rm.check_entry_allowed(
        SAFE_EQUITY, vix=15.999, today=SAFE_DATE, calendar_events=set()
    )
    assert not allowed
    assert reason == "vix_too_low"


def test_vix_at_min_allows_entry(rm):
    """VIX exactly at min_vix=16.0 must be allowed (floor is strict <)."""
    allowed, reason = rm.check_entry_allowed(
        SAFE_EQUITY, vix=16.0, today=SAFE_DATE, calendar_events=set()
    )
    assert allowed, f"Expected allowed at VIX=16.0 but got: {reason}"


def test_vix_floor_zero_disables_floor_check():
    """Setting min_vix=0 effectively disables the floor for all reasonable VIX values."""
    rm_no_floor = RiskManager(min_vix=0.0, max_loss_per_trade=500.0)
    rm_no_floor.peak_equity = 100_000.0
    rm_no_floor.reset_day(100_000.0)
    allowed, reason = rm_no_floor.check_entry_allowed(
        SAFE_EQUITY, vix=8.0, today=SAFE_DATE, calendar_events=set()
    )
    assert allowed, f"Expected allowed with min_vix=0 but got: {reason}"


def test_high_vix_blocks_entry(rm):
    allowed, reason = rm.check_entry_allowed(
        SAFE_EQUITY, vix=26.0, today=SAFE_DATE, calendar_events=set()
    )
    assert not allowed
    assert "VIX" in reason or "vix" in reason.lower()


def test_vix_at_threshold_blocks(rm):
    """Exactly at threshold — should block (≥)."""
    allowed, _ = rm.check_entry_allowed(
        SAFE_EQUITY, vix=25.0, today=SAFE_DATE, calendar_events=set()
    )
    assert not allowed


def test_vix_just_below_threshold_allows(rm):
    allowed, reason = rm.check_entry_allowed(
        SAFE_EQUITY, vix=24.9, today=SAFE_DATE, calendar_events=set()
    )
    assert allowed, f"Expected allowed but got: {reason}"


# ---------------------------------------------------------------------------
# Rule b: Calendar events
# ---------------------------------------------------------------------------

def test_event_day_blocks_entry(rm):
    event = SAFE_DATE
    allowed, reason = rm.check_entry_allowed(
        SAFE_EQUITY, vix=SAFE_VIX, today=event, calendar_events={event}
    )
    assert not allowed
    assert "event" in reason.lower() or str(event) in reason


def test_non_event_day_allows_entry(rm):
    event_day = date(2025, 1, 29)  # FOMC day
    other_day  = date(2025, 3, 1)
    allowed, reason = rm.check_entry_allowed(
        SAFE_EQUITY, vix=SAFE_VIX, today=other_day, calendar_events={event_day}
    )
    assert allowed, f"Expected allowed but got: {reason}"


# ---------------------------------------------------------------------------
# Rule c: Daily loss limit
# ---------------------------------------------------------------------------

def test_daily_loss_limit_blocks_after_two_losses(rm):
    # max_loss_per_trade=500, multiplier=2 → daily limit = $1 000
    rm.record_trade_result(-500.0)   # first loss — still under limit
    rm.record_trade_result(-500.0)   # second loss — at limit
    allowed, reason = rm.check_entry_allowed(
        SAFE_EQUITY - 1000, vix=SAFE_VIX, today=SAFE_DATE, calendar_events=set()
    )
    assert not allowed
    assert "loss" in reason.lower() or "limit" in reason.lower()


def test_single_loss_still_allows_entry(rm):
    rm.record_trade_result(-499.0)   # below daily limit
    allowed, reason = rm.check_entry_allowed(
        SAFE_EQUITY - 499, vix=SAFE_VIX, today=SAFE_DATE, calendar_events=set()
    )
    assert allowed, f"Expected allowed but got: {reason}"


def test_profit_does_not_accumulate_toward_loss_limit(rm):
    rm.record_trade_result(+300.0)   # profit — should not affect loss tracking
    rm.record_trade_result(-499.0)
    allowed, reason = rm.check_entry_allowed(
        SAFE_EQUITY - 499 + 300, vix=SAFE_VIX, today=SAFE_DATE, calendar_events=set()
    )
    assert allowed, f"Expected allowed but got: {reason}"


# ---------------------------------------------------------------------------
# Rule d: Drawdown kill-switch
# ---------------------------------------------------------------------------

def test_drawdown_kill_switch_at_10_pct(rm):
    rm.peak_equity = 100_000.0
    account_value  = 89_999.0  # 10.001% drawdown — triggers
    allowed, reason = rm.check_entry_allowed(
        account_value, vix=SAFE_VIX, today=SAFE_DATE, calendar_events=set()
    )
    assert not allowed
    assert "drawdown" in reason.lower() or "kill" in reason.lower()


def test_drawdown_just_under_10_pct_allows(rm):
    rm.peak_equity = 100_000.0
    account_value  = 90_100.0  # 9.9% drawdown — passes
    allowed, reason = rm.check_entry_allowed(
        account_value, vix=SAFE_VIX, today=SAFE_DATE, calendar_events=set()
    )
    assert allowed, f"Expected allowed but got: {reason}"


def test_new_high_equity_updates_peak(rm):
    rm.peak_equity = 100_000.0
    rm.check_entry_allowed(105_000.0, vix=SAFE_VIX, today=SAFE_DATE, calendar_events=set())
    assert rm.peak_equity == 105_000.0


# ---------------------------------------------------------------------------
# Rule e: Max concurrent positions
# ---------------------------------------------------------------------------

def test_max_positions_blocks_entry(rm):
    rm.open_position()  # now at 1 open position (max = 1)
    allowed, reason = rm.check_entry_allowed(
        SAFE_EQUITY, vix=SAFE_VIX, today=SAFE_DATE, calendar_events=set()
    )
    assert not allowed
    assert "position" in reason.lower()


def test_closing_position_allows_next_entry(rm):
    rm.open_position()
    rm.record_trade_result(100.0)   # closes position + logs result
    allowed, reason = rm.check_entry_allowed(
        SAFE_EQUITY, vix=SAFE_VIX, today=SAFE_DATE, calendar_events=set()
    )
    assert allowed, f"Expected allowed but got: {reason}"


# ---------------------------------------------------------------------------
# All rules pass → entry allowed
# ---------------------------------------------------------------------------

def test_all_rules_pass_allows_entry(rm):
    allowed, reason = rm.check_entry_allowed(
        account_value=SAFE_EQUITY,
        vix=SAFE_VIX,
        today=SAFE_DATE,
        calendar_events=set(),
    )
    assert allowed, f"Expected allowed, got: {reason}"
    assert reason == "OK"


# ---------------------------------------------------------------------------
# State tracking
# ---------------------------------------------------------------------------

def test_record_trade_result_increments_trades(rm):
    rm.record_trade_result(100.0)
    rm.record_trade_result(-200.0)
    assert rm.trades_today == 2


def test_record_trade_result_accumulates_losses(rm):
    rm.record_trade_result(-300.0)
    rm.record_trade_result(-150.0)
    assert rm.losses_today == pytest.approx(450.0)


def test_record_trade_result_updates_equity(rm):
    rm.record_trade_result(250.0)
    assert rm.current_equity == pytest.approx(100_250.0)


def test_summary_contains_expected_keys(rm):
    summary = rm.summary()
    for key in ["peak_equity", "current_equity", "drawdown_pct", "trades_today",
                "losses_today", "open_positions", "daily_loss_limit"]:
        assert key in summary, f"Missing key: {key}"
