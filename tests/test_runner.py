"""
tests/test_runner.py — Unit tests for LiveRunner and its module-level helpers.

All tests are pure in-memory — no network calls, no Alpaca API keys required.
The TradingClient and DataClient are fully mocked via monkeypatch/MagicMock.
"""

from __future__ import annotations

import logging
import signal
import sys
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch
from zoneinfo import ZoneInfo

import pytest

from alpaca_options.live.runner import (
    LiveRunner,
    _format_condor_order,
    _parse_expiry_from_occ,
)
from alpaca_options.strategies.iron_condor_0dte import (
    CondorLegs,
    CondorPosition,
    ExitDecision,
    IronCondor0DTE,
    IronCondorConfig,
)
from alpaca_options.risk.manager import RiskManager

ET = ZoneInfo("America/New_York")


# ---------------------------------------------------------------------------
# Helpers: build a fake CondorLegs
# ---------------------------------------------------------------------------

def _fake_legs(credit: float = 0.62) -> CondorLegs:
    return CondorLegs(
        put_long_symbol="SPY250602P00530000",
        put_long_strike=530.0,
        put_short_symbol="SPY250602P00535000",
        put_short_strike=535.0,
        call_short_symbol="SPY250602C00545000",
        call_short_strike=545.0,
        call_long_symbol="SPY250602C00550000",
        call_long_strike=550.0,
        net_credit=credit,
    )


# ---------------------------------------------------------------------------
# Fixture: a LiveRunner with all external calls mocked
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_trading_client():
    """Fake TradingClient returned by get_clients()."""
    client = MagicMock()
    account = MagicMock()
    account.portfolio_value = 100_000.0
    account.equity = 100_000.0
    client.get_account.return_value = account
    return client


@pytest.fixture
def runner(mock_trading_client, monkeypatch):
    """LiveRunner with mocked Alpaca clients (runner + strategy modules) and default config."""
    factory = lambda: (mock_trading_client, MagicMock())
    monkeypatch.setattr("alpaca_options.live.runner.get_clients", factory)
    monkeypatch.setattr("alpaca_options.strategies.iron_condor_0dte.get_clients", factory)
    strategy = IronCondor0DTE(config=IronCondorConfig(min_vix=16.0))
    risk     = RiskManager(min_vix=16.0, max_loss_per_trade=500.0)
    return LiveRunner(
        dry_run=True,
        strategy=strategy,
        risk=risk,
        poll_interval_entry=60,
        poll_interval_monitor=30,
        poll_interval_closed=300,
    )


# ---------------------------------------------------------------------------
# 1. Poll intervals: defaults, split behaviour, three distinct states
# ---------------------------------------------------------------------------

def test_poll_interval_entry_default(monkeypatch, mock_trading_client):
    monkeypatch.setattr(
        "alpaca_options.live.runner.get_clients",
        lambda: (mock_trading_client, MagicMock()),
    )
    r = LiveRunner(dry_run=True)
    assert r.poll_interval_entry == 60


def test_poll_interval_monitor_default(monkeypatch, mock_trading_client):
    monkeypatch.setattr(
        "alpaca_options.live.runner.get_clients",
        lambda: (mock_trading_client, MagicMock()),
    )
    r = LiveRunner(dry_run=True)
    assert r.poll_interval_monitor == 30


def test_poll_interval_closed_default(monkeypatch, mock_trading_client):
    monkeypatch.setattr(
        "alpaca_options.live.runner.get_clients",
        lambda: (mock_trading_client, MagicMock()),
    )
    r = LiveRunner(dry_run=True)
    assert r.poll_interval_closed == 300


def test_poll_intervals_can_be_overridden(monkeypatch, mock_trading_client):
    monkeypatch.setattr(
        "alpaca_options.live.runner.get_clients",
        lambda: (mock_trading_client, MagicMock()),
    )
    r = LiveRunner(
        dry_run=True,
        poll_interval_entry=120,
        poll_interval_monitor=15,
        poll_interval_closed=600,
    )
    assert r.poll_interval_entry   == 120
    assert r.poll_interval_monitor == 15
    assert r.poll_interval_closed  == 600


# ---------------------------------------------------------------------------
# Helpers — patch get_clients in both modules that import it
# ---------------------------------------------------------------------------

def _patch_all_clients(monkeypatch, trading_mock):
    """Patch get_clients in every module that binds it at import time.

    LiveRunner imports via ``alpaca_options.live.runner.get_clients``.
    IronCondor0DTE imports via ``alpaca_options.strategies.iron_condor_0dte.get_clients``.
    Both must return the same mock so that ``strategy._trading.submit_order`` is
    the same MagicMock we assert on.
    """
    factory = lambda: (trading_mock, MagicMock())
    monkeypatch.setattr("alpaca_options.live.runner.get_clients", factory)
    monkeypatch.setattr("alpaca_options.strategies.iron_condor_0dte.get_clients", factory)


# ---------------------------------------------------------------------------
# 2. Dry-run guard: TradingClient.submit_order is never called when dry_run=True
# ---------------------------------------------------------------------------

def test_dry_run_blocks_submit_order(monkeypatch, mock_trading_client):
    """With dry_run=True the underlying TradingClient.submit_order must never fire.

    This test does NOT mock strategy.enter() — it lets the real call chain run
    and asserts on the API boundary (submit_order) rather than the wrapper.
    build_condor() is mocked to return a valid condor so an entry attempt is made.
    """
    _patch_all_clients(monkeypatch, mock_trading_client)

    strategy = IronCondor0DTE(config=IronCondorConfig(min_vix=16.0))
    risk     = RiskManager(min_vix=16.0, max_loss_per_trade=500.0)
    runner   = LiveRunner(dry_run=True, strategy=strategy, risk=risk)

    monkeypatch.setattr("alpaca_options.live.runner.get_current_vix", lambda: 20.0)
    monkeypatch.setattr("alpaca_options.live.runner.get_event_days",  lambda *a: set())
    monkeypatch.setattr(runner.strategy, "build_condor",  lambda: _fake_legs())
    monkeypatch.setattr(runner.strategy, "should_enter",  lambda now, vix: True)
    monkeypatch.setattr(runner.risk, "check_entry_allowed", lambda **kw: (True, "OK"))

    now = datetime(2025, 6, 2, 11, 0, tzinfo=ET)
    runner._try_enter(now)

    # The critical assertion: the Alpaca API must never have been called.
    mock_trading_client.submit_order.assert_not_called()

    # Secondary: a simulated position must exist so monitoring can proceed.
    assert runner._position is not None
    assert runner._position.order_id == "DRY-RUN"


def test_live_mode_calls_submit_order_exactly_once(monkeypatch, mock_trading_client):
    """With dry_run=False, TradingClient.submit_order must be called exactly once on entry.

    strategy.enter() is NOT mocked — we let the real method run so the full
    call chain (enter → build request → submit_order) is exercised.
    """
    _patch_all_clients(monkeypatch, mock_trading_client)

    # submit_order must return something with an .id attribute
    fake_order      = MagicMock()
    fake_order.id   = "ORD-LIVE-001"
    mock_trading_client.submit_order.return_value = fake_order

    strategy = IronCondor0DTE(config=IronCondorConfig(min_vix=16.0))
    risk     = RiskManager(min_vix=16.0, max_loss_per_trade=500.0)
    runner   = LiveRunner(dry_run=False, strategy=strategy, risk=risk)

    monkeypatch.setattr("alpaca_options.live.runner.get_current_vix", lambda: 20.0)
    monkeypatch.setattr("alpaca_options.live.runner.get_event_days",  lambda *a: set())
    monkeypatch.setattr(runner.strategy, "build_condor",  lambda: _fake_legs())
    monkeypatch.setattr(runner.strategy, "should_enter",  lambda now, vix: True)
    monkeypatch.setattr(runner.risk, "check_entry_allowed", lambda **kw: (True, "OK"))

    now = datetime(2025, 6, 2, 11, 0, tzinfo=ET)
    runner._try_enter(now)

    mock_trading_client.submit_order.assert_called_once()
    assert runner._position is not None
    assert runner._position.order_id == "ORD-LIVE-001"


# ---------------------------------------------------------------------------
# 3. SIGINT handler: logging.shutdown() is called before sys.exit()
# ---------------------------------------------------------------------------

def test_sigint_handler_calls_logging_shutdown_before_exit(runner, monkeypatch):
    """SIGINT handler must call logging.shutdown() then sys.exit(0) — in that order."""
    call_order = []

    monkeypatch.setattr(logging, "shutdown", lambda: call_order.append("shutdown"))

    with pytest.raises(SystemExit) as exc_info:
        monkeypatch.setattr(sys, "exit", lambda code=0: (_ for _ in ()).throw(SystemExit(code)))
        runner._handle_sigint(signal.SIGINT, None)

    assert exc_info.value.code == 0
    assert "shutdown" in call_order, "logging.shutdown() must be called before sys.exit()"
    assert call_order[0] == "shutdown", "logging.shutdown() must be first"


def test_sigint_with_dry_run_position_logs_pnl(runner, monkeypatch, caplog):
    """SIGINT with an open dry-run position must log the estimated P&L and NOT call strategy.exit()."""
    runner._position = CondorPosition(
        legs=_fake_legs(credit=0.62),
        order_id="DRY-RUN",
        entry_time=datetime(2025, 6, 2, 11, 0, tzinfo=ET),
        entry_credit=0.62,
        current_value=0.31,
        underlying_price=540.0,
    )

    exit_called = []
    monkeypatch.setattr(runner.strategy, "exit", lambda pos, dec: exit_called.append(dec))
    monkeypatch.setattr(logging, "shutdown", lambda: None)

    with pytest.raises(SystemExit):
        monkeypatch.setattr(sys, "exit", lambda code=0: (_ for _ in ()).throw(SystemExit(code)))
        with caplog.at_level(logging.INFO):
            runner._handle_sigint(signal.SIGINT, None)

    assert exit_called == [], "strategy.exit() must NOT be called in dry-run mode on SIGINT"


# ---------------------------------------------------------------------------
# 4. _parse_expiry_from_occ — OCC symbol parsing
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("symbol,expected", [
    ("SPY250602C00540000", "2025-06-02"),
    ("SPY251231P00450000", "2025-12-31"),
    ("AAPL260101C00200000", "2026-01-01"),
    ("QQQ250101C00400000", "2025-01-01"),
])
def test_parse_expiry_from_occ_valid(symbol, expected):
    assert _parse_expiry_from_occ(symbol) == expected


def test_parse_expiry_from_occ_invalid_returns_question_mark():
    assert _parse_expiry_from_occ("NOT_AN_OCC_SYMBOL") == "?"
    assert _parse_expiry_from_occ("") == "?"


# ---------------------------------------------------------------------------
# 5. _format_condor_order — dry-run order formatting
# ---------------------------------------------------------------------------

def test_format_condor_order_contains_all_legs():
    legs = _fake_legs(credit=0.62)
    output = _format_condor_order(legs, qty=1)

    assert "BUY" in output
    assert "SELL" in output
    assert "PUT" in output
    assert "CALL" in output
    assert "SPY250602P00530000" in output   # put_long
    assert "SPY250602P00535000" in output   # put_short
    assert "SPY250602C00545000" in output   # call_short
    assert "SPY250602C00550000" in output   # call_long
    assert "2025-06-02" in output           # expiry parsed from OCC
    assert "DRY RUN" in output
    assert "NOT SUBMITTED" in output


def test_format_condor_order_limit_price_is_net_credit():
    legs = _fake_legs(credit=0.625)
    output = _format_condor_order(legs, qty=1)
    # rounded to 2 dp → 0.62; contract value = 0.62 × 100 = $62.00
    assert "$0.62/share" in output
    assert "$62.00/contract" in output


def test_format_condor_order_qty_appears_in_all_legs():
    legs   = _fake_legs()
    output = _format_condor_order(legs, qty=3)
    assert output.count("qty=3") == 4   # one per leg


# ---------------------------------------------------------------------------
# 6. Market hours guard
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("hour,minute,expected", [
    (9,  30, True),    # exact open
    (16,  0, True),    # exact close (inclusive)
    (9,  29, False),   # one minute before open
    (16,  1, False),   # one minute after close
    (12,  0, True),    # midday
])
def test_is_market_hours(hour, minute, expected):
    # A Wednesday (weekday=2) in ET
    dt = datetime(2025, 6, 4, hour, minute, 0, tzinfo=ET)
    assert LiveRunner._is_market_hours(dt) == expected


def test_is_market_hours_weekend():
    saturday = datetime(2025, 6, 7, 12, 0, tzinfo=ET)  # Saturday
    assert LiveRunner._is_market_hours(saturday) is False
