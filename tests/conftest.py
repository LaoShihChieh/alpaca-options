"""
tests/conftest.py — Shared fixtures and mock helpers.

All tests use monkeypatching to avoid real network calls.
The ``_patch_clients`` fixture replaces ``get_clients()`` with mock objects.
"""

from __future__ import annotations

import os
from datetime import date
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Ensure env vars are always present so client.py doesn't raise
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def patch_env(monkeypatch):
    monkeypatch.setenv("ALPACA_API_KEY", "FAKE_KEY")
    monkeypatch.setenv("ALPACA_SECRET_KEY", "FAKE_SECRET")
    monkeypatch.setenv("ALPACA_PAPER", "true")


# ---------------------------------------------------------------------------
# Mock Alpaca clients
# ---------------------------------------------------------------------------

def make_mock_trading_client():
    """Build a MagicMock that looks like TradingClient."""
    client = MagicMock()
    account = MagicMock()
    account.id = "test-account-id"
    account.portfolio_value = 100_000.0
    account.equity = 100_000.0
    account.buying_power = 100_000.0
    account.status = "ACTIVE"
    client.get_account.return_value = account
    return client


def make_mock_data_client():
    """Build a MagicMock that looks like OptionHistoricalDataClient."""
    return MagicMock()


@pytest.fixture
def mock_clients(monkeypatch):
    """Patch ``get_clients`` in every module that imports it."""
    trading = make_mock_trading_client()
    data    = make_mock_data_client()

    def _fake_get_clients():
        return trading, data

    targets = [
        "alpaca_options.client.get_clients",
        "alpaca_options.strategies.iron_condor_0dte.get_clients",
        "alpaca_options.backtest.replay.get_clients",
        "alpaca_options.live.runner.get_clients",
        "alpaca_options.positions.get_clients",
        "alpaca_options.orders.get_clients",
        "alpaca_options.contracts.get_clients",
        "alpaca_options.quotes.get_clients",
    ]
    patches = []
    for target in targets:
        try:
            p = patch(target, side_effect=_fake_get_clients)
            p.start()
            patches.append(p)
        except Exception:
            pass

    yield trading, data

    for p in patches:
        try:
            p.stop()
        except Exception:
            pass
