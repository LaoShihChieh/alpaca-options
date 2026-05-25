"""
client.py — Singleton factory for Alpaca paper-trading clients.

Always operates in paper mode. Raises ``AssertionError`` at init time if
``ALPACA_PAPER`` is not set to ``"true"`` (case-insensitive) in the
environment, acting as a hard guardrail against accidental live trading.

Usage::

    from alpaca_options.client import get_clients
    trading, data = get_clients()
"""

import logging
import os
from functools import lru_cache
from typing import Tuple

from dotenv import load_dotenv
from alpaca.trading.client import TradingClient
from alpaca.data.historical.option import OptionHistoricalDataClient

logger = logging.getLogger(__name__)

# Load .env once at module import time so env vars are available everywhere.
load_dotenv()


def _load_credentials() -> Tuple[str, str]:
    """Read and validate API credentials from the environment."""
    api_key = os.environ.get("ALPACA_API_KEY", "")
    secret_key = os.environ.get("ALPACA_SECRET_KEY", "")
    paper_flag = os.environ.get("ALPACA_PAPER", "false").strip().lower()

    if not api_key:
        raise ValueError("ALPACA_API_KEY is not set in the environment / .env file.")
    if not secret_key:
        raise ValueError("ALPACA_SECRET_KEY is not set in the environment / .env file.")

    assert paper_flag == "true", (
        f"ALPACA_PAPER must be 'true' (got {paper_flag!r}). "
        "This library is paper-only; live trading is not supported."
    )

    return api_key, secret_key


@lru_cache(maxsize=1)
def get_clients() -> Tuple[TradingClient, OptionHistoricalDataClient]:
    """Return a cached ``(TradingClient, OptionHistoricalDataClient)`` pair.

    Both clients are configured for **paper trading only**.

    Returns
    -------
    trading : TradingClient
        Used for order submission, position management, and contract lookup.
    data : OptionHistoricalDataClient
        Used for quotes and bars on option symbols.

    Raises
    ------
    AssertionError
        If ``ALPACA_PAPER`` env var is not ``"true"``.
    ValueError
        If ``ALPACA_API_KEY`` or ``ALPACA_SECRET_KEY`` are missing.
    """
    api_key, secret_key = _load_credentials()

    trading = TradingClient(
        api_key=api_key,
        secret_key=secret_key,
        paper=True,
    )
    data = OptionHistoricalDataClient(
        api_key=api_key,
        secret_key=secret_key,
    )

    logger.info("Alpaca paper-trading clients initialised (paper=True).")
    return trading, data
