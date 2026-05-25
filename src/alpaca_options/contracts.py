"""
contracts.py — Option contract discovery and filtering.

Provides a thin wrapper around :class:`alpaca.trading.requests.GetOptionContractsRequest`
plus convenience helpers ``find_atm_call`` / ``find_atm_put`` that return the
single contract whose strike is closest to a given underlying price.

Usage::

    from alpaca_options.contracts import get_option_contracts, find_atm_call
    from datetime import date

    contracts = get_option_contracts(
        underlying_symbol="SPY",
        expiration_gte=date(2025, 6, 20),
        expiration_lte=date(2025, 6, 20),
        strike_gte=500.0,
        strike_lte=600.0,
    )
    call = find_atm_call("SPY", expiration=date(2025, 6, 20), underlying_price=540.0)
"""

from __future__ import annotations

import logging
from datetime import date
from typing import List, Optional

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import ContractType
from alpaca.trading.models import OptionContract
from alpaca.trading.requests import GetOptionContractsRequest

from alpaca_options.client import get_clients

logger = logging.getLogger(__name__)


def get_option_contracts(
    underlying_symbol: str,
    expiration_gte: Optional[date] = None,
    expiration_lte: Optional[date] = None,
    strike_gte: Optional[float] = None,
    strike_lte: Optional[float] = None,
    contract_type: Optional[ContractType] = None,
    limit: int = 200,
) -> List[OptionContract]:
    """Return option contracts matching the given filters.

    Parameters
    ----------
    underlying_symbol:
        Ticker of the underlying equity (e.g. ``"SPY"``).
    expiration_gte:
        Include contracts expiring on or after this date.
    expiration_lte:
        Include contracts expiring on or before this date.
    strike_gte:
        Lower bound for strike price (inclusive).
    strike_lte:
        Upper bound for strike price (inclusive).
    contract_type:
        ``ContractType.CALL`` or ``ContractType.PUT``.  ``None`` returns both.
    limit:
        Maximum number of contracts to return per API call (1–10 000).

    Returns
    -------
    List[OptionContract]
        Filtered contracts sorted by strike price ascending.
    """
    trading, _ = get_clients()

    req_kwargs: dict = {
        "underlying_symbols": [underlying_symbol.upper()],
        "limit": limit,
    }
    if expiration_gte is not None:
        req_kwargs["expiration_date_gte"] = expiration_gte
    if expiration_lte is not None:
        req_kwargs["expiration_date_lte"] = expiration_lte
    if strike_gte is not None:
        req_kwargs["strike_price_gte"] = str(round(strike_gte, 4))
    if strike_lte is not None:
        req_kwargs["strike_price_lte"] = str(round(strike_lte, 4))
    if contract_type is not None:
        req_kwargs["type"] = contract_type

    request = GetOptionContractsRequest(**req_kwargs)
    response = trading.get_option_contracts(request)

    contracts: List[OptionContract] = response.option_contracts or []
    contracts.sort(key=lambda c: float(c.strike_price or 0))

    logger.info(
        "Found %d %s contracts for %s (exp %s–%s, strike %.2f–%.2f)",
        len(contracts),
        contract_type.value if contract_type else "call+put",
        underlying_symbol.upper(),
        expiration_gte,
        expiration_lte,
        strike_gte or 0,
        strike_lte or 0,
    )
    return contracts


def _find_atm(
    underlying_symbol: str,
    expiration: date,
    underlying_price: float,
    contract_type: ContractType,
    strike_window: float = 50.0,
) -> OptionContract:
    """Internal helper — return the contract whose strike is closest to *underlying_price*."""
    half = strike_window / 2
    contracts = get_option_contracts(
        underlying_symbol=underlying_symbol,
        expiration_gte=expiration,
        expiration_lte=expiration,
        strike_gte=underlying_price - half,
        strike_lte=underlying_price + half,
        contract_type=contract_type,
    )

    if not contracts:
        raise ValueError(
            f"No {contract_type.value} contracts found for {underlying_symbol} "
            f"expiring {expiration} within ±{half:.2f} of {underlying_price:.2f}."
        )

    atm = min(contracts, key=lambda c: abs(float(c.strike_price or 0) - underlying_price))
    logger.info(
        "ATM %s selected: %s (strike=%.2f, exp=%s)",
        contract_type.value.upper(),
        atm.symbol,
        float(atm.strike_price or 0),
        atm.expiration_date,
    )
    return atm


def find_atm_call(
    underlying_symbol: str,
    expiration: date,
    underlying_price: float,
    strike_window: float = 50.0,
) -> OptionContract:
    """Return the ATM call whose strike is closest to *underlying_price*.

    Parameters
    ----------
    underlying_symbol:
        Ticker of the underlying (e.g. ``"SPY"``).
    expiration:
        Exact expiration date to target.
    underlying_price:
        Current price of the underlying (used to measure moneyness).
    strike_window:
        Total width of the strike search band centred on *underlying_price*.
        Defaults to ±$25 ($50 total).

    Returns
    -------
    OptionContract
        The call contract with the nearest strike to *underlying_price*.

    Raises
    ------
    ValueError
        If no contracts are found within the search window.
    """
    return _find_atm(
        underlying_symbol=underlying_symbol,
        expiration=expiration,
        underlying_price=underlying_price,
        contract_type=ContractType.CALL,
        strike_window=strike_window,
    )


def find_atm_put(
    underlying_symbol: str,
    expiration: date,
    underlying_price: float,
    strike_window: float = 50.0,
) -> OptionContract:
    """Return the ATM put whose strike is closest to *underlying_price*.

    Parameters
    ----------
    underlying_symbol:
        Ticker of the underlying (e.g. ``"SPY"``).
    expiration:
        Exact expiration date to target.
    underlying_price:
        Current price of the underlying.
    strike_window:
        Total width of the strike search band.

    Returns
    -------
    OptionContract
        The put contract with the nearest strike to *underlying_price*.

    Raises
    ------
    ValueError
        If no contracts are found within the search window.
    """
    return _find_atm(
        underlying_symbol=underlying_symbol,
        expiration=expiration,
        underlying_price=underlying_price,
        contract_type=ContractType.PUT,
        strike_window=strike_window,
    )
