"""
quotes.py — Market data retrieval for option contracts.

Wraps ``OptionLatestQuoteRequest`` (real-time NBBO) and ``OptionBarsRequest``
(OHLCV) from the Alpaca Options data feed.

Usage::

    from alpaca_options.quotes import get_latest_quote, get_option_bars
    from alpaca.data.timeframe import TimeFrame
    from datetime import datetime, timezone

    quote = get_latest_quote("SPY240620C00540000")
    bars  = get_option_bars(
        "SPY240620C00540000",
        start=datetime(2024, 6, 1, tzinfo=timezone.utc),
        timeframe=TimeFrame.Hour,
    )
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Dict, List, Optional

from alpaca.data.historical.option import OptionHistoricalDataClient
from alpaca.data.models.bars import Bar, BarSet
from alpaca.data.models.quotes import Quote
from alpaca.data.requests import OptionBarsRequest, OptionLatestQuoteRequest
from alpaca.data.timeframe import TimeFrame

from alpaca_options.client import get_clients

logger = logging.getLogger(__name__)


def get_latest_quote(symbol: str) -> Quote:
    """Fetch the latest NBBO quote for a single option symbol.

    Parameters
    ----------
    symbol:
        OCC option symbol (e.g. ``"SPY240620C00540000"``).

    Returns
    -------
    Quote
        The most-recent quote with bid/ask prices, sizes, and timestamp.

    Raises
    ------
    KeyError
        If the API returns no data for *symbol*.
    """
    _, data = get_clients()

    req = OptionLatestQuoteRequest(symbol_or_symbols=symbol)
    result: Dict[str, Quote] = data.get_option_latest_quote(req)  # type: ignore[assignment]

    if symbol not in result:
        raise KeyError(f"No quote data returned for option symbol: {symbol!r}")

    quote = result[symbol]
    logger.info(
        "Latest quote for %s: bid=%.4f ask=%.4f ts=%s",
        symbol,
        float(quote.bid_price or 0),
        float(quote.ask_price or 0),
        quote.timestamp,
    )
    return quote


def get_option_bars(
    symbol: str,
    timeframe: TimeFrame = TimeFrame.Day,
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
    limit: Optional[int] = None,
) -> List[Bar]:
    """Fetch historical OHLCV bars for a single option symbol.

    Parameters
    ----------
    symbol:
        OCC option symbol.
    timeframe:
        Bar resolution.  Defaults to ``TimeFrame.Day``.
    start:
        UTC datetime for the start of the range (inclusive).
    end:
        UTC datetime for the end of the range (inclusive).
    limit:
        Maximum number of bars to return.

    Returns
    -------
    List[Bar]
        Bars sorted oldest-first.  Empty list if no data is available.
    """
    _, data = get_clients()

    req_kwargs: dict = {"symbol_or_symbols": symbol, "timeframe": timeframe}
    if start is not None:
        req_kwargs["start"] = start
    if end is not None:
        req_kwargs["end"] = end
    if limit is not None:
        req_kwargs["limit"] = limit

    req = OptionBarsRequest(**req_kwargs)
    result: BarSet = data.get_option_bars(req)  # type: ignore[assignment]

    bars: List[Bar] = result.data.get(symbol, [])
    logger.info("Fetched %d bars for %s (%s resolution).", len(bars), symbol, timeframe)
    return bars


def midpoint(quote: Quote) -> float:
    """Return the bid/ask midpoint for *quote*.

    Returns 0.0 if both bid and ask are unavailable.
    """
    bid = float(quote.bid_price or 0)
    ask = float(quote.ask_price or 0)
    if bid == 0 and ask == 0:
        return 0.0
    if bid == 0:
        return ask
    if ask == 0:
        return bid
    return (bid + ask) / 2
