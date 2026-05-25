"""
data/vix.py — VIX (CBOE Volatility Index) fetching.

Source priority
---------------
**Live (get_current_vix)**:
  1. yfinance ``^VIX`` — primary; no Alpaca subscription needed.
  2. Alpaca VIXY×10 proxy — fallback if yfinance fails.
  3. Hard constant 20.0 — last resort.

**Historical (get_historical_vix / get_historical_vix_range)**:
  1. yfinance ``^VIX`` daily close for the requested date(s).
  2. Alpaca VIXY daily bars × 10 — fallback per-date.
  3. Hard constant 20.0 — last resort.

For live trading and the risk manager, only the *regime* matters
(above or below the threshold), not the exact VIX value.

Usage::

    from alpaca_options.data.vix import (
        get_current_vix,
        get_historical_vix,
        get_historical_vix_range,
        is_high_vol_regime,
    )

    vix_today = get_current_vix()
    vix_jan   = get_historical_vix(date(2025, 1, 15))
    vix_range = get_historical_vix_range(date(2024, 1, 1), date(2024, 12, 31))
"""

from __future__ import annotations

import logging
import os
from datetime import date, timedelta
from typing import Optional

from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

_VIXY_TO_VIX_SCALE = 10.0
_DEFAULT_VIX = 20.0  # conservative fallback when all sources fail


# ---------------------------------------------------------------------------
# Live VIX helpers
# ---------------------------------------------------------------------------

def _get_vix_from_yfinance_live() -> Optional[float]:
    """Fetch the latest ^VIX price from yfinance."""
    try:
        import yfinance as yf  # type: ignore[import]

        ticker = yf.Ticker("^VIX")
        info   = ticker.fast_info
        # fast_info exposes last_price on yfinance ≥ 0.2
        price  = getattr(info, "last_price", None) or info.get("lastPrice", None)
        if price and float(price) > 0:
            logger.debug("yfinance ^VIX (live): %.2f", float(price))
            return float(price)
    except ImportError:
        logger.debug("yfinance not installed.")
    except Exception as exc:
        logger.debug("yfinance live VIX failed: %s", exc)
    return None


def _get_vix_from_alpaca_live() -> Optional[float]:
    """Fetch VIX from Alpaca using VIXY (VIX futures ETF) as a ×10 proxy."""
    try:
        from alpaca.data.historical.stock import StockHistoricalDataClient
        from alpaca.data.requests import StockLatestQuoteRequest

        api_key    = os.environ.get("ALPACA_API_KEY", "")
        secret_key = os.environ.get("ALPACA_SECRET_KEY", "")
        if not api_key or not secret_key:
            return None

        client = StockHistoricalDataClient(api_key=api_key, secret_key=secret_key)
        req    = StockLatestQuoteRequest(symbol_or_symbols="VIXY")
        result = client.get_stock_latest_quote(req)
        quote  = result.get("VIXY")
        if quote is None:
            return None

        bid = float(quote.bid_price or 0)
        ask = float(quote.ask_price or 0)
        if bid == 0 and ask == 0:
            return None

        mid = (bid + ask) / 2 if (bid > 0 and ask > 0) else max(bid, ask)
        vix_estimate = mid * _VIXY_TO_VIX_SCALE
        logger.debug("Alpaca VIXY mid=%.4f → VIX proxy=%.2f", mid, vix_estimate)
        return vix_estimate
    except Exception as exc:
        logger.debug("Alpaca VIX proxy failed: %s", exc)
        return None


def get_current_vix() -> float:
    """Return the current (or best-available) VIX level.

    Tries sources in order: yfinance → Alpaca VIXY proxy → constant 20.0.

    Returns
    -------
    float
        VIX level (e.g. 18.5 means 18.5%).
    """
    vix = _get_vix_from_yfinance_live()
    if vix is not None:
        logger.info("VIX sourced from yfinance ^VIX (live): %.2f", vix)
        return vix

    vix = _get_vix_from_alpaca_live()
    if vix is not None:
        logger.info("VIX sourced from Alpaca VIXY proxy: %.2f", vix)
        return vix

    logger.warning(
        "All VIX sources failed — using fallback VIX=%.1f. "
        "Ensure yfinance is installed or Alpaca credentials are set.",
        _DEFAULT_VIX,
    )
    return _DEFAULT_VIX


# ---------------------------------------------------------------------------
# Historical VIX helpers
# ---------------------------------------------------------------------------

def get_historical_vix_range(start: date, end: date) -> dict[date, float]:
    """Fetch ^VIX daily close for every trading day in [start, end].

    Tries yfinance first; fills any missing dates from Alpaca VIXY daily bars.
    Missing dates after both sources are omitted — callers should fall back to
    a constant for those.

    Parameters
    ----------
    start, end:
        Inclusive date range.

    Returns
    -------
    dict[date, float]
        ``{calendar_date: vix_close}`` for every date that has data.
    """
    result: dict[date, float] = {}

    # ── yfinance bulk download ────────────────────────────────────────────────
    try:
        import yfinance as yf  # type: ignore[import]

        df = yf.download(
            "^VIX",
            start=start.isoformat(),
            end=(end + timedelta(days=1)).isoformat(),
            progress=False,
            auto_adjust=True,
        )
        if not df.empty:
            # Handle both single- and multi-level column DataFrames
            if hasattr(df.columns, "levels"):
                close_col = ("Close", "^VIX") if ("Close", "^VIX") in df.columns else "Close"
            else:
                close_col = "Close"
            for idx, row in df.iterrows():
                d = idx.date() if hasattr(idx, "date") else idx
                val = float(row[close_col])
                if val > 0:
                    result[d] = val
            logger.info("yfinance ^VIX range: %d days fetched (%s → %s)", len(result), start, end)
    except ImportError:
        logger.debug("yfinance not installed; skipping historical VIX source.")
    except Exception as exc:
        logger.warning("yfinance historical VIX failed: %s", exc)

    # ── Alpaca VIXY daily fill for any missing dates ──────────────────────────
    _fill_from_alpaca_vixy(start, end, result)

    return result


def _fill_from_alpaca_vixy(start: date, end: date, result: dict[date, float]) -> None:
    """Fill *result* in-place using Alpaca VIXY daily bars for dates not yet present."""
    try:
        from datetime import datetime, timezone

        from alpaca.data.historical.stock import StockHistoricalDataClient
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame

        api_key    = os.environ.get("ALPACA_API_KEY", "")
        secret_key = os.environ.get("ALPACA_SECRET_KEY", "")
        if not api_key or not secret_key:
            return

        client = StockHistoricalDataClient(api_key=api_key, secret_key=secret_key)
        req = StockBarsRequest(
            symbol_or_symbols="VIXY",
            timeframe=TimeFrame.Day,
            start=datetime(start.year, start.month, start.day, tzinfo=timezone.utc),
            end=datetime(end.year, end.month, end.day, 23, 59, tzinfo=timezone.utc),
        )
        bars_result = client.get_stock_bars(req)
        bars = bars_result.data.get("VIXY", [])
        from zoneinfo import ZoneInfo
        ET = ZoneInfo("America/New_York")
        filled = 0
        for b in bars:
            d = b.timestamp.astimezone(ET).date()
            if d not in result:
                result[d] = float(b.close) * _VIXY_TO_VIX_SCALE
                filled += 1
        if filled:
            logger.info("Alpaca VIXY filled %d missing VIX dates", filled)
    except Exception as exc:
        logger.debug("Alpaca VIXY historical fill failed: %s", exc)


def get_historical_vix(target_date: date) -> float:
    """Return the ^VIX closing level for *target_date*.

    Fetches a narrow window around *target_date* to handle weekends and
    holidays (returns the most-recent prior trading-day close if the exact
    date has no data).

    Parameters
    ----------
    target_date:
        Calendar date to look up.

    Returns
    -------
    float
        VIX close, or ``_DEFAULT_VIX`` (20.0) if no data is available.
    """
    # Fetch a small window ending on target_date (+1 for inclusive end)
    window_start = target_date - timedelta(days=7)
    mapping = get_historical_vix_range(window_start, target_date)

    if not mapping:
        logger.warning("No historical VIX data for %s — using fallback %.1f", target_date, _DEFAULT_VIX)
        return _DEFAULT_VIX

    # Prefer exact match; otherwise take the most-recent prior date
    if target_date in mapping:
        return mapping[target_date]

    prior_dates = [d for d in mapping if d <= target_date]
    if prior_dates:
        return mapping[max(prior_dates)]

    logger.warning("No VIX data on or before %s — using fallback %.1f", target_date, _DEFAULT_VIX)
    return _DEFAULT_VIX


# ---------------------------------------------------------------------------
# Regime helper
# ---------------------------------------------------------------------------

def is_high_vol_regime(vix: Optional[float] = None, threshold: float = 25.0) -> bool:
    """Return ``True`` when VIX is at or above *threshold*.

    Parameters
    ----------
    vix:
        VIX level.  If ``None``, calls :func:`get_current_vix` automatically.
    threshold:
        VIX level above which we consider the market to be in a high-vol
        regime (default 25).
    """
    current = vix if vix is not None else get_current_vix()
    high    = current >= threshold
    if high:
        logger.warning("High-vol regime: VIX=%.2f ≥ threshold=%.1f", current, threshold)
    return high
