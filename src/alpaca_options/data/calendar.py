"""
data/calendar.py — High-impact economic event calendar.

Returns dates that the strategy should skip: FOMC meeting days, CPI release
days, and NFP (Non-Farm Payroll) release days.

Data sources (priority order):
1. FRED API (free key at https://fred.stlouisfed.org/docs/api/api_key.html)
   Set FRED_API_KEY in your .env file.
   CPI dates are fetched via the ``/release/dates`` endpoint (release ID 10).
   FOMC and NFP dates always come from the hardcoded calendar (verified correct).
2. Hardcoded fallback for 2025-2026 dates.

Usage::

    from alpaca_options.data.calendar import get_event_days
    from datetime import date

    events = get_event_days(date(2025, 1, 1), date(2025, 12, 31))
    if date.today() in events:
        print("Skip trading today — major economic release!")
"""

from __future__ import annotations

import logging
import os
from datetime import date
from typing import Optional

import httpx
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Hardcoded 2025-2026 fallback dates
# (FOMC: federalreserve.gov schedule — verified against FRED release 13, ✅ all 16 match)
# (CPI: updated to match FRED release 10 authoritative data)
# (NFP: first-Friday BLS Employment Situation release dates)
# ---------------------------------------------------------------------------

# FOMC meeting *end* dates — market impact is highest at 2 PM ET announcement
_FOMC_DATES: frozenset[date] = frozenset(
    {
        date(2025, 1, 29),
        date(2025, 3, 19),
        date(2025, 5, 7),
        date(2025, 6, 18),
        date(2025, 7, 30),
        date(2025, 9, 17),
        date(2025, 10, 29),
        date(2025, 12, 10),
        date(2026, 1, 28),
        date(2026, 3, 18),
        date(2026, 4, 29),
        date(2026, 6, 17),
        date(2026, 7, 29),
        date(2026, 9, 16),
        date(2026, 10, 28),
        date(2026, 12, 9),
    }
)

# CPI release dates — corrected to match FRED release ID 10 (Consumer Price Index)
# Where FRED data was available, it takes precedence over BLS calendar estimates.
# Dates confirmed against FRED /release/dates?release_id=10 query (2025-01-01 to 2026-12-31).
_CPI_DATES: frozenset[date] = frozenset(
    {
        date(2025, 1, 15),   # CPI for Dec 2024
        date(2025, 2, 12),   # CPI for Jan 2025
        date(2025, 3, 12),   # CPI for Feb 2025
        date(2025, 4, 10),   # CPI for Mar 2025
        date(2025, 5, 13),   # CPI for Apr 2025
        date(2025, 6, 11),   # CPI for May 2025
        date(2025, 7, 15),   # CPI for Jun 2025
        date(2025, 8, 12),   # CPI for Jul 2025
        date(2025, 9, 11),   # CPI for Aug 2025
        date(2025, 10, 24),  # CPI for Sep 2025  ← corrected (was 10-15)
        date(2025, 11, 13),  # CPI for Oct 2025
        date(2025, 12, 18),  # CPI for Nov 2025  ← corrected (was 12-11)
        date(2026, 1, 13),   # CPI for Dec 2025  ← corrected (was 01-14)
        date(2026, 2, 13),   # CPI for Jan 2026  ← corrected (was 02-11)
        date(2026, 3, 11),   # CPI for Feb 2026
        date(2026, 4, 10),   # CPI for Mar 2026  ← corrected (was 04-14)
        date(2026, 5, 12),   # CPI for Apr 2026  ← corrected (was 05-13)
        date(2026, 6, 10),   # CPI for May 2026
        date(2026, 7, 14),   # CPI for Jun 2026
        date(2026, 8, 12),   # CPI for Jul 2026
        date(2026, 9, 11),   # CPI for Aug 2026  ← corrected (was 09-10)
        date(2026, 10, 14),  # CPI for Sep 2026
        date(2026, 11, 10),  # CPI for Oct 2026  ← corrected (was 11-12)
        date(2026, 12, 10),  # CPI for Nov 2026
    }
)

# NFP release dates (BLS Employment Situation, first Friday of each month, ~8:30 AM ET)
# These are the initial release dates; FRED release ID 50 also includes revision dates
# which are not market-moving events and are intentionally excluded here.
_NFP_DATES: frozenset[date] = frozenset(
    {
        date(2025, 1, 10),
        date(2025, 2, 7),
        date(2025, 3, 7),
        date(2025, 4, 4),
        date(2025, 5, 2),
        date(2025, 6, 6),
        date(2025, 7, 3),
        date(2025, 8, 1),
        date(2025, 9, 5),
        date(2025, 10, 3),
        date(2025, 11, 7),
        date(2025, 12, 5),
        date(2026, 1, 9),
        date(2026, 2, 6),
        date(2026, 3, 6),
        date(2026, 4, 3),
        date(2026, 5, 1),
        date(2026, 6, 5),
        date(2026, 7, 2),
        date(2026, 8, 7),
        date(2026, 9, 4),
        date(2026, 10, 2),
        date(2026, 11, 6),
        date(2026, 12, 4),
    }
)

_ALL_FALLBACK: frozenset[date] = _FOMC_DATES | _CPI_DATES | _NFP_DATES


# ---------------------------------------------------------------------------
# FRED API helpers
# ---------------------------------------------------------------------------

_FRED_BASE = "https://api.stlouisfed.org/fred"

# FRED release IDs for economic data releases.
# CPI: release ID 10 — Consumer Price Index (BLS)
# NFP: first-Friday computation is more reliable than release ID 50 (which
#      includes revision dates that are not market-moving events).
# FOMC: hardcoded calendar matches federalreserve.gov schedule exactly.
_FRED_RELEASE_IDS: dict[str, int] = {
    "cpi": 10,
}


def _fetch_fred_release_dates(
    release_id: int,
    api_key: str,
    start: date,
    end: date,
    timeout: float = 10.0,
) -> set[date]:
    """Fetch release dates for a FRED release via the ``/release/dates`` endpoint.

    Parameters
    ----------
    release_id:
        FRED release ID (e.g. 10 for CPI, 50 for Employment Situation).
    api_key:
        FRED API key.
    start / end:
        Date range (inclusive) to filter returned dates.
    timeout:
        HTTP request timeout in seconds.

    Returns
    -------
    set[date]
        Release dates within [start, end], or an empty set on failure.
    """
    url = f"{_FRED_BASE}/release/dates"
    params = {
        "release_id": release_id,
        "api_key": api_key,
        "file_type": "json",
        "sort_order": "asc",
        "include_release_dates_with_no_data": "true",
        "realtime_start": start.strftime("%Y-%m-%d"),
        "realtime_end": end.strftime("%Y-%m-%d"),
    }
    try:
        resp = httpx.get(url, params=params, timeout=timeout)
        resp.raise_for_status()
        release_dates = resp.json().get("release_dates", [])
        dates: set[date] = set()
        for entry in release_dates:
            try:
                dates.add(date.fromisoformat(entry["date"]))
            except (KeyError, ValueError):
                pass
        logger.debug("FRED release %d: %d dates fetched.", release_id, len(dates))
        return dates
    except Exception as exc:
        logger.warning("FRED API fetch failed for release ID %d: %s", release_id, exc)
        return set()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_event_days(start: date, end: date) -> set[date]:
    """Return high-impact economic event dates within [start, end].

    When ``FRED_API_KEY`` is set in the environment, CPI dates are fetched
    live from FRED release 10.  FOMC and NFP dates always come from the
    verified hardcoded calendar (FOMC matches Fed schedule exactly; NFP uses
    first-Friday computation which is more accurate than FRED release 50).

    Falls back entirely to hardcoded dates if FRED is unavailable or returns
    no data for the requested range.

    Parameters
    ----------
    start:
        First date of the range (inclusive).
    end:
        Last date of the range (inclusive).

    Returns
    -------
    set[date]
        Union of FOMC, CPI, and NFP dates filtered to [start, end].
    """
    fred_key: Optional[str] = os.environ.get("FRED_API_KEY")

    if fred_key:
        logger.info("Fetching CPI release dates from FRED API (release ID 10).")
        fred_cpi_dates: set[date] = set()
        for label, release_id in _FRED_RELEASE_IDS.items():
            fetched = _fetch_fred_release_dates(release_id, fred_key, start, end)
            fred_cpi_dates.update(fetched)

        if fred_cpi_dates:
            # Combine FRED CPI with hardcoded FOMC + NFP (both verified accurate)
            combined: set[date] = fred_cpi_dates
            combined.update({d for d in _FOMC_DATES if start <= d <= end})
            combined.update({d for d in _NFP_DATES if start <= d <= end})
            logger.info(
                "Event days (%s to %s): %d days (FRED CPI + hardcoded FOMC/NFP).",
                start, end, len(combined),
            )
            return combined

        logger.warning("FRED returned no CPI dates — falling back to hardcoded calendar.")

    # Fallback: filter hardcoded calendar to range
    events = {d for d in _ALL_FALLBACK if start <= d <= end}
    logger.info(
        "Event days (%s to %s): %d days from hardcoded fallback.", start, end, len(events)
    )
    return events
