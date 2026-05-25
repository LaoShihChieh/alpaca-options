"""
backtest/_occ.py — OCC option symbol construction.

Format: ``{underlying}{YYMMDD}{C|P}{XXXXXXXX}``
where ``XXXXXXXX`` is the strike × 1000, zero-padded to 8 digits.

Examples:
  SPY expiring 2025-06-02, call, strike $580.00 → ``SPY250602C00580000``
  SPY expiring 2025-06-02, put, strike $560.00  → ``SPY250602P00560000``
"""

from __future__ import annotations

from datetime import date


def occ_symbol(underlying: str, expiry: date, is_call: bool, strike: float) -> str:
    """Return the OCC standardised option symbol.

    Parameters
    ----------
    underlying:
        Ticker (e.g. ``"SPY"``).
    expiry:
        Expiration date.
    is_call:
        ``True`` for a call, ``False`` for a put.
    strike:
        Strike price in dollars (e.g. ``580.0``).

    Returns
    -------
    str
        OCC symbol string.
    """
    strike_int = round(strike * 1000)
    opt_type = "C" if is_call else "P"
    return f"{underlying.upper()}{expiry.strftime('%y%m%d')}{opt_type}{strike_int:08d}"
