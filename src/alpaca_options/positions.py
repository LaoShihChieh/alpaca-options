"""
positions.py — Option position management and P&L reporting.

Wraps Alpaca's ``get_all_positions`` and ``close_all_positions`` to surface
only options (``asset_class == "us_option"``), and computes a per-position
P&L summary as a ``pandas.DataFrame``.

``close_all_options`` accepts a ``dry_run: bool = False`` keyword argument.
When ``dry_run=True`` the close request is logged but not submitted.

Usage::

    from alpaca_options.positions import list_option_positions, close_all_options

    df = list_option_positions()
    print(df.to_string())

    # Emergency close of all options (paper only)
    close_all_options(cancel_orders=True, dry_run=False)
"""

from __future__ import annotations

import logging
from typing import List, Optional

import pandas as pd
from alpaca.trading.models import ClosePositionResponse, Position

from alpaca_options.client import get_clients

logger = logging.getLogger(__name__)

_OPTION_ASSET_CLASS = "us_option"


def list_option_positions() -> pd.DataFrame:
    """Return a DataFrame of current open option positions with P&L columns.

    The DataFrame has one row per open option position and the following
    columns:

    ============= ============================================================
    symbol        OCC option symbol
    qty           Number of contracts (positive = long)
    side          ``"long"`` or ``"short"``
    avg_entry     Average fill price (per share)
    current_price Mark price (per share)
    market_value  Total mark-to-market value (USD)
    cost_basis    Total cost to enter the position (USD)
    unrealized_pl Unrealised P&L in USD
    unrealized_plpc Unrealised P&L as a decimal fraction
    ============= ============================================================

    Returns
    -------
    pd.DataFrame
        Empty DataFrame if no option positions are open.
    """
    trading, _ = get_clients()
    all_positions: List[Position] = trading.get_all_positions()  # type: ignore[assignment]

    option_positions = [p for p in all_positions if str(p.asset_class) == _OPTION_ASSET_CLASS]

    logger.info(
        "Found %d option position(s) out of %d total positions.",
        len(option_positions),
        len(all_positions),
    )

    if not option_positions:
        return pd.DataFrame(
            columns=[
                "symbol", "qty", "side", "avg_entry", "current_price",
                "market_value", "cost_basis", "unrealized_pl", "unrealized_plpc",
            ]
        )

    rows = []
    for pos in option_positions:
        unrealized_pl = float(pos.unrealized_pl or 0)
        unrealized_plpc = float(pos.unrealized_plpc or 0)
        rows.append(
            {
                "symbol": pos.symbol,
                "qty": float(pos.qty or 0),
                "side": str(pos.side.value) if pos.side else "unknown",
                "avg_entry": float(pos.avg_entry_price or 0),
                "current_price": float(pos.current_price or 0),
                "market_value": float(pos.market_value or 0),
                "cost_basis": float(pos.cost_basis or 0),
                "unrealized_pl": unrealized_pl,
                "unrealized_plpc": unrealized_plpc,
            }
        )

    df = pd.DataFrame(rows)
    df.sort_values("symbol", inplace=True, ignore_index=True)
    return df


def close_all_options(
    cancel_orders: bool = True,
    dry_run: bool = False,
) -> Optional[List[ClosePositionResponse]]:
    """Close all open option positions immediately.

    Sends a ``close_all_positions`` request and filters the response to only
    option-related close orders.  Non-option positions are unaffected on
    the Alpaca side (Alpaca's endpoint closes *all* positions, but this
    wrapper is labelled to signal intent).

    .. warning::
        This closes **all** positions in the account — not just options.
        Use with care even in paper mode.

    Parameters
    ----------
    cancel_orders:
        If ``True`` (default), cancel any open orders before closing positions
        so they don't interfere.
    dry_run:
        When ``True``, log what would be submitted and return ``None`` without
        touching the Alpaca API.

    Returns
    -------
    List[ClosePositionResponse] | None
        One response object per closed position, or ``None`` when
        ``dry_run=True``.
    """
    if dry_run:
        logger.info(
            "[DRY RUN] close_all_options: would submit CLOSE ALL POSITIONS  "
            "cancel_orders=%s",
            cancel_orders,
        )
        return None

    trading, _ = get_clients()
    responses: List[ClosePositionResponse] = trading.close_all_positions(  # type: ignore[assignment]
        cancel_orders=cancel_orders
    )
    logger.warning(
        "close_all_options called — %d position(s) submitted for closing.",
        len(responses),
    )
    return responses
