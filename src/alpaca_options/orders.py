"""
orders.py — Order submission for single-leg and multi-leg option strategies.

All functions operate in **paper mode only** (enforced by ``get_clients()``).
Single-leg helpers use ``OrderClass.SIMPLE`` with ``PositionIntent``; multi-leg
strategies (vertical spreads, straddles) use ``OrderClass.MLEG`` with a list of
``OptionLegRequest`` objects.

Every public function accepts a ``dry_run: bool = False`` keyword argument.
When ``dry_run=True`` the order request is **logged but not submitted**;
the function returns ``None`` instead of an ``Order``.

Usage::

    from alpaca_options.orders import buy_to_open_limit, bull_call_spread

    # Dry run — prints order details, submits nothing
    order = buy_to_open_limit("SPY240620C00540000", qty=1, limit_price=3.50, dry_run=True)
    assert order is None

    # Live paper order
    order = buy_to_open_limit("SPY240620C00540000", qty=1, limit_price=3.50, dry_run=False)
    spread = bull_call_spread(
        long_symbol="SPY240620C00540000",
        short_symbol="SPY240620C00545000",
        qty=1,
        net_debit=2.00,
        dry_run=False,
    )
"""

from __future__ import annotations

import logging
from typing import Optional

from alpaca.trading.enums import OrderClass, OrderSide, PositionIntent, TimeInForce
from alpaca.trading.models import Order
from alpaca.trading.requests import LimitOrderRequest, MarketOrderRequest, OptionLegRequest

from alpaca_options.client import get_clients

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _log_order(order: Order) -> None:
    """Log an order submission result at INFO level."""
    logger.info(
        "Order submitted: id=%s symbol=%s side=%s class=%s status=%s",
        order.id,
        order.symbol,
        order.side,
        order.order_class,
        order.status,
    )


def _log_dry_run(fn_name: str, detail: str) -> None:
    """Log a dry-run record at INFO level — identical format for all callers."""
    logger.info("[DRY RUN] %s: would submit — %s", fn_name, detail)


# ---------------------------------------------------------------------------
# Single-leg orders
# ---------------------------------------------------------------------------

def buy_to_open_market(
    symbol: str,
    qty: int,
    dry_run: bool = False,
) -> Optional[Order]:
    """Buy *qty* contracts of *symbol* as a new long position (market order).

    Parameters
    ----------
    symbol:
        OCC option symbol to buy.
    qty:
        Number of contracts (each contract controls 100 shares).
    dry_run:
        When ``True``, log the order that would be sent and return ``None``
        without touching the Alpaca API.

    Returns
    -------
    Order | None
        The submitted order object, or ``None`` when ``dry_run=True``.
    """
    trading, _ = get_clients()
    req = MarketOrderRequest(
        symbol=symbol,
        qty=qty,
        side=OrderSide.BUY,
        time_in_force=TimeInForce.DAY,
        order_class=OrderClass.SIMPLE,
        position_intent=PositionIntent.BUY_TO_OPEN,
    )
    if dry_run:
        _log_dry_run(
            "buy_to_open_market",
            f"BUY TO OPEN  symbol={symbol}  qty={qty}  type=MARKET  class=SIMPLE",
        )
        return None
    order: Order = trading.submit_order(req)  # type: ignore[assignment]
    _log_order(order)
    return order


def buy_to_open_limit(
    symbol: str,
    qty: int,
    limit_price: float,
    dry_run: bool = False,
) -> Optional[Order]:
    """Buy *qty* contracts at or below *limit_price* (limit order).

    Parameters
    ----------
    symbol:
        OCC option symbol to buy.
    qty:
        Number of contracts.
    limit_price:
        Maximum price per contract willing to pay (per-share terms, i.e. $/share).
    dry_run:
        When ``True``, log the order that would be sent and return ``None``
        without touching the Alpaca API.

    Returns
    -------
    Order | None
        The submitted order object, or ``None`` when ``dry_run=True``.
    """
    trading, _ = get_clients()
    req = LimitOrderRequest(
        symbol=symbol,
        qty=qty,
        side=OrderSide.BUY,
        time_in_force=TimeInForce.DAY,
        order_class=OrderClass.SIMPLE,
        position_intent=PositionIntent.BUY_TO_OPEN,
        limit_price=round(limit_price, 2),
    )
    if dry_run:
        _log_dry_run(
            "buy_to_open_limit",
            f"BUY TO OPEN  symbol={symbol}  qty={qty}  type=LIMIT  "
            f"limit=${round(limit_price, 2):.2f}  class=SIMPLE",
        )
        return None
    order: Order = trading.submit_order(req)  # type: ignore[assignment]
    _log_order(order)
    return order


def sell_to_close_market(
    symbol: str,
    qty: int,
    dry_run: bool = False,
) -> Optional[Order]:
    """Sell *qty* contracts of an existing long position (market order).

    Parameters
    ----------
    symbol:
        OCC option symbol to close.
    qty:
        Number of contracts to sell.
    dry_run:
        When ``True``, log the order that would be sent and return ``None``
        without touching the Alpaca API.

    Returns
    -------
    Order | None
        The submitted order object, or ``None`` when ``dry_run=True``.
    """
    trading, _ = get_clients()
    req = MarketOrderRequest(
        symbol=symbol,
        qty=qty,
        side=OrderSide.SELL,
        time_in_force=TimeInForce.DAY,
        order_class=OrderClass.SIMPLE,
        position_intent=PositionIntent.SELL_TO_CLOSE,
    )
    if dry_run:
        _log_dry_run(
            "sell_to_close_market",
            f"SELL TO CLOSE  symbol={symbol}  qty={qty}  type=MARKET  class=SIMPLE",
        )
        return None
    order: Order = trading.submit_order(req)  # type: ignore[assignment]
    _log_order(order)
    return order


def sell_to_close_limit(
    symbol: str,
    qty: int,
    limit_price: float,
    dry_run: bool = False,
) -> Optional[Order]:
    """Sell *qty* contracts at or above *limit_price* (limit order).

    Parameters
    ----------
    symbol:
        OCC option symbol to close.
    qty:
        Number of contracts to sell.
    limit_price:
        Minimum price per contract willing to accept.
    dry_run:
        When ``True``, log the order that would be sent and return ``None``
        without touching the Alpaca API.

    Returns
    -------
    Order | None
        The submitted order object, or ``None`` when ``dry_run=True``.
    """
    trading, _ = get_clients()
    req = LimitOrderRequest(
        symbol=symbol,
        qty=qty,
        side=OrderSide.SELL,
        time_in_force=TimeInForce.DAY,
        order_class=OrderClass.SIMPLE,
        position_intent=PositionIntent.SELL_TO_CLOSE,
        limit_price=round(limit_price, 2),
    )
    if dry_run:
        _log_dry_run(
            "sell_to_close_limit",
            f"SELL TO CLOSE  symbol={symbol}  qty={qty}  type=LIMIT  "
            f"limit=${round(limit_price, 2):.2f}  class=SIMPLE",
        )
        return None
    order: Order = trading.submit_order(req)  # type: ignore[assignment]
    _log_order(order)
    return order


# ---------------------------------------------------------------------------
# Multi-leg orders (OrderClass.MLEG)
# ---------------------------------------------------------------------------

def bull_call_spread(
    long_symbol: str,
    short_symbol: str,
    qty: int,
    net_debit: Optional[float] = None,
    dry_run: bool = False,
) -> Optional[Order]:
    """Submit a bull-call-spread (buy lower strike call, sell higher strike call).

    The two legs are sent as a single MLEG order so Alpaca executes them
    simultaneously.

    Parameters
    ----------
    long_symbol:
        OCC symbol for the *lower-strike* call (the leg you buy).
    short_symbol:
        OCC symbol for the *higher-strike* call (the leg you sell).
    qty:
        Number of spread contracts.
    net_debit:
        If supplied, places a limit order at this net debit (per share).
        Otherwise a market order is used.
    dry_run:
        When ``True``, log the order that would be sent and return ``None``
        without touching the Alpaca API.

    Returns
    -------
    Order | None
        The submitted multi-leg order object, or ``None`` when ``dry_run=True``.
    """
    trading, _ = get_clients()

    legs = [
        OptionLegRequest(
            symbol=long_symbol,
            ratio_qty=1,
            side=OrderSide.BUY,
            position_intent=PositionIntent.BUY_TO_OPEN,
        ),
        OptionLegRequest(
            symbol=short_symbol,
            ratio_qty=1,
            side=OrderSide.SELL,
            position_intent=PositionIntent.SELL_TO_OPEN,
        ),
    ]

    if net_debit is not None:
        req = LimitOrderRequest(
            symbol=long_symbol,       # Alpaca requires a primary symbol even for MLEG
            qty=qty,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.DAY,
            order_class=OrderClass.MLEG,
            limit_price=round(net_debit, 2),
            legs=legs,
        )
        order_type = f"LIMIT  net_debit=${round(net_debit, 2):.2f}"
    else:
        req = MarketOrderRequest(
            symbol=long_symbol,
            qty=qty,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.DAY,
            order_class=OrderClass.MLEG,
            legs=legs,
        )
        order_type = "MARKET"

    if dry_run:
        _log_dry_run(
            "bull_call_spread",
            f"MLEG  qty={qty}  type={order_type}\n"
            f"    Leg 1  BUY  TO OPEN   {long_symbol}\n"
            f"    Leg 2  SELL TO OPEN   {short_symbol}",
        )
        return None
    order: Order = trading.submit_order(req)  # type: ignore[assignment]
    _log_order(order)
    return order


def straddle(
    call_symbol: str,
    put_symbol: str,
    qty: int,
    net_debit: Optional[float] = None,
    dry_run: bool = False,
) -> Optional[Order]:
    """Submit a long straddle (buy an ATM call and put with the same expiration/strike).

    Parameters
    ----------
    call_symbol:
        OCC symbol for the call leg.
    put_symbol:
        OCC symbol for the put leg (same strike and expiration as *call_symbol*).
    qty:
        Number of straddle contracts (buys *qty* calls + *qty* puts).
    net_debit:
        Combined limit price per share for both legs.  If ``None``, market order.
    dry_run:
        When ``True``, log the order that would be sent and return ``None``
        without touching the Alpaca API.

    Returns
    -------
    Order | None
        The submitted multi-leg order object, or ``None`` when ``dry_run=True``.
    """
    trading, _ = get_clients()

    legs = [
        OptionLegRequest(
            symbol=call_symbol,
            ratio_qty=1,
            side=OrderSide.BUY,
            position_intent=PositionIntent.BUY_TO_OPEN,
        ),
        OptionLegRequest(
            symbol=put_symbol,
            ratio_qty=1,
            side=OrderSide.BUY,
            position_intent=PositionIntent.BUY_TO_OPEN,
        ),
    ]

    if net_debit is not None:
        req = LimitOrderRequest(
            symbol=call_symbol,
            qty=qty,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.DAY,
            order_class=OrderClass.MLEG,
            limit_price=round(net_debit, 2),
            legs=legs,
        )
        order_type = f"LIMIT  net_debit=${round(net_debit, 2):.2f}"
    else:
        req = MarketOrderRequest(
            symbol=call_symbol,
            qty=qty,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.DAY,
            order_class=OrderClass.MLEG,
            legs=legs,
        )
        order_type = "MARKET"

    if dry_run:
        _log_dry_run(
            "straddle",
            f"MLEG  qty={qty}  type={order_type}\n"
            f"    Leg 1  BUY TO OPEN  {call_symbol}  (CALL)\n"
            f"    Leg 2  BUY TO OPEN  {put_symbol}   (PUT)",
        )
        return None
    order: Order = trading.submit_order(req)  # type: ignore[assignment]
    _log_order(order)
    return order
