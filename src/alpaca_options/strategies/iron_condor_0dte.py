"""
strategies/iron_condor_0dte.py — Same-day-expiry iron condor on SPY.

Strategy overview
-----------------
An iron condor sells a put-spread below spot and a call-spread above spot.
We collect the net credit upfront; it's ours to keep if SPY stays inside
the short strikes at expiration.

0DTE specifics:
  - Trade only SPY (high liquidity, 5 expiries/week).
  - Filter by: time window (10:00–14:00 ET), VIX < 25, no event days.
  - Strikes selected via Black-Scholes delta targeting.
  - Exits: 50% profit, 2× stop, delta-breach, or 15:30 ET force-close.

Types
-----
CondorLegs     — immutable description of the 4-leg structure at open time.
CondorPosition — mutable live position (updated as market moves).
ExitDecision   — enum returned by ``monitor()``.

Usage (live, paper only)::

    strategy = IronCondor0DTE()
    legs = strategy.build_condor("SPY")
    if legs:
        order_id = strategy.enter(legs)
        position = CondorPosition(legs=legs, order_id=order_id, ...)
        decision = strategy.monitor(position)
        if decision != ExitDecision.HOLD:
            strategy.exit(position, decision)
"""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from enum import Enum
from typing import Optional
from zoneinfo import ZoneInfo

from dotenv import load_dotenv

from alpaca_options.client import get_clients
from alpaca_options.contracts import get_option_contracts
from alpaca_options.data.vix import get_current_vix
from alpaca_options.quotes import get_latest_quote, midpoint
from alpaca_options.utils.black_scholes import bs_delta, strike_for_delta

load_dotenv()
logger = logging.getLogger(__name__)

ET = ZoneInfo("America/New_York")

# SPY option strikes are $1 increments (round to nearest $1)
_STRIKE_INCREMENT = 1.0


# ---------------------------------------------------------------------------
# VIX-adaptive parameter functions
# ---------------------------------------------------------------------------


def target_delta(vix: float) -> float:
    """Return the target absolute delta for the short legs based on current VIX.

    Compressed IV on calm days means 10-delta strikes are so far OTM that the
    real market premium falls below the minimum credit threshold.  Moving
    closer to ATM (higher delta) on low-vol days restores collectible credit
    while keeping the condor structure intact.

    Above 25 the RiskManager blocks entry entirely, so no bucket is needed.

    Parameters
    ----------
    vix:
        Current VIX level (e.g. 14.5 means 14.5%).

    Returns
    -------
    float
        Absolute delta for BS :func:`strike_for_delta` inversion.

    Examples
    --------
    >>> target_delta(10.0)
    0.2
    >>> target_delta(18.0)
    0.12
    """
    if vix <= 12:
        return 0.20
    if vix <= 16:
        return 0.15
    if vix <= 20:
        return 0.12
    return 0.10   # 20 < vix ≤ 25


def adaptive_wing_width(vix: float) -> float:
    """Return the condor wing width (short-to-long strike distance) based on VIX.

    Narrower wings on low-vol days keep max loss proportional to the smaller
    premium collected.  Wider wings at higher vol capture more credit against
    a broader expected move.  The 8 % min-credit-pct gate is applied to
    whatever wing this function returns.

    Parameters
    ----------
    vix:
        Current VIX level.

    Returns
    -------
    float
        Wing width in dollars (e.g. 5.0 → $500 gross max loss per contract).

    Examples
    --------
    >>> adaptive_wing_width(12.0)
    3.0
    >>> adaptive_wing_width(18.0)
    5.0
    >>> adaptive_wing_width(22.0)
    7.0
    """
    if vix <= 14:
        return 3.0
    if vix <= 20:
        return 5.0
    return 7.0    # vix > 20


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class IronCondorConfig:
    """Tunable parameters for :class:`IronCondor0DTE`.

    Attributes
    ----------
    short_delta:
        Target absolute delta for the short legs (0 < delta < 0.5).
        Default 0.10 = ~10-delta, roughly 1 SD OTM for 0DTE.
    wing_width:
        Distance in points from short strike to long strike.
        Defines max loss: wing_width − net_credit (per share).
        Default $5 (so max loss ≈ $500/contract before credit).
    min_credit_pct:
        Minimum net credit as a fraction of wing_width.
        If credit < wing_width × min_credit_pct, skip the trade.
        Default 0.08 → need ≥ $0.40 on a $5-wide condor.
    profit_target_pct:
        Close when we've captured this fraction of the initial credit.
        Default 0.50 (close at 50% profit).
    stop_loss_multiplier:
        Close when spread cost reaches this multiple of initial credit.
        Default 2.0 → close when we'd lose 1× initial credit.
    max_short_delta_breach:
        Close when the short leg's delta exceeds this (in absolute value),
        indicating the underlying is approaching the short strike.
        Default 0.25.
    min_vix:
        Skip entry when VIX is below this level.  At very low VIX the
        credit available (even at higher delta) is insufficient to justify
        the closer-to-ATM risk.  Backtest evidence (per-VIX-bucket analysis)
        shows the 12–16 bucket has negative expected value.
        Default 16.0.
    """

    short_delta: float = 0.10
    wing_width: float = 5.0
    min_credit_pct: float = 0.08
    profit_target_pct: float = 0.50
    stop_loss_multiplier: float = 2.0
    max_short_delta_breach: float = 0.25
    min_vix: float = 16.0


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CondorLegs:
    """Immutable description of the four-leg condor structure.

    The four strikes satisfy:
        put_long < put_short < spot < call_short < call_long

    ``net_credit`` is the total credit per share (in dollars) received at open.
    """

    put_long_symbol: str
    put_short_symbol: str
    call_short_symbol: str
    call_long_symbol: str

    put_long_strike: float
    put_short_strike: float
    call_short_strike: float
    call_long_strike: float

    net_credit: float  # per share, USD


@dataclass
class CondorPosition:
    """Live iron condor position — updated as the market moves.

    Parameters
    ----------
    legs:
        Immutable 4-leg structure from :func:`IronCondor0DTE.build_condor`.
    order_id:
        Alpaca order UUID returned by :func:`IronCondor0DTE.enter`.
    entry_time:
        Datetime (ET) when the order was submitted.
    entry_credit:
        Net credit per share received at open (copy from legs.net_credit).
    current_value:
        Current cost-to-close per share (updated by the monitoring loop).
    underlying_price:
        Latest underlying spot price (updated by the monitoring loop).
    """

    legs: CondorLegs
    order_id: str
    entry_time: datetime
    entry_credit: float       # per share
    current_value: float      # current cost to close, per share
    underlying_price: float   # current spot


# ---------------------------------------------------------------------------
# Exit decision
# ---------------------------------------------------------------------------


class ExitDecision(str, Enum):
    """Decision returned by :meth:`IronCondor0DTE.monitor`."""

    HOLD = "hold"
    CLOSE_PROFIT = "close_profit"      # profit target reached
    CLOSE_STOP = "close_stop"          # stop-loss hit
    CLOSE_DELTA_BREACH = "close_delta_breach"  # short leg delta too high
    CLOSE_TIME = "close_time"          # end-of-day force-close


# ---------------------------------------------------------------------------
# Strategy class
# ---------------------------------------------------------------------------


class IronCondor0DTE:
    """0-DTE iron condor strategy on SPY (paper-only).

    Parameters
    ----------
    config:
        Strategy configuration.  Defaults to :class:`IronCondorConfig`.
    risk_free_rate:
        Annual risk-free rate used in Black-Scholes (default 5%).
    default_iv:
        Implied volatility used for strike selection when VIX is unavailable
        (default 20 %).
    """

    def __init__(
        self,
        config: Optional[IronCondorConfig] = None,
        risk_free_rate: float = 0.05,
        default_iv: float = 0.20,
    ) -> None:
        self.config = config or IronCondorConfig()
        self.risk_free_rate = risk_free_rate
        self.default_iv = default_iv
        self._trading, self._data = get_clients()

    # ------------------------------------------------------------------
    # Entry gate
    # ------------------------------------------------------------------

    def should_enter(self, now: datetime, vix: float) -> bool:
        """Return ``True`` if this is a valid time to open a new condor.

        Rules (evaluated in order):
          1. VIX floor — if VIX < ``config.min_vix`` (default 16), skip.
             Below this level premium is too thin relative to the directional
             risk imposed by moving strikes closer to ATM.
          2. Time window — market must be between 10:00 and 14:00 ET.
          The VIX ceiling (≥ 25) is enforced by the RiskManager.

        Parameters
        ----------
        now:
            Current datetime (any timezone — converted to ET internally).
        vix:
            Current VIX level.
        """
        # 1. VIX floor
        if vix < self.config.min_vix:
            logger.info(
                "should_enter: SKIP — VIX %.1f < min_vix %.1f",
                vix, self.config.min_vix,
            )
            return False

        # 2. Time window
        et_now = now.astimezone(ET)
        entry_open = et_now.replace(hour=10, minute=0, second=0, microsecond=0)
        entry_close = et_now.replace(hour=14, minute=0, second=0, microsecond=0)
        in_window = entry_open <= et_now <= entry_close
        logger.debug(
            "should_enter: VIX=%.1f, ET=%s, window=[10:00, 14:00], in_window=%s",
            vix, et_now.strftime("%H:%M"), in_window,
        )
        return in_window

    # ------------------------------------------------------------------
    # Build condor
    # ------------------------------------------------------------------

    def _get_underlying_price(self, symbol: str) -> float:
        """Fetch the current mid-price of *symbol* using Alpaca stock data."""
        from alpaca.data.historical.stock import StockHistoricalDataClient
        from alpaca.data.requests import StockLatestQuoteRequest

        api_key = os.environ.get("ALPACA_API_KEY", "")
        secret_key = os.environ.get("ALPACA_SECRET_KEY", "")
        stock_client = StockHistoricalDataClient(api_key=api_key, secret_key=secret_key)

        req = StockLatestQuoteRequest(symbol_or_symbols=symbol)
        result = stock_client.get_stock_latest_quote(req)
        q = result.get(symbol)
        if q is None:
            raise RuntimeError(f"No quote returned for {symbol}")
        bid = float(q.bid_price or 0)
        ask = float(q.ask_price or 0)
        return (bid + ask) / 2 if (bid > 0 and ask > 0) else max(bid, ask)

    def _hours_remaining(self) -> float:
        """Hours from now until 4:00 PM ET (option expiry)."""
        now_et = datetime.now(ET)
        close_et = now_et.replace(hour=16, minute=0, second=0, microsecond=0)
        delta = (close_et - now_et).total_seconds()
        return max(delta / 3600, 0.01)

    def _T_from_hours(self, hours: float) -> float:
        """Convert hours remaining to years (for Black-Scholes)."""
        return max(hours / (6.5 * 252), 1e-6)

    def _round_to_strike(self, price: float) -> float:
        """Round *price* to the nearest valid SPY strike ($1 increment)."""
        return round(price / _STRIKE_INCREMENT) * _STRIKE_INCREMENT

    def _fetch_contract(
        self, underlying: str, exp: date, strike: float, is_call: bool
    ) -> Optional[object]:
        """Fetch a specific option contract by underlying/expiry/strike/type."""
        from alpaca.trading.enums import ContractType

        ct = ContractType.CALL if is_call else ContractType.PUT
        contracts = get_option_contracts(
            underlying_symbol=underlying,
            expiration_gte=exp,
            expiration_lte=exp,
            strike_gte=strike - 0.5,
            strike_lte=strike + 0.5,
            contract_type=ct,
            limit=5,
        )
        if not contracts:
            return None
        return min(contracts, key=lambda c: abs(float(c.strike_price or 0) - strike))

    def build_condor(self, underlying: str = "SPY") -> Optional[CondorLegs]:
        """Fetch today's 0DTE chain and build the 4-leg condor structure.

        Steps:
          1. Get current underlying price.
          2. Compute target strikes via Black-Scholes delta inversion.
          3. Fetch each contract from the live chain.
          4. Fetch quotes for all 4 legs; compute net credit.
          5. Return ``None`` if net credit < ``min_credit_pct × wing_width``.

        Parameters
        ----------
        underlying:
            Ticker symbol (default ``"SPY"``).

        Returns
        -------
        CondorLegs | None
            Populated legs structure, or ``None`` if the trade doesn't meet
            minimum credit requirements.
        """
        today = datetime.now(ET).date()
        hours_rem = self._hours_remaining()
        T = self._T_from_hours(hours_rem)

        # Step 1 — current underlying price
        try:
            spot = self._get_underlying_price(underlying)
        except Exception as exc:
            logger.error("Could not fetch %s price: %s", underlying, exc)
            return None
        logger.info("build_condor: %s spot=%.2f, T=%.4f yr, hours_rem=%.1f", underlying, spot, T, hours_rem)

        # Step 2 — VIX-adaptive BS strike selection
        try:
            vix = get_current_vix()
        except Exception:
            vix = self.default_iv * 100
        sigma = vix / 100.0  # annualised vol

        eff_delta = target_delta(vix)
        eff_wing  = adaptive_wing_width(vix)
        logger.info(
            "VIX=%.1f → eff_delta=%.2f  eff_wing=%.1f",
            vix, eff_delta, eff_wing,
        )

        try:
            call_short_raw = strike_for_delta(
                S=spot, T=T, r=self.risk_free_rate, sigma=sigma,
                delta=eff_delta, is_call=True,
            )
            put_short_raw = strike_for_delta(
                S=spot, T=T, r=self.risk_free_rate, sigma=sigma,
                delta=-eff_delta, is_call=False,
            )
        except Exception as exc:
            logger.error("Strike calculation failed: %s", exc)
            return None

        call_short_strike = self._round_to_strike(call_short_raw)
        put_short_strike = self._round_to_strike(put_short_raw)
        call_long_strike = call_short_strike + eff_wing
        put_long_strike = put_short_strike - eff_wing

        logger.info(
            "Target strikes — put_long=%.0f / put_short=%.0f / call_short=%.0f / call_long=%.0f",
            put_long_strike, put_short_strike, call_short_strike, call_long_strike,
        )

        # Step 3 — fetch contracts
        contracts = {}
        for label, strike, is_call in [
            ("put_long",   put_long_strike,   False),
            ("put_short",  put_short_strike,  False),
            ("call_short", call_short_strike, True),
            ("call_long",  call_long_strike,  True),
        ]:
            c = self._fetch_contract(underlying, today, strike, is_call)
            if c is None:
                logger.warning("Contract not found for %s strike=%.0f exp=%s", label, strike, today)
                return None
            contracts[label] = c
            logger.debug("%s → %s (strike=%.1f)", label, c.symbol, float(c.strike_price or 0))

        # Step 4 — quotes and net credit
        credit_components: dict[str, float] = {}
        for label, sign in [
            ("call_short", +1), ("put_short", +1),    # sold
            ("call_long",  -1), ("put_long",  -1),    # bought
        ]:
            sym = contracts[label].symbol
            try:
                q = get_latest_quote(sym)
                mid = midpoint(q)
            except Exception as exc:
                logger.warning("Quote unavailable for %s: %s — using BS estimate", sym, exc)
                k = float(contracts[label].strike_price or 0)
                is_call = label.startswith("call")
                mid = max(
                    bs_delta(spot, k, T, self.risk_free_rate, sigma, is_call=is_call) * 0.01,
                    0.01,
                )
            credit_components[label] = sign * mid

        net_credit = sum(credit_components.values())
        min_credit = eff_wing * self.config.min_credit_pct

        logger.info(
            "Net credit: $%.4f (min required: $%.4f, wing=%.1f, delta=%.2f)",
            net_credit, min_credit, eff_wing, eff_delta,
        )

        if net_credit < min_credit:
            logger.info(
                "Credit $%.4f < minimum $%.4f — skipping condor.", net_credit, min_credit
            )
            return None

        return CondorLegs(
            put_long_symbol=contracts["put_long"].symbol,
            put_short_symbol=contracts["put_short"].symbol,
            call_short_symbol=contracts["call_short"].symbol,
            call_long_symbol=contracts["call_long"].symbol,
            put_long_strike=float(contracts["put_long"].strike_price or put_long_strike),
            put_short_strike=float(contracts["put_short"].strike_price or put_short_strike),
            call_short_strike=float(contracts["call_short"].strike_price or call_short_strike),
            call_long_strike=float(contracts["call_long"].strike_price or call_long_strike),
            net_credit=net_credit,
        )

    # ------------------------------------------------------------------
    # Order submission
    # ------------------------------------------------------------------

    def enter(self, legs: CondorLegs, qty: int = 1) -> str:
        """Submit a 4-leg MLEG limit order to open the iron condor.

        The limit price is the net credit (minimum we'll accept).

        Parameters
        ----------
        legs:
            :class:`CondorLegs` from :meth:`build_condor`.
        qty:
            Number of condor contracts (default 1).

        Returns
        -------
        str
            Alpaca order UUID string.
        """
        from alpaca.trading.enums import OrderClass, OrderSide, PositionIntent, TimeInForce
        from alpaca.trading.requests import LimitOrderRequest, OptionLegRequest

        order_legs = [
            OptionLegRequest(
                symbol=legs.put_long_symbol,
                ratio_qty=1,
                side=OrderSide.BUY,
                position_intent=PositionIntent.BUY_TO_OPEN,
            ),
            OptionLegRequest(
                symbol=legs.put_short_symbol,
                ratio_qty=1,
                side=OrderSide.SELL,
                position_intent=PositionIntent.SELL_TO_OPEN,
            ),
            OptionLegRequest(
                symbol=legs.call_short_symbol,
                ratio_qty=1,
                side=OrderSide.SELL,
                position_intent=PositionIntent.SELL_TO_OPEN,
            ),
            OptionLegRequest(
                symbol=legs.call_long_symbol,
                ratio_qty=1,
                side=OrderSide.BUY,
                position_intent=PositionIntent.BUY_TO_OPEN,
            ),
        ]

        req = LimitOrderRequest(
            symbol=legs.put_short_symbol,  # primary symbol required by Alpaca
            qty=qty,
            side=OrderSide.SELL,
            time_in_force=TimeInForce.DAY,
            order_class=OrderClass.MLEG,
            limit_price=round(legs.net_credit, 2),
            legs=order_legs,
        )

        order = self._trading.submit_order(req)
        order_id = str(order.id)
        logger.info(
            "Condor opened: id=%s credit=$%.2f strikes=[%.0f/%.0f/%.0f/%.0f]",
            order_id, legs.net_credit,
            legs.put_long_strike, legs.put_short_strike,
            legs.call_short_strike, legs.call_long_strike,
        )
        return order_id

    # ------------------------------------------------------------------
    # Monitoring
    # ------------------------------------------------------------------

    def monitor(
        self,
        position: CondorPosition,
        now: Optional[datetime] = None,
    ) -> ExitDecision:
        """Evaluate exit conditions for an open position.

        Checks in priority order:
          1. Time — past 15:30 ET → force-close.
          2. Profit target — current_value ≤ entry_credit × (1 − profit_target_pct).
          3. Stop-loss — current_value ≥ entry_credit × stop_loss_multiplier.
          4. Delta-breach — short leg delta ≥ max_short_delta_breach.

        Parameters
        ----------
        position:
            Live position with up-to-date ``current_value`` and
            ``underlying_price``.
        now:
            Current datetime (defaults to ``datetime.now(ET)``).

        Returns
        -------
        ExitDecision
        """
        if now is None:
            now = datetime.now(ET)
        et_now = now.astimezone(ET)

        # 1. Time-based close (15:30 ET)
        force_close = et_now.replace(hour=15, minute=30, second=0, microsecond=0)
        if et_now >= force_close:
            logger.info("monitor: CLOSE_TIME (%s ≥ 15:30 ET)", et_now.strftime("%H:%M"))
            return ExitDecision.CLOSE_TIME

        credit = position.entry_credit
        cost = position.current_value

        # 2. Profit target
        if credit > 0 and cost <= credit * (1.0 - self.config.profit_target_pct):
            profit_pct = (credit - cost) / credit * 100
            logger.info(
                "monitor: CLOSE_PROFIT — captured %.1f%% of credit ($%.4f → $%.4f)",
                profit_pct, credit, cost,
            )
            return ExitDecision.CLOSE_PROFIT

        # 3. Stop-loss
        if credit > 0 and cost >= credit * self.config.stop_loss_multiplier:
            logger.warning(
                "monitor: CLOSE_STOP — spread cost $%.4f ≥ %.1f× credit $%.4f",
                cost, self.config.stop_loss_multiplier, credit,
            )
            return ExitDecision.CLOSE_STOP

        # 4. Short-leg delta breach
        spot = position.underlying_price
        hours_rem = max((force_close - et_now).total_seconds() / 3600, 0.01)
        T = self._T_from_hours(hours_rem)

        try:
            sigma = get_current_vix() / 100.0
        except Exception:
            sigma = self.default_iv

        call_delta = abs(bs_delta(spot, position.legs.call_short_strike, T, self.risk_free_rate, sigma, is_call=True))
        put_delta = abs(bs_delta(spot, position.legs.put_short_strike, T, self.risk_free_rate, sigma, is_call=False))
        max_delta = max(call_delta, put_delta)

        if max_delta >= self.config.max_short_delta_breach:
            logger.warning(
                "monitor: CLOSE_DELTA_BREACH — short leg delta %.3f ≥ %.3f",
                max_delta, self.config.max_short_delta_breach,
            )
            return ExitDecision.CLOSE_DELTA_BREACH

        logger.debug(
            "monitor: HOLD — cost=$%.4f credit=$%.4f, call_Δ=%.3f, put_Δ=%.3f",
            cost, credit, call_delta, put_delta,
        )
        return ExitDecision.HOLD

    # ------------------------------------------------------------------
    # Exit
    # ------------------------------------------------------------------

    def exit(self, position: CondorPosition, reason: ExitDecision, qty: int = 1) -> str:
        """Submit a 4-leg MLEG market order to close the condor.

        Parameters
        ----------
        position:
            The open :class:`CondorPosition`.
        reason:
            Why we're exiting (for logging).
        qty:
            Number of contracts to close (default 1).

        Returns
        -------
        str
            Alpaca closing order UUID.
        """
        from alpaca.trading.enums import OrderClass, OrderSide, PositionIntent, TimeInForce
        from alpaca.trading.requests import MarketOrderRequest, OptionLegRequest

        legs = position.legs
        closing_legs = [
            OptionLegRequest(
                symbol=legs.put_long_symbol,
                ratio_qty=1,
                side=OrderSide.SELL,
                position_intent=PositionIntent.SELL_TO_CLOSE,
            ),
            OptionLegRequest(
                symbol=legs.put_short_symbol,
                ratio_qty=1,
                side=OrderSide.BUY,
                position_intent=PositionIntent.BUY_TO_CLOSE,
            ),
            OptionLegRequest(
                symbol=legs.call_short_symbol,
                ratio_qty=1,
                side=OrderSide.BUY,
                position_intent=PositionIntent.BUY_TO_CLOSE,
            ),
            OptionLegRequest(
                symbol=legs.call_long_symbol,
                ratio_qty=1,
                side=OrderSide.SELL,
                position_intent=PositionIntent.SELL_TO_CLOSE,
            ),
        ]

        req = MarketOrderRequest(
            symbol=legs.put_short_symbol,
            qty=qty,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.DAY,
            order_class=OrderClass.MLEG,
            legs=closing_legs,
        )

        order = self._trading.submit_order(req)
        order_id = str(order.id)
        logger.info(
            "Condor closed: id=%s reason=%s pnl_estimate=$%.2f",
            order_id,
            reason.value,
            (position.entry_credit - position.current_value) * 100,
        )
        return order_id
