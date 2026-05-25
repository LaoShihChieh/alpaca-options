"""
live/runner.py — Real-time iron-condor execution loop.

Features
--------
- Polls every ``poll_interval_entry`` seconds (default 60) when no position
  is open (looking for entry).
- Polls every ``poll_interval_monitor`` seconds (default 30) when a position
  is open (monitoring).  Shorter because exits need to react faster.
- Checks ``RiskManager.check_entry_allowed()`` and
  ``IronCondor0DTE.should_enter()`` before entry; logs VIX and the exact
  reject reason on every skipped tick.
- ``dry_run=True`` logs all decisions without submitting orders.  The full
  4-leg order (symbol, strike, expiry, side, qty, limit price) is logged so
  you can verify what *would* have been sent.
- Graceful SIGINT: closes any open live position, calls
  ``logging.shutdown()`` to flush all handlers, then ``sys.exit(0)``.

Usage::

    from alpaca_options.live.runner import LiveRunner

    runner = LiveRunner(dry_run=True)
    runner.run()   # blocks until SIGINT

Or via the example script:
    uv run python examples/live_dry_run.py
"""

from __future__ import annotations

import logging
import re
import signal
import sys
import time
from datetime import date, datetime
from typing import Optional
from zoneinfo import ZoneInfo

from alpaca_options.client import get_clients
from alpaca_options.data.calendar import get_event_days
from alpaca_options.data.vix import get_current_vix
from alpaca_options.risk.manager import RiskManager
from alpaca_options.strategies.iron_condor_0dte import (
    CondorLegs,
    CondorPosition,
    ExitDecision,
    IronCondor0DTE,
    IronCondorConfig,
)

logger = logging.getLogger(__name__)

ET = ZoneInfo("America/New_York")
_DEFAULT_POLL_ENTRY   =  60   # seconds between entry-scan ticks (market open, no position)
_DEFAULT_POLL_MONITOR =  30   # seconds between monitoring ticks (position open)
_DEFAULT_POLL_CLOSED  = 300   # seconds between ticks when market is closed (pre/after/weekend)


# ---------------------------------------------------------------------------
# Order formatting helpers (module-level so they are independently testable)
# ---------------------------------------------------------------------------

def _parse_expiry_from_occ(symbol: str) -> str:
    """Extract expiration as ``YYYY-MM-DD`` from an OCC option symbol.

    OCC format: ``{underlying}{YYMMDD}{C|P}{strike*1000:08d}``

    Example: ``SPY250602C00540000`` → ``"2025-06-02"``
    """
    try:
        m = re.search(r"[A-Z](\d{6})[CP]", symbol)
        if m:
            raw = m.group(1)   # YYMMDD
            return f"20{raw[:2]}-{raw[2:4]}-{raw[4:6]}"
    except Exception:
        pass
    return "?"


def _format_condor_order(legs: CondorLegs, qty: int = 1) -> str:
    """Return a multi-line dry-run order summary for the 4-leg condor.

    Shows symbol, option type, expiry, strike, side, qty, and the exact
    limit price (net credit rounded to 2 dp) that *would* be submitted.
    """
    exp    = _parse_expiry_from_occ(legs.call_short_symbol)
    limit  = round(legs.net_credit, 2)   # matches IronCondor0DTE.enter()
    w      = 26                           # symbol column width
    lines  = [
        "  ┌─ CONDOR ORDER (DRY RUN — NOT SUBMITTED) ─────────────────────────────",
        f"  │  Leg 1  BUY   {legs.put_long_symbol:<{w}}  PUT   exp={exp}  K={legs.put_long_strike:.0f}   qty={qty}",
        f"  │  Leg 2  SELL  {legs.put_short_symbol:<{w}}  PUT   exp={exp}  K={legs.put_short_strike:.0f}   qty={qty}",
        f"  │  Leg 3  SELL  {legs.call_short_symbol:<{w}}  CALL  exp={exp}  K={legs.call_short_strike:.0f}  qty={qty}",
        f"  │  Leg 4  BUY   {legs.call_long_symbol:<{w}}  CALL  exp={exp}  K={legs.call_long_strike:.0f}  qty={qty}",
        f"  │  Limit price (net credit): ${limit:.2f}/share  │  ${limit * 100 * qty:.2f}/contract",
        "  └───────────────────────────────────────────────────────────────────────",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

class LiveRunner:
    """Main execution loop for the 0DTE iron condor strategy.

    Parameters
    ----------
    dry_run:
        When ``True`` (default), all order logic is bypassed — decisions are
        logged but no orders are submitted.  Set to ``False`` only after
        thorough paper-trading validation.
    strategy:
        :class:`IronCondor0DTE` instance.  Created with default config if not
        supplied.
    risk:
        :class:`RiskManager` instance.  Created with defaults if not supplied.
    poll_interval_entry:
        Seconds between ticks when the market is open and no position is open
        (looking for entry).  Default 60.
    poll_interval_monitor:
        Seconds between ticks when a position is open (monitoring for exits).
        Default 30.  Shorter than entry because exits need to react faster.
    poll_interval_closed:
        Seconds between ticks when the market is closed (pre-market, after-hours,
        weekends).  No trading decisions are made in this state so a longer
        interval wastes nothing.  Default 300 (5 min).
    """

    def __init__(
        self,
        dry_run: bool = True,
        strategy: Optional[IronCondor0DTE] = None,
        risk: Optional[RiskManager] = None,
        poll_interval_entry: int = _DEFAULT_POLL_ENTRY,
        poll_interval_monitor: int = _DEFAULT_POLL_MONITOR,
        poll_interval_closed: int = _DEFAULT_POLL_CLOSED,
    ) -> None:
        self.dry_run               = dry_run
        self.strategy              = strategy or IronCondor0DTE()
        self.risk                  = risk or RiskManager()
        self.poll_interval_entry   = poll_interval_entry
        self.poll_interval_monitor = poll_interval_monitor
        self.poll_interval_closed  = poll_interval_closed

        self._trading, _          = get_clients()
        self._position: Optional[CondorPosition] = None
        self._shutdown             = False
        self._calendar_events: Optional[set[date]] = None
        self._last_calendar_date: Optional[date]   = None

        if dry_run:
            logger.info(
                "LiveRunner started in DRY-RUN mode — no orders will be submitted. "
                "entry_poll=%ds  monitor_poll=%ds  closed_poll=%ds",
                self.poll_interval_entry, self.poll_interval_monitor,
                self.poll_interval_closed,
            )
        else:
            logger.warning(
                "LiveRunner started in LIVE (paper) mode — orders WILL be submitted. "
                "entry_poll=%ds  monitor_poll=%ds  closed_poll=%ds",
                self.poll_interval_entry, self.poll_interval_monitor,
                self.poll_interval_closed,
            )

    # ------------------------------------------------------------------
    # Signal handling
    # ------------------------------------------------------------------

    def _handle_sigint(self, signum: int, frame) -> None:  # type: ignore[type-arg]
        logger.info("SIGINT received — initiating graceful shutdown.")
        self._shutdown = True

        if self._position is not None and not self.dry_run:
            logger.warning("SHUTDOWN: closing open live position before exit.")
            try:
                self.strategy.exit(self._position, ExitDecision.CLOSE_TIME)
                self.risk.record_trade_result(
                    (self._position.entry_credit - self._position.current_value) * 100
                )
            except Exception as exc:
                logger.error("SHUTDOWN: error closing position: %s", exc)
        elif self._position is not None:
            logger.info(
                "SHUTDOWN [DRY RUN]: would close position %s  estimated_pnl=$%.2f",
                self._position.legs.call_short_symbol,
                (self._position.entry_credit - self._position.current_value) * 100,
            )

        logger.info("SHUTDOWN: flushing log handlers.")
        logging.shutdown()
        sys.exit(0)

    # ------------------------------------------------------------------
    # Market hours guard
    # ------------------------------------------------------------------

    @staticmethod
    def _is_market_hours(now: datetime) -> bool:
        """Return True during official market hours (9:30–16:00 ET, Mon–Fri)."""
        et = now.astimezone(ET)
        market_open  = et.replace(hour=9,  minute=30, second=0, microsecond=0)
        market_close = et.replace(hour=16, minute=0,  second=0, microsecond=0)
        return market_open <= et <= market_close and et.weekday() < 5

    # ------------------------------------------------------------------
    # Account helpers
    # ------------------------------------------------------------------

    def _get_account_value(self) -> float:
        """Fetch current portfolio NAV from Alpaca."""
        try:
            account = self._trading.get_account()
            return float(account.portfolio_value or account.equity or 0)
        except Exception as exc:
            logger.error("Could not fetch account value: %s", exc)
            return self.risk.current_equity or 100_000.0

    # ------------------------------------------------------------------
    # Entry logic
    # ------------------------------------------------------------------

    def _try_enter(self, now: datetime) -> None:
        """Check all conditions and (optionally) submit a new condor order."""
        account_value = self._get_account_value()
        ts = now.strftime("%H:%M:%S")

        vix = 20.0  # fallback if fetch fails
        try:
            vix = get_current_vix()
        except Exception as exc:
            logger.warning(
                "[%s] TICK action=look_for_entry  vix=UNAVAILABLE (fallback=%.1f): %s",
                ts, vix, exc,
            )

        today = now.astimezone(ET).date()

        # ── Risk-manager gate ──────────────────────────────────────────────
        allowed, rm_reason = self.risk.check_entry_allowed(
            account_value=account_value,
            vix=vix,
            today=today,
            calendar_events=self._calendar_events or set(),
        )
        if not allowed:
            logger.info(
                "[%s] TICK action=look_for_entry  vix=%.1f  decision=SKIP  reason=risk_manager  detail=%r",
                ts, vix, rm_reason,
            )
            return

        # ── Strategy time-window / VIX-floor gate ─────────────────────────
        if not self.strategy.should_enter(now, vix):
            if vix < self.strategy.config.min_vix:
                se_reason = (
                    f"vix_floor (VIX={vix:.1f} < min_vix={self.strategy.config.min_vix:.1f})"
                )
            else:
                se_reason = "outside_time_window (entry window 10:00–14:00 ET)"
            logger.info(
                "[%s] TICK action=look_for_entry  vix=%.1f  decision=SKIP  reason=%s",
                ts, vix, se_reason,
            )
            return

        # ── Build condor ───────────────────────────────────────────────────
        logger.info(
            "[%s] TICK action=look_for_entry  vix=%.1f  decision=BUILD_CONDOR",
            ts, vix,
        )
        legs = self.strategy.build_condor()

        if legs is None:
            logger.info(
                "[%s] TICK action=look_for_entry  vix=%.1f  decision=SKIP  reason=credit_too_low",
                ts, vix,
            )
            return

        # ── Enter (or log) ─────────────────────────────────────────────────
        order_detail = _format_condor_order(legs)

        if self.dry_run:
            logger.info(
                "[%s] TICK action=look_for_entry  vix=%.1f  decision=ENTER  [DRY RUN]\n%s",
                ts, vix, order_detail,
            )
            self._position = CondorPosition(
                legs=legs,
                order_id="DRY-RUN",
                entry_time=now,
                entry_credit=legs.net_credit,
                current_value=legs.net_credit,
                underlying_price=0.0,
            )
            self.risk.open_position()
        else:
            order_id = self.strategy.enter(legs)
            logger.info(
                "[%s] TICK action=look_for_entry  vix=%.1f  decision=ENTER  order_id=%s\n%s",
                ts, vix, order_id, order_detail,
            )
            self._position = CondorPosition(
                legs=legs,
                order_id=order_id,
                entry_time=now,
                entry_credit=legs.net_credit,
                current_value=legs.net_credit,
                underlying_price=0.0,
            )
            self.risk.open_position()

    # ------------------------------------------------------------------
    # Exit logic
    # ------------------------------------------------------------------

    def _refresh_position_value(self) -> None:
        """Update current_value and underlying_price on the live position."""
        if self._position is None:
            return
        try:
            from alpaca_options.quotes import get_latest_quote, midpoint
            legs = self._position.legs

            cs = midpoint(get_latest_quote(legs.call_short_symbol))
            cl = midpoint(get_latest_quote(legs.call_long_symbol))
            ps = midpoint(get_latest_quote(legs.put_short_symbol))
            pl = midpoint(get_latest_quote(legs.put_long_symbol))
            self._position.current_value = (cs + ps) - (cl + pl)

            import os
            from alpaca.data.historical.stock import StockHistoricalDataClient
            from alpaca.data.requests import StockLatestQuoteRequest

            stock_client = StockHistoricalDataClient(
                api_key=os.environ.get("ALPACA_API_KEY", ""),
                secret_key=os.environ.get("ALPACA_SECRET_KEY", ""),
            )
            q = stock_client.get_stock_latest_quote(
                StockLatestQuoteRequest(symbol_or_symbols="SPY")
            )
            spy_q = q.get("SPY")
            if spy_q:
                bid = float(spy_q.bid_price or 0)
                ask = float(spy_q.ask_price or 0)
                self._position.underlying_price = (bid + ask) / 2

        except Exception as exc:
            logger.warning("Could not refresh position value: %s", exc)

    def _check_exit(self, now: datetime) -> None:
        """Monitor the open position and exit if a trigger is hit."""
        if self._position is None:
            return

        ts = now.strftime("%H:%M:%S")

        if not self.dry_run:
            self._refresh_position_value()
        else:
            # Simulate theta decay: 0.5% of credit per minute elapsed
            elapsed_mins = (now - self._position.entry_time).total_seconds() / 60
            self._position.current_value = max(
                self._position.entry_credit * (1 - elapsed_mins * 0.005),
                0.0,
            )

        decision = self.strategy.monitor(self._position, now=now)
        pnl      = (self._position.entry_credit - self._position.current_value) * 100

        logger.info(
            "[%s] TICK action=monitor  decision=%s  pnl_estimate=$%.2f  "
            "current_cost=$%.4f  entry_credit=$%.4f",
            ts, decision.value, pnl,
            self._position.current_value,
            self._position.entry_credit,
        )

        if decision == ExitDecision.HOLD:
            return

        if self.dry_run:
            logger.info(
                "[%s] TICK action=exit  [DRY RUN]  decision=%s  estimated_pnl=$%.2f",
                ts, decision.value, pnl,
            )
        else:
            self.strategy.exit(self._position, decision)
            logger.info(
                "[%s] TICK action=exit  decision=%s  estimated_pnl=$%.2f",
                ts, decision.value, pnl,
            )

        self.risk.record_trade_result(pnl)
        self._position = None

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def _tick(self, now: datetime) -> None:
        """One iteration of the main loop."""
        if not self._is_market_hours(now):
            et = now.astimezone(ET)
            day_type = "weekend" if et.weekday() >= 5 else "weekday"
            logger.info(
                "[%s] TICK action=skip  reason=market_closed  day=%s  "
                "next_open=09:30 ET Mon–Fri",
                et.strftime("%H:%M:%S"),
                day_type,
            )
            return

        # Refresh the event calendar once per calendar day (not every tick)
        today = now.astimezone(ET).date()
        if self._last_calendar_date != today:
            self._calendar_events     = get_event_days(today, today)
            self._last_calendar_date  = today

        account_value = self._get_account_value()
        self.risk.reset_day(account_value)

        if self._position is None:
            self._try_enter(now)
        else:
            self._check_exit(now)

    def run(self) -> None:
        """Block and run the trading loop until SIGINT.

        Uses ``poll_interval_entry`` when no position is open and
        ``poll_interval_monitor`` when a position is open.  On SIGINT the
        position is closed (live mode only), all log handlers are flushed
        via ``logging.shutdown()``, and the process exits cleanly.
        """
        signal.signal(signal.SIGINT,  self._handle_sigint)
        signal.signal(signal.SIGTERM, self._handle_sigint)

        logger.info(
            "LiveRunner.run() started — entry_poll=%ds  monitor_poll=%ds  dry_run=%s",
            self.poll_interval_entry, self.poll_interval_monitor, self.dry_run,
        )

        while not self._shutdown:
            now = datetime.now(ET)
            try:
                self._tick(now)
            except Exception as exc:
                logger.error("Unhandled error in tick: %s", exc, exc_info=True)

            # Three distinct sleep intervals — one per runner state:
            #   market closed        → poll_interval_closed  (default 300s)
            #   market open, no pos  → poll_interval_entry   (default  60s)
            #   market open, in pos  → poll_interval_monitor (default  30s)
            if not self._is_market_hours(now):
                interval = self.poll_interval_closed
            elif self._position is not None:
                interval = self.poll_interval_monitor
            else:
                interval = self.poll_interval_entry
            time.sleep(interval)
