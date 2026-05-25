"""
risk/manager.py — Trade-gate and portfolio-level risk management.

``RiskManager`` is a stateful object that tracks intraday P&L, drawdown,
and open-position count.  Before entering any new trade, call
``check_entry_allowed()``; after a trade resolves, call
``record_trade_result()``.

All monetary values are in USD.

Usage::

    from alpaca_options.risk.manager import RiskManager
    from alpaca_options.data.calendar import get_event_days
    from alpaca_options.data.vix import get_current_vix
    from datetime import date

    risk = RiskManager(max_loss_per_trade=500.0, vix_threshold=25.0, min_vix=16.0)
    risk.reset_day(account_value=100_000.0)

    allowed, reason = risk.check_entry_allowed(
        account_value=100_000.0,
        vix=get_current_vix(),
        today=date.today(),
        calendar_events=get_event_days(date.today(), date.today()),
    )
    if allowed:
        # submit order …
        risk.open_position()
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Optional

logger = logging.getLogger(__name__)


class RiskManager:
    """Gate-keeper for iron condor entries.

    Parameters
    ----------
    vix_threshold:
        Block entry when VIX ≥ this level (default 25).
    min_vix:
        Block entry when VIX < this level (default 16).  At low VIX the
        available premium is insufficient to justify the closer-to-ATM risk
        imposed by moving short strikes inward.  Per-bucket backtest evidence
        shows the 12–16 range has negative expected value.
    max_daily_loss_multiplier:
        Block entry when cumulative daily losses ≥ ``max_loss_per_trade ×
        max_daily_loss_multiplier`` (default 2 — i.e. 2 losing trades worth).
    max_drawdown_pct:
        Kill-switch: block all new entries when portfolio has fallen ≥ this
        fraction below its peak equity (default 10 %).
    max_concurrent_positions:
        Maximum number of open condors at any one time (default 1).
    max_loss_per_trade:
        Expected worst-case loss per condor in USD (default $500).  Used to
        calibrate the daily loss limit.
    """

    def __init__(
        self,
        vix_threshold: float = 25.0,
        min_vix: float = 16.0,
        max_daily_loss_multiplier: float = 2.0,
        max_drawdown_pct: float = 0.10,
        max_concurrent_positions: int = 1,
        max_loss_per_trade: float = 500.0,
    ) -> None:
        # Config
        self.vix_threshold = vix_threshold
        self.min_vix = min_vix
        self.max_daily_loss_multiplier = max_daily_loss_multiplier
        self.max_drawdown_pct = max_drawdown_pct
        self.max_concurrent_positions = max_concurrent_positions
        self.max_loss_per_trade = max_loss_per_trade

        # State
        self.peak_equity: float = 0.0
        self.current_equity: float = 0.0
        self.trades_today: int = 0
        self.losses_today: float = 0.0   # cumulative dollar losses today
        self._open_positions: int = 0
        self._trade_date: Optional[date] = None

    # ------------------------------------------------------------------
    # Day lifecycle
    # ------------------------------------------------------------------

    def reset_day(self, account_value: float) -> None:
        """Reset intraday counters and update equity for a new trading day.

        Call this once at the start of each trading session.

        Parameters
        ----------
        account_value:
            Current portfolio NAV (from the broker).
        """
        today = date.today()
        if self._trade_date != today:
            self.trades_today = 0
            self.losses_today = 0.0
            self._trade_date = today
            logger.info("RiskManager: day reset for %s.", today)

        self.current_equity = account_value
        if account_value > self.peak_equity:
            self.peak_equity = account_value
            logger.debug("New peak equity: $%.2f", self.peak_equity)

    # ------------------------------------------------------------------
    # Trade entry gate
    # ------------------------------------------------------------------

    def check_entry_allowed(
        self,
        account_value: float,
        vix: float,
        today: date,
        calendar_events: set[date],
    ) -> tuple[bool, str]:
        """Check all risk rules and return ``(allowed, reason)``.

        Rules are evaluated in priority order — the first failure is returned.

        Parameters
        ----------
        account_value:
            Current portfolio NAV (used for drawdown calculation).
        vix:
            Current VIX level.
        today:
            Today's date (compared against *calendar_events*).
        calendar_events:
            Set of high-impact event dates to skip.

        Returns
        -------
        tuple[bool, str]
            ``(True, "OK")`` when all rules pass; ``(False, reason)`` where
            *reason* describes the blocking rule.
        """
        # Update current equity from the broker value passed in
        self.current_equity = account_value
        if account_value > self.peak_equity:
            self.peak_equity = account_value

        # a. VIX regime filters (floor and ceiling)
        if vix < self.min_vix:
            reason = (
                f"VIX {vix:.1f} < min_vix {self.min_vix:.1f} — "
                "vol too low, premium insufficient."
            )
            logger.info("Entry blocked: %s", reason)
            return False, "vix_too_low"

        if vix >= self.vix_threshold:
            reason = (
                f"VIX {vix:.1f} ≥ threshold {self.vix_threshold:.1f} — "
                "elevated vol regime, skipping."
            )
            logger.info("Entry blocked: %s", reason)
            return False, reason

        # b. Economic calendar filter
        if today in calendar_events:
            reason = f"High-impact economic event on {today} — skipping."
            logger.info("Entry blocked: %s", reason)
            return False, reason

        # c. Daily loss limit (2× max-loss-per-trade)
        daily_loss_limit = self.max_loss_per_trade * self.max_daily_loss_multiplier
        if self.losses_today >= daily_loss_limit:
            reason = (
                f"Daily loss limit hit — losses today ${self.losses_today:.2f} "
                f"≥ limit ${daily_loss_limit:.2f}."
            )
            logger.warning("Entry blocked: %s", reason)
            return False, reason

        # d. Drawdown kill-switch
        if self.peak_equity > 0:
            drawdown = (self.peak_equity - account_value) / self.peak_equity
            if drawdown >= self.max_drawdown_pct:
                reason = (
                    f"Drawdown kill-switch triggered — {drawdown:.1%} drawdown from peak "
                    f"${self.peak_equity:,.2f} ≥ limit {self.max_drawdown_pct:.0%}."
                )
                logger.warning("Entry blocked: %s", reason)
                return False, reason

        # e. Max concurrent positions
        if self._open_positions >= self.max_concurrent_positions:
            reason = (
                f"Already at max concurrent positions ({self._open_positions})."
            )
            logger.info("Entry blocked: %s", reason)
            return False, reason

        logger.debug(
            "Entry allowed: VIX=%.1f, drawdown=%.1%%, losses_today=$%.2f, positions=%d",
            vix,
            (self.peak_equity - account_value) / max(self.peak_equity, 1) * 100,
            self.losses_today,
            self._open_positions,
        )
        return True, "OK"

    # ------------------------------------------------------------------
    # Position lifecycle tracking
    # ------------------------------------------------------------------

    def open_position(self) -> None:
        """Increment the open-position counter (call when an order is filled)."""
        self._open_positions += 1
        logger.debug("Position opened; open=%d", self._open_positions)

    def record_trade_result(self, pnl: float) -> None:
        """Update state after a trade closes.

        Parameters
        ----------
        pnl:
            Realised P&L in USD (positive = profit, negative = loss).
        """
        self.trades_today += 1
        self._open_positions = max(0, self._open_positions - 1)
        self.current_equity += pnl

        if pnl < 0:
            self.losses_today += abs(pnl)
            logger.info(
                "Trade result: LOSS $%.2f | losses today: $%.2f | equity: $%.2f",
                abs(pnl),
                self.losses_today,
                self.current_equity,
            )
        else:
            logger.info(
                "Trade result: PROFIT $%.2f | equity: $%.2f",
                pnl,
                self.current_equity,
            )

        if self.current_equity > self.peak_equity:
            self.peak_equity = self.current_equity

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def summary(self) -> dict:
        """Return a snapshot of current risk state as a plain dict."""
        drawdown = (
            (self.peak_equity - self.current_equity) / self.peak_equity
            if self.peak_equity > 0
            else 0.0
        )
        return {
            "peak_equity": self.peak_equity,
            "current_equity": self.current_equity,
            "drawdown_pct": drawdown,
            "trades_today": self.trades_today,
            "losses_today": self.losses_today,
            "open_positions": self._open_positions,
            "daily_loss_limit": self.max_loss_per_trade * self.max_daily_loss_multiplier,
        }
