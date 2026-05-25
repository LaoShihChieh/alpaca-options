"""
examples/smoke_test_5min.py
============================
Non-interactive 5-minute smoke test of LiveRunner in dry_run=True mode.

- No confirmation prompts.
- Poll every 15 s (entry) / 10 s (monitor) to generate ~20 ticks in 5 min.
- Terminates itself after RUN_SECONDS via SIGALRM (macOS/Linux).
- Writes to logs/smoke_YYYYMMDD_HHMMSS.log AND stdout.
- Exit code 0 on clean shutdown, 1 on unexpected error.

Run with:
    uv run python examples/smoke_test_5min.py
"""

from __future__ import annotations

import logging
import os
import signal
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

RUN_SECONDS = 300   # 5 minutes; SIGALRM fires and runner exits cleanly


# ---------------------------------------------------------------------------
# Logging: console + rotating file
# ---------------------------------------------------------------------------

def _setup_logging() -> Path:
    log_dir  = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp    = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"smoke_{stamp}.log"

    fmt       = "%(asctime)s  %(levelname)-8s  %(name)s: %(message)s"
    datefmt   = "%H:%M:%S"
    formatter = logging.Formatter(fmt, datefmt=datefmt)

    root = logging.getLogger()
    root.setLevel(logging.INFO)

    console_h = logging.StreamHandler(sys.stdout)
    console_h.setFormatter(formatter)
    root.addHandler(console_h)

    file_h = RotatingFileHandler(
        log_path, maxBytes=10 * 1024 * 1024, backupCount=5, encoding="utf-8"
    )
    file_h.setFormatter(formatter)
    root.addHandler(file_h)

    return log_path


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    log_path = _setup_logging()
    log = logging.getLogger(__name__)
    log.info("=== SMOKE TEST START  run_seconds=%d  log=%s ===", RUN_SECONDS, log_path)

    from alpaca_options.live.runner import LiveRunner
    from alpaca_options.risk.manager import RiskManager
    from alpaca_options.strategies.iron_condor_0dte import IronCondor0DTE, IronCondorConfig

    strategy = IronCondor0DTE(config=IronCondorConfig(min_vix=16.0))
    risk = RiskManager(
        vix_threshold=25.0,
        max_loss_per_trade=500.0,
        max_daily_loss_multiplier=2.0,
        max_drawdown_pct=0.10,
        max_concurrent_positions=1,
        min_vix=16.0,
    )
    runner = LiveRunner(
        dry_run=True,
        strategy=strategy,
        risk=risk,
        poll_interval_entry=15,    # shortened for smoke test (production: 60s)
        poll_interval_monitor=10,  # shortened for smoke test (production: 30s)
        poll_interval_closed=15,   # shortened for smoke test (production: 300s)
    )

    # SIGALRM fires after RUN_SECONDS, which triggers the SIGINT handler on
    # the runner (both are registered to _handle_sigint) — clean shutdown.
    signal.signal(signal.SIGALRM, runner._handle_sigint)
    signal.alarm(RUN_SECONDS)

    log.info(
        "Runner started — poll_entry=15s  poll_monitor=10s  "
        "auto_stop_in=%ds  dry_run=True",
        RUN_SECONDS,
    )
    log.info("Log file: %s", log_path.resolve())

    try:
        runner.run()
    except SystemExit:
        log.info("=== SMOKE TEST END (clean shutdown) ===")
        sys.exit(0)


if __name__ == "__main__":
    main()
