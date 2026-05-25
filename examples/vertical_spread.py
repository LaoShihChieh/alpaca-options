"""
examples/vertical_spread.py
============================
Demo: build and submit a SPY bull-call spread ~30 DTE.

Strategy
--------
  Long  lower-strike call  (buy to open)
  Short higher-strike call  (sell to open)
  Net debit = long premium − short premium

Defaults to dry-run mode (``--dry-run``).  Pass ``--no-dry-run`` to submit a
real order to the Alpaca paper account — you will be asked to confirm again
before the order is sent.

Run with:
    uv run python examples/vertical_spread.py           # dry-run (safe)
    uv run python examples/vertical_spread.py --no-dry-run  # paper order
"""

import logging
import sys
from datetime import date, timedelta

import typer
from alpaca.trading.enums import ContractType
from rich import box
from rich import print as rprint
from rich.panel import Panel
from rich.table import Table

from alpaca_options.client import get_clients
from alpaca_options.contracts import find_atm_call, get_option_contracts
from alpaca_options.orders import bull_call_spread
from alpaca_options.quotes import get_latest_quote, midpoint

# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
logging.getLogger("alpaca_options").setLevel(logging.INFO)

UNDERLYING   = "SPY"
TARGET_DTE   = 30
SPREAD_WIDTH = 5.0      # $5-wide spread (buy ATM, sell ATM+5)

# Phrase the user must type verbatim to proceed past the second gate.
_CONFIRM_PHRASE = "I confirm paper trading"

app = typer.Typer(add_completion=False)


def fetch_spy_price() -> float:
    try:
        from alpaca.data.historical.stock import StockHistoricalDataClient
        from alpaca.data.requests import StockLatestQuoteRequest
        import os
        from dotenv import load_dotenv

        load_dotenv()
        stock_client = StockHistoricalDataClient(
            api_key=os.environ["ALPACA_API_KEY"],
            secret_key=os.environ["ALPACA_SECRET_KEY"],
        )
        req = StockLatestQuoteRequest(symbol_or_symbols=UNDERLYING)
        result = stock_client.get_stock_latest_quote(req)
        quote = result[UNDERLYING]
        mid = (float(quote.bid_price or 0) + float(quote.ask_price or 0)) / 2
        if mid > 0:
            return mid
    except Exception as exc:
        logging.getLogger(__name__).warning("Could not fetch SPY price: %s", exc)
    rprint("[yellow]⚠ Could not fetch live SPY price — using fallback $535.00[/yellow]")
    return 535.00


def nearest_friday(target: date) -> date:
    days_ahead = 4 - target.weekday()
    if days_ahead < 0:
        days_ahead += 7
    return target + timedelta(days=days_ahead)


@app.command()
def main(
    dry_run: bool = typer.Option(
        True,
        "--dry-run/--no-dry-run",
        help=(
            "DRY-RUN (default): log the order that would be submitted without "
            "touching the Alpaca API.  NO-DRY-RUN: submit a real paper order."
        ),
    ),
) -> None:
    rprint(Panel.fit(
        "[bold cyan]🦙 Alpaca Paper Options — Bull Call Spread[/bold cyan]",
        box=box.DOUBLE,
    ))

    if dry_run:
        rprint(Panel(
            "[bold yellow]DRY-RUN MODE[/bold yellow]\n\n"
            "Order details will be logged — nothing will be submitted to Alpaca.\n"
            "Pass [bold]--no-dry-run[/bold] to submit a real paper order.",
            border_style="yellow",
        ))
    else:
        rprint(Panel(
            "[bold red]⚠  PAPER LIVE MODE[/bold red]\n\n"
            "A real order WILL be submitted to your Alpaca paper account.\n\n"
            "Two confirmations are required:\n"
            "  1. Y/N prompt — review the order details.\n"
            f'  2. Type [bold]"{_CONFIRM_PHRASE}"[/bold] exactly — one chance, '
            "no retry.",
            border_style="red",
        ))

    # ── 1. Connect ─────────────────────────────────────────────────────────────
    rprint("\n[bold]Step 1 · Connecting to Alpaca paper account…[/bold]")
    trading, _ = get_clients()
    account = trading.get_account()
    rprint(
        f"  Account [green]{account.id}[/green] | "
        f"Buying power: [green]${float(account.buying_power or 0):,.2f}[/green]"
    )

    # ── 2. Find ATM call (~30 DTE) ─────────────────────────────────────────────
    rprint(f"\n[bold]Step 2 · Finding ATM {UNDERLYING} call ~{TARGET_DTE} DTE…[/bold]")
    spy_price = fetch_spy_price()
    rprint(f"  {UNDERLYING} reference price: [yellow]${spy_price:.2f}[/yellow]")

    today = date.today()
    target_exp = nearest_friday(today + timedelta(days=TARGET_DTE))
    rprint(f"  Target expiration: [yellow]{target_exp}[/yellow]")

    long_call = find_atm_call(
        underlying_symbol=UNDERLYING,
        expiration=target_exp,
        underlying_price=spy_price,
        strike_window=60.0,
    )
    long_strike = float(long_call.strike_price or 0)
    rprint(f"  Long leg:  [cyan]{long_call.symbol}[/cyan] (strike ${long_strike:.2f})")

    # ── 3. Find OTM call (spread wing) ─────────────────────────────────────────
    rprint(f"\n[bold]Step 3 · Finding short leg (ATM + ${SPREAD_WIDTH:.0f})…[/bold]")
    short_strike_target = long_strike + SPREAD_WIDTH
    short_calls = get_option_contracts(
        underlying_symbol=UNDERLYING,
        expiration_gte=target_exp,
        expiration_lte=target_exp,
        strike_gte=short_strike_target - 1.0,
        strike_lte=short_strike_target + 1.0,
        contract_type=ContractType.CALL,
    )
    if not short_calls:
        rprint(
            f"[red]No short-leg contract found near ${short_strike_target:.2f}. "
            f"Try adjusting SPREAD_WIDTH.[/red]"
        )
        raise typer.Exit(1)

    short_call   = min(short_calls, key=lambda c: abs(float(c.strike_price or 0) - short_strike_target))
    short_strike = float(short_call.strike_price or 0)
    rprint(f"  Short leg: [cyan]{short_call.symbol}[/cyan] (strike ${short_strike:.2f})")

    # ── 4. Fetch quotes & estimate net debit ───────────────────────────────────
    rprint(f"\n[bold]Step 4 · Fetching quotes for both legs…[/bold]")
    quote_table = Table(box=box.SIMPLE)
    quote_table.add_column("Leg")
    quote_table.add_column("Symbol")
    quote_table.add_column("Bid", justify="right")
    quote_table.add_column("Ask", justify="right")
    quote_table.add_column("Mid", justify="right")

    net_debit  = None
    long_mid   = None
    short_mid  = None
    for label, contract in [("Long (buy)", long_call), ("Short (sell)", short_call)]:
        try:
            q   = get_latest_quote(contract.symbol)
            bid = float(q.bid_price or 0)
            ask = float(q.ask_price or 0)
            mid = midpoint(q)
            quote_table.add_row(label, contract.symbol, f"${bid:.2f}", f"${ask:.2f}", f"${mid:.2f}")
            if label.startswith("Long"):
                long_mid = mid
            else:
                short_mid = mid
        except Exception as exc:
            quote_table.add_row(label, contract.symbol, "n/a", "n/a", "n/a")
            rprint(f"  [yellow]⚠ Quote unavailable for {contract.symbol}: {exc}[/yellow]")

    rprint(quote_table)

    if long_mid is not None and short_mid is not None:
        net_debit = round(long_mid - short_mid + 0.05, 2)   # slight cushion
        rprint(
            f"  Estimated net debit: [yellow]${net_debit:.2f}[/yellow] "
            f"(≈ ${net_debit * 100:.2f} per spread contract)"
        )
    else:
        rprint("  [yellow]Could not compute net debit — will use market order.[/yellow]")

    # ── 5. Confirm before submitting (live mode only) ─────────────────────────
    if not dry_run:
        rprint(
            Panel(
                f"[bold]Strategy:[/bold] Bull Call Spread\n"
                f"  Buy  1 × [cyan]{long_call.symbol}[/cyan]  (${long_strike:.2f} call)\n"
                f"  Sell 1 × [cyan]{short_call.symbol}[/cyan] (${short_strike:.2f} call)\n"
                f"  Net debit limit: [yellow]${net_debit:.2f if net_debit else 'market'}[/yellow]\n"
                f"  Max loss: [red]${(net_debit or 0) * 100:.2f}[/red]  |  "
                f"Max gain: [green]${(SPREAD_WIDTH - (net_debit or 0)) * 100:.2f}[/green]",
                title="[bold red]⚠  Order Confirmation Required — Gate 1 of 2[/bold red]",
                border_style="red",
            )
        )
        # Gate 1: Y/N — gives the user a chance to review the order details.
        if not typer.confirm("Proceed to phrase confirmation?", default=False):
            rprint("[yellow]Order cancelled — no order was submitted.[/yellow]")
            raise typer.Exit(0)

        # Gate 2: typed phrase — one attempt, no retry.
        rprint(
            f"\n[bold red]Gate 2 of 2[/bold red] — type the following phrase exactly "
            f"(case-sensitive, one attempt):\n\n"
            f'  [bold white on red] {_CONFIRM_PHRASE} [/bold white on red]\n'
        )
        entered = typer.prompt("›")
        if entered != _CONFIRM_PHRASE:
            rprint(
                f"[bold red]Phrase did not match.[/bold red] "
                f"Expected: [bold]{_CONFIRM_PHRASE!r}[/bold]\n"
                "[yellow]Order cancelled — no order was submitted.[/yellow]"
            )
            raise typer.Exit(1)

    # ── 6. Submit (or log dry-run) ─────────────────────────────────────────────
    rprint(f"\n[bold]Step 6 · {'Logging dry-run order…' if dry_run else 'Submitting bull call spread…'}[/bold]")
    order = bull_call_spread(
        long_symbol=long_call.symbol,
        short_symbol=short_call.symbol,
        qty=1,
        net_debit=net_debit,
        dry_run=dry_run,          # explicit — never omit this argument
    )

    if dry_run:
        rprint(
            f"\n[bold yellow]DRY RUN complete.[/bold yellow] "
            f"No order was submitted.  Pass [bold]--no-dry-run[/bold] to go live."
        )
        return

    # ── 7. Print confirmation (live mode only) ─────────────────────────────────
    assert order is not None   # guarded by dry_run=False above
    result_table = Table(title="✅ Spread Order Submitted", box=box.ROUNDED, border_style="green")
    result_table.add_column("Field", style="bold")
    result_table.add_column("Value")
    result_table.add_row("Order ID", str(order.id))
    result_table.add_row("Order Class", str(order.order_class))
    result_table.add_row("Status", str(order.status))
    result_table.add_row("Submitted At", str(order.submitted_at))
    if order.legs:
        for i, leg in enumerate(order.legs, 1):
            result_table.add_row(f"Leg {i}", f"{leg.symbol or '?'}  {leg.side}")
    rprint(result_table)

    rprint("\n[bold green]Done! Check your Alpaca paper account dashboard.[/bold green]")


if __name__ == "__main__":
    app()
