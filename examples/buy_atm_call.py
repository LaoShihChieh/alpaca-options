"""
examples/buy_atm_call.py
========================
End-to-end demo: find the ATM SPY call ~30 DTE, fetch its latest quote,
then **ask for confirmation** before submitting a 1-contract limit order.

Defaults to dry-run mode (``--dry-run``).  Pass ``--no-dry-run`` to submit a
real order to the Alpaca paper account — you will be asked to confirm again
before the order is sent.

Run with:
    uv run python examples/buy_atm_call.py           # dry-run (safe)
    uv run python examples/buy_atm_call.py --no-dry-run  # paper order
"""

import logging
import sys
from datetime import date, timedelta

import typer
from rich import box
from rich import print as rprint
from rich.panel import Panel
from rich.table import Table

from alpaca_options.client import get_clients
from alpaca_options.contracts import find_atm_call
from alpaca_options.orders import buy_to_open_limit
from alpaca_options.quotes import get_latest_quote, midpoint

# ---------------------------------------------------------------------------
# Logging — INFO for our modules, WARNING for everything else
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)s %(name)s: %(message)s",
)
logging.getLogger("alpaca_options").setLevel(logging.INFO)

UNDERLYING = "SPY"
TARGET_DTE = 30

# Phrase the user must type verbatim to proceed past the second gate.
# Changing this string also changes what the prompt asks for — keep them in sync.
_CONFIRM_PHRASE = "I confirm paper trading"

app = typer.Typer(add_completion=False)


def fetch_spy_price() -> float:
    """Get the latest SPY mid-price.

    Falls back to a hardcoded reasonable value if unavailable.
    """
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
    """Advance *target* to the nearest Friday on or after that date."""
    days_ahead = 4 - target.weekday()  # Friday = 4
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
        "[bold cyan]🦙 Alpaca Paper Options — Buy ATM Call[/bold cyan]",
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

    # ── 1. Connect ────────────────────────────────────────────────────────────
    rprint("\n[bold]Step 1 · Connecting to Alpaca paper account…[/bold]")
    trading, _ = get_clients()
    account = trading.get_account()
    rprint(
        f"  Account [green]{account.id}[/green] | "
        f"Buying power: [green]${float(account.buying_power or 0):,.2f}[/green] | "
        f"Status: [green]{account.status}[/green]"
    )

    # ── 2. Find ATM call ~30 DTE ──────────────────────────────────────────────
    rprint(f"\n[bold]Step 2 · Finding ATM {UNDERLYING} call ~{TARGET_DTE} DTE…[/bold]")
    spy_price = fetch_spy_price()
    rprint(f"  {UNDERLYING} reference price: [yellow]${spy_price:.2f}[/yellow]")

    today = date.today()
    target_exp = nearest_friday(today + timedelta(days=TARGET_DTE))
    rprint(f"  Target expiration: [yellow]{target_exp}[/yellow]")

    atm_call = find_atm_call(
        underlying_symbol=UNDERLYING,
        expiration=target_exp,
        underlying_price=spy_price,
        strike_window=60.0,
    )

    info_table = Table(box=box.SIMPLE, show_header=False)
    info_table.add_column("Field", style="bold")
    info_table.add_column("Value")
    info_table.add_row("Symbol", atm_call.symbol)
    info_table.add_row("Strike", f"${float(atm_call.strike_price or 0):.2f}")
    info_table.add_row("Expiration", str(atm_call.expiration_date))
    info_table.add_row("Type", str(atm_call.type))
    info_table.add_row("Underlying", atm_call.underlying_symbol)
    rprint(info_table)

    # ── 3. Fetch latest quote ─────────────────────────────────────────────────
    rprint(f"\n[bold]Step 3 · Fetching latest quote for {atm_call.symbol}…[/bold]")
    try:
        quote = get_latest_quote(atm_call.symbol)
        bid = float(quote.bid_price or 0)
        ask = float(quote.ask_price or 0)
        mid = midpoint(quote)
        rprint(
            f"  Bid: [red]${bid:.2f}[/red]  Ask: [green]${ask:.2f}[/green]  "
            f"Mid: [yellow]${mid:.2f}[/yellow]  (×100 = ${mid*100:.2f}/contract)"
        )
        limit_price = round(mid + 0.05, 2)   # slight cushion above mid
    except Exception as exc:
        rprint(f"  [yellow]⚠ Quote unavailable ({exc}) — using fallback limit of $1.00[/yellow]")
        limit_price = 1.00

    rprint(f"  Proposed limit price: [bold yellow]${limit_price:.2f}[/bold yellow]")

    # ── 4. Confirm before submitting (live mode only) ─────────────────────────
    if not dry_run:
        rprint(
            Panel(
                f"[bold]About to submit:[/bold]\n"
                f"  BUY 1 × [cyan]{atm_call.symbol}[/cyan] @ limit [yellow]${limit_price:.2f}[/yellow]\n"
                f"  Max cost ≈ [yellow]${limit_price * 100:.2f}[/yellow] (1 contract × 100 shares)",
                title="[bold red]⚠  Order Confirmation Required — Gate 1 of 2[/bold red]",
                border_style="red",
            )
        )
        # Gate 1: Y/N — gives the user a chance to review the order details.
        if not typer.confirm("Proceed to phrase confirmation?", default=False):
            rprint("[yellow]Order cancelled — no order was submitted.[/yellow]")
            raise typer.Exit(0)

        # Gate 2: typed phrase — one attempt, no retry.
        # Requires deliberate keystrokes; "y" reflex cannot trigger it.
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

    # ── 5. Submit order (or log dry-run) ──────────────────────────────────────
    step = "5" if dry_run else "5"
    rprint(f"\n[bold]Step {step} · {'Logging dry-run order…' if dry_run else 'Submitting order…'}[/bold]")
    order = buy_to_open_limit(
        symbol=atm_call.symbol,
        qty=1,
        limit_price=limit_price,
        dry_run=dry_run,          # explicit — never omit this argument
    )

    if dry_run:
        rprint(
            f"\n[bold yellow]DRY RUN complete.[/bold yellow] "
            f"No order was submitted.  Pass [bold]--no-dry-run[/bold] to go live."
        )
        return

    # ── 6. Print confirmation (live mode only) ────────────────────────────────
    assert order is not None   # guarded by dry_run=False above
    result_table = Table(title="✅ Order Submitted", box=box.ROUNDED, border_style="green")
    result_table.add_column("Field", style="bold")
    result_table.add_column("Value")
    result_table.add_row("Order ID", str(order.id))
    result_table.add_row("Symbol", order.symbol or atm_call.symbol)
    result_table.add_row("Side", str(order.side))
    result_table.add_row("Qty", str(order.qty))
    result_table.add_row("Limit Price", f"${float(order.limit_price or limit_price):.2f}")
    result_table.add_row("Status", str(order.status))
    result_table.add_row("Submitted At", str(order.submitted_at))
    rprint(result_table)

    rprint("\n[bold green]Done! Check your Alpaca paper account dashboard.[/bold green]")


if __name__ == "__main__":
    app()
