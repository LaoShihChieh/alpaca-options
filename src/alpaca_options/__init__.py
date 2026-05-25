"""
alpaca_options — paper-trading options toolkit built on alpaca-py.

Public re-exports so consumers can do:
    from alpaca_options import get_clients, find_atm_call, ...
"""

from alpaca_options.client import get_clients
from alpaca_options.contracts import (
    get_option_contracts,
    find_atm_call,
    find_atm_put,
)
from alpaca_options.quotes import get_latest_quote, get_option_bars
from alpaca_options.orders import (
    buy_to_open_market,
    buy_to_open_limit,
    sell_to_close_market,
    sell_to_close_limit,
    bull_call_spread,
    straddle,
)
from alpaca_options.positions import list_option_positions, close_all_options

__all__ = [
    "get_clients",
    "get_option_contracts",
    "find_atm_call",
    "find_atm_put",
    "get_latest_quote",
    "get_option_bars",
    "buy_to_open_market",
    "buy_to_open_limit",
    "sell_to_close_market",
    "sell_to_close_limit",
    "bull_call_spread",
    "straddle",
    "list_option_positions",
    "close_all_options",
]


def main() -> None:
    """CLI entry-point (placeholder — use the example scripts directly)."""
    print("alpaca-options: run examples/buy_atm_call.py or examples/vertical_spread.py")
