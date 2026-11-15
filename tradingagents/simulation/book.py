"""The simulated book: buy, sell, value, and save it as portfolio.json (cash and positions only)."""

from __future__ import annotations

import json
from pathlib import Path

from tradingagents.portfolio import Position


def buy(book, ticker, shares, price):
    cost = shares * price
    if cost > book.cash:
        raise ValueError(f"not enough cash to buy {shares} {ticker} at {price}: {book.cash} available")
    position = Position(ticker=ticker.upper(), quantity=shares, average_price=price)
    return book.model_copy(update={"cash": round(book.cash - cost, 2), "positions": book.positions + [position]})


def sell_all(book, ticker, price):
    held = book.position_in(ticker)
    if held is None:
        return book
    return book.model_copy(update={"cash": round(book.cash + held.quantity * price, 2),
                                   "positions": [p for p in book.positions if p is not held]})


def market_value(book, prices):
    return round(book.cash + sum(p.quantity * prices[p.ticker.upper()] for p in book.positions), 2)


def save_portfolio(book, path):
    Path(path).write_text(json.dumps(book.model_dump(), indent=2) + "\n", encoding="utf-8")  # no BOM
