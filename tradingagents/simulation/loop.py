"""Simulation loop: analyze the leader, allocate, hold with a stop, rotate. One stock at a time.

The backtest command keeps the portfolio constant; this loop updates it. Simplifications:
a decision uses the data up to the rotation date and is executed at that day's close; a stop
fills at the stop price (or at the open on a gap below it) when a day's low reaches it;
no fees, no slippage.

bars(ticker, date) returns the daily OHLC data up to that date, like load_ohlcv.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd

from tradingagents.agents.allocator import log_allocation
from tradingagents.agents.allocator.schemas import Allocation
from tradingagents.simulation.book import buy, market_value, save_portfolio, sell_all
from tradingagents.simulation.rotation import rotation_step


def make_analyzer(graph):
    return lambda ticker, date, portfolio: graph.propagate(ticker, date, portfolio=portfolio)[0]


def create_buy_all(bars):
    """Allocator for the pure Magic Formula benchmark: all the cash in the leader, no stop."""
    def allocate(state):
        ticker, date, book = state["company_of_interest"], state["trade_date"], state["portfolio"]
        price = _close(bars, ticker, date)
        shares = math.floor(book.cash / price)
        cost = shares * price
        return {"allocation": Allocation(
            ticker=ticker, date=date, action="buy" if shares else "none", shares=shares, price=price,
            weight=cost / book.cash, cash_before=book.cash, cash_after=round(book.cash - cost, 2))}
    return allocate


def _close(bars, ticker, date):
    return float(bars(ticker, date).iloc[-1]["Close"])


def _log(path, **fields):
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(fields) + "\n")


def _stop_fill(bars, ticker, start, end, stop):
    """(date, price) of the first session after start, up to end, whose low reaches the stop."""
    df = bars(ticker, end)
    for row in df[pd.to_datetime(df["Date"]) > pd.Timestamp(start)].itertuples():
        if row.Low <= stop:
            return pd.Timestamp(row.Date).strftime("%Y-%m-%d"), min(stop, row.Open)
    return None


def run_simulation(dates, end_date, ranking_for_date, allocate, bars, portfolio, out_dir, analyze=None, top_n=20):
    """Go through the rotation dates and return the final book.

    Writes to out_dir: portfolio.json (state), allocator_log.jsonl (every decision with its
    stop, weight and reasons) and events.jsonl (buys, sells, stops and the equity at each date).
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    allocator_log, events = out / "allocator_log.jsonl", out / "events.jsonl"
    holding, stop = (portfolio.positions[0].ticker if portfolio.positions else None), None

    for date, next_date in zip(dates, list(dates[1:]) + [end_date]):
        step = rotation_step(holding, ranking_for_date(date), top_n)

        if step.sell:
            price = _close(bars, step.sell, date)
            portfolio = sell_all(portfolio, step.sell, price)
            _log(events, date=date, event="sell", ticker=step.sell, price=price)
            holding, stop = None, None

        if step.buy:
            state = analyze(step.buy, date, portfolio) if analyze else {}
            state = {**state, "company_of_interest": step.buy, "trade_date": date, "portfolio": portfolio}
            allocation = allocate(state)["allocation"]
            log_allocation(allocator_log, allocation)
            if allocation.action == "buy":
                portfolio = buy(portfolio, step.buy, allocation.shares, allocation.price)
                holding, stop = step.buy, allocation.stop_loss
                _log(events, date=date, event="buy", ticker=step.buy, shares=allocation.shares,
                     price=allocation.price, stop=stop)
            else:
                _log(events, date=date, event="stay in cash", ticker=step.buy)

        if holding and stop:
            hit = _stop_fill(bars, holding, date, next_date, stop)
            if hit:
                portfolio = sell_all(portfolio, holding, hit[1])
                _log(events, date=hit[0], event="stop", ticker=holding, price=hit[1])
                holding, stop = None, None

        prices = {p.ticker.upper(): _close(bars, p.ticker, next_date) for p in portfolio.positions}
        _log(events, date=next_date, event="equity", value=market_value(portfolio, prices),
             cash=portfolio.cash, holding=holding)
        save_portfolio(portfolio, out / "portfolio.json")
    return portfolio
