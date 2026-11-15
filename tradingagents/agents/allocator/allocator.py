"""Allocator: decides whether to enter the top-ranked stock and with how much of the cash.

The model only picks Enter/Skip and a fraction of the cash. Stop, share count and cash
left are computed in sizing.py. Selling is the rotation rule's job, not this agent's.
"""

from __future__ import annotations

import re
from pathlib import Path

from stockstats import wrap

from tradingagents.agents.allocator.schemas import Allocation, AllocatorAction, AllocatorDecision
from tradingagents.agents.allocator.sizing import Limits, resolve_stop, size_long
from tradingagents.agents.context import get_instrument_context_from_state, get_language_instruction
from tradingagents.agents.rating import extract_rating
from tradingagents.agents.structured import NO_EXTERNAL_TOOLS, bind_structured, invoke_structured
from tradingagents.dataflows.vendors.yahoo.ohlcv import load_ohlcv

# Long only: with an empty book these ratings mean "do not enter"
NO_ENTRY_RATINGS = {"Sell", "Underweight"}


def price_and_atr(symbol, curr_date):
    df = wrap(load_ohlcv(symbol, curr_date))
    return float(df["close"].iloc[-1]), float(df["atr"].iloc[-1])


def _trader_stop(trader_plan):
    match = re.search(r"\*\*Stop Loss\*\*:\s*([\d,]+\.?\d*)", trader_plan or "")
    return float(match.group(1).replace(",", "")) if match else None


def create_allocator(llm, limits=Limits(), price_source=price_and_atr):
    structured_llm = bind_structured(llm, AllocatorDecision, "Allocator")

    def allocator_node(state):
        ticker, date, book = state["company_of_interest"], state["trade_date"], state["portfolio"]
        decision = state.get("final_trade_decision", "")
        plan = state.get("trader_investment_plan", "")
        rating = extract_rating(decision)
        order = Allocation(ticker=ticker, date=date, action="none", rating=rating,
                           cash_before=book.cash, cash_after=book.cash)

        def result(*notes, **fields):
            return {"allocation": order.model_copy(update={"notes": list(notes), **fields})}

        if book.positions:
            return result("the book is not flat: sell before allocating to a new stock")
        if book.cash is None or book.cash <= 0:
            return result("no free cash")
        if rating is None or rating in NO_ENTRY_RATINGS:
            return result(f"Portfolio Manager rating is {rating or 'unreadable'}: do not enter")
        try:
            price, atr = price_source(ticker, date)
        except Exception as e:
            return result(f"no price data ({e})")
        stop, stop_note = resolve_stop(price, _trader_stop(plan), atr)
        if stop is None:
            return result(stop_note, price=price)

        messages = [
            {
                "role": "system",
                "content": (
                    "You are an allocation agent for a small investor who holds one stock at a time. "
                    "Decide whether to open a position in this stock now or stay in cash, and what "
                    "fraction of the free cash to commit: more when the case is strong and the stop is "
                    "close, less when it is weak or the stop is far. Do not compute share counts, "
                    "stops or totals; the system does that. "
                    + NO_EXTERNAL_TOOLS
                    + get_language_instruction()
                ),
            },
            {
                "role": "user",
                "content": (
                    f"{get_instrument_context_from_state(state)}\n\n"
                    f"{book.render(ticker)}\n\n"
                    f"Last close: {price:.2f}. Stop-loss that will be used: {stop:.2f} "
                    f"({(price - stop) / price:.1%} below the price).\n\n"
                    f"Portfolio Manager decision:\n{decision}\n\n"
                    f"Trader proposal:\n{plan}\n\n"
                    "Decide: Enter or Skip, and the fraction of free cash to invest."
                ),
            },
        ]
        answer = invoke_structured(structured_llm, messages, "Allocator")
        if answer is None:
            return result("no readable decision from the model", price=price, stop_loss=stop)
        if answer.action != AllocatorAction.ENTER:
            return result(price=price, stop_loss=stop, reasoning=answer.reasoning)

        shares, notes = size_long(book.cash, price, stop, answer.weight, limits)
        notes = ([stop_note] if stop_note else []) + notes
        if shares == 0:
            return result(*notes, price=price, stop_loss=stop, reasoning=answer.reasoning)
        cost = shares * price
        return result(*notes, action="buy", shares=shares, price=price, stop_loss=stop,
                      reasoning=answer.reasoning, weight=cost / book.cash, cash_after=round(book.cash - cost, 2))

    return allocator_node


def log_allocation(path, allocation):
    """One JSON line per decision: stops, weights and reasons live here, not in portfolio.json."""
    with open(path, "a", encoding="utf-8") as f:
        f.write(allocation.model_dump_json() + "\n")
