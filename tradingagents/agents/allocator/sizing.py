"""Position sizing for the Allocator. All the numbers come from here, not from the model."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class Limits:
    max_weight: float = 1.0  # share of the cash that can go into the position
    max_loss: float = 0.05  # share of the cash that can be lost if the stop is hit


def resolve_stop(price, trader_stop, atr, multiple=2.0):
    """The Trader's stop if it is below the price, else price - multiple x ATR. Returns (stop, note)."""
    if trader_stop is not None and 0 < trader_stop < price:
        return trader_stop, None
    if atr and price - multiple * atr > 0:
        reason = "no Trader stop" if trader_stop is None else f"Trader stop {trader_stop:g} is not below the price"
        return round(price - multiple * atr, 2), f"{reason}: using price - {multiple:g} x ATR"
    return None, "no usable stop"


def size_long(cash, price, stop, weight, limits=Limits()):
    """Whole shares to buy, and the notes explaining a reduction."""
    if cash <= 0 or price <= 0 or stop >= price:
        return 0, ["cannot size: bad cash, price or stop"]
    weight = min(max(weight, 0), limits.max_weight)
    shares = math.floor(weight * cash / price)
    cap = math.floor(limits.max_loss * cash / (price - stop))
    notes = []
    if cap < shares:
        notes.append(f"cut from {shares} to {cap} shares so the loss at the stop is at most {limits.max_loss:.0%} of cash")
        shares = cap
    if shares < 1:
        return 0, notes + ["less than one share"]
    return shares, notes
