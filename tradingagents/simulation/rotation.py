"""Rotation rule: a stock that leaves the top N is sold and the new leader is bought."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RotationStep:
    sell: str | None = None  # ticker to sell now
    buy: str | None = None  # ticker to analyze and, if the allocator agrees, buy


def rotation_step(holding: str | None, ranking: list[str], top_n: int = 20) -> RotationStep:
    """What to do on a rotation date, given the ticker held (None if flat) and the ranking (best first)."""
    top = [t.upper() for t in ranking[:top_n]]
    if not top:
        return RotationStep()
    if holding is None:
        return RotationStep(buy=top[0])
    if holding.upper() in top:
        return RotationStep()  # still in the top N: keep it
    return RotationStep(sell=holding.upper(), buy=top[0])
