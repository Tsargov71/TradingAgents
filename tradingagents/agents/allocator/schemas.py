"""Allocator: what the model decides, and the order built from it."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field, field_validator


class AllocatorAction(StrEnum):
    ENTER = "Enter"
    SKIP = "Skip"


class AllocatorDecision(BaseModel):
    action: AllocatorAction = Field(
        description="Enter to open a position in this stock now, Skip to stay in cash.",
    )
    weight: float = Field(
        default=0.0,
        description="Fraction of the free cash to invest, from 0 to 1 (0.6 means 60%). 0 when skipping.",
    )
    reasoning: str = Field(description="Why, in two to four sentences.")

    @field_validator("weight", mode="before")
    @classmethod
    def _as_fraction(cls, v):
        value = float(str(v or 0).strip().rstrip("%"))
        return value / 100 if value > 1 else value  # "60" and "60%" mean 60%


class Allocation(BaseModel):
    ticker: str
    date: str
    action: str  # "buy" or "none"
    shares: int = 0
    price: float | None = None
    stop_loss: float | None = None
    weight: float = 0.0  # share of the free cash actually invested
    cash_before: float | None = None
    cash_after: float | None = None
    rating: str | None = None  # the Portfolio Manager rating that was read
    reasoning: str = ""
    notes: list[str] = Field(default_factory=list)
