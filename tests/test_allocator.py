"""Allocator: model decides Enter/Skip and a cash fraction; the code does stop, shares and cash."""
import json

import numpy as np
import pandas as pd
import pytest

from tradingagents.agents.allocator import Limits, create_allocator, log_allocation
from tradingagents.agents.allocator import allocator as allocator_module
from tradingagents.agents.allocator.schemas import AllocatorAction, AllocatorDecision
from tradingagents.agents.allocator.sizing import resolve_stop, size_long
from tradingagents.portfolio import PortfolioContext

pytestmark = pytest.mark.unit

PM_BUY = "**Rating**: Buy\n\n**Executive Summary**: x\n\n**Investment Thesis**: y\n\n**Price Target**: 120\n\n**Time Horizon**: 1-3 months"
PM_SELL = PM_BUY.replace("Buy", "Sell", 1)
PLAN = "**Action**: Buy\n\n**Reasoning**: r\n\n**Entry Price**: 100\n\n**Stop Loss**: 94.5\n\n**Position Sizing**: 50%"


class FakeLLM:
    """Stands in for the chat model: with_structured_output(...).invoke(...) returns a canned decision."""

    def __init__(self, decision=None):
        self.decision, self.calls = decision, 0

    def with_structured_output(self, schema):
        return self

    def invoke(self, prompt):
        self.calls += 1
        return self.decision


def _state(plan=PLAN, decision=PM_BUY, portfolio=None):
    return {"company_of_interest": "SNA", "trade_date": "2026-08-17", "final_trade_decision": decision,
            "trader_investment_plan": plan,
            "portfolio": portfolio or PortfolioContext(cash=2000.0, currency="USD")}


def _node(llm, **kw):
    return create_allocator(llm, price_source=lambda t, d: (100.0, 2.0), **kw)


def _enter(weight=0.6):
    return AllocatorDecision(action="Enter", weight=weight, reasoning="strong case")


def test_enter_sizes_in_code():
    out = _node(FakeLLM(_enter(0.6)))(_state())["allocation"]
    assert (out.action, out.shares, out.stop_loss) == ("buy", 12, 94.5)   # floor(0.6 * 2000 / 100)
    assert out.cash_after == 800.0 and out.weight == pytest.approx(0.6)


def test_loss_at_the_stop_is_capped():
    out = _node(FakeLLM(_enter(1.0)), limits=Limits(max_loss=0.02))(_state())["allocation"]
    assert out.shares == 7                         # floor(0.02 * 2000 / 5.5), not the 20 the weight allows
    assert any("at most" in n for n in out.notes)


def test_stop_on_the_wrong_side_is_recomputed_with_2_atr():
    out = _node(FakeLLM(_enter()))(_state(plan="**Stop Loss**: 108.0"))["allocation"]
    assert out.stop_loss == 96.0 and any("using price" in n for n in out.notes)   # 100 - 2 x 2
    out = _node(FakeLLM(_enter()))(_state(plan="**Stop Loss**: not provided"))["allocation"]
    assert out.stop_loss == 96.0


def test_percent_weight_is_understood():
    assert AllocatorDecision(action="Enter", weight="60%", reasoning="r").weight == 0.6
    assert AllocatorDecision(action="Enter", weight=60, reasoning="r").weight == 0.6


def test_model_skip_means_no_order():
    out = _node(FakeLLM(AllocatorDecision(action="Skip", reasoning="weak")))(_state())["allocation"]
    assert out.action == "none" and out.shares == 0 and out.cash_after == 2000.0 and out.reasoning == "weak"


def test_unreadable_model_output_stays_in_cash():
    out = _node(FakeLLM(None))(_state())["allocation"]
    assert out.action == "none" and any("no readable decision" in n for n in out.notes)


@pytest.mark.parametrize("state, why", [
    (_state(decision=PM_SELL), "do not enter"),
    (_state(portfolio=PortfolioContext(cash=2000.0, positions=[{"ticker": "CHD", "quantity": 5}])), "not flat"),
    (_state(portfolio=PortfolioContext(cash=0.0)), "no free cash"),
])
def test_checks_that_need_no_model_do_not_call_it(state, why):
    llm = FakeLLM(_enter())
    out = _node(llm)(state)["allocation"]
    assert out.action == "none" and llm.calls == 0 and why in out.notes[0]


def test_less_than_one_share_is_no_order():
    out = _node(FakeLLM(_enter(1.0)))(_state(portfolio=PortfolioContext(cash=50.0)))["allocation"]
    assert out.action == "none" and out.shares == 0


def test_sizing_helpers():
    assert resolve_stop(100, 95, 2) == (95, None)
    assert resolve_stop(100, None, None)[0] is None
    assert size_long(2000, 100, 96, 5.0)[0] == 20          # weight above the limit is clipped to 100%
    assert size_long(2000, 100, 96, 0.5, Limits(max_loss=0.05))[0] == 10


def test_price_and_atr_reads_the_data_known_at_the_date(monkeypatch):
    idx = pd.bdate_range(end="2026-08-17", periods=60)
    close = 100 + np.arange(60, dtype=float)
    frame = pd.DataFrame({"Date": idx, "Open": close, "High": close + 1, "Low": close - 1,
                          "Close": close, "Volume": 1000})
    monkeypatch.setattr(allocator_module, "load_ohlcv", lambda symbol, date: frame)
    price, atr = allocator_module.price_and_atr("SNA", "2026-08-17")
    assert price == 159.0 and atr == pytest.approx(2.0, abs=0.2)


def test_log_is_one_json_line_per_decision(tmp_path):
    out = _node(FakeLLM(_enter()))(_state())["allocation"]
    path = tmp_path / "allocator_log.jsonl"
    log_allocation(path, out)
    log_allocation(path, out)
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(rows) == 2 and rows[0]["shares"] == 12 and rows[0]["stop_loss"] == 94.5
