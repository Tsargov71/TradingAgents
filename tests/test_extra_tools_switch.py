"""TRADINGAGENTS_EXTRA_TOOLS=0 gives the baseline: none of the fork's three added tools."""
import importlib

import pytest

from tradingagents.agents.analysts import fundamentals_analyst, market_analyst

ADDED = {"get_ma_crossover", "get_volatility_analysis", "get_competitor_comparison"}


def _names(module):
    importlib.reload(module)
    return {t.name for t in module.TOOLS}


@pytest.fixture(autouse=True)
def _restore_default():
    yield
    importlib.reload(market_analyst)
    importlib.reload(fundamentals_analyst)


@pytest.mark.unit
def test_added_tools_are_on_by_default(monkeypatch):
    monkeypatch.delenv("TRADINGAGENTS_EXTRA_TOOLS", raising=False)
    assert ADDED & _names(market_analyst) == {"get_ma_crossover", "get_volatility_analysis"}
    assert ADDED & _names(fundamentals_analyst) == {"get_competitor_comparison"}
    assert market_analyst.EXTRA_TOOLS_PROMPT.startswith("Two dedicated tools")


@pytest.mark.unit
@pytest.mark.parametrize("value", ["0", "false", "OFF"])
def test_switch_removes_the_tools_from_both_analysts(monkeypatch, value):
    monkeypatch.setenv("TRADINGAGENTS_EXTRA_TOOLS", value)
    assert not ADDED & _names(market_analyst)
    assert not ADDED & _names(fundamentals_analyst)
    assert market_analyst.EXTRA_TOOLS is False and fundamentals_analyst.EXTRA_TOOLS is False


@pytest.mark.unit
def test_blank_value_means_on(monkeypatch):
    monkeypatch.setenv("TRADINGAGENTS_EXTRA_TOOLS", "")
    assert ADDED & _names(market_analyst)
