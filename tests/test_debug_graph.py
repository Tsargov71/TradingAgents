"""tools_debug's sequential runner, end to end, with scripted models and no network.

Pins that the debug runner follows the production graph: the three added tools
execute through their ToolNodes (state injection included), a run reaches a
logged decision, and a second run is served entirely from the disk caches.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest
from langchain_core.globals import set_llm_cache

from tests.test_graph_end_to_end import ARGS, TRADE_DATE, ScriptedModel, _Client, offline  # noqa: F401
from tools_debug.debug_graph import TradingAgentsGraphDebug
from tradingagents.agents import ma_crossover_tool, volatility_tool
from tradingagents.dataflows import router
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.graph import trading_graph


@pytest.fixture(autouse=True)
def _no_global_llm_cache():
    """The debug graph registers a process-wide LLM cache; do not leak it to other tests."""
    yield
    set_llm_cache(None)


def _ledger(out_dir):
    lines = (out_dir / "ledger.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines]


@pytest.mark.unit
def test_a_debug_run_reaches_a_decision_and_the_second_run_is_all_cache_hits(
        tmp_path, monkeypatch, offline):  # noqa: F811
    prices = pd.DataFrame({
        "Date": pd.bdate_range(end=TRADE_DATE, periods=300),
        "Open": 100.0, "High": 101.0, "Low": 99.0, "Close": [100.0 + i * 0.1 for i in range(300)],
        "Volume": 1_000_000,
    })
    for module in (ma_crossover_tool, volatility_tool):
        monkeypatch.setattr(module, "load_ohlcv", lambda *a, **k: prices.copy())
    # The competitor tool takes the peers from the model's own call and reads
    # their fundamentals through the router's raw-fundamentals method.
    monkeypatch.setitem(ARGS, "competitor_tickers", ["AMD"])
    raw = {"market_cap": 1.0e12, "pe_ratio": 30.0, "profit_margin": 0.25}
    for vendor in router.VENDOR_METHODS["get_fundamentals_raw"]:
        monkeypatch.setitem(router.VENDOR_METHODS["get_fundamentals_raw"], vendor, lambda *a, **k: dict(raw))

    cfg = dict(DEFAULT_CONFIG)
    cfg.update(results_dir=str(tmp_path / "results"), data_cache_dir=str(tmp_path / "cache"),
               memory_log_path=str(tmp_path / "log.md"))
    monkeypatch.setattr(trading_graph, "create_tier_client",
                        lambda config, tier, **k: _Client(ScriptedModel()))
    out_dir = tmp_path / "debug_out"

    graph = TradingAgentsGraphDebug(["market", "fundamentals"], config=cfg, out_dir=out_dir)
    state, rating = graph.run_sequential("NVDA", TRADE_DATE)

    assert rating == "Overweight"
    assert state["market_report"] and state["fundamentals_report"]

    tools_run = {e["name"] for e in _ledger(out_dir) if e["kind"] == "tool"}
    assert {"get_ma_crossover", "get_volatility_analysis", "get_competitor_comparison"} <= tools_run
    errors = [e for e in _ledger(out_dir)
              if e["kind"] == "tool" and ("ERROR" in e["detail"] or "Error" in e["detail"])]
    assert not errors, errors
    comparison = next((out_dir / "cache" / "tools").glob("get_competitor_comparison_*.json"))
    assert "AMD" in json.loads(comparison.read_text(encoding="utf-8"))["output"]

    # A second run, same inputs: every tool and LLM call comes from disk.
    before = len(_ledger(out_dir))
    graph2 = TradingAgentsGraphDebug(["market", "fundamentals"], config=cfg, out_dir=out_dir)
    graph2.run_sequential("NVDA", TRADE_DATE)
    again = _ledger(out_dir)[before:]
    assert [e for e in again if e["kind"] in ("tool", "llm") and e["status"] == "miss"] == []
    assert any(e["kind"] == "tool" and e["status"] == "hit" for e in again)
