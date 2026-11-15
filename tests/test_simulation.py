"""Simulation loop: rotation, allocator, stops and portfolio.json, on fake prices and a fake model."""
import json

import pandas as pd
import pytest

from tradingagents.agents.allocator import create_allocator
from tradingagents.agents.allocator.schemas import AllocatorDecision
from tradingagents.portfolio import PortfolioContext, load_portfolio
from tradingagents.simulation import buy_and_hold, create_buy_all, rotation_step, run_simulation
from tradingagents.simulation.book import buy, market_value, save_portfolio, sell_all

pytestmark = pytest.mark.unit

PM_BUY = "**Rating**: Buy\n\n**Executive Summary**: x\n\n**Investment Thesis**: y"
PLAN = "**Action**: Buy\n\n**Stop Loss**: 94.5"
DATES, END = ["2026-08-17", "2026-08-31"], "2026-09-14"


class FakeLLM:
    def __init__(self, decision):
        self.decision = decision

    def with_structured_output(self, schema):
        return self

    def invoke(self, prompt):
        return self.decision


def make_bars(**series):
    """series: ticker -> {date: (open, high, low, close)} overrides over a flat default of 100."""
    def bars(ticker, as_of):
        days = pd.bdate_range("2026-08-03", "2026-09-14")
        base = series.get(ticker, {})
        rows = [dict(Date=d, **dict(zip(("Open", "High", "Low", "Close"),
                                       base.get(d.strftime("%Y-%m-%d"), (100.0,) * 4)))) for d in days]
        df = pd.DataFrame(rows)
        return df[df["Date"] <= pd.Timestamp(as_of)].reset_index(drop=True)
    return bars


def agent_allocator(bars, decision=None):
    llm = FakeLLM(decision or AllocatorDecision(action="Enter", weight=0.6, reasoning="strong"))
    return create_allocator(llm, price_source=lambda t, d: (float(bars(t, d).iloc[-1]["Close"]), 2.0))


def fake_analyzer(calls):
    def analyze(ticker, date, portfolio):
        calls.append((ticker, date))
        return {"final_trade_decision": PM_BUY, "trader_investment_plan": PLAN}
    return analyze


def events_of(out):
    return [json.loads(line) for line in (out / "events.jsonl").read_text().splitlines()]


def test_buy_hold_and_equity(tmp_path):
    bars, calls = make_bars(), []
    book = run_simulation(DATES, END, lambda d: ["SNA", "CHD"], agent_allocator(bars), bars,
                          PortfolioContext(cash=2000.0, currency="USD"), tmp_path, analyze=fake_analyzer(calls))
    assert calls == [("SNA", "2026-08-17")]                      # held and still in the top list: no new analysis
    assert (book.cash, book.positions[0].ticker, book.positions[0].quantity) == (800.0, "SNA", 12)
    saved = load_portfolio(tmp_path / "portfolio.json")          # state file is readable by the framework
    assert saved.cash == 800.0 and saved.position_in("SNA").average_price == 100.0
    assert events_of(tmp_path)[-1]["value"] == 2000.0


def test_stop_fills_at_the_stop_or_at_the_open_on_a_gap(tmp_path):
    bars = make_bars(SNA={"2026-08-20": (99.0, 99.0, 90.0, 92.0)})           # low below the 94.5 stop
    run_simulation(DATES, END, lambda d: ["SNA"], agent_allocator(bars), bars,
                   PortfolioContext(cash=2000.0), tmp_path, analyze=fake_analyzer([]))
    stop = next(e for e in events_of(tmp_path) if e["event"] == "stop")
    assert (stop["date"], stop["price"]) == ("2026-08-20", 94.5)
    gap = make_bars(SNA={"2026-08-20": (90.0, 91.0, 88.0, 90.0)})            # opens below the stop
    run_simulation(DATES, END, lambda d: ["SNA"], agent_allocator(gap), gap,
                   PortfolioContext(cash=2000.0), tmp_path / "gap", analyze=fake_analyzer([]))
    assert next(e for e in events_of(tmp_path / "gap") if e["event"] == "stop")["price"] == 90.0


def test_stopped_out_run_buys_the_leader_again_on_the_next_rotation_date(tmp_path):
    bars = make_bars(SNA={"2026-08-20": (99.0, 99.0, 90.0, 92.0)})
    calls = []
    run_simulation(DATES, END, lambda d: ["SNA"], agent_allocator(bars), bars,
                   PortfolioContext(cash=2000.0), tmp_path, analyze=fake_analyzer(calls))
    assert [e["event"] for e in events_of(tmp_path) if e["event"] != "equity"] == ["buy", "stop", "buy"]
    assert calls == [("SNA", "2026-08-17"), ("SNA", "2026-08-31")]


def test_leaving_the_top_list_sells_and_buys_the_new_leader(tmp_path):
    bars = make_bars()
    rankings = {"2026-08-17": ["SNA", "CHD"], "2026-08-31": ["CHD", "IEX"]}      # SNA drops out
    book = run_simulation(DATES, END, rankings.get, agent_allocator(bars), bars,
                          PortfolioContext(cash=2000.0), tmp_path, analyze=fake_analyzer([]), top_n=2)
    assert [e["event"] for e in events_of(tmp_path) if e["event"] != "equity"] == ["buy", "sell", "buy"]
    assert [p.ticker for p in book.positions] == ["CHD"]


def test_skip_stays_in_cash_and_tries_again(tmp_path):
    bars, calls = make_bars(), []
    skip = AllocatorDecision(action="Skip", reasoning="weak")
    book = run_simulation(DATES, END, lambda d: ["SNA"], agent_allocator(bars, skip), bars,
                          PortfolioContext(cash=2000.0), tmp_path, analyze=fake_analyzer(calls))
    assert book.cash == 2000.0 and not book.positions and len(calls) == 2
    log = [json.loads(line) for line in (tmp_path / "allocator_log.jsonl").read_text().splitlines()]
    assert [r["action"] for r in log] == ["none", "none"] and log[0]["reasoning"] == "weak"


def test_pure_magic_formula_buys_everything_without_agents_or_stop(tmp_path):
    bars = make_bars(SNA={"2026-08-20": (99.0, 99.0, 50.0, 60.0)})            # a crash: no stop, so no exit
    book = run_simulation(DATES, END, lambda d: ["SNA"], create_buy_all(bars), bars,
                          PortfolioContext(cash=2000.0), tmp_path)
    assert (book.cash, book.positions[0].quantity) == (0.0, 20)
    assert not any(e["event"] == "stop" for e in events_of(tmp_path))


def test_rotation_step():
    assert rotation_step(None, ["a", "b"]).buy == "A"
    assert rotation_step("B", ["a", "b"]).buy is None and rotation_step("B", ["a", "b"]).sell is None   # keep it
    step = rotation_step("Z", ["a", "b"], top_n=2)
    assert (step.sell, step.buy) == ("Z", "A")
    assert rotation_step("Z", ["a", "b", "z"], top_n=2).sell == "Z"           # rank 3 is outside the top 2
    assert rotation_step("Z", []).sell is None


def test_book_and_portfolio_file(tmp_path):
    book = buy(PortfolioContext(cash=2000.0, currency="USD"), "sna", 10, 100.0)
    assert book.cash == 1000.0 and (book.positions[0].ticker, book.positions[0].quantity) == ("SNA", 10)
    assert market_value(book, {"SNA": 120.0}) == 2200.0
    assert sell_all(book, "SNA", 120.0).cash == 2200.0
    with pytest.raises(ValueError):
        buy(book, "CHD", 100, 100.0)                                               # more than the cash
    path = tmp_path / "p.json"
    save_portfolio(book, path)
    assert not path.read_bytes().startswith(b"\xef\xbb\xbf") and load_portfolio(path).cash == 1000.0


def test_buy_and_hold_benchmark():
    bars = make_bars(AAA={"2026-08-31": (110.0,) * 4}, BBB={})
    bars2 = lambda t, d: bars(t, d).assign(Close=lambda x: x["Close"] * (0.5 if t == "BBB" else 1.0))
    result = buy_and_hold(["AAA", "BBB"], "2026-08-17", ["2026-08-31"], 1000.0, bars2)
    assert result["values"] == [{"date": "2026-08-31", "value": 1050.0}] and result["skipped"] == []


def test_runner_pure_magic_end_to_end(tmp_path, monkeypatch):
    from tools_debug import run_simulation as runner
    pd.DataFrame({"ticker": ["SNA", "CHD"]}).to_csv(tmp_path / "rank.csv", index=False)
    bars = make_bars()
    monkeypatch.setattr(runner, "load_ohlcv", bars)
    book = runner.main(["--ranking", str(tmp_path / "rank.csv"), "--dates", "2026-08-17", "--end", "2026-08-31",
                        "--cash", "2000", "--out-dir", str(tmp_path / "sim"), "--pure-magic"])
    assert book.positions[0].ticker == "SNA" and book.positions[0].quantity == 20
    assert load_portfolio(tmp_path / "sim" / "portfolio.json").cash == 0.0
