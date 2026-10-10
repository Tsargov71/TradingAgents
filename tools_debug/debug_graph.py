"""
debug_graph.py — TradingAgentsGraphDebug

A rewrite of TradingAgentsGraph's execution path that runs the pipeline
SEQUENTIALLY (explicit node-by-node calls, no LangGraph stream) and persists
an artifact for every operation, so that:

  * every tool call and every LLM answer is cached on disk;
  * re-running skips everything already computed (resume where you stopped);
  * deleting artifacts forces recomputation from exactly that point onward.

Design notes
------------
Cache keys are CONTENT HASHES, not sequence numbers. That matters: if you
edit a tool's implementation, its output changes -> the analyst's next prompt
changes -> that LLM call's hash changes -> it recomputes automatically, while
every step before it stays a cache hit. Invalidation cascades on its own; you
don't have to reason about which downstream steps went stale.

Agent node code is NOT duplicated here. The create_*_node factories from
tradingagents.agents are reused as-is, the analysts' tools come from the same
TOOLS tuples the production graph builds its ToolNodes from, and
ConditionalLogic is reused for routing, so this debug runner exercises the
same prompts, tools and branching as production. What's rewritten is only the
orchestration: an explicit Python loop instead of a compiled StateGraph.

Differences from production worth knowing (graph layout of tradingagents 0.6):
  * production runs the analysts and the Memory Log step in parallel; here they
    run one after another, in the order given by --analysts;
  * each analyst keeps a PRIVATE message history and contributes only its
    report, exactly like its sub-graph in graph/setup.py, including the
    wrap-up turn once config["max_tool_rounds"] is spent;
  * tools run through a LangGraph ToolNode (not a bare tool.invoke), because
    the data tools take the ticker and trade date from the graph state
    (InjectedState). The tool cache key therefore includes both.

Layout produced under <out_dir>:
    ledger.jsonl              append-only record of every op, in order
    steps/NNN_<node>.json     state delta produced by each node execution
    cache/tools/<name>_<h>.json   tool call: args + output
    cache/llm/<hash>.json         LLM call: prompt + completion
    reports/<key>.md          each report section as it is produced
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

from langchain_core.caches import BaseCache
from langchain_core.globals import set_llm_cache
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage, convert_to_messages
from langchain_core.outputs import ChatGeneration
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from tradingagents.agents import (
    create_aggressive_debator,
    create_bear_researcher,
    create_bull_researcher,
    create_conservative_debator,
    create_fundamentals_analyst,
    create_market_analyst,
    create_neutral_debator,
    create_news_analyst,
    create_portfolio_manager,
    create_research_manager,
    create_sentiment_analyst,
    create_trader,
)
from tradingagents.agents.analysts.turn import WRAP_UP
from tradingagents.agents.rating import run_rating
from tradingagents.agents.state import AgentState
from tradingagents.dataflows.config import run_config
from tradingagents.graph.analyst_execution import build_analyst_execution_plan
from tradingagents.graph.trading_graph import TradingAgentsGraph, _validate_trade_date

REPORT_KEYS = ("market_report", "fundamentals_report", "sentiment_report", "news_report")


# --------------------------------------------------------------------------
# LLM cache
# --------------------------------------------------------------------------
def _sha(*parts: str) -> str:
    h = hashlib.sha256()
    for p in parts:
        h.update(p.encode("utf-8", errors="replace"))
    return h.hexdigest()[:16]


# --------------------------------------------------------------------------
# LLM cache
# --------------------------------------------------------------------------
class FileLLMCache(BaseCache):
    """Disk-backed LangChain LLM cache: one JSON file per (prompt, model) pair.

    Registered globally via set_llm_cache(), so it transparently covers every
    `prompt | llm.bind_tools(tools)` chain inside the agent nodes without
    wrapping or patching them.

    Generations are reconstructed field-by-field rather than via langchain's
    generic serializer, so a langchain-core version bump can't silently break
    round-tripping of tool_calls.
    """

    def __init__(self, cache_dir: Path, on_event=None):
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.on_event = on_event or (lambda **kw: None)

    @staticmethod
    def _stable(prompt: str) -> str:
        """The prompt with per-message ``usage_metadata`` removed.

        langchain stamps a message it serves from a cache with its own
        ``usage_metadata``, so the same history reads differently after a fresh
        call and after a cache hit, and every later prompt would get a new key
        on the second run (a miss), only settling on the third. The token
        counts say nothing about what the model is asked, so they are left out
        of the key.
        """
        try:
            data = json.loads(prompt)
        except (TypeError, ValueError):
            return prompt

        def strip(node):
            if isinstance(node, dict):
                return {k: strip(v) for k, v in node.items() if k != "usage_metadata"}
            if isinstance(node, list):
                return [strip(v) for v in node]
            return node

        return json.dumps(strip(data), sort_keys=True, ensure_ascii=False)

    def _path(self, prompt: str, llm_string: str) -> Path:
        return self.cache_dir / f"{_sha(self._stable(prompt), llm_string)}.json"

    def lookup(self, prompt: str, llm_string: str):
        path = self._path(prompt, llm_string)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None  # corrupt/partial file: treat as a miss and recompute
        gens = []
        for g in data.get("generations", []):
            msg = AIMessage(
                content=g.get("content", ""),
                additional_kwargs=g.get("additional_kwargs", {}) or {},
                response_metadata=g.get("response_metadata", {}) or {},
                tool_calls=g.get("tool_calls", []) or [],
                id=g.get("id"),
            )
            gens.append(ChatGeneration(message=msg))
        if not gens:
            return None
        self.on_event(kind="llm", status="hit", artifact=path.name,
                      detail=f"{len(data.get('generations', []))} generation(s)")
        return gens

    def update(self, prompt: str, llm_string: str, return_val) -> None:
        path = self._path(prompt, llm_string)
        gens = []
        for gen in return_val:
            msg = getattr(gen, "message", None)
            gens.append({
                "content": getattr(msg, "content", getattr(gen, "text", "")),
                "additional_kwargs": getattr(msg, "additional_kwargs", {}),
                "response_metadata": getattr(msg, "response_metadata", {}),
                "tool_calls": getattr(msg, "tool_calls", []),
                "id": getattr(msg, "id", None),
            })
        payload = {
            "cached_at": time.time(),
            "llm_string": llm_string,
            "prompt": prompt,          # full prompt kept for inspection
            "generations": gens,
        }
        path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        if gens:
            calls = gens[0].get("tool_calls") or []
            # A tool-calling turn has empty content, so show the requested
            # tools instead of an empty line.
            preview = (gens[0]["content"] or "")[:120] or (
                "-> " + ", ".join(c.get("name", "?") for c in calls) if calls else "(empty)"
            )
        else:
            preview = ""
        self.on_event(kind="llm", status="miss", artifact=path.name, detail=preview)

    def clear(self, **kwargs) -> None:
        for f in self.cache_dir.glob("*.json"):
            f.unlink()


# --------------------------------------------------------------------------
# The debug graph
# --------------------------------------------------------------------------
class TradingAgentsGraphDebug(TradingAgentsGraph):
    """Sequential, artifact-producing, resumable version of TradingAgentsGraph.

    Use `run_sequential()` instead of `propagate()`. Everything else
    (LLM clients, memory log, settlement, report/state logging) is inherited
    unchanged from the production class.
    """

    def __init__(self, selected_analysts=("fundamentals",), config=None,
                 out_dir: str | Path = "tools_debug/run_output", **kwargs):
        super().__init__(selected_analysts, debug=False, config=config, **kwargs)

        self.out_dir = Path(out_dir)
        self.steps_dir = self.out_dir / "steps"
        self.reports_dir = self.out_dir / "reports"
        self.tool_cache_dir = self.out_dir / "cache" / "tools"
        self.llm_cache_dir = self.out_dir / "cache" / "llm"
        for d in (self.steps_dir, self.reports_dir, self.tool_cache_dir, self.llm_cache_dir):
            d.mkdir(parents=True, exist_ok=True)
        self.ledger_path = self.out_dir / "ledger.jsonl"

        self._seq = self._last_seq_in_ledger()

        # Register the disk LLM cache globally. This is what makes LLM answers
        # survive across runs without touching a single line of agent code.
        set_llm_cache(FileLLMCache(self.llm_cache_dir, on_event=self._record))

        # One ToolNode per analyst, built from the same TOOLS tuple the
        # production graph uses (graph/setup.py), so any tool added to an
        # analyst's TOOLS (e.g. get_ma_crossover, get_volatility_analysis,
        # get_competitor_comparison) is picked up automatically.
        plan = build_analyst_execution_plan(self.selected_analysts)
        self._tool_nodes: dict[str, ToolNode] = {
            spec.key: ToolNode(list(spec.tools)) for spec in plan.specs if spec.tools
        }
        # A ToolNode needs a graph runtime to inject state into the tools, so
        # each one is run inside a one-node graph (START -> tools -> END).
        self._tool_graphs = {key: self._one_node_graph(node) for key, node in self._tool_nodes.items()}

    @staticmethod
    def _one_node_graph(node: ToolNode):
        graph = StateGraph(AgentState)
        graph.add_node("tools", node)
        graph.add_edge(START, "tools")
        graph.add_edge("tools", END)
        return graph.compile()

    # ---------------- ledger + artifacts ----------------
    def _last_seq_in_ledger(self) -> int:
        if not self.ledger_path.exists():
            return 0
        last = 0
        for line in self.ledger_path.read_text(encoding="utf-8").splitlines():
            try:
                last = max(last, json.loads(line).get("seq", 0))
            except json.JSONDecodeError:
                continue
        return last

    def _record(self, kind: str, status: str, artifact: str = "",
                name: str = "", detail: str = "") -> int:
        self._seq += 1
        entry = {
            "seq": self._seq,
            "ts": time.strftime("%H:%M:%S"),
            "kind": kind,          # tool | llm | node
            "name": name,
            "status": status,      # hit | miss | done
            "artifact": artifact,
            "detail": (detail or "")[:400],
        }
        with open(self.ledger_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
            f.flush()
        flag = {"hit": "CACHED", "miss": "ran", "done": "done"}.get(status, status)
        label = f"{kind}:{name}" if name else kind
        print(f"[{entry['ts']}] #{self._seq:03d} {label} [{flag}] {detail[:160]}", flush=True)
        return self._seq

    # ---------------- cached tool execution ----------------
    def _run_tool(self, spec_key: str, call: dict, state: dict) -> ToolMessage:
        """Execute one tool call (through the analyst's ToolNode), or return its cached output.

        Cache key is (tool name, args, ticker, trade date): the data tools read
        the last two from the graph state, so they are part of what the call
        depends on. Delete the file to force one specific call to re-run.
        Errors are returned to the model as tool output but never cached, so
        fixing the tool makes the next run recompute it.
        """
        name, args, call_id = call["name"], call.get("args", {}), call["id"]
        key = json.dumps(
            {"args": args, "ticker": state["company_of_interest"], "date": state["trade_date"]},
            sort_keys=True, default=str,
        )
        path = self.tool_cache_dir / f"{name}_{_sha(name, key)}.json"

        if path.exists():
            try:
                out = json.loads(path.read_text(encoding="utf-8"))["output"]
                self._record("tool", "hit", path.name, name, f"{len(out)} chars")
                return ToolMessage(content=out, name=name, tool_call_id=call_id)
            except (json.JSONDecodeError, KeyError, OSError):
                pass  # corrupt artifact: fall through and recompute

        node = self._tool_nodes.get(spec_key)
        if node is None or name not in node.tools_by_name:
            # Bound to the LLM but absent from the analyst's ToolNode -> would
            # also fail in production. Surfaced explicitly instead of silently.
            err = f"ERROR: tool '{name}' is not registered in the {spec_key} ToolNode"
            self._record("tool", "miss", "", name, err)
            return ToolMessage(content=err, name=name, tool_call_id=call_id)

        request = AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": call_id,
                                                      "type": "tool_call"}])
        try:
            result = self._tool_graphs[spec_key].invoke(
                {**state, "messages": [*state["messages"], request]})
            reply = result["messages"][-1]
        except Exception as e:  # keep the run alive; the error becomes the tool output
            out = f"ERROR calling {name}: {e}"
            self._record("tool", "miss", "", name, out)
            return ToolMessage(content=out, name=name, tool_call_id=call_id)

        out = reply.content if isinstance(reply.content, str) else str(reply.content)
        if getattr(reply, "status", "success") == "error":
            self._record("tool", "miss", "", name, out)
            return ToolMessage(content=out, name=name, tool_call_id=call_id)

        path.write_text(
            json.dumps({"tool": name, "args": args, "ticker": state["company_of_interest"],
                        "trade_date": state["trade_date"], "output": out,
                        "computed_at": time.time()}, indent=2, default=str),
            encoding="utf-8",
        )
        self._record("tool", "miss", path.name, name, f"{len(out)} chars")
        return ToolMessage(content=out, name=name, tool_call_id=call_id)

    # ---------------- state helpers ----------------
    @staticmethod
    def _merge(state: dict, delta: dict) -> dict:
        """Apply a node's return value to the state.

        Uses LangGraph's own `add_messages` reducer for the `messages` key
        rather than a hand-rolled append, so raw ("human", "TEXT") tuples are
        coerced into Message objects and RemoveMessage deletions apply, as in
        the compiled graph. Every other key is replaced.
        """
        for key, val in (delta or {}).items():
            if key == "messages":
                incoming = val if isinstance(val, list) else [val]
                state["messages"] = add_messages(state.get("messages", []), incoming)
            else:
                state[key] = val
        return state

    def _save_step(self, node_name: str, delta: dict) -> str:
        fname = self.steps_dir / f"{self._seq + 1:03d}_{node_name.replace(' ', '_')}.json"
        fname.write_text(json.dumps(delta, indent=2, default=str), encoding="utf-8")
        return fname.name

    def _run_node(self, node_name: str, node_fn, state: dict) -> dict:
        delta = node_fn(state)
        artifact = self._save_step(node_name, delta)
        summary = ", ".join(k for k in (delta or {}) if k != "messages") or "messages only"
        self._record("node", "done", artifact, node_name, summary)
        self._merge(state, delta)
        for key in REPORT_KEYS:
            if (delta or {}).get(key):
                (self.reports_dir / f"{key}.md").write_text(delta[key], encoding="utf-8")
        return state

    def _run_analyst(self, spec, agent, state: dict) -> str:
        """One analyst's tool loop on a private history; returns its report.

        Mirrors the sub-graph in graph/setup.py: the model answers, its tool
        calls run, and once `max_tool_rounds` rounds of calls are spent it is
        asked to write its report (the wrap-up turn), which ends the analyst.
        """
        local = {**state, "messages": list(state["messages"])}
        max_rounds = self.config["max_tool_rounds"]
        while True:
            self._run_node(spec.agent_node, agent, local)
            calls = getattr(local["messages"][-1], "tool_calls", None) or []
            if not calls or spec.key not in self._tool_nodes:
                break
            for call in calls:
                local["messages"].append(self._run_tool(spec.key, call, local))
            rounds = sum(1 for m in local["messages"] if getattr(m, "tool_calls", None))
            if rounds >= max_rounds:
                self._run_node(
                    f"{spec.agent_node} wrap-up",
                    lambda s: agent({**s, "messages": [*s["messages"], HumanMessage(WRAP_UP)]}),
                    local,
                )
                break
        return local.get(spec.report_key, "")

    # ---------------- the sequential pipeline ----------------
    def run_sequential(self, ticker: str, trade_date: str, asset_type: str = "stock",
                       portfolio=None):
        """Run the whole pipeline as explicit sequential steps.

        Same nodes and same routing decisions as the compiled graph, but every
        step is observable, cached and resumable. Returns (state, rating).
        """
        trade_date = _validate_trade_date(trade_date)
        self.ticker = ticker

        with run_config(self.config):
            state = self.create_run_state(ticker, trade_date, asset_type, portfolio)
            # create_initial_state seeds messages as [("human", ticker)]; the
            # compiled graph coerces that through its reducer, here it is done
            # up front so no bare tuple reaches the agents.
            state["messages"] = convert_to_messages(state["messages"])
            self._record("node", "done", "", "resolve_instrument_context",
                         state["instrument_context"][:120])

            q, d = self.quick_thinking_llm, self.deep_thinking_llm

            # --- 0. memory log: settle past decisions, load lessons ---
            state = self._run_node("Memory Log", self._memory_step, state)

            # --- 1. analysts, each with its own tool loop ---
            analyst_factories = {
                "market": lambda: create_market_analyst(q),
                "social": lambda: create_sentiment_analyst(q),
                "news": lambda: create_news_analyst(q),
                "fundamentals": lambda: create_fundamentals_analyst(q),
            }
            plan = build_analyst_execution_plan(self.selected_analysts)
            for spec in plan.specs:
                agent = analyst_factories[spec.key]()
                state[spec.report_key] = self._run_analyst(spec, agent, state)

            # --- 2. bull/bear debate, routed by ConditionalLogic ---
            debate_nodes = {
                "Bull Researcher": create_bull_researcher(q),
                "Bear Researcher": create_bear_researcher(q),
            }
            current = "Bull Researcher"
            while True:
                state = self._run_node(current, debate_nodes[current], state)
                nxt = self.conditional_logic.should_continue_debate(state)
                if nxt == "Research Manager":
                    break
                current = nxt

            state = self._run_node("Research Manager", create_research_manager(d), state)
            state = self._run_node("Trader", create_trader(q), state)

            # --- 3. risk debate ---
            risk_nodes = {
                "Aggressive Analyst": create_aggressive_debator(q),
                "Conservative Analyst": create_conservative_debator(q),
                "Neutral Analyst": create_neutral_debator(q),
            }
            current = "Aggressive Analyst"
            while True:
                state = self._run_node(current, risk_nodes[current], state)
                nxt = self.conditional_logic.should_continue_risk_analysis(state)
                if nxt == "Portfolio Manager":
                    break
                current = nxt

            state = self._run_node("Portfolio Manager", create_portfolio_manager(d), state)

            # --- 4. same persistence tail as _run_graph() ---
            self.curr_state = state
            self.record_decision(ticker, trade_date, state)
            rating = run_rating(state)
            (self.out_dir / "final_decision.md").write_text(str(rating), encoding="utf-8")
            self._record("node", "done", "final_decision.md", "FINISHED", str(rating))
            return state, rating

    # ---------------- invalidation ----------------
    def invalidate_from(self, seq: int) -> int:
        """Delete every cached artifact recorded at ledger seq >= `seq`.

        Use this to rewind: `invalidate_from(12)` makes the next run replay
        from operation 12 onward while keeping everything before it cached.
        Returns the number of artifacts removed.
        """
        if not self.ledger_path.exists():
            return 0
        kept_lines, removed = [], 0
        for line in self.ledger_path.read_text(encoding="utf-8").splitlines():
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if entry.get("seq", 0) < seq:
                kept_lines.append(line)
                continue
            art = entry.get("artifact") or ""
            for d in (self.tool_cache_dir, self.llm_cache_dir, self.steps_dir):
                p = d / art
                if art and p.exists():
                    p.unlink()
                    removed += 1
        self.ledger_path.write_text("\n".join(kept_lines) + ("\n" if kept_lines else ""),
                                    encoding="utf-8")
        self._seq = self._last_seq_in_ledger()
        return removed