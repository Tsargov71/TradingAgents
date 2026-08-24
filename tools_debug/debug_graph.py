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
tradingagents.agents are reused as-is, and ConditionalLogic is reused for
routing, so this debug runner exercises the same prompts and the same
branching as production. What's rewritten is only the orchestration: an
explicit Python loop instead of a compiled StateGraph.

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
from langchain_core.messages import AIMessage, ToolMessage, convert_to_messages
from langchain_core.outputs import ChatGeneration
from langgraph.graph.message import add_messages

from tradingagents.agents import (
    create_aggressive_debator,
    create_bear_researcher,
    create_bull_researcher,
    create_conservative_debator,
    create_fundamentals_analyst,
    create_market_analyst,
    create_msg_delete,
    create_neutral_debator,
    create_news_analyst,
    create_portfolio_manager,
    create_research_manager,
    create_sentiment_analyst,
    create_trader,
)
from tradingagents.graph.analyst_execution import build_analyst_execution_plan
from tradingagents.graph.trading_graph import TradingAgentsGraph

REPORT_KEYS = ("market_report", "fundamentals_report", "sentiment_report", "news_report")


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

    def _path(self, prompt: str, llm_string: str) -> Path:
        return self.cache_dir / f"{_sha(prompt, llm_string)}.json"

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
    (LLM clients, tool nodes, memory log, signal processing) is inherited
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

        # name -> tool, taken from the ToolNodes the production class already
        # built, so any tool added there (e.g. the new ma_crossover /
        # volatility / competitor_comparison tools) is picked up automatically.
        self._tools_by_name: dict[str, Any] = {}
        for node in self.tool_nodes.values():
            self._tools_by_name.update(getattr(node, "tools_by_name", {}) or {})

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
    def _run_tool(self, name: str, args: dict) -> str:
        """Execute one tool call, or return its cached output.

        Cache key is (tool name, args), so identical calls across runs are
        free. Delete the file to force one specific tool call to re-run.
        """
        args_json = json.dumps(args, sort_keys=True, default=str)
        path = self.tool_cache_dir / f"{name}_{_sha(name, args_json)}.json"

        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                out = data["output"]
                self._record("tool", "hit", path.name, name, f"{len(out)} chars")
                return out
            except (json.JSONDecodeError, KeyError, OSError):
                pass  # corrupt artifact: fall through and recompute

        tool = self._tools_by_name.get(name)
        if tool is None:
            # Bound to the LLM but absent from every ToolNode -> would also
            # fail in production. Surfaced explicitly instead of silently.
            err = f"ERROR: tool '{name}' is not registered in any ToolNode"
            self._record("tool", "miss", "", name, err)
            return err

        try:
            out = str(tool.invoke(args))
        except Exception as e:  # keep the run alive; the error becomes the tool output
            out = f"ERROR calling {name}: {e}"
            self._record("tool", "miss", "", name, out)
            return out

        path.write_text(
            json.dumps({"tool": name, "args": args, "output": out,
                        "computed_at": time.time()}, indent=2, default=str),
            encoding="utf-8",
        )
        self._record("tool", "miss", path.name, name, f"{len(out)} chars")
        return out

    # ---------------- state helpers ----------------
    @staticmethod
    def _merge(state: dict, delta: dict) -> dict:
        """Apply a node's return value to the state.

        Uses LangGraph's own `add_messages` reducer for the `messages` key
        rather than a hand-rolled append. That reducer is what the compiled
        StateGraph uses via MessagesState, and it does two things a naive
        append does not: it COERCES raw ("human", "TEXT") tuples into real
        Message objects (create_initial_state seeds one), and it applies
        RemoveMessage deletions. Every other key is replaced.
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

    def _drain_tool_calls(self, state: dict) -> bool:
        """If the last message requested tools, run them all and append results.

        Returns True if any tool ran (i.e. the analyst should be called again).
        """
        last = state["messages"][-1]
        calls = getattr(last, "tool_calls", None) or []
        if not calls:
            return False
        for call in calls:
            output = self._run_tool(call["name"], call.get("args", {}))
            state["messages"].append(
                ToolMessage(content=output, name=call["name"], tool_call_id=call["id"])
            )
        return True

    # ---------------- the sequential pipeline ----------------
    def run_sequential(self, ticker: str, trade_date: str, asset_type: str = "stock"):
        """Run the whole pipeline as explicit sequential steps.

        Same node order and same routing decisions as the compiled graph,
        but every step is observable, cached and resumable.
        """
        self.ticker = ticker

        self._record("node", "done", "", "resolve_pending_entries", "memory log")
        self._resolve_pending_entries(ticker)

        past_context = self.memory_log.get_past_context(ticker)
        instrument_context = self.resolve_instrument_context(ticker, asset_type)
        self._record("node", "done", "", "resolve_instrument_context", instrument_context[:120])

        state = self.propagator.create_initial_state(
            ticker, trade_date, asset_type=asset_type,
            past_context=past_context, instrument_context=instrument_context,
        )
        # create_initial_state seeds messages as [("human", ticker)]. The
        # compiled graph coerces that via its reducer on first update; here the
        # first analyst node reads the list before any merge happens, so coerce
        # it now rather than letting a bare tuple reach the agents.
        state["messages"] = convert_to_messages(state["messages"])

        q, d = self.quick_thinking_llm, self.deep_thinking_llm
        analyst_factories = {
            "market": lambda: create_market_analyst(q),
            "social": lambda: create_sentiment_analyst(q),
            "news": lambda: create_news_analyst(q),
            "fundamentals": lambda: create_fundamentals_analyst(q),
        }
        msg_clear = create_msg_delete()

        # --- 1. analysts, each with its own tool loop ---
        plan = build_analyst_execution_plan(self.selected_analysts)
        for spec in plan.specs:
            node_fn = analyst_factories[spec.key]()
            router = getattr(self.conditional_logic, f"should_continue_{spec.key}")
            while True:
                state = self._run_node(spec.agent_node, node_fn, state)
                if router(state) != spec.tool_node:
                    break
                if not self._drain_tool_calls(state):
                    break
            state = self._run_node(spec.clear_node, msg_clear, state)

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
        self._log_state(trade_date, state)
        self.memory_log.store_decision(
            ticker=ticker, trade_date=trade_date,
            final_trade_decision=state["final_trade_decision"],
        )
        decision = self.process_signal(state["final_trade_decision"])
        (self.out_dir / "final_decision.md").write_text(str(decision), encoding="utf-8")
        self._record("node", "done", "final_decision.md", "FINISHED", str(decision))
        return state, decision

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