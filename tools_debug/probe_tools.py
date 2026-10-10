"""
probe_tools.py — invoke the three added tools DIRECTLY, with no LLM in the loop.

Separates two questions that a full pipeline run conflates:

  1. Does the tool work?          <- this script answers it, in seconds
  2. Does the model choose it?    <- only a full run answers that, and the
                                     answer is non-deterministic

Run 1 first. A tool that errors here will never produce anything useful no
matter how often the model calls it; a tool that works here but never shows up
in ledger.jsonl is a prompt problem, not a code problem.

Usage:
    python -u -m tools_debug.probe_tools --ticker SNA --date 2026-08-21
    python -u -m tools_debug.probe_tools --ticker LDO.MI --competitors CAT,SWK
"""
import argparse
import traceback

from tradingagents.dataflows.config import set_config
from tradingagents.default_config import DEFAULT_CONFIG

from tradingagents.agents.ma_crossover_tool import get_ma_crossover
from tradingagents.agents.volatility_tool import get_volatility_analysis
from tradingagents.agents.competitor_comparison_tool import get_competitor_comparison


def run(label: str, tool, args: dict, preview: int):
    print(f"\n{'=' * 70}\n{label}  args={args}\n{'=' * 70}")
    try:
        # .invoke() goes through the StructuredTool arg validation, i.e. the
        # same path a real tool call from the model would take -- not the bare
        # Python function. A schema mismatch shows up here too.
        out = str(tool.invoke(args))
    except Exception:
        print("RAISED:")
        traceback.print_exc()
        return False
    print(f"OK — {len(out)} chars")
    print(out[:preview] + ("\n...[truncated]" if len(out) > preview else ""))
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticker", default="SNA")
    ap.add_argument("--date", default="2026-08-21")
    ap.add_argument("--competitors", default="CAT,SWK,DOV",
                    help="comma-separated competitor tickers")
    ap.add_argument("--preview", type=int, default=1500,
                    help="chars of output to print per tool")
    args = ap.parse_args()

    # The tools read vendor settings (yfinance vs alpha_vantage) from the
    # dataflows config, which TradingAgentsGraph normally sets at startup.
    # Without this they'd run against an unset config.
    set_config(DEFAULT_CONFIG.copy())

    competitors = [c.strip() for c in args.competitors.split(",") if c.strip()]
    results = {}

    results["get_ma_crossover"] = run(
        "get_ma_crossover", get_ma_crossover,
        {"symbol": args.ticker, "curr_date": args.date}, args.preview)

    results["get_volatility_analysis"] = run(
        "get_volatility_analysis", get_volatility_analysis,
        {"symbol": args.ticker, "curr_date": args.date}, args.preview)

    results["get_competitor_comparison"] = run(
        "get_competitor_comparison", get_competitor_comparison,
        {"ticker": args.ticker, "competitor_tickers": competitors,
         "curr_date": args.date}, args.preview)

    print(f"\n{'=' * 70}\nSUMMARY")
    for name, ok in results.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")


if __name__ == "__main__":
    main()