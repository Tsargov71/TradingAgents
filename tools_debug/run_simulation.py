"""Run the one-stock-at-a-time simulation (rotation + allocator).

Prova A / B: same command with TRADINGAGENTS_EXTRA_TOOLS=0 / 1, a different --out-dir and a
different TRADINGAGENTS_MEMORY_LOG_PATH. --pure-magic runs the benchmark without agents.

  python -m tools_debug.run_simulation --ranking results_T/magic_formula_with_trend.csv \\
      --dates 2026-08-17 --end 2026-10-19 --cash 2000 --out-dir sim_A
"""

from __future__ import annotations

import argparse

import pandas as pd

from tradingagents.dataflows.vendors.yahoo.ohlcv import load_ohlcv
from tradingagents.portfolio import PortfolioContext
from tradingagents.simulation import create_buy_all, run_simulation


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ranking", required=True, help="CSV with a 'ticker' column, best first; {date} allowed")
    parser.add_argument("--dates", required=True, help="rotation dates, comma separated")
    parser.add_argument("--end", required=True, help="last day of the simulation")
    parser.add_argument("--cash", type=float, default=2000.0)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--pure-magic", action="store_true")
    args = parser.parse_args(argv)

    if args.pure_magic:
        allocate, analyze = create_buy_all(load_ohlcv), None
    else:
        from tradingagents.agents.allocator import create_allocator
        from tradingagents.default_config import DEFAULT_CONFIG
        from tradingagents.graph.trading_graph import TradingAgentsGraph
        from tradingagents.simulation import make_analyzer

        graph = TradingAgentsGraph(debug=False, config=DEFAULT_CONFIG.copy())
        allocate, analyze = create_allocator(graph.quick_thinking_llm), make_analyzer(graph)

    def ranking_for_date(date):
        return pd.read_csv(args.ranking.format(date=date))["ticker"].dropna().tolist()

    return run_simulation(args.dates.split(","), args.end, ranking_for_date, allocate, load_ohlcv,
                          PortfolioContext(cash=args.cash, currency="USD"), args.out_dir,
                          analyze=analyze, top_n=args.top_n)


if __name__ == "__main__":
    main()
