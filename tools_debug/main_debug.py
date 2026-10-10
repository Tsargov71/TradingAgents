"""
main_debug.py — runner for TradingAgentsGraphDebug.

Replicates the CLI selections (WSM / 2026-08-21 / English / Fundamentals
Analyst only / Shallow) but runs the pipeline sequentially with every tool
call and LLM answer cached to disk.

Typical loop while debugging a tool:

    python -u tools_debug/main_debug.py                # first run: computes everything
    python -u tools_debug/main_debug.py                # second run: all cache hits, instant

    # now edit tradingagents/agents/competitor_comparison_tool.py ...
    python -u tools_debug/main_debug.py                # the edited tool re-runs on its own:
                                                       # its output changed -> the analyst
                                                       # prompt changed -> that LLM call and
                                                       # everything after it recompute, while
                                                       # earlier steps stay cached.

    # to rewind explicitly to a given operation number from ledger.jsonl:
    python -u tools_debug/main_debug.py --from-step 12

    # to start completely clean:
    python -u tools_debug/main_debug.py --fresh

    # example with a ticker:
    python -u -m tools_debug.main_debug \
        --ticker SNA \
        --analysts market,fundamentals \
        --out-dir tools_debug/run_output_SNA

Inspect ledger.jsonl to see the ordered list of operations and pick the
number to rewind to.
"""
import argparse
import shutil
from pathlib import Path

from tradingagents.default_config import DEFAULT_CONFIG

from tools_debug.debug_graph import TradingAgentsGraphDebug

DEFAULTS = dict(ticker="WSM", date="2026-08-21", asset_type="stock")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ticker", default=DEFAULTS["ticker"])
    ap.add_argument("--date", default=DEFAULTS["date"])
    ap.add_argument("--asset-type", default=DEFAULTS["asset_type"])
    ap.add_argument("--analysts", default="fundamentals",
                    help="comma-separated: market,social,news,fundamentals")
    ap.add_argument("--from-step", type=int, default=None, metavar="N",
                    help="delete cached artifacts from ledger op N onward, then run")
    ap.add_argument("--fresh", action="store_true",
                    help="wipe run_output entirely before running")
    ap.add_argument("--out-dir", default=str(Path(__file__).parent / "run_output"))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    if args.fresh and out_dir.exists():
        shutil.rmtree(out_dir)
        print(f"Wiped {out_dir}")

    config = DEFAULT_CONFIG.copy()
    config["max_debate_rounds"] = 1        # "Shallow"
    config["max_risk_discuss_rounds"] = 1  # "Shallow"
    config["output_language"] = "English"
    # llm_provider / base url come from the container env
    # (TRADINGAGENTS_LLM_PROVIDER=ollama, OLLAMA_BASE_URL) via docker-compose.yml

    selected = [a.strip() for a in args.analysts.split(",") if a.strip()]

    ta = TradingAgentsGraphDebug(selected, config=config, out_dir=out_dir)

    if args.from_step is not None:
        n = ta.invalidate_from(args.from_step)
        print(f"Invalidated {n} artifact(s) from op #{args.from_step} onward")

    state, decision = ta.run_sequential(args.ticker, args.date, asset_type=args.asset_type)

    print("\n=== FINAL DECISION ===")
    print(decision)
    print(f"\nArtifacts: {out_dir}")
    print(f"Ledger:    {out_dir / 'ledger.jsonl'}")


if __name__ == "__main__":
    main()