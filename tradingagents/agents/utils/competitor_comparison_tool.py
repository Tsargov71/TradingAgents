from langchain_core.tools import tool
from typing import Annotated
from datetime import datetime
from tradingagents.dataflows.interface import route_to_vendor

COMPARISON_FIELDS = [
    ("market_cap", "Market Cap"),
    ("pe_ratio", "PE Ratio (TTM)"),
    ("forward_pe", "Forward PE"),
    ("peg_ratio", "PEG Ratio"),
    ("price_to_book", "Price to Book"),
    ("revenue_ttm", "Revenue (TTM)"),
    ("gross_profit", "Gross Profit"),
    ("profit_margin", "Profit Margin"),
    ("operating_margin", "Operating Margin"),
    ("return_on_equity", "Return on Equity"),
    ("return_on_assets", "Return on Assets"),
    ("debt_to_equity", "Debt to Equity"),
    ("current_ratio", "Current Ratio"),
    ("free_cash_flow", "Free Cash Flow"),
]


def _fmt(value) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, (int, float)):
        if abs(value) >= 1000:
            return f"{value:,.0f}"
        return f"{value:.2f}"
    return str(value)


def _is_valid_fundamental_data(raw_data: dict) -> bool:
    """
    Verifica se il dizionario dei dati fondamentali contiene almeno un dato utile.
    Evita il problema dei ticker delistati/acquisiti (es. DFS) che restituiscono
    dizionari popolati interamente da valori None.
    """
    if not raw_data or not isinstance(raw_data, dict):
        return False

    # Controlla se almeno uno dei campi di confronto ha un valore non None
    return any(raw_data.get(key) is not None for key, _ in COMPARISON_FIELDS)


@tool
def get_competitor_comparison(
    ticker: Annotated[str, "ticker symbol"],
    competitor_tickers: Annotated[
        list[str],
        "ticker symbols of 2-4 direct, publicly traded competitors identified for the company",
    ],
    curr_date: Annotated[str, "current date you are trading at, yyyy-mm-dd"],
) -> str:
    """
    Compare key fundamental metrics (valuation, margins, profitability, leverage)
    of a company against a list of competitor tickers, using the configured
    fundamental_data vendor.

    The caller is expected to identify plausible direct competitors from its
    own knowledge and pass their tickers here for data-backed comparison —
    this tool does not discover competitors on its own.

    Note: field coverage depends on the configured vendor. Alpha Vantage does
    not expose Debt to Equity, Current Ratio or Free Cash Flow — those cells
    show N/A when Alpha Vantage is the active vendor.

    Args:
        ticker (str): Ticker symbol of the company under analysis
        competitor_tickers (list[str]): Tickers of comparable companies, e.g. ["RGTI", "QBTS"]
        curr_date (str): Current date you are trading at, yyyy-mm-dd
    Returns:
        str: A markdown comparison table of the company vs. each resolved
             competitor, plus a note listing any competitor tickers that
             failed to resolve so the caller can drop them from the analysis.
    """
    if not competitor_tickers:
        return f"Error: no competitor tickers provided for {ticker}."

    target_raw = route_to_vendor("get_fundamentals_raw", ticker, curr_date)
    if not _is_valid_fundamental_data(target_raw):
        return f"Error: no valid fundamentals data found for target {ticker}."

    competitor_data = {}
    failed = []
    for comp_ticker in competitor_tickers:
        try:
            comp_raw = route_to_vendor("get_fundamentals_raw", comp_ticker, curr_date)
        except Exception as e:
            failed.append(f"{comp_ticker} ({e})")
            continue

        if not _is_valid_fundamental_data(comp_raw):
            failed.append(f"{comp_ticker} (no valid fundamental data found)")
            continue

        competitor_data[comp_ticker] = comp_raw

    if not competitor_data:
        return (
            f"Error: could not resolve any of the provided competitor tickers "
            f"for {ticker}: {', '.join(failed)}"
        )

    header = f"# Competitor Comparison for {ticker.upper()}\n"
    header += f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

    col_names = [ticker.upper()] + [c.upper() for c in competitor_data]
    rows = [" | ".join(["Metric"] + col_names)]
    rows.append(" | ".join(["---"] * (len(col_names) + 1)))

    for key, label in COMPARISON_FIELDS:
        row = [label, _fmt(target_raw.get(key))]
        for comp_raw in competitor_data.values():
            row.append(_fmt(comp_raw.get(key)))
        rows.append(" | ".join(row))

    body = "\n".join(rows)

    footer = ""
    if failed:
        footer = (
            f"\n\nCould not resolve the following competitor tickers "
            f"(excluded from the comparison): {', '.join(failed)}"
        )

    return header + body + footer