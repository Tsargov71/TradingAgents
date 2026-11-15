from typing import Annotated
from datetime import datetime
from langchain_core.tools import tool

from tradingagents.dataflows.date_window import is_historical
from tradingagents.dataflows.router import route_to_vendor
from tradingagents.dataflows.vendors import sec_edgar as edgar
from tradingagents.dataflows.vendors.yahoo.ohlcv import load_ohlcv

# Fields for today's date (snapshot from the configured vendor)
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

# Fields for past dates (only what was public on curr_date, from SEC EDGAR)
PIT_FIELDS = [
    ("sic", "Industry (SIC)"),
    ("fy_end", "Last 10-K fiscal year end"),
    ("market_cap", "Market Cap (shares x close at date)"),
    ("pe_ratio", "P/E (close / last FY diluted EPS)"),
    ("price_to_book", "Price to Book (last FY equity)"),
    ("revenue", "Revenue (last FY, USD M)"),
    ("revenue_growth", "Revenue growth YoY"),
    ("gross_margin", "Gross Margin"),
    ("operating_margin", "Operating Margin"),
    ("net_margin", "Net Margin"),
    ("return_on_equity", "Return on Equity (last FY)"),
    ("return_on_assets", "Return on Assets (last FY)"),
    ("liabilities_to_equity", "Total Liabilities / Equity"),
    ("current_ratio", "Current Ratio (latest filing)"),
    ("free_cash_flow", "Free Cash Flow (OCF - capex, last FY, USD M)"),
]

SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"


PERCENT_KEYS = {"revenue_growth", "gross_margin", "operating_margin", "net_margin",
                "return_on_equity", "return_on_assets"}


def _fmt_pct(value) -> str:
    return "N/A" if value is None else f"{value * 100:.1f}%"


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
    return any(raw_data.get(key) is not None for key, _ in COMPARISON_FIELDS)


def _annual(us_gaap: dict, tags: tuple, curr_date: str) -> dict:
    """{fiscal year end: value} from 10-Ks filed on or before curr_date (no future data)."""
    values, _ = edgar._as_of(us_gaap, tags, curr_date, (edgar._SPANS["annual"],), edgar._ANNUAL_FORMS)
    return {end: val for (end, _), val in values.items()}


def _ratio(num, den):
    return None if num is None or not den else num / den


def _close_at(ticker: str, curr_date: str):
    # load_ohlcv already cuts the data at curr_date
    try:
        data = load_ohlcv(ticker, curr_date)
        return None if data is None or data.empty else float(data["Close"].iloc[-1])
    except Exception:
        return None


def _shares_at(facts: dict, curr_date: str):
    """Shares outstanding from the latest cover page filed on or before curr_date."""
    dei = (facts.get("facts") or {}).get("dei") or {}
    entries = ((dei.get("EntityCommonStockSharesOutstanding") or {}).get("units") or {}).get("shares", [])
    known = [e for e in entries if e.get("filed", "9999") <= curr_date]
    return max(known, key=lambda e: e["filed"])["val"] if known else None


def _get_pit_metrics(ticker: str, curr_date: str):
    """
    Metrics of a ticker as known on curr_date, from SEC EDGAR.
    Returns (metrics, None) or (None, reason for the failure).
    """
    try:
        cik = edgar.cik_for(ticker)
        if cik is None:
            return None, "not a US SEC filer"
        facts = edgar._cached_json(edgar._FACTS_URL.format(cik=cik), f"CIK{cik}.json")
    except Exception as e:
        return None, f"EDGAR unavailable: {e}"

    us_gaap = (facts.get("facts") or {}).get("us-gaap")
    if not us_gaap:
        return None, "no us-gaap facts"

    revenue = _annual(us_gaap, ("RevenueFromContractWithCustomerExcludingAssessedTax", "Revenues", "SalesRevenueNet"), curr_date)
    net_income = _annual(us_gaap, ("NetIncomeLoss",), curr_date)
    if not (revenue or net_income):
        return None, f"no 10-K filed by {curr_date}"
    fy_end = max(revenue or net_income)  # latest fiscal year already filed

    def fy(*tags):
        return _annual(us_gaap, tags, curr_date).get(fy_end)

    rev, ni = revenue.get(fy_end), net_income.get(fy_end)
    gross = fy("GrossProfit")
    cost = fy("CostOfRevenue", "CostOfGoodsAndServicesSold", "CostOfGoodsSold")
    if gross is None and rev is not None and cost is not None:
        gross = rev - cost
    equity = fy("StockholdersEquity", "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest")
    ocf = fy("NetCashProvidedByUsedInOperatingActivities", "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations")
    capex = fy("PaymentsToAcquirePropertyPlantAndEquipment", "PaymentsToAcquireProductiveAssets")
    eps = fy("EarningsPerShareDiluted")
    assets = fy("Assets")
    liabilities = fy("Liabilities")
    if liabilities is None:
        # Many filers never tag total liabilities: use assets minus total equity
        total_equity = fy("StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest") or equity
        liabilities = assets - total_equity if assets is not None and total_equity is not None else None

    # Revenue growth vs the fiscal year about one year earlier
    prior = [v for e, v in revenue.items()
             if 330 <= (datetime.fromisoformat(fy_end) - datetime.fromisoformat(e)).days <= 400]
    growth = rev / prior[0] - 1 if rev is not None and prior and prior[0] else None

    # Current ratio from the latest balance sheet of any form (10-Qs included)
    def latest(tag):
        vals, _ = edgar._as_of(us_gaap, (tag,), curr_date, (edgar._SPANS["annual"],), ())
        return max(vals.items(), key=lambda kv: kv[0][0]) if vals else None
    cur_a, cur_l = latest("AssetsCurrent"), latest("LiabilitiesCurrent")
    current_ratio = _ratio(cur_a[1], cur_l[1]) if cur_a and cur_l and cur_a[0] == cur_l[0] else None

    shares, price = _shares_at(facts, curr_date), _close_at(ticker, curr_date)
    market_cap = shares * price if shares and price else None
    good_equity = equity if equity and equity > 0 else None

    sic = None
    try:
        sub = edgar._cached_json(SUBMISSIONS_URL.format(cik=cik), f"submissions-{cik}.json")
        sic = f"{sub['sic']} {sub.get('sicDescription', '')}".strip() if sub.get("sic") else None
    except Exception:
        pass

    return {
        "sic": sic,
        "fy_end": fy_end,
        "market_cap": market_cap,
        "pe_ratio": _ratio(price, eps) if eps and eps > 0 else None,
        "price_to_book": _ratio(market_cap, good_equity),
        "revenue": None if rev is None else rev / 1e6,
        "revenue_growth": growth,
        "gross_margin": _ratio(gross, rev),
        "operating_margin": _ratio(fy("OperatingIncomeLoss"), rev),
        "net_margin": _ratio(ni, rev),
        "return_on_equity": _ratio(ni, good_equity),
        "return_on_assets": _ratio(ni, assets),
        "liabilities_to_equity": _ratio(liabilities, good_equity),
        "current_ratio": current_ratio,
        "free_cash_flow": None if ocf is None else (ocf - (capex or 0.0)) / 1e6,
    }, None


def _historical_comparison(ticker: str, competitor_tickers: list, curr_date: str) -> str:
    target, reason = _get_pit_metrics(ticker, curr_date)
    if target is None:
        return f"Error: no point-in-time fundamentals found for target {ticker} as of {curr_date} ({reason})."

    competitor_data, failed, warnings = {}, [], []
    for comp_ticker in competitor_tickers[:4]:
        comp = comp_ticker.strip().upper()
        if comp == ticker.upper() or comp in competitor_data:
            continue
        metrics, reason = _get_pit_metrics(comp, curr_date)
        if metrics is None:
            failed.append(f"{comp} ({reason})")
            continue
        competitor_data[comp] = metrics
        # Same SIC group (first 2 digits) = compatible industry; otherwise warn
        if target["sic"] and metrics["sic"] and target["sic"][:2] != metrics["sic"][:2]:
            warnings.append(
                f"{comp} is classified as SIC {metrics['sic']}, a different industry group from "
                f"{ticker.upper()} (SIC {target['sic']}): check it is really a competitor"
            )

    if not competitor_data:
        return (
            f"Error: could not verify any of the provided competitor tickers "
            f"for {ticker} as of {curr_date}: {', '.join(failed)}"
        )

    header = f"# Competitor Comparison for {ticker.upper()}\n"
    header += f"# Point-in-time as of: {curr_date} (SEC EDGAR filings public on that date, close price of that date)\n"
    header += "# Competitors were proposed by the analyst; this tool only verifies them (SEC filer, SIC code)\n\n"

    col_names = [ticker.upper()] + list(competitor_data)
    rows = [" | ".join(["Metric"] + col_names)]
    rows.append(" | ".join(["---"] * (len(col_names) + 1)))
    for key, label in PIT_FIELDS:
        fmt = _fmt_pct if key in PERCENT_KEYS else _fmt
        rows.append(" | ".join([label, fmt(target.get(key))] + [fmt(m.get(key)) for m in competitor_data.values()]))

    footer = ""
    if failed:
        footer += f"\n\nCould not verify the following competitor tickers (excluded from the comparison): {', '.join(failed)}"
    if warnings:
        footer += "\n\nWarnings:\n" + "\n".join(f"- {w}" for w in warnings)

    return header + "\n".join(rows) + footer


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
    of a company against a list of competitor tickers.

    The caller is expected to identify plausible direct competitors from its
    own knowledge and pass their tickers here for data-backed comparison —
    this tool does not discover competitors on its own and does not search the web.
    It does verify them: each competitor must be an SEC filer with data public by
    curr_date, and its SIC industry code is compared with the target's (a
    mismatch is reported as a warning).

    For a past curr_date every figure comes from SEC EDGAR filings public on that
    date plus the closing price of that date, never from today's data. For today's
    date the configured fundamental_data vendor is used.

    Args:
        ticker (str): Ticker symbol of the company under analysis
        competitor_tickers (list[str]): Tickers of comparable companies, e.g. ["SWK", "DORM"]
        curr_date (str): Current date you are trading at, yyyy-mm-dd
    Returns:
        str: A markdown comparison table of the company vs. each verified
             competitor, plus a note listing any competitor tickers that
             failed verification so the caller can drop them from the analysis.
    """
    if not competitor_tickers:
        return f"Error: no competitor tickers provided for {ticker}."

    if is_historical(curr_date):
        return _historical_comparison(ticker, competitor_tickers, curr_date)

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
