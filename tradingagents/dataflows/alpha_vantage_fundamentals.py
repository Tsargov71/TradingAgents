import json

from .alpha_vantage_common import _make_api_request

def _to_float(value):
    """Alpha Vantage ritorna spesso 'None' come stringa letterale per i campi
    mancanti, oltre a valori numerici come stringhe. Normalizza entrambi i casi."""
    if value is None or value == "None" or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None

def get_fundamentals_raw(ticker: str, curr_date: str = None) -> dict:
    """
    Get company fundamentals as a raw dict with canonical field names, from
    Alpha Vantage's OVERVIEW endpoint. Same canonical keys as
    y_finance.get_fundamentals_raw, so route_to_vendor callers can consume
    either vendor's output identically.
 
    Returns None if no data is found or the response is empty/malformed.
    """
    try:
        result = _make_api_request("OVERVIEW", {"symbol": ticker})
 
        if not result or not isinstance(result, dict) or not result.get("Symbol"):
            return None
 
        return {
            "name": result.get("Name"),
            "sector": result.get("Sector"),
            "industry": result.get("Industry"),
            "market_cap": _to_float(result.get("MarketCapitalization")),
            "pe_ratio": _to_float(result.get("PERatio")),
            "forward_pe": _to_float(result.get("ForwardPE")),
            "peg_ratio": _to_float(result.get("PEGRatio")),
            "price_to_book": _to_float(result.get("PriceToBookRatio")),
            "eps": _to_float(result.get("EPS")),
            "forward_eps": None,  # non disponibile in OVERVIEW
            "dividend_yield": _to_float(result.get("DividendYield")),
            "beta": _to_float(result.get("Beta")),
            "fifty_two_week_high": _to_float(result.get("52WeekHigh")),
            "fifty_two_week_low": _to_float(result.get("52WeekLow")),
            "fifty_day_avg": _to_float(result.get("50DayMovingAverage")),
            "two_hundred_day_avg": _to_float(result.get("200DayMovingAverage")),
            "revenue_ttm": _to_float(result.get("RevenueTTM")),
            "gross_profit": _to_float(result.get("GrossProfitTTM")),
            "ebitda": _to_float(result.get("EBITDA")),
            "net_income": None,  # non disponibile in OVERVIEW
            "profit_margin": _to_float(result.get("ProfitMargin")),
            "operating_margin": _to_float(result.get("OperatingMarginTTM")),
            "return_on_equity": _to_float(result.get("ReturnOnEquityTTM")),
            "return_on_assets": _to_float(result.get("ReturnOnAssetsTTM")),
            "debt_to_equity": None,  # non disponibile in OVERVIEW
            "current_ratio": None,  # non disponibile in OVERVIEW
            "book_value": _to_float(result.get("BookValue")),
            "free_cash_flow": None,  # non disponibile in OVERVIEW
        }
    except Exception:
        return None

def _filter_reports_by_date(result, curr_date: str):
    """Drop annual/quarterly reports dated after curr_date to prevent look-ahead.

    ``_make_api_request`` returns the fundamentals payload as a JSON string, so
    parse, filter, and re-serialize. A non-JSON body or an unset ``curr_date`` is
    returned unchanged.
    """
    if not curr_date or not isinstance(result, str):
        return result
    try:
        payload = json.loads(result)
    except json.JSONDecodeError:
        return result
    if not isinstance(payload, dict):
        return result
    for key in ("annualReports", "quarterlyReports"):
        if isinstance(payload.get(key), list):
            payload[key] = [
                r for r in payload[key]
                if r.get("fiscalDateEnding", "") <= curr_date
            ]
    return json.dumps(payload)


def get_fundamentals(ticker: str, curr_date: str = None) -> str:
    """
    Retrieve comprehensive fundamental data for a given ticker symbol using Alpha Vantage.

    Args:
        ticker (str): Ticker symbol of the company
        curr_date (str): Current date you are trading at, yyyy-mm-dd (not used for Alpha Vantage)

    Returns:
        str: Company overview data including financial ratios and key metrics
    """
    params = {
        "symbol": ticker,
    }

    return _make_api_request("OVERVIEW", params)


def get_balance_sheet(ticker: str, freq: str = "quarterly", curr_date: str = None):
    """Retrieve balance sheet data for a given ticker symbol using Alpha Vantage."""
    result = _make_api_request("BALANCE_SHEET", {"symbol": ticker})
    return _filter_reports_by_date(result, curr_date)


def get_cashflow(ticker: str, freq: str = "quarterly", curr_date: str = None):
    """Retrieve cash flow statement data for a given ticker symbol using Alpha Vantage."""
    result = _make_api_request("CASH_FLOW", {"symbol": ticker})
    return _filter_reports_by_date(result, curr_date)


def get_income_statement(ticker: str, freq: str = "quarterly", curr_date: str = None):
    """Retrieve income statement data for a given ticker symbol using Alpha Vantage."""
    result = _make_api_request("INCOME_STATEMENT", {"symbol": ticker})
    return _filter_reports_by_date(result, curr_date)

