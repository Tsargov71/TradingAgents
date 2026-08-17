from typing import Annotated
from datetime import datetime
from langchain_core.tools import tool
import pandas as pd
from stockstats import wrap

# load_ohlcv gestisce giÃ  cache (5 anni di storico) e taglio look-ahead a curr_date
from tradingagents.dataflows.stockstats_utils import load_ohlcv


def _get_stock_stats_bulk(symbol: str, indicator: str, curr_date: str) -> dict:
    """
    Calcola l'indicatore stockstats per tutte le date disponibili fino a curr_date
    e ritorna un vero dict {data_str: valore_str}, senza passare da output formattato
    a stringa (che get_stock_stats_indicators_window produce per uso "display", non
    per essere ri-consumato programmaticamente).
    """
    data = load_ohlcv(symbol, curr_date)
    df = wrap(data)
    df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")

    df[indicator]

    result = {}
    for _, row in df.iterrows():
        value = row[indicator]
        result[row["Date"]] = "N/A" if pd.isna(value) else str(value)
    return result


@tool
def get_ma_crossover(
    symbol: Annotated[str, "ticker symbol of the company"],
    curr_date: Annotated[str, "The current trading date you are trading on, YYYY-mm-dd"],
    fast_period: Annotated[int, "period (days) of the fast moving average"] = 20,
    slow_period: Annotated[int, "period (days) of the slow moving average"] = 50,
) -> str:
    """
    Analyze moving average crossover (golden cross / death cross) for a given ticker,
    using SMA values computed via stockstats (same engine as get_indicators).

    Args:
        symbol (str): Ticker symbol of the company, e.g. AAPL, TSM
        curr_date (str): The current trading date you are trading on, YYYY-mm-dd
        fast_period (int): Period in days for the fast SMA, default 20
        slow_period (int): Period in days for the slow SMA, default 50
            (50/200 gives the "textbook" golden/death cross; 20/50 is more responsive
            for swing-trading style signals)

    Returns:
        str: Current trend, most recent crossover event and its recency, and the
             current distance between the two moving averages.
    """
    if fast_period >= slow_period:
        return f"Error: fast_period ({fast_period}) must be smaller than slow_period ({slow_period})."

    fast_key = f"close_{fast_period}_sma"
    slow_key = f"close_{slow_period}_sma"

    try:
        fast_dict = _get_stock_stats_bulk(symbol, fast_key, curr_date)
        slow_dict = _get_stock_stats_bulk(symbol, slow_key, curr_date)
    except Exception as e:
        return f"Error computing MA crossover for {symbol}: {e}"

    common_dates = sorted(
        d for d in (set(fast_dict) & set(slow_dict))
        if fast_dict[d] not in ("N/A", None) and slow_dict[d] not in ("N/A", None)
    )

    if len(common_dates) < 2:
        return (
            f"Not enough history to compute a {fast_period}/{slow_period} crossover "
            f"for {symbol} as of {curr_date}."
        )

    signal = {d: (1 if float(fast_dict[d]) > float(slow_dict[d]) else -1) for d in common_dates}

    last_date = common_dates[-1]
    last_fast, last_slow = float(fast_dict[last_date]), float(slow_dict[last_date])
    trend = "bullish" if signal[last_date] == 1 else "bearish"
    distance_pct = (last_fast - last_slow) / last_slow * 100

    cross_summary = "No crossover detected in the available history"
    for i in range(len(common_dates) - 1, 0, -1):
        d_curr, d_prev = common_dates[i], common_dates[i - 1]
        if signal[d_curr] != signal[d_prev]:
            cross_type = "Golden cross" if signal[d_curr] == 1 else "Death cross"
            days_since = len(common_dates) - 1 - i
            recency = "today" if days_since == 0 else f"{days_since} trading days ago"
            cross_summary = f"{cross_type} on {d_curr} ({recency})"
            break

    header = f"# MA Crossover Analysis for {symbol.upper()} (fast={fast_period}, slow={slow_period})\n"
    header += f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

    return (
        header
        + f"Current trend: {trend} â {fast_key} is {'above' if distance_pct >= 0 else 'below'} "
        f"{slow_key} by {abs(distance_pct):.2f}%\n"
        + f"Last crossover: {cross_summary}\n"
        + f"{fast_key}: {last_fast:.2f} | {slow_key}: {last_slow:.2f} (as of {last_date})"
    )