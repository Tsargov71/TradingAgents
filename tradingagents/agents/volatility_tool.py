from typing import Annotated
from datetime import datetime
from langchain_core.tools import tool
import pandas as pd
from stockstats import wrap

from tradingagents.dataflows.stockstats_utils import load_ohlcv


def _get_volatility_frame(symbol: str, curr_date: str) -> pd.DataFrame:
    data = load_ohlcv(symbol, curr_date)
    df = wrap(data)
    
    # Se Date è datetime, formattalo
    if pd.api.types.is_datetime64_any_dtype(df["Date"]):
        df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")
            
    for col in ("boll", "boll_ub", "boll_lb", "atr"):
        df[col] 
        close_col = "close" if "close" in df.columns else "Close"
        return df[["Date", close_col, "boll", "boll_ub", "boll_lb", "atr"]].rename(
            columns={close_col: "close"}
        ).dropna()

@tool
def get_volatility_analysis(
    symbol: Annotated[str, "ticker symbol of the company"],
    curr_date: Annotated[str, "The current trading date you are trading on, YYYY-mm-dd"],
    atr_lookback: Annotated[int, "days used to compute the average ATR for comparison"] = 20,
    atr_stop_multiple: Annotated[float, "multiple of ATR used to suggest a stop-loss distance"] = 2.0,
) -> str:
    """
    Analyze volatility for a given ticker using Bollinger Bands and ATR (same
    stockstats engine as get_indicators), returning an interpreted summary instead
    of raw indicator series: where price sits relative to the bands, whether the
    bands are squeezing or expanding, whether ATR is rising or falling versus its
    recent average, and a suggested ATR-based stop-loss distance.

    Args:
        symbol (str): Ticker symbol of the company, e.g. AAPL, TSM
        curr_date (str): The current trading date you are trading on, YYYY-mm-dd
        atr_lookback (int): Days used to compute the average ATR for comparison, default 20
        atr_stop_multiple (float): Multiple of ATR used to suggest a stop-loss distance, default 2.0

    Returns:
        str: A formatted summary of band position, squeeze/expansion state,
             ATR trend, and a suggested stop-loss level.
    """
    try:
        df = _get_volatility_frame(symbol, curr_date)
    except Exception as e:
        return f"Error computing volatility analysis for {symbol}: {e}"

    if len(df) < atr_lookback + 1:
        return f"Not enough history to compute volatility analysis for {symbol} as of {curr_date}."

    last = df.iloc[-1]    
    # Gestione sicura del nome della colonna Close (minuscola o maiuscola)
    close_key = "close" if "close" in last.index else "Close"    
    close, mid, ub, lb, atr = last[close_key], last["boll"], last["boll_ub"], last["boll_lb"], last["atr"]

    # %B: 0 = sulla banda inferiore, 1 = sulla banda superiore, 0.5 = sulla media
    band_range = ub - lb
    percent_b = (close - lb) / band_range if band_range != 0 else 0.5

    if percent_b >= 0.8:
        band_position = "vicino alla banda superiore (zona di potenziale ipercomprato)"
    elif percent_b <= 0.2:
        band_position = "vicino alla banda inferiore (zona di potenziale ipervenduto)"
    else:
        band_position = "in zona centrale, senza estremi"

    # Ampiezza bande attuale vs media storica (proxy per squeeze/espansione)
    df = df.copy()
    df["band_width_pct"] = (df["boll_ub"] - df["boll_lb"]) / df["boll"] * 100
    current_width = df["band_width_pct"].iloc[-1]
    avg_width = df["band_width_pct"].tail(atr_lookback).mean()
    width_ratio = current_width / avg_width if avg_width else 1.0

    if width_ratio <= 0.7:
        squeeze_state = f"bande in compressione ({current_width:.2f}% vs media {avg_width:.2f}%) è possibile preludio a un breakout"
    elif width_ratio >= 1.3:
        squeeze_state = f"bande in espansione ({current_width:.2f}% vs media {avg_width:.2f}%) volatilità  elevata in corso"
    else:
        squeeze_state = f"ampiezza bande nella norma ({current_width:.2f}% vs media {avg_width:.2f}%)"

    # ATR attuale vs media storica
    avg_atr = df["atr"].tail(atr_lookback).mean()
    atr_ratio = atr / avg_atr if avg_atr else 1.0
    if atr_ratio >= 1.2:
        atr_trend = f"in aumento ({atr:.2f} vs media {avg_atr:.2f} sugli ultimi {atr_lookback} giorni) volatilità  in crescita"
    elif atr_ratio <= 0.8:
        atr_trend = f"in calo ({atr:.2f} vs media {avg_atr:.2f} sugli ultimi {atr_lookback} giorni) volatilità  in contrazione"
    else:
        atr_trend = f"stabile ({atr:.2f} vs media {avg_atr:.2f} sugli ultimi {atr_lookback} giorni)"

    suggested_stop = close - (atr * atr_stop_multiple)

    header = f"# Volatility Analysis for {symbol.upper()}\n"
    header += f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

    return (
        header
        + f"Close: {close:.2f} | Bollinger Mid: {mid:.2f} | Upper: {ub:.2f} | Lower: {lb:.2f}\n"
        + f"Band position (%B={percent_b:.2f}): {band_position}\n"
        + f"Band width: {squeeze_state}\n"
        + f"ATR: {atr_trend}\n"
        + f"Suggested stop-loss (Close - {atr_stop_multiple}x ATR): {suggested_stop:.2f} "
        f"(distance: {atr * atr_stop_multiple:.2f}, {atr_stop_multiple}x current ATR of {atr:.2f})"
    )