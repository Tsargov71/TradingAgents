from typing import Annotated
from datetime import datetime
from langchain_core.tools import tool
import pandas as pd
from stockstats import wrap

from tradingagents.dataflows.vendors.yahoo.ohlcv import load_ohlcv


def _get_volatility_frame(symbol: str, curr_date: str) -> pd.DataFrame:
    data = load_ohlcv(symbol, curr_date)
    df = wrap(data)

    # If Date is a datetime, format it as a string
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
    # Safe handling of the Close column name (lowercase or uppercase)
    close_key = "close" if "close" in last.index else "Close"
    close, mid, ub, lb, atr = last[close_key], last["boll"], last["boll_ub"], last["boll_lb"], last["atr"]

    # %B: 0 = on the lower band, 1 = on the upper band, 0.5 = on the middle band
    band_range = ub - lb
    percent_b = (close - lb) / band_range if band_range != 0 else 0.5

    if percent_b >= 0.8:
        band_position = "near the upper band (potential overbought zone)"
    elif percent_b <= 0.2:
        band_position = "near the lower band (potential oversold zone)"
    else:
        band_position = "in the middle zone, no extremes"

    # Current band width vs historical average (proxy for squeeze/expansion)
    df = df.copy()
    df["band_width_pct"] = (df["boll_ub"] - df["boll_lb"]) / df["boll"] * 100
    current_width = df["band_width_pct"].iloc[-1]
    avg_width = df["band_width_pct"].tail(atr_lookback).mean()
    width_ratio = current_width / avg_width if avg_width else 1.0

    if width_ratio <= 0.7:
        squeeze_state = f"bands squeezing ({current_width:.2f}% vs average {avg_width:.2f}%), possible prelude to a breakout"
    elif width_ratio >= 1.3:
        squeeze_state = f"bands expanding ({current_width:.2f}% vs average {avg_width:.2f}%), high volatility under way"
    else:
        squeeze_state = f"band width normal ({current_width:.2f}% vs average {avg_width:.2f}%)"

    # Current ATR vs historical average
    avg_atr = df["atr"].tail(atr_lookback).mean()
    atr_ratio = atr / avg_atr if avg_atr else 1.0
    if atr_ratio >= 1.2:
        atr_trend = f"rising ({atr:.2f} vs average {avg_atr:.2f} over the last {atr_lookback} days), volatility increasing"
    elif atr_ratio <= 0.8:
        atr_trend = f"falling ({atr:.2f} vs average {avg_atr:.2f} over the last {atr_lookback} days), volatility contracting"
    else:
        atr_trend = f"stable ({atr:.2f} vs average {avg_atr:.2f} over the last {atr_lookback} days)"

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