"""Buy-and-hold benchmarks: an equal-weight basket, or one ticker (SPY as a proxy for the S&P 500)."""

from __future__ import annotations


def buy_and_hold(tickers, start, dates, cash, bars):
    """Value at each date of `cash` split equally across the tickers at start's close (fractional units).

    A ticker with no data at start is skipped and listed in the result.
    """
    start_prices, skipped = {}, []
    for ticker in tickers:
        try:
            start_prices[ticker] = float(bars(ticker, start).iloc[-1]["Close"])
        except Exception:
            skipped.append(ticker)
    units = {t: cash / len(start_prices) / price for t, price in start_prices.items()}
    values = [{"date": d, "value": round(sum(u * float(bars(t, d).iloc[-1]["Close"]) for t, u in units.items()), 2)}
              for d in dates]
    return {"values": values, "skipped": skipped}
