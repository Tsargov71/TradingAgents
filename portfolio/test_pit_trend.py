"""Offline tests: price_trend_entry.py must not see prices after the threshold date T.

Run from the pipeline folder:  python -m pytest test_pit_trend.py -q
"""
import numpy as np
import pandas as pd

import price_trend_entry as pte

T = "2026-08-17"


def _prices(end, n, start_price=100.0, drift=0.001):
    idx = pd.bdate_range(end=end, periods=n)
    return pd.DataFrame({"Close": start_price * np.exp(drift * np.arange(n))}, index=idx)


def _run(tmp_path, tickers, as_of=T):
    out = tmp_path / "trend.csv"
    df = pte.compute_trend_for_all(tickers, price_cache_dir=str(tmp_path / "cache"), output_path=str(out),
                                   blacklist_path=str(tmp_path / "bl.txt"), as_of=as_of, sleep_between=0)
    return df.set_index("ticker")


def test_download_stops_at_T_and_ignores_later_prices(tmp_path, monkeypatch):
    seen = {}

    def fake_download(ticker, **kw):
        seen.update(kw)
        full = _prices("2026-10-20", 600)                     # data up to "today"
        return full[full.index < pd.Timestamp(kw["end"])]     # yfinance end is exclusive

    monkeypatch.setattr(pte.yf, "download", fake_download)
    res = _run(tmp_path, ["AAA"])
    assert seen["end"] == "2026-08-18" and "period" not in seen
    expected = _prices("2026-10-20", 600)
    expected = expected[expected.index <= T]["Close"].iloc[-1]
    assert abs(res.loc["AAA", "current_price"] - expected) < 1e-9


def test_cache_written_without_a_cutoff_is_cut_at_T(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    full = _prices("2026-10-20", 600)
    full.loc[full.index > T, "Close"] *= 50                   # crazy post-T prices: must not leak
    full.to_parquet(cache / "AAA.parquet")
    res = _run(tmp_path, ["AAA"])
    expected = full[full.index <= T]["Close"].iloc[-1]
    assert abs(res.loc["AAA", "current_price"] - expected) < 1e-9
    assert res.loc["AAA", "price_history_days"] == (full.index <= T).sum()


def test_ticker_without_prices_near_T_is_dropped(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    _prices("2026-03-31", 400).to_parquet(cache / "OLD.parquet")  # stopped trading months before T
    res = _run(tmp_path, ["OLD"])
    assert pd.isna(res.loc["OLD", "price_monotonicity"])          # NaN -> excluded by combine_and_rank
