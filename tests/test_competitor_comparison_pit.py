"""The competitor comparison must not use today's data in historical runs."""
import pandas as pd

from tradingagents.agents import competitor_comparison_tool as cct
from tradingagents.dataflows.vendors import sec_edgar as edgar
from tradingagents.dataflows.vendors.yahoo import fundamentals as yf_fund


def _fact(end, val, filed, form="10-K", start=None):
    f = {"end": end, "val": val, "filed": filed, "form": form}
    if start:
        f["start"] = start
    return f


def _facts(scale=1.0, shares=100e6):
    def dur(tag_vals):
        return {"units": {"USD": [_fact(e, v * scale, fl, start=s) for e, v, fl, s in tag_vals]}}

    def inst(tag_vals):
        return {"units": {"USD": [_fact(e, v * scale, fl) for e, v, fl in tag_vals]}}

    return {"facts": {
        "us-gaap": {
            "Revenues": dur([("2024-12-31", 800e6, "2025-02-20", "2024-01-01"),
                             ("2025-12-31", 1000e6, "2026-02-20", "2025-01-01"),
                             # filed AFTER the analysis date: must not count
                             ("2026-12-31", 5000e6, "2027-02-20", "2026-01-01")]),
            "NetIncomeLoss": dur([("2025-12-31", 100e6, "2026-02-20", "2025-01-01")]),
            "OperatingIncomeLoss": dur([("2025-12-31", 150e6, "2026-02-20", "2025-01-01")]),
            "GrossProfit": dur([("2025-12-31", 400e6, "2026-02-20", "2025-01-01")]),
            "EarningsPerShareDiluted": {"units": {"USD/shares": [
                _fact("2025-12-31", 1.0, "2026-02-20", start="2025-01-01")]}},
            "Assets": inst([("2025-12-31", 1200e6, "2026-02-20")]),
            "Liabilities": inst([("2025-12-31", 600e6, "2026-02-20")]),
            "StockholdersEquity": inst([("2025-12-31", 600e6, "2026-02-20")]),
            "AssetsCurrent": inst([("2026-06-30", 500e6, "2026-08-05")]),
            "LiabilitiesCurrent": inst([("2026-06-30", 250e6, "2026-08-05")]),
            "NetCashProvidedByUsedInOperatingActivities": dur([("2025-12-31", 140e6, "2026-02-20", "2025-01-01")]),
            "PaymentsToAcquirePropertyPlantAndEquipment": dur([("2025-12-31", 40e6, "2026-02-20", "2025-01-01")]),
        },
        "dei": {"EntityCommonStockSharesOutstanding": {"units": {"shares": [
            {"end": "2026-02-10", "val": shares, "filed": "2026-02-20"},
            {"end": "2026-10-10", "val": shares * 10, "filed": "2026-10-15"}]}}},
    }}


def _install(monkeypatch, sics):
    facts = {"AAA": _facts(), "BBB": _facts(0.5), "CCC": _facts(2.0)}
    ciks = {"AAA": "0000000001", "BBB": "0000000002", "CCC": "0000000003"}
    monkeypatch.setattr(edgar, "cik_for", lambda t: ciks.get(t))

    def cached(url, name):
        for t, c in ciks.items():
            if name == f"CIK{c}.json":
                return facts[t]
            if name == f"submissions-{c}.json":
                return {"sic": sics[t][0], "sicDescription": sics[t][1]}
        raise AssertionError(name)

    monkeypatch.setattr(edgar, "_cached_json", cached)
    monkeypatch.setattr(cct, "_close_at", lambda t, d: 20.0)
    monkeypatch.setattr(cct, "is_historical", lambda d: True)


def test_historical_uses_only_filings_public_at_date(monkeypatch):
    _install(monkeypatch, {"AAA": ("3420", "Cutlery"), "BBB": ("3420", "Cutlery"), "CCC": ("3714", "Motor parts")})
    out = cct.get_competitor_comparison.invoke(
        {"ticker": "AAA", "competitor_tickers": ["BBB", "CCC", "ZZZ"], "curr_date": "2026-08-17"})
    assert "2025-12-31" in out and "2026-12-31" not in out
    # revenue 1000M, +25% vs 800M: the 5000M filed in 2027 is excluded
    assert "1,000" in out and "25.0%" in out and "5,000" not in out
    # market cap = shares from the 2026-02 cover (100M) x 20, not the 1000M of 2026-10
    assert "2,000,000,000" in out and "20,000,000,000" not in out
    assert "ZZZ (not a US SEC filer)" in out          # competitor that cannot be verified
    assert "CCC is classified as SIC 3714" in out                   # different industry flagged
    assert "Point-in-time as of: 2026-08-17" in out and "Data retrieved on" not in out


def test_historical_never_touches_live_vendor(monkeypatch):
    _install(monkeypatch, {"AAA": ("3420", "x"), "BBB": ("3420", "x"), "CCC": ("3420", "x")})

    def boom(*a, **k):
        raise AssertionError("live vendor used in a historical run")

    monkeypatch.setattr(cct, "route_to_vendor", boom)
    out = cct.get_competitor_comparison.invoke(
        {"ticker": "AAA", "competitor_tickers": ["BBB"], "curr_date": "2026-08-17"})
    assert "Error" not in out


def test_yahoo_raw_fundamentals_withheld_for_past_dates():
    assert yf_fund.get_fundamentals_raw("AAPL", "2020-01-02") is None


def test_liabilities_fall_back_to_assets_minus_equity(monkeypatch):
    _install(monkeypatch, {"AAA": ("3420", "x"), "BBB": ("3420", "x"), "CCC": ("3420", "x")})
    facts = _facts()
    del facts["facts"]["us-gaap"]["Liabilities"]            # filer does not tag total liabilities
    monkeypatch.setattr(edgar, "_cached_json",
                        lambda url, name: facts if name.startswith("CIK") else {"sic": "3420"})
    metrics, why = cct._get_pit_metrics("AAA", "2026-08-17")
    assert why is None and metrics["liabilities_to_equity"] == 1.0   # (1200 - 600) / 600
