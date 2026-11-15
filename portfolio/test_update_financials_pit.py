"""Offline tests for update_financials_pit.py (synthetic SEC bulk files, no network).

Run from the pipeline folder:  python -m pytest test_update_financials_pit.py -q
"""
import json
import os
import pickle
import zipfile

import pandas as pd

import update_financials_pit as pit
from magic_formula import compute_magic_formula_from_yfdir

T = "2026-08-17"


def _flow(end, val, filed, form="10-K"):
    year = int(end[:4])
    return {"start": f"{year}-01-01", "end": end, "val": val, "filed": filed, "form": form, "accn": filed}


def _inst(end, val, filed, form="10-K"):
    return {"end": end, "val": val, "filed": filed, "form": form, "accn": filed}


def _facts(cik, ebits, **extra):
    """ebits: {fiscal year: (value, filing date)} (several entries per year = restatements)."""
    def usd(entries):
        return {"units": {"USD": entries}}
    us = {
        "OperatingIncomeLoss": usd([_flow(f"{y}-12-31", v, f) for y, v, f in ebits]),
        "NetIncomeLoss": usd([_flow(f"{y}-12-31", v * 0.7, f) for y, v, f in ebits]),
        "AssetsCurrent": usd([_inst(f"{y}-12-31", 500.0, f) for y, _, f in ebits]),
        "LiabilitiesCurrent": usd([_inst(f"{y}-12-31", 200.0, f) for y, _, f in ebits]),
        "PropertyPlantAndEquipmentNet": usd([_inst(f"{y}-12-31", 300.0, f) for y, _, f in ebits]),
        "Assets": usd([_inst(f"{y}-12-31", 1000.0, f) for y, _, f in ebits]),
        "CashAndCashEquivalentsAtCarryingValue": usd([_inst(f"{y}-12-31", 100.0, f) for y, _, f in ebits]),
        "LongTermDebtNoncurrent": usd([_inst(f"{y}-12-31", 150.0, f) for y, _, f in ebits]),
        "LongTermDebtCurrent": usd([_inst(f"{y}-12-31", 50.0, f) for y, _, f in ebits]),
        "NetCashProvidedByUsedInOperatingActivities": usd([_flow(f"{y}-12-31", 120.0, f) for y, _, f in ebits]),
        "PaymentsToAcquirePropertyPlantAndEquipment": usd([_flow(f"{y}-12-31", 30.0, f) for y, _, f in ebits]),
    }
    dei = {"EntityCommonStockSharesOutstanding": {"units": {"shares": [
        {"end": "2026-02-10", "val": 100.0, "filed": "2026-02-20", "accn": "a1", "form": "10-K"},
        {"end": "2026-08-01", "val": 110.0, "filed": "2026-08-05", "accn": "a2", "form": "10-Q"},
        {"end": "2026-10-20", "val": 999.0, "filed": "2026-10-25", "accn": "a3", "form": "10-Q"},  # after T
    ]}}}
    return {"cik": cik, "entityName": f"Company {cik}", "facts": {"us-gaap": us, "dei": dei}}


def _make_zips(tmp_path):
    aaa = _facts(1, [(2023, 100.0, "2024-02-20"), (2024, 120.0, "2025-02-20"), (2025, 140.0, "2026-02-20"),
                     (2025, 1.0, "2026-11-30"),          # restatement filed AFTER T: must be ignored
                     (2026, 5000.0, "2027-02-20")])      # fiscal year 2026 filed AFTER T: must not exist
    bnk = _facts(2, [(2025, 50.0, "2026-02-20")])
    late = _facts(3, [(2026, 70.0, "2027-02-20")])        # first 10-K only after T
    sub = {1: {"sic": "3420", "sicDescription": "Cutlery", "name": "AAA Inc"},
           2: {"sic": "6022", "sicDescription": "State commercial banks", "name": "Bank"},
           3: {"sic": "3420", "sicDescription": "Cutlery", "name": "Late"}}
    cf, sb = tmp_path / "cf.zip", tmp_path / "sub.zip"
    with zipfile.ZipFile(cf, "w") as z1, zipfile.ZipFile(sb, "w") as z2:
        for cik, facts in {1: aaa, 2: bnk, 3: late}.items():
            z1.writestr(f"CIK{cik:010d}.json", json.dumps(facts))
            z2.writestr(f"CIK{cik:010d}.json", json.dumps(sub[cik]))
    universe = pd.DataFrame({"cik": [1, 2, 3, 4], "ticker": ["AAA", "BNK", "LATE", "OTCX"],
                             "exchange": ["Nasdaq", "NYSE", "Nasdaq", "OTC"]})
    return str(cf), str(sb), universe


def _run(tmp_path):
    cf, sb, universe = _make_zips(tmp_path)
    out = tmp_path / "out"
    manifest = pit.run(T, cf, sb, str(out), universe=universe,
                       price_fn=lambda tickers, as_of: {"AAA": (20.0, "2026-08-14")})
    return out, manifest


def test_only_filings_public_at_T_are_used_and_banks_are_skipped(tmp_path):
    out, manifest = _run(tmp_path)
    assert sorted(f for f in os.listdir(out) if f.endswith(".pkl")) == ["AAA.pkl"]
    reasons = json.load(open(out / "pit_manifest.json"))["skipped_reasons"]
    assert "SIC 6022" in reasons["BNK"] and "no 10-K" in reasons["LATE"]
    assert "OTCX" not in reasons                                  # OTC tickers never enter the universe

    rec = pickle.load(open(out / "AAA.pkl", "rb"))
    fin = rec["financials"]
    assert [c.year for c in fin.columns] == [2025, 2024, 2023]    # most recent year on the LEFT
    assert fin.loc["Operating Income"].tolist() == [140.0, 120.0, 100.0]   # restatement filed after T ignored
    assert rec["balance_sheet"].loc["Total Debt"].iloc[0] == 200.0         # 150 non-current + 50 current
    assert rec["cashflow"].loc["Capital Expenditure"].iloc[0] == -30.0     # negative, as in yfinance


def test_market_cap_uses_shares_and_price_known_at_T(tmp_path):
    out, manifest = _run(tmp_path)
    info = pickle.load(open(out / "AAA.pkl", "rb"))["info"]
    assert info["sharesOutstanding"] == 110.0                     # 10-Q cover of 2026-08-05, not the one of 2026-10-25
    assert info["marketCap"] == 110.0 * 20.0
    assert info["as_of"] == T and manifest["with_market_cap"] == 1


def test_price_older_than_a_week_is_discarded(tmp_path):
    cf, sb, universe = _make_zips(tmp_path)
    pit.run(T, cf, sb, str(tmp_path / "o"), universe=universe,
            price_fn=lambda tickers, as_of: {"AAA": (20.0, "2026-07-01")})
    assert pickle.load(open(tmp_path / "o" / "AAA.pkl", "rb"))["info"]["marketCap"] is None


def test_output_is_readable_by_magic_formula(tmp_path):
    out, _ = _run(tmp_path)
    df = compute_magic_formula_from_yfdir(str(out))
    row = df[df["ticker"] == "AAA"].iloc[0]
    assert row["ebits_count"] == 3 and row["MarketCap"] == 2200.0 and row["EV"] == 2300.0   # EV = 2200 + 200 debt - 100 cash


def test_download_resumes_after_a_dropped_connection(tmp_path, monkeypatch):
    """First request dies halfway; the retry must continue from the .part file (HTTP Range)."""
    import http.server
    import io
    import threading

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_STORED) as z:
        z.writestr("CIK0000000001.json", os.urandom(300_000))
    payload, requests_seen = buf.getvalue(), []

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            rng = self.headers.get("Range")
            requests_seen.append(rng)
            start = int(rng.split("=")[1].rstrip("-")) if rng else 0
            body = payload[start:]
            self.send_response(206 if rng else 200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            if not rng:  # first attempt: send only half, then drop the connection
                self.wfile.write(body[:len(body) // 2])
                self.wfile.flush()
                self.connection.close()
            else:
                self.wfile.write(body)

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setenv("SEC_EDGAR_USER_AGENT", "Test test@example.com")
    monkeypatch.setattr(pit.time, "sleep", lambda s: None)
    target = tmp_path / "file.zip"
    try:
        pit._download(f"http://127.0.0.1:{server.server_port}/file.zip", str(target))
    finally:
        server.shutdown()
    assert target.read_bytes() == payload and not os.path.exists(str(target) + ".part")
    assert requests_seen[0] is None and requests_seen[1].startswith("bytes=")
