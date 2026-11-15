"""Point-in-time version of update_financials.py.

For a threshold date T (--as_of) it writes one pickle per ticker in the same format as
update_financials.py, so magic_formula.py can read the folder unchanged. Only data public on T is used:
10-K filings filed on or before T (SEC bulk files), shares from the latest cover page filed on or before T
times the last close on or before T. Finance and real-estate filers (SIC 6000-6799) are skipped.

Run with --download the first time and set SEC_EDGAR_USER_AGENT="Your Name your.email@domain".

Limits to state in the report: the ticker list is today's, so companies delisted since are missing;
Yahoo closes are split-adjusted up to today while the share counts are not; debt and cash come from the
last 10-K; only USD-reporting 10-K filers are kept; a company with cash but no debt tag gets debt = 0.
"""
import argparse
import json
import os
import time
import zipfile
from datetime import date

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from tqdm import tqdm

from magic_utils import get_all_companies
from update_financials import _atomic_pickle_dump

COMPANYFACTS_URL = "https://www.sec.gov/Archives/edgar/daily-index/xbrl/companyfacts.zip"
SUBMISSIONS_URL = "https://www.sec.gov/Archives/edgar/daily-index/bulkdata/submissions.zip"
EXCHANGES = {"Nasdaq", "NYSE", "CBOE"}

# XBRL tags, in order of preference (the first tag that covers a period wins)
EBIT_TAGS = ("OperatingIncomeLoss",)
PRETAX_TAGS = (
    "IncomeLossFromContinuingOperationsBeforeIncomeTaxesExtraordinaryItemsNoncontrollingInterest",
    "IncomeLossFromContinuingOperationsBeforeIncomeTaxesMinorityInterestAndIncomeLossFromEquityMethodInvestments",
)
INTEREST_TAGS = ("InterestExpense", "InterestExpenseDebt")
NET_INCOME_TAGS = ("NetIncomeLoss",)
OCF_TAGS = ("NetCashProvidedByUsedInOperatingActivities",
            "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations")
CAPEX_TAGS = ("PaymentsToAcquirePropertyPlantAndEquipment", "PaymentsToAcquireProductiveAssets")
ASSETS_TAGS = ("Assets",)
CURRENT_ASSETS_TAGS = ("AssetsCurrent",)
CURRENT_LIABILITIES_TAGS = ("LiabilitiesCurrent",)
PPE_TAGS = ("PropertyPlantAndEquipmentNet",
            "PropertyPlantAndEquipmentAndFinanceLeaseRightOfUseAssetAfterAccumulatedDepreciationAndAmortization")
CASH_TAGS = ("CashAndCashEquivalentsAtCarryingValue",)
DEBT_NONCURRENT_TAGS = ("LongTermDebtNoncurrent",)
DEBT_CURRENT_TAGS = ("LongTermDebtCurrent", "DebtCurrent")
DEBT_TOTAL_TAGS = ("LongTermDebt",)


def _series(us_gaap, tags, as_of, flow):
    """{period end: value}: USD facts from 10-Ks filed on or before as_of.

    flow=True keeps annual-duration facts (income/cash-flow items), flow=False
    keeps instant facts (balance sheet). Per period the latest filing <= as_of
    wins, so a restatement filed after as_of is ignored."""
    out = {}
    for tag in tags:
        best = {}
        for f in (us_gaap.get(tag) or {}).get("units", {}).get("USD", []):
            if f["filed"] > as_of or not f.get("form", "").startswith("10-K"):
                continue
            if flow:
                if "start" not in f:
                    continue
                days = (date.fromisoformat(f["end"]) - date.fromisoformat(f["start"])).days
                if not 300 <= days <= 400:
                    continue
            elif "start" in f:
                continue
            if f["end"] in out:  # an earlier (preferred) tag already covers this period
                continue
            cur = best.get(f["end"])
            if cur is None or f["filed"] >= cur["filed"]:
                best[f["end"]] = f
        out.update({end: f["val"] for end, f in best.items()})
    return out


def _shares_at(facts, as_of):
    """Shares outstanding from the latest cover page filed on or before as_of (all share classes summed)."""
    entries = (((facts.get("facts") or {}).get("dei") or {}).get("EntityCommonStockSharesOutstanding") or {}) \
        .get("units", {}).get("shares", [])
    known = [e for e in entries if e["filed"] <= as_of]
    if not known:
        return None
    last = max(known, key=lambda e: e["filed"])
    return float(sum(e["val"] for e in known if e.get("accn") == last.get("accn") and e["filed"] == last["filed"]))


def _frame(rows, ends):
    """DataFrame with one row per label and one column per fiscal year end (most recent first)."""
    return pd.DataFrame({pd.Timestamp(e): {lab: vals.get(e, np.nan) for lab, vals in rows.items() if vals}
                         for e in ends})


def build_record(facts, sub, as_of, n_years=5):
    """Pickle-compatible dict for one company as known on as_of, or (None, reason)."""
    us = (facts.get("facts") or {}).get("us-gaap")
    if not us:
        return None, "no us-gaap facts"

    ebit = _series(us, EBIT_TAGS, as_of, flow=True)
    interest = _series(us, INTEREST_TAGS, as_of, flow=True)
    for end, pretax in _series(us, PRETAX_TAGS, as_of, flow=True).items():
        if end not in ebit:  # no operating income line: EBIT = pre-tax income + interest expense
            ebit[end] = pretax + interest.get(end, 0.0)
    ends = sorted(ebit, reverse=True)[:n_years]
    if not ends:
        return None, f"no 10-K with EBIT filed by {as_of}"

    nc = _series(us, DEBT_NONCURRENT_TAGS, as_of, flow=False)
    cur = _series(us, DEBT_CURRENT_TAGS, as_of, flow=False)
    tot = _series(us, DEBT_TOTAL_TAGS, as_of, flow=False)
    debt = {e: nc[e] + cur.get(e, 0.0) if e in nc else tot[e] for e in set(nc) | set(tot)}
    cash = _series(us, CASH_TAGS, as_of, flow=False)
    debt_assumed_zero = not debt and bool(cash)
    if debt_assumed_zero:
        debt = {e: 0.0 for e in cash}

    capex = {e: -abs(v) for e, v in _series(us, CAPEX_TAGS, as_of, flow=True).items()}  # negative, as in yfinance
    fin = _frame({"Operating Income": {e: ebit[e] for e in ends},
                  "Net Income": _series(us, NET_INCOME_TAGS, as_of, flow=True)}, ends)
    bs = _frame({"Total Assets": _series(us, ASSETS_TAGS, as_of, flow=False),
                 "Total Current Assets": _series(us, CURRENT_ASSETS_TAGS, as_of, flow=False),
                 "Total Current Liabilities": _series(us, CURRENT_LIABILITIES_TAGS, as_of, flow=False),
                 "Net PPE": _series(us, PPE_TAGS, as_of, flow=False),
                 "Total Debt": debt,
                 "Cash And Cash Equivalents": cash}, ends)
    cf = _frame({"Operating Cash Flow": _series(us, OCF_TAGS, as_of, flow=True),
                 "Capital Expenditure": capex}, ends)

    sic = str(sub.get("sic") or "")
    info = {
        "longName": sub.get("name") or facts.get("entityName"),
        "cik": facts.get("cik"),
        "sic": sic,
        "sector": sub.get("sicDescription"),
        "marketCap": None,  # filled in once the price at as_of is known
        "enterpriseValue": None,  # magic_formula.py derives it from marketCap + totalDebt - totalCash
        "totalDebt": debt.get(ends[0]),
        "totalCash": cash.get(ends[0]),
        "currency": "USD",
        "financialCurrency": "USD",
        "sharesOutstanding": _shares_at(facts, as_of),
        "as_of": as_of,
        "latest_fiscal_year_end": ends[0],
        "debt_assumed_zero": debt_assumed_zero,
    }
    return {"info": info, "financials": fin, "balance_sheet": bs, "cashflow": cf, "granularity": "yearly"}, None


def _yf_closes(tickers, as_of, chunk=100):
    """{ticker: (last close on or before as_of, its date)} (closes are split-adjusted by Yahoo)."""
    out = {}
    start = (pd.Timestamp(as_of) - pd.Timedelta(days=10)).strftime("%Y-%m-%d")
    end = (pd.Timestamp(as_of) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")  # end is exclusive
    for i in tqdm(range(0, len(tickers), chunk), desc="Prices"):
        part = tickers[i:i + chunk]
        try:
            data = yf.download(part, start=start, end=end, interval="1d", auto_adjust=False,
                               progress=False, group_by="column", threads=True)
            close = data["Close"]
        except Exception:
            continue
        if isinstance(close, pd.Series):
            close = close.to_frame(part[0])
        for t in close.columns:
            s = close[t].dropna()
            if not s.empty:
                out[t] = (float(s.iloc[-1]), s.index[-1].strftime("%Y-%m-%d"))
    return out


def _download(url, path, retries=30):
    """Download url to path, resuming from path + '.part' after a timeout or dropped connection."""
    ua = os.getenv("SEC_EDGAR_USER_AGENT")
    if not ua:
        raise SystemExit('Set SEC_EDGAR_USER_AGENT="Your Name your.email@domain" before using --download')
    if os.path.exists(path):
        print(f"{path} already present, skipping download")
        return
    part = path + ".part"
    for attempt in range(1, retries + 1):
        done = os.path.getsize(part) if os.path.exists(part) else 0
        headers = {"User-Agent": ua, **({"Range": f"bytes={done}-"} if done else {})}
        try:
            with requests.get(url, headers=headers, stream=True, timeout=(30, 120)) as r:
                if r.status_code == 416:  # the .part file already holds the whole file
                    break
                r.raise_for_status()
                resumed = r.status_code == 206  # 200 means the server ignored Range: start over
                total = int(r.headers.get("Content-Length", 0)) + (done if resumed else 0)
                with open(part, "ab" if resumed else "wb") as f, \
                        tqdm(total=total or None, initial=done if resumed else 0, unit="B", unit_scale=True,
                             desc=os.path.basename(path)) as bar:
                    for block in r.iter_content(1 << 16):
                        f.write(block)
                        bar.update(len(block))
            break
        except (requests.exceptions.RequestException, OSError) as e:
            print(f"Download interrupted ({type(e).__name__}), retry {attempt}/{retries} in a moment, resuming...")
            time.sleep(min(5 * attempt, 60))
    else:
        raise SystemExit(f"Download failed after {retries} attempts: rerun the same command, it resumes from {part}")
    if not zipfile.is_zipfile(part):
        raise SystemExit(f"{part} is not a valid zip file: delete it and run the download again")
    os.replace(part, path)


def run(as_of, companyfacts_zip, submissions_zip, out_dir, n_years=5, exclude_sic=(6000, 6799),
        universe=None, price_fn=_yf_closes):
    os.makedirs(out_dir, exist_ok=True)
    if universe is None:
        universe = get_all_companies()
    universe = universe[universe["exchange"].isin(EXCHANGES)].drop_duplicates("cik")  # one ticker per company

    records, skipped = {}, {}
    with zipfile.ZipFile(companyfacts_zip) as zf_facts, zipfile.ZipFile(submissions_zip) as zf_sub:
        for row in tqdm(universe.itertuples(), total=len(universe), desc="Reading EDGAR"):
            name = f"CIK{int(row.cik):010d}.json"
            try:
                facts = json.loads(zf_facts.read(name))
                sub = json.loads(zf_sub.read(name))
            except KeyError:
                skipped[row.ticker] = "not in the bulk files"
                continue
            sic = int(sub["sic"]) if str(sub.get("sic") or "").isdigit() else None
            if sic is not None and exclude_sic[0] <= sic <= exclude_sic[1]:
                skipped[row.ticker] = f"excluded SIC {sic}"
                continue
            rec, why = build_record(facts, sub, as_of, n_years)
            if rec is None:
                skipped[row.ticker] = why
            else:
                records[row.ticker] = rec

    # Prices only for companies magic_formula.py could rank (>= 2 EBIT values, all positive)
    def rankable(rec):
        ebits = rec["financials"].loc["Operating Income"].dropna()
        return len(ebits) >= 2 and (ebits > 0).all() and rec["info"]["sharesOutstanding"]
    candidates = [t for t, r in records.items() if rankable(r)]
    prices = price_fn(candidates, as_of)
    stale = date.fromisoformat(as_of).toordinal() - 7
    for t in candidates:
        if t in prices and date.fromisoformat(prices[t][1]).toordinal() >= stale:
            info = records[t]["info"]
            info["marketCap"] = info["sharesOutstanding"] * prices[t][0]
            info["priceAtAsOf"], info["priceDate"] = prices[t]

    for t, rec in records.items():
        _atomic_pickle_dump(rec, os.path.join(out_dir, f"{t}.pkl"))
    manifest = {"as_of": as_of, "written": len(records), "rankable": len(candidates),
                "with_market_cap": sum(r["info"]["marketCap"] is not None for r in records.values()),
                "skipped": len(skipped)}
    with open(os.path.join(out_dir, "pit_manifest.json"), "w") as f:
        json.dump({**manifest, "skipped_reasons": skipped}, f, indent=1)
    print(manifest)
    return manifest


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--as_of", type=str, default="2026-08-17", help="threshold date T, yyyy-mm-dd")
    ap.add_argument("--out_dir", type=str, default=None, help="default: yf_data_pit_<as_of>")
    ap.add_argument("--companyfacts_zip", type=str, default="companyfacts.zip")
    ap.add_argument("--submissions_zip", type=str, default="submissions.zip")
    ap.add_argument("--n_years", type=int, default=5)
    ap.add_argument("--download", action="store_true", help="download the two SEC bulk files first (~GB)")
    args = ap.parse_args()
    if args.download:
        _download(COMPANYFACTS_URL, args.companyfacts_zip)
        _download(SUBMISSIONS_URL, args.submissions_zip)
    run(args.as_of, args.companyfacts_zip, args.submissions_zip,
        args.out_dir or f"yf_data_pit_{args.as_of}", n_years=args.n_years)
