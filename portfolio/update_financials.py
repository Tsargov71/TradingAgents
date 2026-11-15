""" Run this code first to download financial data from yahoo finance.
Author:
    Giovanni Pio Delvecchio
"""
import os
import pickle
import time
import yfinance as yf
from tqdm import tqdm
import pandas as pd
from magic_utils import get_all_companies
import argparse


def _is_pkl_valid(file_path):
    """Returns True if file_path exists, is non-empty, and unpickles cleanly."""
    if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
        return False
    try:
        with open(file_path, "rb") as f:
            pickle.load(f)
        return True
    except Exception:
        return False


def _load_pkl_safe(file_path):
    """Loads a pickle, returning None if missing/corrupted instead of raising."""
    if not _is_pkl_valid(file_path):
        return None
    with open(file_path, "rb") as f:
        return pickle.load(f)


def _atomic_pickle_dump(obj, file_path):
    """Writes obj to file_path atomically: dump to a temp file in the same
    directory, then os.replace() it into place. This guarantees file_path
    either has the old complete content or the new complete content, never
    a half-written/corrupt file, even if the process is killed mid-write."""
    tmp_path = file_path + ".tmp"
    with open(tmp_path, "wb") as f:
        pickle.dump(obj, f)
    os.replace(tmp_path, file_path)


def _download_one(ticker_obj, granularity, n_years, old_data):
    """Fetches fresh data for a single ticker and merges it with old_data.
    Raises on failure (caller is responsible for retrying)."""
    if granularity == "yearly":
        fin = ticker_obj.financials
        bs = ticker_obj.balance_sheet
        cf = ticker_obj.cashflow
    else:
        fin = ticker_obj.quarterly_financials
        bs = ticker_obj.quarterly_balance_sheet
        cf = ticker_obj.quarterly_cashflow

    # Mantieni solo gli ultimi n_years
    fin = fin.iloc[:, :n_years] if not fin.empty else fin
    bs = bs.iloc[:, :n_years] if not bs.empty else bs
    cf = cf.iloc[:, :n_years] if not cf.empty else cf

    def merge_frames(old, new):
        if old is None or not isinstance(old, pd.DataFrame):
            return new
        if new is None or new.empty:
            return old
        merged = pd.concat([new, old], axis=1)
        merged = merged.loc[:, ~merged.columns.duplicated()]
        return merged.iloc[:, :n_years]

    if old_data:
        fin = merge_frames(old_data.get("financials"), fin)
        bs = merge_frames(old_data.get("balance_sheet"), bs)
        cf = merge_frames(old_data.get("cashflow"), cf)

    info = ticker_obj.info

    # Se yfinance non solleva ma restituisce dati vuoti/inutilizzabili, trattalo
    # come un fallimento cosi' che venga ritentato invece di scrivere un pickle
    # tecnicamente valido ma privo di contenuto.
    fin_empty = fin is None or (isinstance(fin, pd.DataFrame) and fin.empty)
    bs_empty = bs is None or (isinstance(bs, pd.DataFrame) and bs.empty)
    info_empty = not info or not isinstance(info, dict)
    if fin_empty and bs_empty and info_empty:
        raise ValueError("empty response from yfinance (financials, balance_sheet and info all empty)")

    return {
        "info": info,
        "financials": fin,
        "balance_sheet": bs,
        "cashflow": cf,
        "granularity": granularity
    }


def update_yf_data(tickers, n_years=5, granularity="yearly", save_dir="yf_data",
                    n_retry=5, retry_backoff=2.0, corrupted_log="corrupted.txt"):
    """ Updates financial reports downloaded from yahoo finance.

    Any existing .pkl that fails to unpickle (empty/truncated/corrupt) is
    treated as missing and re-downloaded. Each ticker gets up to n_retry
    attempts; if all attempts fail, the ticker is appended to corrupted_log
    and skipped. Writes are atomic, so an interrupted run can never leave a
    half-written .pkl on disk.
    """
    os.makedirs(save_dir, exist_ok=True)

    # 1) Individua i file gia' presenti e valida quelli esistenti
    existing = [f for f in os.listdir(save_dir) if f.endswith(".pkl")]
    print(f"🔍 {len(existing)} files present, checking integrity...")

    corrupt_found = []
    for f in tqdm(existing, desc="Validating"):
        file_path = os.path.join(save_dir, f)
        if not _is_pkl_valid(file_path):
            corrupt_found.append(f)

    if corrupt_found:
        print(f"⚠️  {len(corrupt_found)} corrupted file(s) found, will re-download: {corrupt_found}")
    else:
        print("✅ All existing files are valid.")

    failed_tickers = []

    for t in tqdm(tickers, desc="Updating"):
        file_path = os.path.join(save_dir, f"{t}.pkl")

        # Se il file e' gia' presente e valido, non serve nulla da fare
        if _is_pkl_valid(file_path):
            continue

        old_data = _load_pkl_safe(file_path)  # None se assente/corrotto

        success = False
        last_error = None
        for attempt in range(1, n_retry + 1):
            try:
                ticker_obj = yf.Ticker(t)
                obj_data = _download_one(ticker_obj, granularity, n_years, old_data)
                _atomic_pickle_dump(obj_data, file_path)

                # Verifica che quanto appena scritto sia effettivamente valido
                if not _is_pkl_valid(file_path):
                    raise IOError("file written but failed integrity check")

                success = True
                break
            except Exception as e:
                last_error = e
                if attempt < n_retry:
                    time.sleep(retry_backoff * attempt)  # backoff crescente
                continue

        if not success:
            print(f"[WARN] Error with {t} after {n_retry} attempts: {last_error}")
            failed_tickers.append(t)

    # 2) Scrivi i ticker falliti in corrupted.txt
    if failed_tickers:
        with open(corrupted_log, "w") as f:
            for t in failed_tickers:
                f.write(f"{t}\n")
        print(f"❌ {len(failed_tickers)} ticker(s) failed after {n_retry} retries, logged to {corrupted_log}")
    else:
        # Nessun fallimento: rimuovi un eventuale log obsoleto per evitare confusione
        if os.path.exists(corrupted_log):
            os.remove(corrupted_log)
        print("✅ No failed tickers.")

    print("🏁 Update complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="yf_data")
    parser.add_argument("--n_retry", type=int, default=5)
    parser.add_argument("--corrupted_log", type=str, default="corrupted.txt")
    args = parser.parse_args()

    all_companies_df = get_all_companies()
    tickers = all_companies_df["ticker"].tolist()
    update_yf_data(
        tickers,
        n_years=5,
        granularity="yearly",
        save_dir=args.data_dir,
        n_retry=args.n_retry,
        corrupted_log=args.corrupted_log,
    )