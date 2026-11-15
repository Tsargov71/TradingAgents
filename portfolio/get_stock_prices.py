""" Run this script after magic_formula.py to gather the stock prices of 
    the top N companies.

Author:
    Giovanni Pio Delvecchio
"""

import os
import yfinance as yf
import pandas as pd
from tqdm import tqdm
from datetime import datetime
import argparse


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--top_n", type=int, default=10)
    parser.add_argument("--data_dir", type=str, default="stock_prices")
    parser.add_argument("--start_date", type=str, default="2020-11-11")
    parser.add_argument("--magic_ranking_file", type=str, default="magic_formula_ranked_filtered.csv")
    parser.add_argument("--re_download", action="store_true")
    parser.add_argument("--end_date", type=str, default=None, help="last day to download (default: today)")
    args = parser.parse_args()

    TOP_N = args.top_n
    DATA_DIR = args.data_dir
    START_DATE = args.start_date
    END_DATE = pd.Timestamp(args.end_date) if args.end_date else pd.Timestamp(datetime.today().date())


    df_valid = pd.read_csv(args.magic_ranking_file)
    tickers = df_valid["ticker"].head(TOP_N).dropna().unique().tolist()

    os.makedirs(DATA_DIR, exist_ok=True)

    existing_files = {
        os.path.splitext(f)[0]
        for f in os.listdir(DATA_DIR)
        if f.endswith(".parquet")
    }

    if args.re_download:
        for existing_file in existing_files:
            to_remove = os.path.join(DATA_DIR, existing_file + ".parquet")
            os.remove(to_remove)
        existing_files = []

    tickers_to_download = [t for t in tickers if t not in existing_files]

    print(f"Total number of tickers: {len(tickers)}")
    print(f"Downloaded: {len(existing_files)}")
    print(f"To download: {len(tickers_to_download)}")

    # === Loop principale ===
    for t in tqdm(tickers_to_download, desc="Downloading"):
        try:
            data = yf.download(
                t,
                start=START_DATE,
                end=END_DATE,
                interval="1d",       # dati giornalieri
                progress=False,
                auto_adjust=True,    # usa prezzi aggiustati per split/dividendi
            )
            if data.empty:
                print(f"No data for {t}, skipping...")
                continue

            data = data.rename(columns=str.lower)
            data = data[["open", "high", "low", "close", "volume"]]
            data["target"] = data["close"]
            data.reset_index(inplace=True)
            data.to_parquet(f"{DATA_DIR}/{t}.parquet")

        except Exception as e:
            print(f"❌ Error for {t}: {e}")


