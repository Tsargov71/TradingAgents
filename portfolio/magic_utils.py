import pandas as pd
import numpy as np
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
import pdb
from datetime import datetime, timedelta
from tqdm import tqdm
import os
import requests

"""
def aggregate_stock_predictions_alt(forecasts_dict, test_data_dict, prediction_length=1, aggregation='median'):
    predictions_by_stock = {}
    labels_by_stock = {}

    for item_id, forecasts in forecasts_dict.items():
        test_data = test_data_dict[item_id]

        all_preds = []
        all_labels = []
        all_dates = []

        for ts, forecast in zip(test_data, forecasts):
            # Date: partire dalla start_date del forecast
            dates = [forecast.start_date + pd.Timedelta(days=i) for i in range(prediction_length)]
            preds = forecast.mean[:prediction_length]  # previsione
            labels = ts["future_target"][:prediction_length]  # ground truth

            all_dates.extend(dates)
            all_preds.extend(preds)
            all_labels.extend(labels)

        temp_df = pd.DataFrame({
            "date": all_dates,
            "pred": all_preds,
            "label": all_labels
        })

        # Aggregazione
        if aggregation == 'last':
            temp_df = temp_df.groupby("date").last()
        elif aggregation == 'mean':
            temp_df = temp_df.groupby("date").mean()
        elif aggregation == 'median':
            temp_df = temp_df.groupby("date").median()
        else:
            raise ValueError(f"Unknown aggregation: {aggregation}")

        predictions_by_stock[item_id] = temp_df["pred"]
        labels_by_stock[item_id] = temp_df["label"]

    # Converti in DataFrame cross-sezionale
    predictions_df = pd.DataFrame(predictions_by_stock)
    labels_df = pd.DataFrame(labels_by_stock)

    # Mantieni solo date comuni
    common_dates = predictions_df.index.intersection(labels_df.index)
    predictions_df = predictions_df.loc[common_dates].sort_index()
    labels_df = labels_df.loc[common_dates].sort_index()

    print(f"\n[aggregate_stock_predictions] Aggregation: {aggregation}")
    print(f"  Total stocks: {len(predictions_by_stock)}")
    print(f"  Date range: {predictions_df.index.min()} to {predictions_df.index.max()}")

    return predictions_df, labels_df
"""

def get_aggregated_forecasts(forecasts_entry, aggregation="median"):
    start_dates = []
    aggregated_forecasts = []
    for curr_forecast in forecasts_entry:
        date_to_append = curr_forecast.start_date
        start_dates.append(date_to_append)

        if aggregation == "median":
            agg_vec_to_append = curr_forecast.median
        elif aggregation == "mean":
            agg_vec_to_append = curr_forecast.mean
        else:
            agg_vec_to_append = curr_forecast.samples[-1]
        aggregated_forecasts.append(agg_vec_to_append)
    return aggregated_forecasts, start_dates
        


def aggregate_stock_predictions_alt(forecasts_dict, test_data_dict, prediction_length=1, aggregation='median'):
    """
    Costruisce due DataFrame:
      - predictions_df: valori previsti per ogni ticker e data di inizio forecast
      - labels_df: valori ground-truth corrispondenti (future_target)
    """
    predictions_dict = {}
    labels_dict = {}

    tickers = list(forecasts_dict.keys())

    for ticker in tickers:
        # recupera forecasts e dati associati
        forecasts_entry = forecasts_dict[ticker]
        aggregated_forecasts, start_dates = get_aggregated_forecasts(forecasts_entry, aggregation)

        test_data_entry = test_data_dict[ticker]
        future_target_list = [elem["future_target"] for elem in test_data_entry]

        # salva come dizionario {data -> lista di valori}
        predictions_dict[ticker] = {
            str(date): list(forecast) for date, forecast in zip(start_dates, aggregated_forecasts)
        }
        labels_dict[ticker] = {
            str(date): list(gt) for date, gt in zip(start_dates, future_target_list)
        }

    # costruisci i DataFrame finali
    predictions_df = pd.DataFrame.from_dict(predictions_dict, orient="index")
    labels_df = pd.DataFrame.from_dict(labels_dict, orient="index")

    return predictions_df, labels_df


SEC_URL = "https://www.sec.gov/files/company_tickers_exchange.json"
def get_all_companies() -> pd.DataFrame:
    """Scarica e restituisce il DataFrame con tutte le società SEC."""
    headers = {
        "User-Agent": "Valerii Sargov valerii.sargov@studio.unibo.it",  # 👈 Cambia con nome + email reale
        "Accept-Encoding": "gzip, deflate",
        "Host": "www.sec.gov"
    }

    try:
        print("Downloading data from SEC...")
        r = requests.get(SEC_URL, headers=headers)
        r.raise_for_status()
        json_data = r.json()

        # Converti da fields + data a DataFrame
        df = pd.DataFrame(json_data["data"], columns=json_data["fields"])

        print(f"Download completato. Totale aziende: {len(df):,}")
        return df

    except Exception as e:
        print(f"Errore durante il download: {e}")
        return pd.DataFrame(columns=["cik", "ticker", "name", "exchange"])



import pandas as pd


def compute_target_on_instrument_data(instrument_data, price_column="target", forward_days = 1):
    instrument_data = instrument_data.sort_index()
    
    # Return futuro
    future_close = instrument_data[price_column].shift(-forward_days)
    current_close = instrument_data[price_column]
    instrument_labels = (future_close / current_close) - 1
    
    label_df = pd.DataFrame({'target': instrument_labels})
    return label_df


def calc_labels(df: pd.DataFrame, 
                price_column: str = 'target', 
                forward_days: int = 1) -> pd.DataFrame:
    """
    Calcola il return futuro come label.
    Label verificata: (close_{t+1} / close_t) - 1
    """
    result_df = df.copy()
    
    if price_column not in result_df.columns:
        raise ValueError(f"Colonna '{price_column}' non trovata")
    
    labels = []
    
    for instrument, curr_subdf in result_df.groupby('item_id'):
        label_df = compute_target_on_instrument_data(curr_subdf, price_column=price_column, forward_days=forward_days)
        labels.append(label_df)
    
    all_labels = pd.concat(labels)
    result_df['target'] = all_labels[price_column]
    
    return result_df

def fill_missing_dates_zero(df, date_col="Date", group_col="item_id"):
    filled = []

    for symbol, group in df.groupby(group_col):
        # Creiamo l'indice completo di date
        full_index = pd.date_range(group[date_col].min(), group[date_col].max(), freq="D")

        # Reindex sul nuovo indice
        group_full = group.set_index(date_col).reindex(full_index)

        # Ripristina il nome della colonna item_id
        group_full[group_col] = symbol

        # Tutte le colonne numeriche diventano 0.0 dove mancanti
        for col in group_full.columns:
            if col != group_col:
                group_full[col] = group_full[col].fillna(0.0)

        # Reset index per riportare la colonna date
        group_full = group_full.reset_index().rename(columns={"index": date_col})

        filled.append(group_full)

    # Concatenazione finale
    return pd.concat(filled, ignore_index=True)


def apply_transforms_to_dict(data_dict, price_cols=["target"], volume_col=None, forward_days=1, Q1=None, Q3=None):
    """
    Applica le trasformazioni finanziarie (ritorni e log-volume) a ogni DataFrame nel dizionario.
    È una versione adattata per DataFrame caricati da pickle (senza parsing stringhe).
    """
    transformed_dict = {}

    for ticker, df in data_dict.items():
        try:
            df_copy = df.copy()
            df_copy["item_id"] = ticker
            df_copy = calc_labels(df_copy)
            df_copy = fill_missing_dates_zero(df_copy)
            df_copy, Q1, Q3 = remove_quantiles(df_copy, Q1, Q3)
            transformed_dict[ticker] = df_copy
            print(f"✓ Trasformato {ticker}: {len(df)} → {len(df_copy)} righe")

        except Exception as e:
            print(f"✗ Errore con {ticker}: {e}")

    return transformed_dict


import ast

def linearize_df_from_csv(preds_labels_df):
    tickers = preds_labels_df.index.tolist()
    cols = preds_labels_df.columns.tolist()
    linearized_dict = {}
    max_len = 0

    for ticker in tickers:
        series = []

        for col in cols:
            val = preds_labels_df.loc[ticker, col]

            # --- Converti stringhe in liste, se serve ---
            if isinstance(val, str):
                val = val.strip()
                if val.lower() in ("nan", "none", ""):
                    val = np.nan
                else:
                    try:
                        val = ast.literal_eval(val)  # converte "[0.1, 0.2]" in [0.1, 0.2]
                    except Exception:
                        val = np.nan

            # --- Gestisci array/list ---
            if isinstance(val, (list, np.ndarray)) and len(val) > 0:
                arr = np.array(val, dtype=float)
                arr = arr[~np.isnan(arr)]  # rimuovi eventuali NaN
                if len(arr) > 0:
                    series.append(arr)
                    max_len = max(max_len, len(arr))

        # --- Concatena tutto per ticker ---
        if len(series) > 0:
            linearized_dict[ticker] = np.concatenate(series)

    return linearized_dict, max_len


def linearize_df(preds_labels_df):
    tickers = preds_labels_df.index.tolist()
    cols = preds_labels_df.columns.tolist()
    linearized_dict = {}
    max_len = 0
    for ticker in tickers:
        # ignora celle vuote o non numeriche
        series = []
        for col in cols:
            val = preds_labels_df.loc[ticker, col]
            if isinstance(val, (list, np.ndarray)) and len(val) > 0:
                arr = np.array(val, dtype=float)
                arr = arr[~np.isnan(arr)]  # rimuovi NaN
                if len(arr) > 0:
                    series.append(arr)
                if arr.shape[0] > max_len:
                    max_len = arr.shape[0]
        if len(series) > 0:
            linearized_dict[ticker] = np.concatenate(series)
    return linearized_dict, max_len


def remove_quantiles(df_chronos, Q1=None, Q3=None):
    # Calcolo IQR
    if Q1 is None or Q3 is None:
        Q1 = df_chronos["target"].quantile(0.25)
        Q3 = df_chronos["target"].quantile(0.75)
    IQR = Q3 - Q1
    print(f"Q1: {Q1:.5f}, Q3: {Q3:.5f}, IQR: {IQR:.5f}")

    # Definizione dei limiti
    bounds_factor = 1.5 # 1.5
    lower = Q1 - bounds_factor * IQR
    upper = Q3 + bounds_factor * IQR
    print(f"Lower bound: {lower:.5f}, Upper bound: {upper:.5f}")
    # Clipping values (Winsorization)
    df_cut = df_chronos.copy()
    df_cut["target"] = df_chronos["target"].clip(lower=lower, upper=upper)
    n_outliers = (df_cut["target"] != df_chronos["target"]).sum()
    print(f"N outliers: {n_outliers}, df length: {len(df_chronos)}")
    return df_cut, Q1, Q3


def pct_change(some_arr):
    some_arr = np.array(some_arr, dtype=float)
    some_arr = some_arr[~np.isnan(some_arr)]  # rimuovi eventuali NaN
    if len(some_arr) < 2:
        return np.array([], dtype=float)
    pct = np.diff(some_arr) / np.where(some_arr[:-1] == 0, np.nan, some_arr[:-1])
    pct = np.where(np.isfinite(pct), pct, np.nan)  # rimuovi inf
    return pct

import traceback





def compute_ic_trends(preds_df, labels_df, q=0.3, min_obs = 5):
    """
    Calcola IC, RankIC, ICIR e RankICIR in modo robusto ai NaN.
    """
    
    # 1️⃣ Linearizza le sequenze ticker→array
    linearized_preds, max_len_preds = linearize_df(preds_df)
    linearized_labels, max_len_labels = linearize_df(labels_df)

    # 2️⃣ Calcola le variazioni percentuali giornaliere
    #pct_change_preds = {k: pct_change(v) for k, v in linearized_preds.items()}
    #pct_change_labels = {k: pct_change(v) for k, v in linearized_labels.items()}
    pct_change_preds = linearized_preds
    pct_change_labels = linearized_labels

    # 3️⃣ Allinea ticker e lunghezze
    common_tickers = list(set(pct_change_preds.keys()) & set(pct_change_labels.keys()))

    preds_aligned, labels_aligned = {}, {}
    for t in common_tickers:
        p, l = pct_change_preds[t], pct_change_labels[t]
        if len(p) != len(l):
            continue
        if len(p) < max_len_preds:
            continue
        preds_aligned[t] = p
        labels_aligned[t] = l

    if not preds_aligned:
        raise ValueError("Nessun ticker valido per IC computation.")

    preds_trend_df = pd.DataFrame.from_dict(preds_aligned, orient='index')
    labels_trend_df = pd.DataFrame.from_dict(labels_aligned, orient='index')

    # 5️⃣ Calcola IC, RankIC e AR per ogni giorno (colonna)
    results = {}
    ar_values = []

    for day in preds_trend_df.columns:
        preds_today = preds_trend_df[day]
        labels_today = labels_trend_df[day]

        valid_mask = preds_today.notna() & labels_today.notna()
        if valid_mask.sum() < 3:
            continue

        preds_valid = preds_today[valid_mask]
        labels_valid = labels_today[valid_mask]

        # escludi giorni a varianza zero
        if labels_valid.var() == 0:
            continue

        try:
            ic, _ = pearsonr(preds_valid, labels_valid)
        except Exception:
            ic = np.nan

        try:
            rank_ic, _ = spearmanr(preds_valid, labels_valid)
        except Exception:
            rank_ic = np.nan

        # === AR: rendimento medio long-short ===
        ar = np.nan
        try:
            threshold_low = preds_valid.quantile(q)
            threshold_high = preds_valid.quantile(1 - q)
            if threshold_low == 0 or threshold_high == 0:
                print(f"Null threshold, preds variance is too low: {preds_valid.var(axis=0)} compared to labels variance: {labels_valid.var(axis=0)}")
            long_mask = preds_valid >= threshold_high
            short_mask = preds_valid <= threshold_low

            n_long, n_short = long_mask.sum(), short_mask.sum()
            if n_long >= min_obs and n_short >= min_obs:
                long_ret = labels_valid[long_mask].mean(skipna=True)
                short_ret = labels_valid[short_mask].mean(skipna=True)
                if np.isfinite(long_ret) and np.isfinite(short_ret):
                    ar = long_ret - short_ret

        except Exception as e:
            print(f"[warn] AR skipped on day {day}: {e}")

        results[day] = {"IC": ic, "RankIC": rank_ic, "AR": ar}
        if np.isfinite(ar):
            ar_values.append(ar)

    ic_values = np.array([v["IC"] for v in results.values() if np.isfinite(v["IC"])])
    rank_ic_values = np.array([v["RankIC"] for v in results.values() if np.isfinite(v["RankIC"])])
    ar_values = np.array(ar_values)

    ic_mean = np.nanmean(ic_values) if len(ic_values) else np.nan
    ic_std = np.nanstd(ic_values) if len(ic_values) > 1 else np.nan
    rank_ic_mean = np.nanmean(rank_ic_values) if len(rank_ic_values) else np.nan
    rank_ic_std = np.nanstd(rank_ic_values) if len(rank_ic_values) > 1 else np.nan
    ar_mean = np.nanmean(ar_values) if len(ar_values) else np.nan
    ar_std = np.nanstd(ar_values) if len(ar_values) > 1 else np.nan

    metrics = {
        "IC_mean": ic_mean,
        "IC_std": ic_std,
        "ICIR": ic_mean / (ic_std + 1e-8) if np.isfinite(ic_std) and ic_std > 0 else np.nan,
        "RankIC_mean": rank_ic_mean,
        "RankIC_std": rank_ic_std,
        "RankICIR": rank_ic_mean / (rank_ic_std + 1e-8) if np.isfinite(rank_ic_std) and rank_ic_std > 0 else np.nan,
        "AR_mean": ar_mean,
        "AR_std": ar_std,
        "IR": ar_mean / (ar_std + 1e-8) if np.isfinite(ar_std) and ar_std > 0 else np.nan,
        "per_date": results,
        "n_periods": len(results),
    }

    return metrics



def split_df_by_column(df: pd.DataFrame, group_col: str) -> dict:
    """
    Groups a DataFrame by a column and returns a dictionary of sub-DataFrames.
    
    Args:
        df (pd.DataFrame): The input DataFrame.
        group_col (str): Column name to group by.
    
    Returns:
        dict[str, pd.DataFrame]: A dictionary where keys are unique values of `group_col`
                                 and values are the corresponding sub-DataFrames.
    """
    grouped_dict = {}
    
    for key, group in df.groupby(group_col):
        grouped_dict[key] = group.reset_index(drop=True)
    
    return grouped_dict


def gather_series(files, DATA_DIR):

    all_dfs = []
    #offset_days = 2 # Be careful here as the dataset has to be continuously updated...
    #yesterday = pd.Timestamp(datetime.today().date() - timedelta(days=offset_days))

    for f in tqdm(files):
        #pdb.set_trace()
        try:
            df = pd.read_parquet(os.path.join(DATA_DIR, f))
            if df is None:
                continue 

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            df = df.sort_values("Date")
            df["item_id"] = os.path.splitext(f)[0]

            all_dfs.append(df)

        except Exception as e:
            print(f"Errore con {f}: {e}")

    print(f"Number of valid training series: {len(all_dfs)}")
    dfs = pd.concat(all_dfs, ignore_index=True)
    return dfs