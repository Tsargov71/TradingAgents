""" Run this script after after magic_formula.py, instead of running get_stock_prices.py
    use the flag --re_download.

    Computes a price-trend "monotonicity" factor for the FULL universe of
    tickers in magic_formula_ranked_filtered.csv, saves it to its own
    reusable CSV (price_trend.csv), and combines it with the fundamentals
    ranking to produce a top-N stockpicking shortlist.

    Monotonicity is measured with the Spearman rank correlation between
    trading-day index and closing price:
        rho -> +1  : price has moved up almost every step (strong, consistent uptrend)
        rho ->  0  : no consistent direction
        rho -> -1  : consistent decline

    TWO monotonicity factors are computed: price_monotonicity (full
    available history -- captures "historical compounder" behaviour, but for
    tickers with decades of history this saturates near +1 regardless of
    RECENT momentum) and price_monotonicity_recent (last `recent_window`
    trading days only, default ~1 year -- captures current momentum, and is
    comparable across tickers with very different listing ages). Both are
    ranked and combined separately in combine_and_rank.

    In addition, an ENTRY PRICE ESTIMATE is computed for each ticker: a
    linear regression on the recent log-price gives a "fair value" trend
    line, and the current price is expressed as a z-score of deviation from
    that line. See estimate_entry_price() for the full rationale. This is
    informational by default (does not change the ranking unless
    --w_entry_discount is set > 0).

    Design for ~1000+ tickers:
    - price_trend.csv is a STANDALONE artifact (ticker + trend metrics only).
      It is meant to be reused: once built, `combine_and_rank` can be rerun
      instantly with different weights without re-downloading anything.
    - Resumable: if price_trend.csv already has a ticker, it's skipped on
      the next run (unless --re_download), and progress is flushed to disk
      every 25 tickers, so an interrupted run over ~1000 tickers doesn't
      lose everything (mirrors the corrupted-pickle lesson from
      update_financials.py).
    - Per-ticker price history is cached as .parquet with an atomic write
      (temp file + os.replace), same pattern used elsewhere in this
      pipeline, so a crash mid-download can't leave a corrupt cache file.

Author:
    Giovanni Pio Delvecchio
"""
import os
import time
import argparse
import numpy as np
import pandas as pd
import yfinance as yf
from tqdm import tqdm
from scipy.stats import spearmanr


def _download_full_history(ticker, retries=3, backoff=2.0, as_of=None):
    """Scarica tutta la storia disponibile per un ticker. period='max' fa
    si' che yfinance rilevi automaticamente la data di IPO/quotazione per
    ciascun titolo, invece di usare una data di inizio fissa.
    Se as_of (yyyy-mm-dd) e' dato, scarica solo fino a quella data inclusa
    (point-in-time: nessun prezzo successivo)."""
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            if as_of:
                end = (pd.Timestamp(as_of) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")  # end e' esclusivo
                data = yf.download(ticker, start="1970-01-01", end=end, interval="1d",
                                    progress=False, auto_adjust=True)
            else:
                data = yf.download(ticker, period="max", interval="1d",
                                    progress=False, auto_adjust=True)
            if data is None or data.empty:
                raise ValueError("empty history")
            return data
        except Exception as e:
            last_err = e
            if attempt < retries:
                time.sleep(backoff * attempt)
    raise RuntimeError(f"failed to download {ticker} after {retries} attempts: {last_err}")


# Frasi tipiche che yfinance/yahoo restituiscono quando un ticker non esiste
# piu' (delisting, cambio ticker, acquisizione, ecc.), a differenza di un
# errore di rete transitorio. Riconoscerle serve a smettere di ritentare
# ogni volta il download di un ticker morto (3 retry + backoff sprecati a
# ogni run) e a escluderlo stabilmente dall'universo.
_DELISTED_MARKERS = (
    "possibly delisted",
    "delisted",
    "no timezone found",
    "no price data found",
    "symbol may be delisted",
)


def _looks_delisted(err):
    msg = str(err).lower()
    return any(marker in msg for marker in _DELISTED_MARKERS)


def _load_blacklist(path):
    """Legge la blacklist come un semplice file di testo, un ticker per riga."""
    if not os.path.exists(path):
        return set()
    with open(path, "r") as f:
        return set(line.strip() for line in f if line.strip())


def _append_blacklist(path, ticker):
    """Append semplice: non c'e' concorrenza in questo script (un solo
    processo, loop sequenziale), quindi non serve write atomico qui come per
    i pickle/parquet -- al peggio, un crash a meta' scrittura perde solo
    l'ultima riga aggiunta, che verra' semplicemente ritentata al prossimo run."""
    with open(path, "a") as f:
        f.write(ticker + "\n")


def price_monotonicity(close_prices, min_points=252):
    """
    Ritorna (rho, p-value, n_punti) per la serie di prezzi di chiusura.
    Con meno di min_points osservazioni (default ~1 anno di trading) il
    risultato e' troppo rumoroso: ritorna NaN invece di un numero
    fuorviante basato su pochissimi punti (es. IPO recentissime).
    """
    close_prices = np.asarray(close_prices, dtype=float)
    close_prices = close_prices[np.isfinite(close_prices)]
    n = len(close_prices)
    if n < min_points:
        return np.nan, np.nan, n
    x = np.arange(n)
    rho, pval = spearmanr(x, close_prices)
    return float(rho), float(pval), n


def estimate_entry_price(close_prices, window=252, z_entry=-1.0, min_points=60):
    """
    Stima un prezzo di entrata basato sulla deviazione del prezzo attuale
    da una retta di trend (regressione lineare sul log-prezzo), calcolata
    sulla finestra piu' recente (default ultimo anno di trading, non tutta
    la storia come per il rho: una retta calcolata dall'IPO non
    rappresenta piu' il trend "attuale" per titoli con anni di storico).

    Perche' sul log-prezzo e non sul prezzo grezzo: i rendimenti azionari
    sono moltiplicativi, non additivi. Una regressione sul prezzo grezzo
    darebbe lo stesso peso assoluto a un titolo che passa da 10 a 20
    (raddoppio) e a uno che passa da 100 a 110 (+10%), pur essendo episodi
    di rendimento molto diversi. Sul log-prezzo la pendenza e' direttamente
    il rendimento composto medio per periodo, e i residui sono deviazioni
    percentuali confrontabili tra titoli diversi.

    Ritorna un dizionario con:
    - trend_fair_value: prezzo "atteso" oggi se il titolo avesse seguito
      esattamente la sua retta di trend recente.
    - price_zscore: quante deviazioni standard (dei residui storici della
      regressione) il prezzo attuale si trova SOPRA (+) o SOTTO (-) quella
      retta. Negativo = prezzo attualmente a sconto rispetto al proprio
      trend recente.
    - entry_price: prezzo suggerito per un eventuale ordine di acquisto,
      ottenuto spostando trend_fair_value verso il basso di z_entry
      deviazioni standard. Default z_entry=-1.0: un pullback modesto
      rispetto al trend, non "compra al prezzo di oggi" ne' un tentativo
      di intercettare un crollo enorme. Un z_entry piu' negativo richiede
      un pullback piu' marcato prima che il prezzo sia considerato "buono".
    - discount_pct: quanto il prezzo attuale e' sotto (negativo) o sopra
      (positivo) trend_fair_value, in percentuale.
    - current_price: ultimo prezzo di chiusura disponibile, per confronto.

    ATTENZIONE - questa metrica NON dice se il trend e' salutare (per
    quello c'e' price_monotonicity/rho): dice solo se, DATO il trend
    osservato, il prezzo attuale e' relativamente conveniente o costoso
    rispetto ad esso. Le due cose vanno lette insieme: uno sconto marcato
    (z molto negativo) su un titolo con rho fortemente negativo (trend
    discendente) non e' un'occasione, e' semplicemente un titolo che
    continua a scendere lungo la sua stessa traiettoria.
    """
    close_prices = np.asarray(close_prices, dtype=float)
    close_prices = close_prices[np.isfinite(close_prices) & (close_prices > 0)]
    n = len(close_prices)

    empty = {
        "trend_fair_value": np.nan,
        "price_zscore": np.nan,
        "entry_price": np.nan,
        "discount_pct": np.nan,
        "current_price": np.nan,
    }
    if n < min_points:
        return empty

    recent = close_prices[-window:] if n > window else close_prices
    log_p = np.log(recent)
    x = np.arange(len(log_p))

    try:
        slope, intercept = np.polyfit(x, log_p, 1)
    except Exception:
        return empty

    fitted = slope * x + intercept
    residuals = log_p - fitted
    resid_std = float(np.std(residuals, ddof=1)) if len(residuals) > 2 else np.nan

    last_x = x[-1]
    trend_log_fair_value = slope * last_x + intercept
    trend_fair_value = float(np.exp(trend_log_fair_value))

    current_price = float(recent[-1])

    if resid_std and not np.isnan(resid_std) and resid_std > 0:
        price_zscore = float((np.log(current_price) - trend_log_fair_value) / resid_std)
        entry_log_price = trend_log_fair_value + z_entry * resid_std
    else:
        price_zscore = np.nan
        entry_log_price = trend_log_fair_value  # nessuna dispersione stimabile: fair value stesso

    entry_price = float(np.exp(entry_log_price))
    discount_pct = float((current_price / trend_fair_value - 1.0) * 100.0) if trend_fair_value else np.nan

    return {
        "trend_fair_value": trend_fair_value,
        "price_zscore": price_zscore,
        "entry_price": entry_price,
        "discount_pct": discount_pct,
        "current_price": current_price,
    }


def compute_trend_for_all(tickers, price_cache_dir, output_path, retries=3,
                           re_download=False, min_points=252, sleep_between=0.15,
                           flush_every=25, entry_window=252, entry_min_points=60,
                           z_entry=-1.0, recent_window=252, recent_min_points=120,
                           blacklist_path="delisted_tickers.txt", as_of=None):
    """Calcola (o riusa) il trend di prezzo (storico completo + finestra
    recente) e la stima di entry price per TUTTI i ticker forniti, salvando
    incrementalmente in output_path cosi' che l'elaborazione di
    centinaia/migliaia di ticker sia interrompibile e riprendibile.

    I ticker gia' noti come delistati (blacklist_path) vengono saltati SENZA
    nemmeno tentare il download: niente retry/backoff sprecati su un ticker
    che non esiste piu'."""
    os.makedirs(price_cache_dir, exist_ok=True)

    # Tutte le colonne che questa versione dello script produce per riga.
    # Se un price_trend.csv esistente non le ha TUTTE (tipicamente perche'
    # scritto da una versione precedente dello script, prima dell'entry
    # price o del rho recente), non e' una cache valida per il resume: va
    # ricalcolato, altrimenti le righe "riusate" arrivano nel DataFrame
    # finale senza queste colonne e il merge/rank successivo (es. su
    # price_zscore) esplode con un KeyError.
    REQUIRED_FIELDS = {
        "price_monotonicity", "price_monotonicity_pval", "price_history_days",
        "price_monotonicity_recent", "price_monotonicity_recent_pval", "price_history_days_recent",
        "trend_fair_value", "price_zscore", "entry_price", "discount_pct", "current_price",
    }

    blacklist = _load_blacklist(blacklist_path)
    if blacklist:
        print(f"🚫 {len(blacklist)} ticker(s) in blacklist ({blacklist_path}), verranno saltati senza tentare il download.")

    existing = {}
    if os.path.exists(output_path) and not re_download:
        try:
            prev = pd.read_csv(output_path)
            missing_cols = REQUIRED_FIELDS - set(prev.columns)
            if missing_cols:
                print(f"⚠️  {output_path} manca delle colonne {sorted(missing_cols)} "
                      f"(probabilmente scritto da una versione precedente dello script). "
                      f"Ricalcolo tutti i ticker per completarle (uso la cache .parquet, "
                      f"quindi senza re-download).")
            else:
                existing = {row["ticker"]: row.to_dict() for _, row in prev.iterrows()}
                print(f"🔍 {len(existing)} ticker(s) already in {output_path}, will skip them.")
        except Exception:
            existing = {}

    results = []
    failed = []
    newly_blacklisted = []
    n_downloaded = 0
    for t in tqdm(tickers, desc="Computing price trend (full universe)"):
        if t in existing:
            results.append(existing[t])
            continue

        if t in blacklist:
            # niente riga in output: il ticker resta semplicemente assente
            # da price_trend.csv (e quindi escluso dal merge in
            # combine_and_rank), esattamente come un fallimento di download,
            # ma senza nemmeno provarci.
            continue

        cache_path = os.path.join(price_cache_dir, f"{t}.parquet")
        try:
            if re_download and os.path.exists(cache_path):
                os.remove(cache_path)
            if os.path.exists(cache_path):
                hist = pd.read_parquet(cache_path)
            else:
                hist = _download_full_history(t, retries=retries, as_of=as_of)
                tmp_path = cache_path + ".tmp"
                hist.to_parquet(tmp_path)
                os.replace(tmp_path, cache_path)
                n_downloaded += 1
                time.sleep(sleep_between)  # non martellare l'API

            if as_of:
                # Point-in-time: anche una cache scaricata senza limite di data viene tagliata a as_of,
                # e un titolo senza prezzi vicini a as_of (non quotato a quella data) viene scartato.
                idx = hist.index.tz_localize(None) if getattr(hist.index, "tz", None) else hist.index
                hist = hist[idx <= pd.Timestamp(as_of)]
                if hist.empty:
                    raise ValueError("empty history")
                last_day = idx[idx <= pd.Timestamp(as_of)][-1]
                if (pd.Timestamp(as_of) - last_day).days > 10:
                    raise ValueError(f"no price near {as_of} (last price {last_day.date()})")

            close_col = "Close" if "Close" in hist.columns else hist.columns[0]
            close = hist[close_col].dropna().values
            rho, pval, n = price_monotonicity(close, min_points=min_points)
            close_recent = close[-recent_window:] if len(close) > recent_window else close
            rho_recent, pval_recent, n_recent = price_monotonicity(close_recent, min_points=recent_min_points)
            entry_info = estimate_entry_price(
                close, window=entry_window, z_entry=z_entry, min_points=entry_min_points
            )
        except Exception as e:
            if _looks_delisted(e):
                print(f"[BLACKLIST] {t}: sembra delistato ({e}) — aggiunto a {blacklist_path}, "
                      f"verra' saltato senza retry nei prossimi run.")
                blacklist.add(t)
                _append_blacklist(blacklist_path, t)
                newly_blacklisted.append(t)
                continue  # nessuna riga per questo ticker, come sopra

            print(f"[WARN] {t}: {e}")
            rho, pval, n = np.nan, np.nan, 0
            rho_recent, pval_recent, n_recent = np.nan, np.nan, 0
            entry_info = {
                "trend_fair_value": np.nan, "price_zscore": np.nan,
                "entry_price": np.nan, "discount_pct": np.nan, "current_price": np.nan,
            }
            failed.append(t)

        row = {
            "ticker": t,
            "price_monotonicity": rho,
            "price_monotonicity_pval": pval,
            "price_history_days": n,
            "price_monotonicity_recent": rho_recent,
            "price_monotonicity_recent_pval": pval_recent,
            "price_history_days_recent": n_recent,
        }
        row.update(entry_info)
        results.append(row)

        if len(results) % flush_every == 0:
            pd.DataFrame(results).to_csv(output_path, index=False)

    df_trend = pd.DataFrame(results)
    df_trend.to_csv(output_path, index=False)

    if failed:
        print(f"\n⚠️  {len(failed)} ticker(s) failed price download (errore non riconosciuto come delisting): {failed}")
    if newly_blacklisted:
        print(f"🚫 {len(newly_blacklisted)} ticker(s) aggiunto/i alla blacklist in questo run: {newly_blacklisted}")
    print(f"✅ {output_path} written ({len(df_trend)} tickers, {n_downloaded} freshly downloaded).")

    return df_trend


def combine_and_rank(magic_ranking_file, trend_file, w_fundamentals, w_monotonicity,
                      w_monotonicity_recent=1.0, w_entry_discount=0.0):
    """Unisce il ranking fondamentale con il trend di prezzo (storico
    completo + finestra recente) e, se w_entry_discount > 0, con lo sconto
    rispetto al trend line, per produrre un punteggio combinato. Tutti i
    fattori vengono ri-espressi come rank ALL'INTERNO del set combinato (non
    della shortlist), cosi' i pesi sono confrontabili indipendentemente da
    quanti ticker sopravvivono al join.

    Monotonicity_rank (storico completo) e Monotonicity_recent_rank (ultimo
    anno circa) sono fattori DISTINTI e vengono pesati separatamente: il
    primo premia i "compounder" di lungo periodo, il secondo il momentum
    attuale, che per titoli con decenni di storia puo' divergere non poco
    dal primo (rho storico quasi sempre altissimo, rho recente molto piu'
    discriminante).

    w_entry_discount di default e' 0.0: l'entry price e' pensato
    principalmente come informazione da leggere accanto al ranking
    (a che prezzo entrare su un nome che hai gia' scelto per fondamentali +
    trend), non come un fattore di scoring aggiuntivo di default. Impostalo
    > 0 solo se vuoi che lo score finale privilegi esplicitamente i ticker
    che al momento scambiano a sconto rispetto al proprio trend recente
    (ricordando l'avvertenza in estimate_entry_price: uno sconto su un trend
    negativo non e' un'occasione)."""
    df_magic = pd.read_csv(magic_ranking_file)
    df_trend = pd.read_csv(trend_file)

    df = df_magic.merge(df_trend, on="ticker", how="inner")

    # Tieni solo i ticker con uno storico sufficiente per un rho affidabile
    # (sia storico completo che recente)
    n_before = len(df)
    df = df[df["price_monotonicity"].notna()].copy()
    n_after = len(df)
    if n_after < n_before:
        print(f"ℹ️  {n_before - n_after} ticker(s) escluso/i per storico prezzi insufficiente o mancante.")

    df["Fundamentals_rank"] = df["magic_score"].rank(ascending=True, method="min")
    df["Monotonicity_rank"] = df["price_monotonicity"].rank(ascending=False, method="min", na_option="bottom")
    df["Monotonicity_recent_rank"] = df["price_monotonicity_recent"].rank(ascending=False, method="min", na_option="bottom")
    # ascending=True: z-score piu' negativo (prezzo piu' scontato rispetto
    # al proprio trend recente) ottiene rank 1.
    df["Entry_discount_rank"] = df["price_zscore"].rank(ascending=True, method="min", na_option="bottom")

    df["combined_score"] = (
        w_fundamentals * df["Fundamentals_rank"] +
        w_monotonicity * df["Monotonicity_rank"] +
        w_monotonicity_recent * df["Monotonicity_recent_rank"] +
        w_entry_discount * df["Entry_discount_rank"]
    )
    df = df.sort_values("combined_score").reset_index(drop=True)
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--magic_ranking_file", type=str, default="magic_formula_ranked_filtered.csv")
    parser.add_argument("--trend_file", type=str, default="price_trend.csv",
                         help="CSV separato coi soli dati di trend di prezzo, riusabile senza riscaricare tutto")
    parser.add_argument("--price_cache_dir", type=str, default="price_history_full")
    parser.add_argument("--w_monotonicity", type=float, default=1.0,
                         help="peso del rho calcolato su TUTTA la storia disponibile")
    parser.add_argument("--w_monotonicity_recent", type=float, default=1.0,
                         help="peso del rho calcolato solo sulla finestra recente (--recent_window)")
    parser.add_argument("--recent_window", type=int, default=252,
                         help="finestra (giorni di trading) usata per il rho di momentum recente")
    parser.add_argument("--recent_min_points", type=int, default=120,
                         help="minimo storico richiesto per calcolare il rho recente")
    parser.add_argument("--w_fundamentals", type=float, default=1.0)
    parser.add_argument("--w_entry_discount", type=float, default=0.0,
                         help="peso dello sconto rispetto al trend line nello score combinato (0 = solo informativo)")
    parser.add_argument("--min_points", type=int, default=252)
    parser.add_argument("--entry_window", type=int, default=252,
                         help="finestra (giorni di trading) usata per la retta di trend dell'entry price")
    parser.add_argument("--entry_min_points", type=int, default=60,
                         help="minimo storico richiesto per stimare l'entry price")
    parser.add_argument("--z_entry", type=float, default=-1.0,
                         help="deviazioni standard sotto il trend fair value per il prezzo di entrata suggerito")
    parser.add_argument("--top_n", type=int, default=40, help="quanti risultati mostrare/salvare nel combinato finale")
    parser.add_argument("--re_download", action="store_true",
                         help="ignora price_trend.csv e la cache .parquet esistenti, ricalcola tutto da zero")
    parser.add_argument("--skip_compute", action="store_true",
                         help="salta il calcolo del trend e usa trend_file gia' esistente cosi' com'e' (solo combine)")
    parser.add_argument("--blacklist_file", type=str, default="delisted_tickers.txt",
                         help="file di testo (un ticker per riga) dei ticker noti come delistati, da saltare senza retry")
    parser.add_argument("--output", type=str, default="magic_formula_with_trend.csv")
    parser.add_argument("--as_of", type=str, default=None,
                         help="data soglia T (yyyy-mm-dd): usa solo prezzi fino a T incluso (point-in-time)")
    args = parser.parse_args()

    df_magic_full = pd.read_csv(args.magic_ranking_file)
    all_tickers = df_magic_full["ticker"].dropna().unique().tolist()
    print(f"Universe: {len(all_tickers)} tickers from {args.magic_ranking_file}")

    if not args.skip_compute:
        compute_trend_for_all(
            all_tickers,
            price_cache_dir=args.price_cache_dir,
            output_path=args.trend_file,
            re_download=args.re_download,
            min_points=args.min_points,
            entry_window=args.entry_window,
            entry_min_points=args.entry_min_points,
            z_entry=args.z_entry,
            recent_window=args.recent_window,
            recent_min_points=args.recent_min_points,
            blacklist_path=args.blacklist_file,
            as_of=args.as_of,
        )

    df_combined = combine_and_rank(
        args.magic_ranking_file, args.trend_file,
        args.w_fundamentals, args.w_monotonicity,
        w_monotonicity_recent=args.w_monotonicity_recent,
        w_entry_discount=args.w_entry_discount,
    )
    df_combined.to_csv(args.output, index=False)

    cols = ["ticker", "magic_score", "Fundamentals_rank",
            "price_monotonicity", "Monotonicity_rank",
            "price_monotonicity_recent", "Monotonicity_recent_rank",
            "current_price", "trend_fair_value", "discount_pct", "price_zscore",
            "entry_price", "Entry_discount_rank", "combined_score"]
    print(f"\nTop {args.top_n} (combined fundamentals + price trend):")
    print(df_combined.head(args.top_n)[cols].to_string())


if __name__ == "__main__":
    main()