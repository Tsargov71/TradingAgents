""" Run this script after update_financials.py to rank companies
    according to the magic formula (slightly revised to take EBIT growth into account).

Author:
    Giovanni Pio Delvecchio
"""


import os
import pickle
import re
import numpy as np
import pandas as pd
from difflib import get_close_matches
from tqdm import tqdm
import argparse

import pdb

# Lista alias probabili per i principali campi
EBIT_CANDIDATES = [
    "EBIT", "Operating Income", "Operating Income (Loss)",
    "Income from operations", "OperatingIncome", "OperatingIncomeLoss",
    "OperatingIncomeLoss", "OperatingIncomeLossNet", "OperatingIncomeLossGross",
    "OperatingIncomeOrLoss", "OperatingIncomeOrLoss"
]
NET_INCOME_CANDIDATES = ["Net Income", "NetIncome", "Net Income Common Stockholders", "Net Income Available to Common Stockholders"]

TOTAL_CURRENT_ASSETS_CANDIDATES = [
    "Total Current Assets", "TotalCurrentAssets", "Current Assets", "TotalCurrentAssetsNet",
    "Total Current Assets (Net)", "TotalCurrentAssetsNet"
]
TOTAL_CURRENT_LIABILITIES_CANDIDATES = [
    "Total Current Liabilities", "TotalCurrentLiabilities", "Current Liabilities",
    "TotalCurrentLiabilitiesNet"
]
PPE_CANDIDATES = [
    "Property Plant Equipment", "PropertyPlantEquipment", "Net PPE", "NetPPE", "Gross PPE", "GrossPPE",
    "NetPPEExcludingRevaluation", "GrossPropertyPlantAndEquipment", "NetPropertyPlantAndEquipment",
    "NetFixedAssets", "NetFixedAssets"
]
TOTAL_ASSETS_CANDIDATES = ["Total Assets", "TotalAssets"]
TOTAL_DEBT_CANDIDATES = ["Total Debt", "TotalDebt", "Long Term Debt", "LongTermDebt", "Long Term Debt And Capital Lease Obligation"]
CASH_CANDIDATES = ["Cash And Cash Equivalents", "CashAndCashEquivalents", "Cash", "Cash Cash Equivalents And Federal Funds Sold", "Cash Financial"]

# Nuovi alias per il free cash flow. yfinance a volte espone direttamente
# "Free Cash Flow" nel cashflow statement; se non c'e', lo ricostruiamo come
# Operating Cash Flow + CapEx (il CapEx e' quasi sempre gia' riportato con
# segno negativo, quindi va SOMMATO, non sottratto, per ottenere FCF = OCF - |CapEx|).
FCF_CANDIDATES = ["Free Cash Flow", "FreeCashFlow", "FCF"]
OCF_CANDIDATES = [
    "Operating Cash Flow", "OperatingCashFlow", "Total Cash From Operating Activities",
    "CashFlowFromContinuingOperatingActivities", "Cash Flow From Continuing Operating Activities",
    "Net Cash Provided By Operating Activities"
]
CAPEX_CANDIDATES = [
    "Capital Expenditure", "CapitalExpenditure", "CapitalExpenditures",
    "Purchase Of PPE", "PurchaseOfPPE", "Net PPE Purchase And Sale"
]


def _tail(arr, n):
    """Safe equivalent of arr[-n:] that returns an empty array when n == 0,
    instead of Python's arr[-0:] == arr[0:] (the whole array). Without this,
    whenever one series was empty and minlen collapsed to 0, the OTHER series
    would come back unsliced, causing shape-mismatch errors during subtraction."""
    if n <= 0:
        return np.array([])
    return arr[-n:]


def _normalize_label(s):
    s = str(s)
    s = s.strip()
    s = re.sub(r'[^0-9a-zA-Z]', '', s).lower()
    return s

def find_best_match(index_iterable, candidates, cutoff=0.6):
    """
    Cerca la migliore corrispondenza tra labels (index_iterable) e una lista di candidates.
    Ritorna il label originale se trovato, altrimenti None.
    """
    if index_iterable is None:
        return None
    # normalizzazione delle etichette disponibili
    mapping = { _normalize_label(lbl): lbl for lbl in index_iterable }
    keys = list(mapping.keys())
    # prova esatti normalizzati dei candidates
    for cand in candidates:
        nc = _normalize_label(cand)
        if nc in mapping:
            return mapping[nc]
    # fuzzy match
    cand_norms = [_normalize_label(c) for c in candidates]
    for cn in cand_norms:
        matches = get_close_matches(cn, keys, n=1, cutoff=cutoff)
        if matches:
            return mapping[matches[0]]
    # fallback: try contains strategy (candidate token in label)
    for cand in candidates:
        token = _normalize_label(cand)
        for k, orig in mapping.items():
            if token in k or k in token:
                return orig
    return None


def _series_from_financials(fin_df, label):
    """
    Restituisce una numpy array con gli ultimi valori numerici per la riga 'label'.
    Se la riga non esiste o contiene solo NaN, restituisce array([]).
    """
    if fin_df is None or fin_df.empty or label is None:
        return np.array([])
    try:
        row = fin_df.loc[label]
    except Exception:
        # se l'index/label non esiste
        return np.array([])
    # row può essere Series o DataFrame (se più etichette uguali)
    if isinstance(row, pd.DataFrame):
        # somma/aggrega le righe duplicate prendendo la prima non-nan per colonna
        row = row.apply(lambda col: pd.to_numeric(col, errors='coerce')).iloc[0]
    # converti in numerico e togli NaN
    ser = pd.to_numeric(row, errors='coerce').dropna()
    if ser.empty:
        return np.array([])
    # IMPORTANTE: update_financials.py memorizza le colonne con la piu' recente
    # a SINISTRA (vedi merge_frames: pd.concat([new, old], axis=1) mette le
    # colonne nuove per prime, poi iloc[:, :n_years] tiene le prime a sinistra).
    # Quindi ser.values, cosi' com'e', va da PIU' RECENTE a PIU' VECCHIO.
    # Tutte le funzioni a valle (weighted_growth_from_ebits, leverage_trend, ecc.)
    # assumono invece un ordine cronologico crescente (dal piu' vecchio al piu'
    # recente) quando costruiscono x = np.arange(len(...)) per le regressioni.
    # Senza questa inversione, ogni slope/growth veniva calcolato con l'asse
    # temporale invertito, capovolgendo il segno di ogni trend nell'intera pipeline.
    vals = ser.values[::-1]
    return vals


def weighted_growth_from_ebits(ebits_arr, alpha=0.75):
    """
    Calcola la crescita media pesata nel tempo degli EBIT.
    alpha = 0.75 dà maggior peso ai dati recenti.
    """
    if len(ebits_arr) <= 1:
        return 0.0
    
    growths = []
    for i in range(1, len(ebits_arr)):
        prev = ebits_arr[i-1]
        curr = ebits_arr[i]
        if prev == 0 or np.isnan(prev):
            continue
        growths.append((curr - prev) / abs(prev))

    if not growths:
        return 0.0
    # growths è dal più vecchio → al più recente
    weights = np.array([alpha ** (len(growths) - i - 1) for i in range(len(growths))])
    weights /= weights.sum()
    return float(np.dot(growths, weights))


def safe_log_growth_from_ebits(ebits_arr, eps=1e-6):
    """
    Calcola la crescita media anno su anno di EBIT usando una metrica logaritmica stabilizzata.

    Usa log(1 + delta) per smorzare outlier e gestire casi con EBIT molto piccolo o negativo.
    Ritorna 0.0 se non ci sono informazioni utili.

    Args:
        ebits_arr (array-like): serie di EBIT ordinata temporalmente (dal più vecchio al più recente).
        eps (float): valore minimo per evitare divisioni per zero e log(0).
    """
    ebits_arr = np.array(ebits_arr, dtype=float)
    if len(ebits_arr) <= 1:
        return 0.0

    log_growths = []
    for i in range(1, len(ebits_arr)):
        prev = ebits_arr[i - 1]
        curr = ebits_arr[i]
        if np.isnan(prev) or np.isnan(curr):
            continue
        
        # Evita overflow se prev ≈ 0 e cambia segno
        denom = np.sign(prev) * max(abs(prev), eps)
        delta = (curr - prev) / abs(denom)
        
        # Usa log(1 + delta) per smorzare grandi salti
        try:
            log_g = np.log1p(delta)
        except ValueError:
            # in caso di delta < -1 (es. passaggio da +X a -Y)
            continue
        log_growths.append(log_g)

    if not log_growths:
        return 0.0

    return float(np.mean(log_growths))


def safe_log_signed_growth_from_ebits(ebits_arr, eps=1e-6):
    """
    Calcola la crescita media anno su anno su scala logaritmica, 
    mantenendo il segno delle variazioni (anche negative).

    - Usa log(1 + delta) per stabilizzare la scala.
    - Se prev == 0, salta il punto (non informativo).
    - Penalizza i cali forti ma senza divergere.
    - Restituisce 0.0 se non ci sono dati utili.
    """
    if len(ebits_arr) <= 1:
        return 0.0
    
    deltas = []
    for i in range(1, len(ebits_arr)):
        prev = ebits_arr[i-1]
        curr = ebits_arr[i]
        if np.isnan(prev) or np.isnan(curr) or abs(prev) < eps:
            continue
        delta = (curr - prev) / abs(prev)
        # clampare delta a [-0.9, +10] per evitare esplosioni e log negativi
        delta = np.clip(delta, -0.9, 10.0)
        # log signed: mantieni il segno ma riduci scala
        signed_log = np.sign(delta) * np.log1p(abs(delta))
        deltas.append(signed_log)
    
    if not deltas:
        return 0.0
    return float(np.mean(deltas))


def ebit_stability_score(ebits):
    """
    Ritorna un punteggio di stabilità EBIT = mean / (std + eps).
    Se ci sono meno di 2 anni → ritorna NaN (stabilità non definibile).
    """
    ebits = np.array(ebits, dtype=float)

    # almeno 2 anni
    if len(ebits) < 2:
        return np.nan

    mean = np.mean(ebits)
    std = np.std(ebits, ddof=1)

    # Se std è zero (EBIT identico ogni anno), penalizziamo meno
    if std == 0:
        return mean  # più EBIT = più stabilità

    return mean / std


def leverage_trend(net_debt_series, ebit_series, cap=50.0):
    """
    Ritorna la pendenza (slope) della serie temporale di leverage = net_debt / ebit,
    calcolata con una regressione LINEARE SEMPLICE (non piu' log-lineare).
    Negativo = leverage in calo nel tempo (buono), positivo = leverage in
    aumento (cattivo).

    Perche' non piu' log-lineare: la versione precedente clampava il leverage
    vicino a zero (floor) per gestire i periodi di cassa netta, poi applicava
    log(). Per aziende con debito quasi nullo (es. WSM, leverage vicino a
    zero quasi ogni anno), pero', il log amplifica enormemente variazioni
    assolute minuscole: passare da un leverage clampato vicino a zero a un
    leverage leggermente positivo in un solo anno produceva uno slope enorme
    anche se il cambiamento in termini assoluti era trascurabile. Il risultato
    erano classifiche rumorose proprio per le aziende piu' sane (basso debito),
    che sono esattamente quelle che questo fattore dovrebbe premiare.

    Una regressione lineare diretta sul leverage grezzo gestisce i valori
    negativi (cassa netta) senza bisogno di floor/clamp e non amplifica
    artificialmente piccole variazioni vicino allo zero. Il cap serve solo a
    limitare l'influenza di outlier estremi (es. EBIT quasi zero che fa
    esplodere il rapporto), non a schiacciare i valori piccoli.
    """
    net_debt_series = np.asarray(net_debt_series, dtype=float)
    ebit_series = np.asarray(ebit_series, dtype=float)

    if len(net_debt_series) < 3 or len(ebit_series) < 3:
        return np.nan

    valid = np.isfinite(net_debt_series) & np.isfinite(ebit_series) & (ebit_series > 0)
    if valid.sum() < 3:
        return np.nan

    nd = net_debt_series[valid]
    eb = ebit_series[valid]
    x = np.arange(len(nd))

    leverage = nd / eb
    leverage = np.clip(leverage, -cap, cap)

    slope, _ = np.polyfit(x, leverage, 1)

    # Restituisco lo slope (negativo = buono, leverage in calo)
    return float(slope)


def fcf_coverage_score(fcf_mean, net_debt_latest, min_debt=1e6):
    """
    Punteggio di "copertura" del debito netto tramite il free cash flow.
    A differenza di leverage_trend, questo NON e' un rank da invertire:
    piu' alto = meglio, esattamente come ROC/EY/Growth/Stability.

    Perche' non basta fare semplicemente FCF / net_debt:

    1) Debito netto positivo, FCF positivo -> ratio positivo, piu' alto e'
       meglio (l'azienda ripaga il debito piu' in fretta). Nessun problema.
    2) Debito netto positivo, FCF negativo -> ratio negativo. Corretto: e'
       il caso peggiore (debito + cassa che si brucia).
    3) Cassa netta (net_debt <= 0), FCF positivo -> QUI il rapporto grezzo si
       rompe: FCF positivo diviso un net debt NEGATIVO da' un ratio negativo,
       come se fosse il caso peggiore, quando in realta' e' il MIGLIORE
       possibile (niente debito, e per giunta genera cassa). E' lo stesso
       tipo di inversione di segno vicino allo zero gia' visto in
       leverage_trend.
    4) Cassa netta, FCF negativo -> il segno "sbagliato" del ratio grezzo
       darebbe un risultato positivo (negativo/negativo), facendolo sembrare
       un caso buono, quando l'azienda sta comunque bruciando cassa.

    La correzione e' usare abs(net_debt) al denominatore: cosi' il SEGNO del
    punteggio dipende solo dal segno del FCF (generare cassa e' sempre un
    bene, bruciarla e' sempre un male, a prescindere da quanto debito/cassa
    netta ha l'azienda), mentre la MAGNITUDINE resta scalata dalla dimensione
    del debito/cassa netta (payback piu' rapido = punteggio piu' alto in
    valore assoluto). Il floor min_debt evita che un net debt vicino a zero
    faccia esplodere il rapporto, stesso principio del cap in leverage_trend.

    Nota: il livello assoluto di leva (chi ha tanto o poco debito) e' gia'
    catturato da leverage_trend/net debt altrove nella pipeline; questo
    fattore misura specificamente quanto la generazione di cassa e' adeguata
    a coprire (o non coprire) l'esposizione debitoria/di cassa attuale.
    """
    if fcf_mean is None or np.isnan(fcf_mean) or net_debt_latest is None or np.isnan(net_debt_latest):
        return np.nan
    denom = max(abs(net_debt_latest), min_debt)
    return float(fcf_mean / denom)


def compute_magic_formula_from_yfdir(yf_data_dir, n_years=5, w1=1.0, w2=1.0, w3=1.0, w4=1.0, w5=1.0, w6=1.0, verbose=False):
    """
    Estrae campi dai pickle salvati in yf_data_dir e costruisce un DataFrame di output
    con informazioni diagnostiche (flags) per capire cosa è stato trovato.
    """
    results = []
    files = [f for f in os.listdir(yf_data_dir) if f.endswith(".pkl")]
    for file in tqdm(files, desc="Parsing yf_data"):
        ticker = file.replace(".pkl", "")
        path = os.path.join(yf_data_dir, file)
        try:
            with open(path, "rb") as fh:
                data = pickle.load(fh)
        except Exception as e:
            if verbose:
                print(f"[WARN] cannot open {file}: {e}")
            continue

        fin = data.get("financials")
        bs = data.get("balance_sheet")
        info = data.get("info", {})
        # ATTENZIONE: il free cash flow vive nel cash flow statement, che
        # update_financials.py potrebbe non salvare ancora nel pickle. Provo
        # qualche nome di chiave plausibile; se nessuna e' presente, cf resta
        # None e FCF_NetDebt sara' NaN per il ticker (finisce in fondo al
        # rank, non rompe nulla, ma va aggiunto a update_financials.py per
        # avere dati reali: tipicamente ticker.cashflow / ticker.cash_flow
        # in yfinance).
        cf = data.get("cashflow", data.get("cash_flow", data.get("cashflow_statement")))

        # Trova le label effettive
        ebit_label = find_best_match(fin.index if isinstance(fin, pd.DataFrame) else [], EBIT_CANDIDATES)
        # fallback: se non trovi EBIT prova Net Income (meno desiderabile)
        if ebit_label is None:
            ebit_label = find_best_match(fin.index if isinstance(fin, pd.DataFrame) else [], NET_INCOME_CANDIDATES)

        # estrai valori
        ebits = _series_from_financials(fin, ebit_label)
        if ebits.size > 0:
            ebits = ebits[-n_years:]  # ultimi n_years
        # growth
        #growth = safe_growth_from_ebits(ebits)
        #growth = safe_log_growth_from_ebits(ebits)
        #growth = safe_log_signed_growth_from_ebits(ebits)

        growth = weighted_growth_from_ebits(ebits)
        
        # Balance sheet fields
        tca_label = find_best_match(bs.index if isinstance(bs, pd.DataFrame) else [], TOTAL_CURRENT_ASSETS_CANDIDATES)
        tcl_label = find_best_match(bs.index if isinstance(bs, pd.DataFrame) else [], TOTAL_CURRENT_LIABILITIES_CANDIDATES)
        ppe_label = find_best_match(bs.index if isinstance(bs, pd.DataFrame) else [], PPE_CANDIDATES)
        ta_label = find_best_match(bs.index if isinstance(bs, pd.DataFrame) else [], TOTAL_ASSETS_CANDIDATES)
        debt_label = find_best_match(bs.index if isinstance(bs, pd.DataFrame) else [], TOTAL_DEBT_CANDIDATES)
        cash_label = find_best_match(bs.index if isinstance(bs, pd.DataFrame) else [], CASH_CANDIDATES)

        # estrai serie NWC e PPE
        # NWC: preferiamo TCA - TCL; altrimenti tentiamo una somma semplice
        nwc_vals = np.array([])
        if tca_label and tcl_label:
            tca_series = _series_from_financials(bs, tca_label)
            tcl_series = _series_from_financials(bs, tcl_label)
            # allineiamo lunghezze (usa _tail per evitare il bug di arr[-0:])
            if tca_series.size and tcl_series.size:
                minlen = min(len(tca_series), len(tcl_series))
                nwc_vals = _tail(tca_series, minlen) - _tail(tcl_series, minlen)
        # se manca NWC, proviamo con alcune componenti (receivables - payables)
        if nwc_vals.size == 0:
            # tentativo: receipts - payables
            rec_label = find_best_match(bs.index if isinstance(bs, pd.DataFrame) else [], ["Receivables", "Accounts Receivable", "AccountsReceivable"])
            pay_label = find_best_match(bs.index if isinstance(bs, pd.DataFrame) else [], ["Payables", "Accounts Payable", "Payables And Accrued Expenses"])
            if rec_label and pay_label:
                rec = _series_from_financials(bs, rec_label)
                pay = _series_from_financials(bs, pay_label)
                if rec.size and pay.size:
                    minlen = min(len(rec), len(pay))
                    nwc_vals = _tail(rec, minlen) - _tail(pay, minlen)

        # net debt (debt - cash): sempre inizializzato, anche se le label non si trovano,
        # cosi' il controllo successivo (net_debt_series.shape[0]) non solleva UnboundLocalError
        net_debt_series = np.array([])
        if debt_label and cash_label:
            debt_series = _series_from_financials(bs, debt_label)
            cash_series = _series_from_financials(bs, cash_label)
            minl_debt_cash = min(len(debt_series), len(cash_series))
            net_debt_series = _tail(debt_series, minl_debt_cash) - _tail(cash_series, minl_debt_cash)

        if net_debt_series.shape[0] > 0:
            minl_lev = min(len(net_debt_series), len(ebits))
            leverage = leverage_trend(_tail(net_debt_series, minl_lev), _tail(ebits, minl_lev))
        else:
            leverage = np.nan
        #pdb.set_trace()

        # PPE
        ppe_vals = _series_from_financials(bs, ppe_label) if ppe_label else np.array([])

        # Free Cash Flow: prova prima la riga diretta, altrimenti OCF + CapEx
        # (CapEx e' quasi sempre gia' negativo nelle fonti yfinance-like, quindi
        # va sommato, non sottratto: FCF = OCF - |CapEx| = OCF + CapEx).
        fcf_label = find_best_match(cf.index if isinstance(cf, pd.DataFrame) else [], FCF_CANDIDATES)
        fcf_vals = _series_from_financials(cf, fcf_label) if fcf_label else np.array([])
        if fcf_vals.size == 0:
            ocf_label = find_best_match(cf.index if isinstance(cf, pd.DataFrame) else [], OCF_CANDIDATES)
            capex_label = find_best_match(cf.index if isinstance(cf, pd.DataFrame) else [], CAPEX_CANDIDATES)
            ocf_series = _series_from_financials(cf, ocf_label) if ocf_label else np.array([])
            capex_series = _series_from_financials(cf, capex_label) if capex_label else np.array([])
            if ocf_series.size and capex_series.size:
                minlen = min(len(ocf_series), len(capex_series))
                fcf_vals = _tail(ocf_series, minlen) + _tail(capex_series, minlen)
        if fcf_vals.size > 0:
            fcf_vals = fcf_vals[-n_years:]
        fcf_mean = float(np.nan) if fcf_vals.size == 0 else float(np.nanmean(fcf_vals))

        # net debt PIU' RECENTE (non una media: net debt e' uno stock di
        # bilancio, non un flusso, quindi ha senso guardare l'ultimo dato
        # disponibile piuttosto che mediarlo nel tempo).
        net_debt_latest = float(net_debt_series[-1]) if net_debt_series.size > 0 else np.nan
        fcf_netdebt = fcf_coverage_score(fcf_mean, net_debt_latest)

        # EV: prefer info['enterpriseValue'] o calcolo da info
        ev = None
        currency = None
        financial_currency = None
        currency_mismatch = False
        if info and isinstance(info, dict):
            currency = info.get("currency")
            financial_currency = info.get("financialCurrency")
            # Se la valuta di trading differisce dalla valuta dei bilanci
            # (tipico di ADR/OTC esteri, es. Korea/Giappone), sommare
            # marketCap (in currency) a totalDebt/cash (spesso in
            # financialCurrency) produce un EV completamente sbagliato
            # (es. migliaia di miliardi contro pochi miliardi di MarketCap).
            currency_mismatch = bool(currency and financial_currency and currency != financial_currency)

            ev = info.get("enterpriseValue", None)
            market_cap = info.get("marketCap", None)
            total_debt_info = info.get("totalDebt", None)
            cash_info = info.get("totalCash", None) or info.get("cash", None)
            if ev is None and market_cap is not None:
                # prova a calcolare EV da info, fallback
                try:
                    ev = market_cap + (total_debt_info or 0) - (cash_info or 0)
                except Exception:
                    ev = None
        # se non in info, prova a ottenere da balance sheet (Total Assets o somma NWC+PPE)
        if ev is None:
            if ta_label:
                ta_vals = _series_from_financials(bs, ta_label)
                if ta_vals.size:
                    ev = float(ta_vals[-1])
            elif (nwc_vals.size or ppe_vals.size):
                # somma ultimo valore disponibile
                nv = (nwc_vals[-1] if nwc_vals.size else 0.0) + (ppe_vals[-1] if ppe_vals.size else 0.0)
                ev = float(nv) if nv != 0 else None

        # Guardia di plausibilita': indipendentemente dalla causa (mismatch di
        # valuta, dato sporco, ecc.), un EV che si discosta enormemente dal
        # MarketCap (fuori da un range 0.1x-10x) non e' affidabile. Meglio
        # scartarlo (NaN, cosi' EY diventa NaN e il ticker viene penalizzato
        # in modo trasparente) che lasciarlo avvelenare EY e il magic_score
        # con un denominatore assurdo.
        ev_mcap_ratio = np.nan
        market_cap_for_check = info.get("marketCap") if isinstance(info, dict) else None
        if ev not in (None, np.nan, 0.0) and market_cap_for_check not in (None, np.nan, 0.0):
            ev_mcap_ratio = ev / market_cap_for_check
            if not (0.1 <= ev_mcap_ratio <= 10.0):
                ev = np.nan


        # ebit mean
        ebit_mean = float(np.nan) if ebits.size==0 else float(np.nanmean(ebits))

        # denom per ROC = mean(NWC + PPE) (allineiamo lunghezze)
        denom = np.nan
        combined = None
        if nwc_vals.size and ppe_vals.size:
            minlen = min(len(nwc_vals), len(ppe_vals))
            combined = _tail(nwc_vals, minlen) + _tail(ppe_vals, minlen)
        elif nwc_vals.size:
            combined = nwc_vals
        elif ppe_vals.size:
            combined = ppe_vals
        if combined is not None and len(combined)>0:
            denom = float(np.nanmean(combined[-n_years:] if len(combined)>=n_years else combined))

        roc = None
        ey = None
        # sicurezza contro zero / NaN
        if not np.isnan(ebit_mean) and denom not in (None, np.nan, 0.0):
            roc = ebit_mean / denom
        else:
            roc = np.nan
        if ev not in (None, np.nan, 0.0):
            ey = ebit_mean / ev if not np.isnan(ebit_mean) else np.nan
        else:
            ey = np.nan

        ebit_stability = ebit_stability_score(ebits)

        #pdb.set_trace()

        results.append({
            "ticker": ticker,
            "ebit_label": ebit_label,
            "ebits_count": int(ebits.size),
            "ebit_mean": ebit_mean,
            "leverage": leverage,
            "debt_label": debt_label,
            "cash_label": cash_label,
            "net_debt_series": net_debt_series,
            "net_debt_latest": net_debt_latest,
            "fcf_label": fcf_label,
            "FCF_mean": fcf_mean,
            "FCF_NetDebt": fcf_netdebt,
            "ebit_stability": ebit_stability,
            "ebits": ebits,
            "ROC": roc,
            "EY": ey,
            "Growth": growth,
            "MarketCap": info.get("marketCap", np.nan) if isinstance(info, dict) else np.nan,
            "EV": ev,
            "ev_mcap_ratio": ev_mcap_ratio,
            "currency": currency,
            "financial_currency": financial_currency,
            "currency_mismatch": currency_mismatch,
            "nwc_found": bool(nwc_vals.size),
            "ppe_found": bool(ppe_vals.size),
            "total_assets_label": ta_label,
            "ppe_label": ppe_label,
            "nwc_labels": (tca_label, tcl_label),
            "info_keys": list(info.keys()) if isinstance(info, dict) else [],
            "path": path
        })

    df = pd.DataFrame(results)
    # ranks - NOT filtering: but puoi filtrare dopo
    df["ROC_rank"] = df["ROC"].rank(ascending=False, method="min", na_option="bottom")
    df["EY_rank"] = df["EY"].rank(ascending=False, method="min", na_option="bottom")
    df["Growth_rank"] = df["Growth"].rank(ascending=False, method="min", na_option="bottom")
    df["Stability_rank"] = df["ebit_stability"].rank(ascending=False, method="min", na_option="bottom")
    # ascending=True: lo slope piu' NEGATIVO (leverage in calo, cioe' il caso
    # migliore) deve ottenere rank 1. Con ascending=False (versione precedente)
    # accadeva l'opposto: lo slope piu' POSITIVO (leverage in aumento, il caso
    # peggiore) otteneva rank 1, penalizzando esattamente le aziende che
    # stavano riducendo il proprio indebitamento.
    df["Leverage_trend"] = df["leverage"].rank(ascending=True, method="min", na_option="bottom")
    # ascending=False: FCF_NetDebt e' gia' costruito in modo che "piu' alto
    # sia sempre meglio" (vedi fcf_coverage_score), quindi qui si comporta
    # come ROC/EY/Growth/Stability e non come leverage_trend.
    df["FCF_NetDebt_rank"] = df["FCF_NetDebt"].rank(ascending=False, method="min", na_option="bottom")
    df["magic_score"] = (
        w1 * df["ROC_rank"] +
        w2 * df["EY_rank"] +
        w3 * df["Growth_rank"] +
        w4 * df["Stability_rank"] +
        w5 * df["Leverage_trend"] +
        w6 * df["FCF_NetDebt_rank"]
    )
    return df


# df = pd.read_csv("magic_formula_debug.csv")  # se parti dal CSV
# oppure usa direttamente df prodotto dalla funzione

def clamp_growth(x, low=-2.0, high=2.0):
    if pd.isna(x):
        return np.nan
    return max(low, min(high, x))


def prepare_and_rank(
    df,
    min_ebit_years=2,
    min_marketcap=5e9,          # 5 miliardi
    max_marketcap=50e9,          # 50 miliardi
    exclude_negative_ebit=True,
    clamp_growth_bounds=(-2, 2)
):
    df2 = df.copy()

    # 1) Filtri base
    df2['valid_ebits'] = df2['ebits_count'] >= min_ebit_years

    # Range market cap: tra min e max
    df2['valid_marketcap'] = df2['MarketCap'].fillna(0).between(min_marketcap, max_marketcap)

    if exclude_negative_ebit:
        df2['all_ebits_positive'] = df2['ebits'].apply(
            lambda arr: np.all(np.array(arr) > 0) if isinstance(arr, (list, np.ndarray)) else False
        )
    else:
        df2['all_ebits_positive'] = True

    # Esclude ticker con EV implausibile rispetto al MarketCap (tipicamente
    # ADR/OTC estere dove la valuta di trading e quella dei bilanci
    # differiscono, es. ASBRF/KB/SURDF con EV migliaia di volte il MarketCap).
    # Un EV cosi' rotto rende EY (e quindi il magic_score) inutilizzabile per
    # quel ticker, quindi va escluso invece di essere semplicemente penalizzato.
    df2['valid_ev'] = df2['EV'].notna()

    df2['include'] = (
        df2['valid_ebits'] &
        df2['valid_marketcap'] &
        df2['all_ebits_positive'] &
        df2['valid_ev']
    )

    # 2) Winsor / clamp growth per stabilità
    df2['Growth_clamped'] = df2['Growth'].apply(lambda x: clamp_growth(x, *clamp_growth_bounds))

    # 3) Sostituisci NaN in ROC/EY/Growth con -inf (così i NaN finiscono in fondo)
    df2['ROC_filled'] = df2['ROC'].fillna(-np.inf)
    df2['EY_filled'] = df2['EY'].fillna(-np.inf)
    df2['Growth_filled'] = df2['Growth_clamped'].fillna(-np.inf)

    # 4) Calcolo rank solo sul sottoinsieme includibile
    df_valid = df2[df2['include']].copy()

    if df_valid.empty:
        raise ValueError("Nessun ticker soddisfa i filtri. Abbassa le soglie per debug.")

    df_valid['ROC_rank'] = df_valid['ROC_filled'].rank(ascending=False, method='min')
    df_valid['EY_rank'] = df_valid['EY_filled'].rank(ascending=False, method='min')
    df_valid['Growth_rank'] = df_valid['Growth_filled'].rank(ascending=False, method='min')
    # Stability_rank / Leverage_trend / FCF_NetDebt_rank gia' presenti in df2
    # (copiati da df), riallineati automaticamente per indice sul sottoinsieme filtrato.
    df_valid["Stability_rank"] = df2["Stability_rank"]
    df_valid["Leverage_trend"] = df2["Leverage_trend"]
    df_valid["FCF_NetDebt_rank"] = df2["FCF_NetDebt_rank"]

    # 5) Calcolo Magic Formula (ROC + EY + Growth + Stability + Leverage + FCF/NetDebt)
    w1, w2, w3, w4, w5, w6 = 1.0, 1.0, 1.0, 1.5, 1.5, 1.5
    df_valid['magic_score'] = (
        w1 * df_valid['ROC_rank'] +
        w2 * df_valid['EY_rank'] +
        w3 * df_valid['Growth_rank'] + 
        w4 * df_valid["Stability_rank"] +
        w5 * df_valid["Leverage_trend"] +
        w6 * df_valid["FCF_NetDebt_rank"]
    )

    # 6) Ordina per punteggio
    df_valid = df_valid.sort_values('magic_score').reset_index(drop=True)

    return df2, df_valid


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="yf_data")
    parser.add_argument("--min_ebit_years", type=int, default=2)
    parser.add_argument("--n_years", type=int, default=5)
    parser.add_argument("--top_n", type=int, default=40)

    args = parser.parse_args()

    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)

    df = compute_magic_formula_from_yfdir(args.data_dir, n_years=args.n_years, w1=1.0, w2=1.0, w3=1.0, w4=1.5, w5=1.5, w6=1.5, verbose=True)
    #df.to_csv("magic_formula_ranking.csv", index=False)
    df_all, df_ranked = prepare_and_rank(df, min_ebit_years=args.min_ebit_years, min_marketcap=5e9, max_marketcap=50e9, exclude_negative_ebit=True)
    # salva debug + ranking
    df_all.to_csv("magic_formula_debug_flags.csv", index=False)
    df_ranked.to_csv("magic_formula_ranked_filtered.csv", index=False)

    #pdb.set_trace()

    print(f"Top {args.top_n}:")
    print(df_ranked.head(args.top_n)[['ticker','ROC','EY','Growth_clamped','MarketCap','EV', "Leverage_trend", "Stability_rank", "FCF_NetDebt_rank", 'magic_score']])