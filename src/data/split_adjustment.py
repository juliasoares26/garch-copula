from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

THRESHOLD = 0.35
CONFIRM_WINDOW = 5
REVERT_TOL = 0.15


# Lê um xlsx do Economatica (cabeçalho linha 3, colunas Data/Fechamento).
def read_economatica_xlsx(path: str, col_name: str) -> pd.Series:
    df = pd.read_excel(path, header=2, usecols=[0, 4])
    df.columns = ["Data", "Fechamento"]
    df = df[df["Data"] != "Data"]
    df = df[df["Fechamento"].notna() & (df["Fechamento"] != "-")]
    df["Data"] = pd.to_datetime(df["Data"], errors="coerce")
    df["Fechamento"] = pd.to_numeric(df["Fechamento"], errors="coerce")
    df = df.dropna().sort_values("Data").set_index("Data")
    return df["Fechamento"].rename(col_name)


# Detecta e corrige splits/grupamentos não ajustados numa série de preços.
def detect_and_adjust_splits(
    s: pd.Series,
    ticker: str,
    threshold: float = THRESHOLD,
    confirm_window: int = CONFIRM_WINDOW,
    revert_tol: float = REVERT_TOL,
) -> Tuple[pd.Series, List[dict], List[dict]]:
    s = s.copy()
    events: List[dict] = []
    unconfirmed: List[dict] = []
    processed_dates = set()
    changed = True
    guard = 0
    while changed and guard < 50:
        guard += 1
        changed = False
        lr = np.log(s / s.shift(1))
        candidates = sorted(d for d in lr[lr.abs() > threshold].index if d not in processed_dates)
        if not candidates:
            break
        dt = candidates[0]
        pos = s.index.get_loc(dt)
        if pos == 0:
            processed_dates.add(dt)
            changed = True
            continue
        pre_price = s.iloc[pos - 1]
        post_price = s.iloc[pos]
        avg_after = s.iloc[pos: pos + confirm_window].mean()
        stability = abs(np.log(avg_after / post_price)) if post_price > 0 else np.inf
        logret_raw = np.log(post_price / pre_price)
        if stability < revert_tol:
            factor = post_price / pre_price
            s.iloc[:pos] = s.iloc[:pos] * factor
            events.append({
                "ticker": ticker,
                "date": dt.date().isoformat(),
                "pre_price_raw": round(float(pre_price), 4),
                "post_price_raw": round(float(post_price), 4),
                "factor_applied_to_history_before_date": round(float(factor), 6),
                "raw_logret": round(float(logret_raw), 4),
                "implied_ratio": f"{1/factor:.2f}:1" if factor < 1 else f"1:{factor:.2f}",
            })
        else:
            unconfirmed.append({
                "ticker": ticker,
                "date": dt.date().isoformat(),
                "pre_price_raw": round(float(pre_price), 4),
                "post_price_raw": round(float(post_price), 4),
                "raw_logret": round(float(logret_raw), 4),
                "stability_check": round(float(stability), 4),
                "reason": "not persistent / possible data error or genuine one-off market move",
            })
        processed_dates.add(dt)
        changed = True
    return s, events, unconfirmed


# Lê todos os xlsx em `raw_dir`, aplica detect_and_adjust_splits a cada
def build_adjusted_panel(
    raw_dir: str,
    pattern: str = "*.xlsx",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    files = sorted(Path(raw_dir).glob(pattern))
    raw_series: Dict[str, pd.Series] = {}
    adj_series: Dict[str, pd.Series] = {}
    all_events: List[dict] = []
    all_unconfirmed: List[dict] = []

    for f in files:
        ticker = f.stem
        s_raw = read_economatica_xlsx(str(f), ticker)
        raw_series[ticker] = s_raw
        s_adj, events, unconfirmed = detect_and_adjust_splits(s_raw, ticker)
        adj_series[ticker] = s_adj
        all_events.extend(events)
        all_unconfirmed.extend(unconfirmed)

    raw_panel = pd.DataFrame(raw_series)
    adj_panel = pd.DataFrame(adj_series)
    events_df = pd.DataFrame(all_events).sort_values(["ticker", "date"]) if all_events else pd.DataFrame(
        columns=["ticker", "date", "pre_price_raw", "post_price_raw",
                 "factor_applied_to_history_before_date", "raw_logret", "implied_ratio"])
    unconfirmed_df = pd.DataFrame(all_unconfirmed).sort_values(["ticker", "date"]) if all_unconfirmed else pd.DataFrame(
        columns=["ticker", "date", "pre_price_raw", "post_price_raw", "raw_logret", "stability_check", "reason"])

    return raw_panel, adj_panel, events_df, unconfirmed_df
