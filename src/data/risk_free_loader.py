
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_this_dir = Path(__file__).resolve().parent
root_dir  = _this_dir.parent
sys.path.insert(0, str(root_dir))

try:
    from utils.config import PATHS
    _DEFAULT_DATA_DIR = Path(PATHS["raw_data"])
except Exception:
    _DEFAULT_DATA_DIR = root_dir / "data" / "raw"


_RATE_FILES = {
    "selic": {
        "file": "Selic_252d.xlsx",
        "desc": "Selic % a.a. base 252",
    },
    "cdi": {
        "file": "cdi252.xlsx",
        "desc": "CDI % a.a. base 252",
    },
}


# Leitura interna

# Lê o xlsx correspondente e retorna a taxa anual (% a.a.) como Series
def _read_rate_series(source: str, data_dir: Path) -> pd.Series:
    if source not in _RATE_FILES:
        raise ValueError(f"source deve ser 'selic' ou 'cdi', recebido: '{source}'")

    info = _RATE_FILES[source]
    path = data_dir / info["file"]

    if not path.exists():
        raise FileNotFoundError(
            f"Arquivo não encontrado: {path}\n"
            f"Coloque '{info['file']}' em {data_dir}"
        )

    df = pd.read_excel(path, header=3, usecols=["Data", "Fechamento"])
    df.columns = ["Date", "Rate_aa_pct"]

    df["Date"]       = pd.to_datetime(df["Date"], errors="coerce")
    df["Rate_aa_pct"] = pd.to_numeric(df["Rate_aa_pct"], errors="coerce")
    df = df.dropna().set_index("Date").sort_index()

    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    logger.info(
        f"[{source.upper()}] carregado: {df.index.min().date()} → "
        f"{df.index.max().date()}  ({len(df)} obs)  "
        f"último={df['Rate_aa_pct'].iloc[-1]:.2f}% a.a."
    )

    return df["Rate_aa_pct"]


# API pública

# Carrega a taxa livre de risco e retorna um DataFrame com colunas:
def load_risk_free_rate(
    start_date: Optional[str] = None,
    end_date:   Optional[str] = None,
    source:     str = "cdi",
    frequency:  str = "daily",
    data_dir:   Optional[Path] = None,
) -> pd.DataFrame:
    dir_ = Path(data_dir) if data_dir else _DEFAULT_DATA_DIR
    s    = _read_rate_series(source, dir_)

    if start_date:
        s = s[s.index >= pd.Timestamp(start_date)]
    if end_date:
        s = s[s.index <= pd.Timestamp(end_date)]

    df = s.to_frame("Rate_aa_pct").copy()
    df["Rate_aa"]       = df["Rate_aa_pct"] / 100
    df["Rate_daily"]    = (1 + df["Rate_aa"]) ** (1 / 252) - 1
    df["Risk_Free_Rate"] = df["Rate_daily"]

    if frequency == "monthly":
        df = df.resample("ME").last()
    elif frequency == "annual":
        df = df.resample("YE").last()

    return df


# Retorna diretamente a Series de retorno diário (base 252).
def get_daily_rf_series(
    start_date: Optional[str] = None,
    end_date:   Optional[str] = None,
    source:     str = "cdi",
    data_dir:   Optional[Path] = None,
) -> pd.Series:
    df = load_risk_free_rate(
        start_date=start_date,
        end_date=end_date,
        source=source,
        data_dir=data_dir,
    )
    return df["Rate_daily"]


# Retorna a taxa anual (decimal) mais recente disponível no arquivo.
def get_current_rf(
    source:   str = "cdi",
    data_dir: Optional[Path] = None,
) -> float:
    try:
        df = load_risk_free_rate(source=source, data_dir=data_dir)
        annual = float(df["Rate_aa"].iloc[-1])
        logger.info(f"taxa {source.upper()} atual: {annual*100:.2f}% a.a.")
        return annual
    except Exception as e:
        logger.warning(f"não foi possível ler taxa atual ({e}). Usando 10.75% a.a.")
        return 0.1075


# Retorna DataFrame comparativo Selic vs CDI (% a.a.) para análise.
def compare_selic_cdi(
    start_date: Optional[str] = None,
    end_date:   Optional[str] = None,
    data_dir:   Optional[Path] = None,
) -> pd.DataFrame:
    dir_ = Path(data_dir) if data_dir else _DEFAULT_DATA_DIR

    selic = _read_rate_series("selic", dir_).rename("Selic_aa_pct")
    cdi   = _read_rate_series("cdi",   dir_).rename("CDI_aa_pct")

    df = pd.concat([selic, cdi], axis=1).dropna()

    if start_date:
        df = df[df.index >= pd.Timestamp(start_date)]
    if end_date:
        df = df[df.index <= pd.Timestamp(end_date)]

    df["Diff_bps"] = (df["CDI_aa_pct"] - df["Selic_aa_pct"]) * 100

    return df


if __name__ == "__main__":
    print("=" * 55)
    print("  RISK FREE LOADER — smoke test")
    print("=" * 55)

    for src in ("cdi", "selic"):
        df = load_risk_free_rate(start_date="2015-01-01", source=src)
        print(f"\n[{src.upper()}]  {len(df)} obs")
        print(df.tail(3).to_string())
        print(f"  Taxa atual : {get_current_rf(source=src)*100:.2f}% a.a.")

    print("\nComparação Selic vs CDI (últimas 5 linhas):")
    comp = compare_selic_cdi(start_date="2020-01-01")
    print(comp.tail(5).to_string())
    print(f"  Diferença média: {comp['Diff_bps'].mean():.2f} bps")
