
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


# Resolução de caminhos (sem config)

def _find_raw_dir(data_dir: Optional[Path] = None) -> Path:
    if data_dir is not None:
        return Path(data_dir)
    candidate = Path(__file__).resolve()
    for _ in range(5):
        candidate = candidate.parent
        raw = candidate / "data" / "raw"
        if raw.exists():
            return raw
    return Path("data") / "raw"


def _find_processed_dir(output_dir: Optional[Path] = None) -> Path:
    if output_dir is not None:
        return Path(output_dir)
    candidate = Path(__file__).resolve()
    for _ in range(5):
        candidate = candidate.parent
        if (candidate / "data").exists():
            return candidate / "data" / "processed"
    return Path("data") / "processed"


UNIVERSE_FILES: Dict[str, Dict] = {
    "IBRX50": {"file": "IBRX50.xlsx", "desc": "IBrX-50", "currency": "BRL"},
}

BENCHMARK_FILES: Dict[str, Dict] = {
    "^BVSP": {"file": "IBOV.xlsx",   "desc": "Ibovespa",            "currency": "BRL"},
    "EWZ":   {"file": "EWZ.xlsx",    "desc": "iShares MSCI Brazil", "currency": "USD"},
    "GLD":   {"file": "GLD.xlsx",    "desc": "SPDR Gold Shares",    "currency": "USD"},
    "SPY":   {"file": "SPY.xlsx",    "desc": "SPDR S&P 500 ETF",    "currency": "USD"},
}

RATE_FILES: Dict[str, Dict] = {
    "SELIC252": {"file": "Selic_252d.xlsx", "desc": "Selic % a.a. base 252"},
    "CDI252":   {"file": "cdi252.xlsx",     "desc": "CDI % a.a. base 252"},
}

_ALL_FILES = {**UNIVERSE_FILES, **BENCHMARK_FILES}


# Leitura do formato Economatica

def _read_economatica(path: Path, col: str = "Fechamento") -> pd.Series:
    if not path.exists():
        raise FileNotFoundError(f"arquivo não encontrado: {path}")
    df = pd.read_excel(path, header=3, usecols=["Data", col])
    df.columns = ["Date", "Value"]
    df["Date"]  = pd.to_datetime(df["Date"], errors="coerce")
    df["Value"] = pd.to_numeric(df["Value"], errors="coerce")
    df = df.dropna().set_index("Date").sort_index()
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    return df["Value"].rename(col)


# Lógica comum de carregamento para qualquer mapeamento de símbolos.
def _load_symbols(
    symbol_map: Dict[str, Dict],
    data_dir: Path,
    tickers: Optional[List[str]],
    start_date: Optional[str],
    end_date: Optional[str],
) -> Dict[str, pd.Series]:
    targets = tickers or list(symbol_map.keys())
    raw: Dict[str, pd.Series] = {}

    for sym in targets:
        if sym not in symbol_map:
            logger.warning(f"símbolo {sym!r} não mapeado — ignorado")
            continue
        info = symbol_map[sym]
        path = data_dir / info["file"]
        try:
            s = _read_economatica(path, col="Fechamento")
            if start_date:
                s = s[s.index >= pd.Timestamp(start_date)]
            if end_date:
                s = s[s.index <= pd.Timestamp(end_date)]
            raw[sym] = s
            logger.info(
                f"carregado {sym:10} ({info['desc']:25})  "
                f"{s.index.min().date()} → {s.index.max().date()}  "
                f"({len(s)} obs)"
            )
        except FileNotFoundError as e:
            logger.warning(str(e))
        except Exception as e:
            logger.error(f"erro ao carregar {sym}: {e}")

    return raw


def _to_dataframe(raw: Dict[str, pd.Series]) -> Optional[pd.DataFrame]:
    if not raw:
        return None
    df = pd.DataFrame(raw)
    nan_before = int(df.isnull().sum().sum())
    logger.info(f"shape: {df.shape}  NaN antes: {nan_before}")
    df = df.ffill().bfill()
    nan_after = int(df.isnull().sum().sum())
    if nan_after > 0:
        logger.warning(f"ainda {nan_after} NaN — removendo linhas")
        df = df.dropna()
    return df


# DataLoader

# Carrega preços de fechamento dos xlsx do Economatica.
class DataLoader:

    def __init__(self, data_dir: Optional[Path] = None):
        self.data_dir:   Path                    = _find_raw_dir(data_dir)
        self.prices:     Optional[pd.DataFrame]  = None
        self.benchmarks: Optional[pd.DataFrame]  = None
        self.volumes:    None                    = None
        self._raw:       Dict[str, pd.Series]    = {}
        self._raw_bench: Dict[str, pd.Series]    = {}

    # Carrega o universo principal (UNIVERSE_FILES).
    def load_data(
        self,
        tickers:    Optional[List[str]] = None,
        start_date: Optional[str]       = None,
        end_date:   Optional[str]       = None,
    ) -> Dict[str, pd.Series]:
        self._raw = _load_symbols(
            UNIVERSE_FILES, self.data_dir, tickers, start_date, end_date
        )
        if not self._raw:
            logger.warning("nenhum arquivo do universo carregado")
            return {}

        self.prices = _to_dataframe(self._raw)
        logger.info(f"load_data: {len(self._raw)} ticker(s) do universo carregados")
        return self._raw

    # Carrega benchmarks e ativos auxiliares (BENCHMARK_FILES).
    def load_benchmarks(
        self,
        tickers:    Optional[List[str]] = None,
        start_date: Optional[str]       = None,
        end_date:   Optional[str]       = None,
    ) -> Dict[str, pd.Series]:
        self._raw_bench = _load_symbols(
            BENCHMARK_FILES, self.data_dir, tickers, start_date, end_date
        )
        if not self._raw_bench:
            logger.warning("nenhum benchmark carregado")
            return {}

        self.benchmarks = _to_dataframe(self._raw_bench)
        logger.info(f"load_benchmarks: {len(self._raw_bench)} benchmark(s) carregados")
        return self._raw_bench

    def get_adjusted_close(self) -> pd.DataFrame:
        if self.prices is None:
            raise ValueError("execute load_data() primeiro")
        return self.prices

    def get_benchmark_prices(self) -> pd.DataFrame:
        if self.benchmarks is None:
            raise ValueError("execute load_benchmarks() primeiro")
        return self.benchmarks

    # Busca em universo e benchmarks.
    def get_ticker_data(self, ticker: str) -> pd.Series:
        if ticker in self._raw:
            return self._raw[ticker]
        if ticker in self._raw_bench:
            return self._raw_bench[ticker]
        raise ValueError(f"ticker {ticker!r} não carregado")

    # Atalho para carregar um único benchmark.
    def get_index_data(
        self,
        ticker: str,
        start_date: Optional[str] = None,
    ) -> pd.Series:
        self.load_benchmarks(tickers=[ticker], start_date=start_date)
        return self.get_ticker_data(ticker)

    def save_consolidated(
        self,
        filename:   str           = "portfolio_raw.parquet",
        fmt:        str           = "parquet",
        output_dir: Optional[Path] = None,
    ) -> Path:
        if self.prices is None:
            raise ValueError("nenhum dado para salvar")
        out = _find_processed_dir(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        filepath = out / filename
        if fmt == "parquet":
            self.prices.to_parquet(filepath)
        elif fmt == "csv":
            self.prices.to_csv(filepath)
        else:
            raise ValueError(f"formato não suportado: {fmt!r}")
        logger.info(f"dados consolidados salvos em {filepath}")
        return filepath

    def get_data_summary(self) -> pd.DataFrame:
        rows = []

        def _add_rows(raw: Dict[str, pd.Series], group: str, file_map: Dict):
            for sym, s in raw.items():
                rows.append({
                    "Grupo":     group,
                    "Ticker":    sym,
                    "Descrição": file_map.get(sym, {}).get("desc", sym),
                    "Moeda":     file_map.get(sym, {}).get("currency", "?"),
                    "Obs":       len(s),
                    "Início":    s.index.min().date(),
                    "Fim":       s.index.max().date(),
                    "Último":    round(float(s.iloc[-1]), 4),
                    "NaN":       int(s.isnull().sum()),
                })

        if self._raw:
            _add_rows(self._raw, "Universo", UNIVERSE_FILES)
        if self._raw_bench:
            _add_rows(self._raw_bench, "Benchmark", BENCHMARK_FILES)

        if not rows:
            raise ValueError("execute load_data() e/ou load_benchmarks() primeiro")

        return pd.DataFrame(rows)


# Conveniência

def load_benchmark_prices(
    data_dir:   Optional[Path] = None,
    tickers:    Optional[List[str]] = None,
    start_date: Optional[str] = None,
    end_date:   Optional[str] = None,
) -> pd.DataFrame:
    loader = DataLoader(data_dir=data_dir)
    loader.load_benchmarks(tickers=tickers, start_date=start_date, end_date=end_date)
    return loader.get_benchmark_prices()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    loader = DataLoader()
    loader.load_data(start_date="2010-01-01")
    loader.load_benchmarks(start_date="2010-01-01")
    if loader.prices is not None:
        print(loader.get_data_summary().to_string(index=False))
        print(f"\nUniverso (últimas 3 linhas):\n{loader.prices.tail(3)}")
        if loader.benchmarks is not None:
            print(f"\nBenchmarks (últimas 3 linhas):\n{loader.benchmarks.tail(3)}")
