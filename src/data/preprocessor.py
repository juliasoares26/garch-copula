import pandas as pd
import numpy as np
from typing import Optional, Tuple, List
import logging
from scipy import stats
import sys
import os
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import PROCESSED_DATA_DIR, RAW_DATA_DIR, DATA_DIR


# Selic

# Carrega série histórica da Selic de data/external/selic.parquet ou .csv.
def load_selic_rate(data_dir=None, fallback_annual: float = 0.1075) -> pd.Series:
    external_dir = (Path(data_dir) if data_dir else DATA_DIR) / "external"
    for fname in ["selic.parquet", "selic.csv", "cdi.parquet", "cdi.csv"]:
        fpath = external_dir / fname
        if fpath.exists():
            try:
                df = (
                    pd.read_parquet(fpath)
                    if fname.endswith(".parquet")
                    else pd.read_csv(fpath, index_col=0, parse_dates=True)
                )
                col = next(
                    (c for c in ["taxa_anual", "selic", "cdi", "rate"] if c in df.columns),
                    df.columns[0],
                )
                series = df[col].squeeze()
                if series.max() > 1:
                    series = series / 100
                daily = series / 252
                logging.getLogger(__name__).info(
                    f"Selic carregada de {fpath}  média anual={series.mean()*100:.2f}%"
                )
                return daily
            except Exception as e:
                logging.getLogger(__name__).warning(f"Falha ao carregar {fpath}: {e}")
    logging.getLogger(__name__).warning(
        f"Selic não encontrada. Usando fallback: {fallback_annual*100:.2f}% a.a."
    )
    return pd.Series(dtype=float, name="selic_daily")


# Carrega série histórica de taxa livre de risco em USD (ex.: 3-month
def load_usd_risk_free_rate(data_dir=None, fallback_annual: float = 0.045) -> pd.Series:
    external_dir = (Path(data_dir) if data_dir else DATA_DIR) / "external"
    for fname in ["usd_risk_free.parquet", "usd_risk_free.csv", "dgs3mo.parquet", "dgs3mo.csv"]:
        fpath = external_dir / fname
        if fpath.exists():
            try:
                df = (
                    pd.read_parquet(fpath)
                    if fname.endswith(".parquet")
                    else pd.read_csv(fpath, index_col=0, parse_dates=True)
                )
                col = next(
                    (c for c in ["taxa_anual", "rate", "dgs3mo"] if c in df.columns),
                    df.columns[0],
                )
                series = df[col].squeeze()
                if series.max() > 1:
                    series = series / 100
                daily = series / 252
                logging.getLogger(__name__).info(
                    f"Taxa livre de risco USD carregada de {fpath}  média anual={series.mean()*100:.2f}%"
                )
                return daily
            except Exception as e:
                logging.getLogger(__name__).warning(f"Falha ao carregar {fpath}: {e}")
    logging.getLogger(__name__).warning(
        f"Taxa livre de risco USD não encontrada. Usando fallback: {fallback_annual*100:.2f}% a.a."
    )
    return pd.Series(dtype=float, name="usd_rf_daily")


# Retorna taxa livre de risco anualizada média no período do index,
def get_risk_free_rate(
    index: pd.DatetimeIndex,
    data_dir=None,
    fallback_annual: float = 0.1075,
    currency: str = "BRL",
) -> float:
    if currency == "USD":
        rf_daily = load_usd_risk_free_rate(data_dir, fallback_annual=0.045)
        default = 0.045
    else:
        rf_daily = load_selic_rate(data_dir, fallback_annual)
        default = fallback_annual

    if rf_daily.empty:
        return default
    common = rf_daily.index.intersection(index)
    if len(common) == 0:
        return default
    return float(rf_daily.loc[common].mean()) * 252


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# DataPreprocessor

# Pré-processador focado em log-retornos diários para o pipeline GARCH-Cópula.
class DataPreprocessor:

    def __init__(self, prices: pd.DataFrame):
        self.prices = self._normalize_index(prices.copy())
        self.returns: Optional[pd.DataFrame] = None
        self.log_returns: Optional[pd.DataFrame] = None
        self.clean_prices: Optional[pd.DataFrame] = None

    # Colapsa timestamps intradiários/timezone-mistos para um único
    @staticmethod
    def _normalize_index(prices: pd.DataFrame) -> pd.DataFrame:
        idx = pd.to_datetime(prices.index)
        if idx.tz is not None:
            idx = idx.tz_localize(None)
        idx = idx.normalize()
        n_before = len(prices)
        out = prices.copy()
        out.index = idx
        out = out[~out.index.duplicated(keep="last")].sort_index()
        n_after = len(out)
        if n_after < n_before:
            logger.warning(
                f"_normalize_index: {n_before - n_after} linhas duplicadas "
                f"(mesmo dia, horários diferentes) removidas "
                f"({n_before} -> {n_after})"
            )
        return out

    # Missing data

    # Trata dados faltantes em preços.
    def handle_missing_data(
        self,
        method: str = "forward_fill",
        max_consecutive_missing: int = 5,
    ) -> pd.DataFrame:
        logger.info(f"Tratando dados faltantes com método: {method}")

        if method == "forward_fill":
            filled = self.prices.ffill(limit=max_consecutive_missing).bfill()
        elif method == "interpolate":
            filled = self.prices.interpolate(method="linear", limit=max_consecutive_missing)
            filled = filled.ffill().bfill()
        elif method == "drop":
            filled = self.prices.dropna()
            logger.info(f"Descartadas {len(self.prices) - len(filled)} linhas com NaN")
        else:
            raise ValueError(f"Método desconhecido: {method}")

        for col in self.prices.columns:
            mask = self.prices[col].isnull()
            if mask.any():
                runs = mask.astype(int).groupby((~mask).astype(int).cumsum()).sum()
                max_gap = int(runs.max())
                if max_gap > max_consecutive_missing:
                    logger.warning(
                        f"{col}: {max_gap} dias consecutivos sem dados "
                        f"(limite={max_consecutive_missing})"
                    )

        remaining_na = filled.isnull().sum().sum()
        if remaining_na > 0:
            logger.warning(
                f"{remaining_na} valores ainda faltantes após {method}. Descartando linhas."
            )
            filled = filled.dropna()

        self.clean_prices = filled
        logger.info(f"Preços limpos: {self.clean_prices.shape}")
        return self.clean_prices

    # Log-retornos diários

    # Calcula retornos diários.
    def compute_returns(
        self,
        method: str = "log",
        periods: int = 1,
    ) -> pd.DataFrame:
        if method != "log":
            raise ValueError(
                "Apenas log-retornos ('log') são suportados no pipeline GARCH-Cópula. "
                "Retornos simples distorcem a PIT e a estimação GARCH."
            )

        if self.clean_prices is None:
            self.handle_missing_data()

        log_ret = np.log(self.clean_prices / self.clean_prices.shift(periods))
        log_ret = log_ret.iloc[periods:]

        extreme_mask = (log_ret.abs() > 0.5)
        n_extreme = int(extreme_mask.sum().sum())
        if n_extreme > 0:
            logger.warning(
                f"{n_extreme} retornos com |r| > 50% detectados. "
                "Verifique se os preços têm splits/erros não ajustados."
            )

        self.log_returns = log_ret
        self.returns = log_ret
        logger.info(f"Log-retornos diários calculados: {log_ret.shape}")
        return log_ret

    # Outliers — apenas diagnóstico, sem remoção automática

    # Identifica outliers nos log-retornos (apenas diagnóstico).
    def detect_outliers(
        self,
        method: str = "iqr",
        threshold: float = 3.0,
    ) -> pd.DataFrame:
        if self.returns is None:
            self.compute_returns()

        outliers = pd.DataFrame(
            False, index=self.returns.index, columns=self.returns.columns
        )

        for col in self.returns.columns:
            series = self.returns[col].dropna()

            if method == "iqr":
                q1, q3 = series.quantile(0.25), series.quantile(0.75)
                iqr = q3 - q1
                lower, upper = q1 - threshold * iqr, q3 + threshold * iqr
                outliers[col] = (self.returns[col] < lower) | (self.returns[col] > upper)

            elif method == "zscore":
                z = np.abs(stats.zscore(series, nan_policy="omit"))
                outliers.loc[series.index, col] = z > threshold

            elif method == "mad":
                median = series.median()
                mad = np.median(np.abs(series - median))
                if mad > 0:
                    mod_z = 0.6745 * (series - median) / mad
                    outliers.loc[series.index, col] = np.abs(mod_z) > threshold

            else:
                raise ValueError(f"Método desconhecido: {method}")

        n_outliers = int(outliers.sum().sum())
        pct = n_outliers / (outliers.shape[0] * outliers.shape[1]) * 100
        logger.info(
            f"Outliers detectados (método={method}, threshold={threshold}): "
            f"{n_outliers} ({pct:.2f}% do total) — apenas informativos, não removidos."
        )
        return outliers

    # Estatísticas descritivas

    # Teste ADF para estacionariedade dos log-retornos.
    def check_stationary(self, alpha: float = 0.05) -> pd.DataFrame:
        from statsmodels.tsa.stattools import adfuller

        if self.returns is None:
            self.compute_returns()

        records = []
        for col in self.returns.columns:
            series = self.returns[col].dropna()
            try:
                adf_stat, p_val, *_ = adfuller(series, autolag="AIC")
                records.append(
                    dict(ticker=col, adf_statistic=adf_stat, p_value=p_val,
                         is_stationary=p_val < alpha)
                )
            except Exception as e:
                logger.warning(f"ADF falhou para {col}: {e}")
                records.append(
                    dict(ticker=col, adf_statistic=np.nan, p_value=np.nan,
                         is_stationary=False)
                )

        df = pd.DataFrame(records)
        n_stat = int(df["is_stationary"].sum())
        logger.info(f"ADF: {n_stat}/{len(df)} séries estacionárias (α={alpha})")
        return df

    # Estatísticas descritivas dos log-retornos.
    def get_descriptive_stats(self, currency_map: Optional[dict] = None) -> pd.DataFrame:
        if self.returns is None:
            self.compute_returns()

        stats_dict: dict = {
            "mean":       self.returns.mean(),
            "std":        self.returns.std(),
            "skewness":   self.returns.skew(),
            "kurtosis":   self.returns.kurtosis(),
            "min":        self.returns.min(),
            "q01":        self.returns.quantile(0.01),
            "q05":        self.returns.quantile(0.05),
            "median":     self.returns.median(),
            "q95":        self.returns.quantile(0.95),
            "q99":        self.returns.quantile(0.99),
            "max":        self.returns.max(),
        }

        stats_dict["annual_return"]     = self.returns.mean() * 252
        stats_dict["annual_volatility"] = self.returns.std() * np.sqrt(252)

        currency_map = currency_map or {}
        rf_by_col = {
            col: get_risk_free_rate(
                self.returns.index,
                currency=currency_map.get(col, "BRL"),
            )
            for col in self.returns.columns
        }
        rf = pd.Series(rf_by_col)
        stats_dict["risk_free_rate"] = rf
        stats_dict["sharpe_ratio"] = (
            (stats_dict["annual_return"] - rf) / stats_dict["annual_volatility"]
        )

        jb_stats, jb_pvals = [], []
        for col in self.returns.columns:
            series = self.returns[col].dropna()
            try:
                jb_s, jb_p = stats.jarque_bera(series)
                jb_stats.append(jb_s)
                jb_pvals.append(jb_p)
            except Exception:
                jb_stats.append(np.nan)
                jb_pvals.append(np.nan)

        stats_dict["jb_statistic"] = pd.Series(jb_stats, index=self.returns.columns)
        stats_dict["jb_pvalue"]    = pd.Series(jb_pvals, index=self.returns.columns)
        stats_dict["is_normal"]    = pd.Series([p > 0.05 for p in jb_pvals], index=self.returns.columns)

        logger.info("Estatísticas descritivas calculadas")
        return pd.DataFrame(stats_dict, index=self.returns.columns)

    # Utilitários

    def align_data(
        self,
        other_data: pd.DataFrame,
        method: str = "inner",
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        if method == "inner":
            common = self.returns.index.intersection(other_data.index)
            return self.returns.loc[common], other_data.loc[common]
        elif method == "outer":
            all_idx = self.returns.index.union(other_data.index)
            return self.returns.reindex(all_idx), other_data.reindex(all_idx)
        else:
            raise ValueError(f"Método desconhecido: {method}")

    def save_processed_data(self, filename: str = "returns.parquet"):
        if self.returns is None:
            raise ValueError("Nenhum retorno computado.")
        filepath = PROCESSED_DATA_DIR / filename
        if filename.endswith(".parquet"):
            self.returns.to_parquet(filepath)
        elif filename.endswith(".csv"):
            self.returns.to_csv(filepath)
        else:
            raise ValueError(f"Formato não suportado: {filename}")
        logger.info(f"Dados salvos em {filepath}")


# Helpers de alto nível

def separate_stocks_and_indices(
    prices: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    try:
        from data.data_loader import UNIVERSE_FILES, BENCHMARK_FILES
    except ImportError:
        from data_loader import UNIVERSE_FILES, BENCHMARK_FILES
    stock_cols = [c for c in prices.columns if c in UNIVERSE_FILES]
    index_cols = [c for c in prices.columns if c in BENCHMARK_FILES]
    unrecognized = [c for c in prices.columns if c not in stock_cols and c not in index_cols]
    if unrecognized:
        raise ValueError(
            f"Colunas não reconhecidas em UNIVERSE_FILES nem BENCHMARK_FILES: "
            f"{unrecognized}. Adicione-as a um dos dois mapeamentos em "
            f"data_loader.py (ou trate-as separadamente) antes de chamar "
            f"separate_stocks_and_indices."
        )
    logger.info(f"Separados: {len(stock_cols)} ações/universo, {len(index_cols)} índices/benchmarks")
    return prices[stock_cols], prices[index_cols]


# Pipeline de pré-processamento para o contexto GARCH-Cópula.
def preprocess_stocks_separately(
    prices: pd.DataFrame,
    handle_missing: str = "forward_fill",
    save: bool = True,
    data_dir=None,
) -> dict:
    stocks, indices = separate_stocks_and_indices(prices)
    results: dict = {}

    try:
        from data.data_loader import UNIVERSE_FILES, BENCHMARK_FILES
    except ImportError:
        from data_loader import UNIVERSE_FILES, BENCHMARK_FILES
    _currency_lookup = {**UNIVERSE_FILES, **BENCHMARK_FILES}
    currency_map = {
        ticker: meta.get("currency", "BRL")
        for ticker, meta in _currency_lookup.items()
    }

    if len(stocks.columns) > 0:
        logger.info(f"\n[AÇÕES] Pré-processando {len(stocks.columns)} ativos")
        sp = DataPreprocessor(stocks)
        sp.handle_missing_data(method=handle_missing)
        sp.compute_returns(method="log")

        _ = sp.detect_outliers(method="iqr", threshold=3.0)

        stocks_stationary = sp.check_stationary()
        stocks_stats      = sp.get_descriptive_stats(currency_map=currency_map)

        results["stocks_returns"]    = sp.returns
        results["stocks_stats"]      = stocks_stats
        results["stocks_stationary"] = stocks_stationary
        results["risk_free_rate"]    = (
            float(stocks_stats["risk_free_rate"].iloc[0])
            if "risk_free_rate" in stocks_stats.columns
            else 0.1075
        )

        if save:
            sp.save_processed_data("stocks_returns.parquet")
            stocks_stats.to_csv(PROCESSED_DATA_DIR / "stocks_descriptive_stats.csv")
            stocks_stationary.to_csv(
                PROCESSED_DATA_DIR / "stocks_stationary_tests.csv", index=False
            )
            logger.info("Dados das ações salvos.")

    if len(indices.columns) > 0:
        logger.info(f"\n[ÍNDICES] Pré-processando {len(indices.columns)} benchmarks")
        ip = DataPreprocessor(indices)
        ip.handle_missing_data(method=handle_missing)
        ip.compute_returns(method="log")

        indices_stationary = ip.check_stationary()
        indices_stats      = ip.get_descriptive_stats(currency_map=currency_map)

        results["indices_returns"]    = ip.returns
        results["indices_stats"]      = indices_stats
        results["indices_stationary"] = indices_stationary

        if save:
            ip.save_processed_data("indices_returns.parquet")
            indices_stats.to_csv(PROCESSED_DATA_DIR / "indices_descriptive_stats.csv")
            indices_stationary.to_csv(
                PROCESSED_DATA_DIR / "indices_stationary_tests.csv", index=False
            )
            logger.info("Dados dos índices salvos.")

    if (
        save
        and len(stocks.columns) > 0
        and len(indices.columns) > 0
    ):
        with open(PROCESSED_DATA_DIR / "comparative_summary.txt", "w", encoding="utf-8") as f:
            f.write(f"Período: {prices.index.min()} até {prices.index.max()}\n\n")

            ss = results["stocks_stats"]
            f.write("=== AÇÕES BRASILEIRAS ===\n")
            f.write(f"Quantidade: {len(stocks.columns)}\n")
            f.write(f"Retorno médio anualizado : {ss['annual_return'].mean()*100:.2f}%\n")
            f.write(f"Volatilidade média anual  : {ss['annual_volatility'].mean()*100:.2f}%\n")
            f.write(f"Sharpe ratio médio        : {ss['sharpe_ratio'].mean():.3f}\n")
            f.write(
                f"Séries estacionárias      : "
                f"{results['stocks_stationary']['is_stationary'].sum()}"
                f"/{len(results['stocks_stationary'])}\n\n"
            )

            f.write("Top 5 por retorno anualizado:\n")
            for ticker, ret in ss["annual_return"].sort_values(ascending=False).head(5).items():
                f.write(f"  {ticker:15} {ret*100:7.2f}%\n")

            f.write("\nTop 5 por Sharpe ratio:\n")
            for ticker, sr in ss["sharpe_ratio"].sort_values(ascending=False).head(5).items():
                f.write(f"  {ticker:15} {sr:7.3f}\n")

            idx_s = results["indices_stats"]
            f.write("\n=== ÍNDICES / BENCHMARKS ===\n")
            f.write(f"Quantidade: {len(indices.columns)}\n")
            f.write(f"Retorno médio anualizado : {idx_s['annual_return'].mean()*100:.2f}%\n")
            f.write(f"Volatilidade média anual  : {idx_s['annual_volatility'].mean()*100:.2f}%\n")
            f.write(f"Sharpe ratio médio        : {idx_s['sharpe_ratio'].mean():.3f}\n\n")
            for ticker in idx_s.index:
                r = idx_s.loc[ticker]
                f.write(
                    f"  {ticker:15} | Retorno: {r['annual_return']*100:7.2f}% "
                    f"| Vol: {r['annual_volatility']*100:6.2f}% "
                    f"| Sharpe: {r['sharpe_ratio']:6.3f}\n"
                )
        logger.info("Resumo comparativo salvo.")

    logger.info("Pré-processamento concluído.")
    return results


# Volatilidade realizada / correlação rolling

def calculate_realized_volatility(
    returns: pd.DataFrame,
    window: int = 20,
) -> pd.DataFrame:
    logger.info(f"Volatilidade realizada (janela={window})")
    return returns.rolling(window=window).std() * np.sqrt(252)


def calculate_rolling_correlation(
    returns: pd.DataFrame,
    window: int = 60,
) -> pd.Series:
    logger.info(f"Correlação rolling (janela={window})")
    rolling_corr = returns.rolling(window=window).corr()
    avg_corr = []

    for date in returns.index[window:]:
        try:
            cm = rolling_corr.loc[date]
            mask = np.triu(np.ones(cm.shape, dtype=bool), k=1)
            avg_corr.append(float(cm.where(mask).stack().mean()))
        except Exception:
            avg_corr.append(np.nan)

    return pd.Series(avg_corr, index=returns.index[window:], name="avg_correlation")


if __name__ == "__main__":
    from data_loader import DataLoader

    try:
        from utils.config import PROCESSED_DATA_DIR
        parquet_path = PROCESSED_DATA_DIR / "portfolio_raw.parquet"
        prices = pd.read_parquet(parquet_path)
        print("Dados carregados do parquet.")
    except FileNotFoundError:
        print("Parquet não encontrado — carregando xlsx da pasta raw.")
        loader = DataLoader()
        loader.load_data()
        prices = loader.get_adjusted_close()
    print(f"Shape: {prices.shape} | Período: {prices.index.min()} – {prices.index.max()}")
    print(f"Ativos: {list(prices.columns)}")

    results = preprocess_stocks_separately(prices, save=True)

    if "stocks_stats" in results:
        print("\nEstatísticas das ações:")
        print(
            results["stocks_stats"][
                ["mean", "std", "annual_return", "annual_volatility", "sharpe_ratio",
                 "kurtosis", "skewness"]
            ].round(4)
        )

    if "indices_stats" in results:
        print("\nEstatísticas dos índices:")
        print(
            results["indices_stats"][
                ["mean", "std", "annual_return", "annual_volatility", "sharpe_ratio"]
            ].round(4)
        )

    print("\nPré-processamento concluído.")
