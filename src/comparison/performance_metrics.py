
from __future__ import annotations

import logging
import warnings
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from scipy.stats import norm, t as t_dist
from scipy.optimize import minimize

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


# Utilitários

def _annualize(mu: float, sigma: float, n: int = 252) -> Tuple[float, float]:
    return mu * n, sigma * np.sqrt(n)


def _drawdown_series(returns: np.ndarray) -> np.ndarray:
    cum = np.cumprod(1 + returns)
    peak = np.maximum.accumulate(cum)
    return (cum - peak) / np.maximum(peak, 1e-10)


def _sig_stars(p: float) -> str:
    if np.isnan(p):
        return ""
    if p < 0.01:
        return "***"
    if p < 0.05:
        return "**"
    if p < 0.10:
        return "*"
    return ""


# Métricas escalares

# Desvio-padrão (amostral) dos retornos.
def standard_deviation(
    returns: np.ndarray,
    annualize: bool = True,
    periods: int = 252,
) -> float:
    sd = float(np.std(returns, ddof=1))
    return sd * np.sqrt(periods) if annualize else sd


# Sharpe ratio anualizado.
def sharpe_ratio(
    returns: np.ndarray,
    risk_free: float = 0.0,
    periods: int = 252,
) -> float:
    rf_daily = risk_free / periods
    excess   = returns - rf_daily
    mu_ann, vol_ann = _annualize(np.mean(excess), np.std(excess, ddof=1), periods)
    return mu_ann / max(vol_ann, 1e-8)


# Sortino ratio anualizado.
def sortino_ratio(
    returns: np.ndarray,
    risk_free: float = 0.0,
    periods: int = 252,
    mar: Optional[float] = None,
) -> float:
    rf_daily = risk_free / periods
    threshold = mar if mar is not None else rf_daily
    excess    = np.mean(returns) - rf_daily
    downside_dev = np.minimum(returns - threshold, 0)
    downside_vol = np.sqrt(np.mean(downside_dev ** 2)) * np.sqrt(periods)
    return (excess * periods) / max(downside_vol, 1e-8)


# Drawdown máximo (valor negativo).
def max_drawdown(returns: np.ndarray) -> float:
    dd = _drawdown_series(returns)
    return float(dd.min())


# Calmar ratio = retorno anualizado / |max drawdown|.
def calmar_ratio(returns: np.ndarray, periods: int = 252) -> float:
    ann_ret = float(np.mean(returns) * periods)
    mdd     = abs(max_drawdown(returns))
    return ann_ret / max(mdd, 1e-8)


# Turnover médio diário: E[ Σ_i |w_{it} - w_{i,t-1}| ] / 2
def average_turnover(weights_df: pd.DataFrame) -> float:
    diff = weights_df.diff().dropna()
    return float(np.abs(diff).sum(axis=1).mean() / 2)


# SSPW — Sum of Squared Portfolio Weights (Kirby & Ostdiek, 2012).
def sspw(weights_df: pd.DataFrame) -> float:
    return float((weights_df ** 2).sum(axis=1).mean())


# Ω = E[max(r - L, 0)] / E[max(L - r, 0)].
def omega_ratio(returns: np.ndarray, threshold: float = 0.0) -> float:
    gains  = np.sum(np.maximum(returns - threshold, 0))
    losses = np.sum(np.maximum(threshold - returns, 0))
    return gains / max(losses, 1e-10)


# |percentil (1-q)| / |percentil q| — assimetria das caudas.
def tail_ratio(returns: np.ndarray, q: float = 0.05) -> float:
    p_up   = abs(np.percentile(returns, (1 - q) * 100))
    p_down = abs(np.percentile(returns, q * 100))
    return p_up / max(p_down, 1e-10)


# (VaR, ES) histórico — perdas positivas.
def historical_var_es(
    returns: np.ndarray,
    confidence_level: float = 0.99,
) -> Tuple[float, float]:
    var  = float(-np.quantile(returns, 1 - confidence_level))
    tail = returns[returns <= -var]
    es   = float(-tail.mean()) if len(tail) > 0 else var
    return var, es


# Bootstrap de Politis-Romano (1994) — stationary bootstrap

# Gera índice de bloco por Stationary Bootstrap.
def _geometric_block_length(T: int, expected_block: float) -> np.ndarray:
    rng = np.random.default_rng()
    indices = []
    while len(indices) < T:
        start = rng.integers(0, T)
        length = rng.geometric(1 / expected_block)
        for j in range(length):
            indices.append((start + j) % T)
    return np.array(indices[:T])


# Bootstrap estacionário (Politis & Romano, 1994) para IC de qualquer métrica.
def bootstrap_metric(
    returns: np.ndarray,
    metric_fn,
    n_bootstrap: int = 1_000,
    block_size: float = 20.0,
    ci: float = 0.95,
    random_state: Optional[int] = None,
) -> Tuple[float, float, float]:
    if random_state is not None:
        np.random.seed(random_state)

    T    = len(returns)
    stat = metric_fn(returns)
    boot = np.empty(n_bootstrap)

    for b in range(n_bootstrap):
        idx  = _geometric_block_length(T, block_size)
        boot[b] = metric_fn(returns[idx])

    alpha = (1 - ci) / 2
    lo    = float(np.percentile(boot, alpha * 100))
    hi    = float(np.percentile(boot, (1 - alpha) * 100))
    return float(stat), lo, hi


# IC bootstrap estacionário para Sharpe ratio anualizado.
def bootstrap_sharpe_ci(
    returns: np.ndarray,
    risk_free: float = 0.0,
    periods: int = 252,
    n_bootstrap: int = 1_000,
    block_size: float = 20.0,
    ci: float = 0.95,
    random_state: int = 42,
) -> Tuple[float, float, float]:
    fn = lambda r: sharpe_ratio(r, risk_free=risk_free, periods=periods)
    return bootstrap_metric(returns, fn, n_bootstrap, block_size, ci, random_state)


# IC bootstrap estacionário para Sortino ratio anualizado.
def bootstrap_sortino_ci(
    returns: np.ndarray,
    risk_free: float = 0.0,
    periods: int = 252,
    n_bootstrap: int = 1_000,
    block_size: float = 20.0,
    ci: float = 0.95,
    random_state: int = 42,
) -> Tuple[float, float, float]:
    fn = lambda r: sortino_ratio(r, risk_free=risk_free, periods=periods)
    return bootstrap_metric(returns, fn, n_bootstrap, block_size, ci, random_state)


# Teste de Ledoit & Wolf (2008) para Sharpe ratios

# Teste HAC de Ledoit & Wolf (2008) para H0: SR_1 = SR_2.
class LedoitWolfSharpeTest:

    # Parameters
    def __init__(self, kernel: str = "bartlett", max_lag: Optional[int] = None):
        self.kernel  = kernel
        self.max_lag = max_lag

    # Estimativa HAC de Var(sqrt(T) * mean(u)).
    def _hac_variance(self, u: np.ndarray) -> float:
        T   = len(u)
        L   = self.max_lag if self.max_lag is not None else max(1, int(T ** (1 / 3)))
        gamma_0 = float(np.mean(u ** 2))
        total   = gamma_0
        for j in range(1, L + 1):
            gamma_j = float(np.mean(u[j:] * u[:-j]))
            if self.kernel == "bartlett":
                w = 1 - j / (L + 1)
            else:
                x = j / (L + 1)
                w = 1 - 6 * x ** 2 + 6 * x ** 3 if x <= 0.5 else 2 * (1 - x) ** 3
            total += 2 * w * gamma_j
        return max(total, 1e-10)

    # Testa H0: SR(returns1) = SR(returns2) (bilateral).
    def test(
        self,
        returns1: np.ndarray,
        returns2: np.ndarray,
        risk_free: float = 0.0,
        periods: int = 252,
        name1: str = "Portfolio 1",
        name2: str = "Portfolio 2",
    ) -> Dict:
        T = min(len(returns1), len(returns2))
        r1 = returns1[-T:]
        r2 = returns2[-T:]
        rf = risk_free / periods

        sr1 = sharpe_ratio(r1, risk_free, periods)
        sr2 = sharpe_ratio(r2, risk_free, periods)

        sigma1 = float(np.std(r1, ddof=1))
        sigma2 = float(np.std(r2, ddof=1))
        u = (r1 - rf) / max(sigma1, 1e-8) - (r2 - rf) / max(sigma2, 1e-8)

        sr1_d = (np.mean(r1) - rf) / max(sigma1, 1e-8)
        sr2_d = (np.mean(r2) - rf) / max(sigma2, 1e-8)

        d = (r1 - rf) / max(sigma1, 1e-8) - (r2 - rf) / max(sigma2, 1e-8) \
            - 0.5 * sr1_d * ((r1 - rf) ** 2 / sigma1 ** 2 - 1) \
            + 0.5 * sr2_d * ((r2 - rf) ** 2 / sigma2 ** 2 - 1)

        hac_var = self._hac_variance(d)
        se      = np.sqrt(hac_var / T) * np.sqrt(periods)
        diff_sr = sr1 - sr2
        z_stat  = diff_sr / max(se, 1e-8)
        pvalue  = float(2 * (1 - norm.cdf(abs(z_stat))))

        return {
            "name1":    name1,
            "name2":    name2,
            "SR1":      round(sr1, 4),
            "SR2":      round(sr2, 4),
            "diff_SR":  round(diff_sr, 4),
            "se_diff":  round(se, 4),
            "z_stat":   round(z_stat, 4),
            "pvalue":   round(pvalue, 4),
            "sig":      _sig_stars(pvalue),
            "H0_rejected": bool(pvalue < 0.05),
            "n_obs":    T,
            "hac_lags": self.max_lag if self.max_lag else int(T ** (1 / 3)),
        }

    # Matriz de p-valores pairwise para todos os pares de estratégias.
    def matrix(
        self,
        returns_dict: Dict[str, np.ndarray],
        risk_free: float = 0.0,
        periods: int = 252,
    ) -> pd.DataFrame:
        names = list(returns_dict.keys())
        n     = len(names)
        mat   = pd.DataFrame(np.nan, index=names, columns=names)

        for i, ni in enumerate(names):
            for j, nj in enumerate(names):
                if i == j:
                    mat.loc[ni, nj] = 1.0
                    continue
                try:
                    res = self.test(
                        returns_dict[ni], returns_dict[nj],
                        risk_free=risk_free, periods=periods,
                        name1=ni, name2=nj,
                    )
                    mat.loc[ni, nj] = res["pvalue"]
                except Exception as exc:
                    logger.debug(f"LW test {ni} vs {nj}: {exc}")

        return mat


# PerformanceMetrics — classe principal

# Calcula, consolida e compara métricas de performance para múltiplas estratégias.
class PerformanceMetrics:

    def __init__(
        self,
        risk_free: float = 0.1075,
        periods: int = 252,
        n_bootstrap: int = 1_000,
        block_size: float = 20.0,
        bootstrap_ci: float = 0.95,
        random_state: int = 42,
    ):
        self.rf            = risk_free
        self.periods       = periods
        self.n_bootstrap   = n_bootstrap
        self.block_size    = block_size
        self.bootstrap_ci  = bootstrap_ci
        self.random_state  = random_state

        self._returns:  Dict[str, np.ndarray]           = {}
        self._weights:  Dict[str, Optional[pd.DataFrame]] = {}
        self._metrics:  Dict[str, Dict]                 = {}

    # ── API de adição ────────────────────────────────────────────────────────

    # Registra uma estratégia.
    def add(
        self,
        name: str,
        returns: Union[np.ndarray, pd.Series],
        weights: Optional[pd.DataFrame] = None,
    ) -> "PerformanceMetrics":
        r = np.asarray(returns, dtype=float)
        r = r[~np.isnan(r)]
        self._returns[name]  = r
        self._weights[name]  = weights
        self._metrics[name]  = self._compute(name, r, weights)
        return self

    # Integração direta com BacktestResult de backtesting_engine.py.
    def add_from_backtest_result(
        self,
        name: str,
        result,
    ) -> "PerformanceMetrics":
        returns = result.returns_series.dropna().values
        weights = getattr(result, "weights_df", None)
        return self.add(name, returns, weights)

    # ── Cálculo de métricas ──────────────────────────────────────────────────

    def _compute(
        self,
        name: str,
        r: np.ndarray,
        weights: Optional[pd.DataFrame],
    ) -> Dict:
        T = len(r)
        if T < 10:
            logger.warning(f"'{name}': série muito curta (T={T}). Pulando.")
            return {}

        sd = standard_deviation(r, annualize=True, periods=self.periods)
        mdd = max_drawdown(r)
        om  = omega_ratio(r, threshold=self.rf / self.periods)
        tr  = tail_ratio(r)

        var95, es95 = historical_var_es(r, 0.95)
        var99, es99 = historical_var_es(r, 0.99)

        tot_ret = float(np.prod(1 + r) - 1)
        skew    = float(stats.skew(r))
        kurt    = float(stats.kurtosis(r))

        ann_ret = float((1 + tot_ret) ** (self.periods / T) - 1)
        rf_daily = self.rf / self.periods
        excess_cagr = ann_ret - self.rf
        downside_dev = np.minimum(r - rf_daily, 0)
        downside_vol = float(np.sqrt(np.mean(downside_dev ** 2)) * np.sqrt(self.periods))
        sr  = excess_cagr / max(sd, 1e-8)
        so  = excess_cagr / max(downside_vol, 1e-8)
        cal = ann_ret / max(abs(mdd), 1e-8)

        sr_est, sr_lo, sr_hi = bootstrap_sharpe_ci(
            r, self.rf, self.periods,
            self.n_bootstrap, self.block_size, self.bootstrap_ci, self.random_state,
        )
        so_est, so_lo, so_hi = bootstrap_sortino_ci(
            r, self.rf, self.periods,
            self.n_bootstrap, self.block_size, self.bootstrap_ci, self.random_state,
        )

        metrics: Dict = {
            "n_obs":        T,
            "annual_return": round(ann_ret, 4),
            "total_return":  round(tot_ret, 4),
            "annual_vol":    round(sd, 4),
            "sharpe":        round(sr, 4),
            "sharpe_arith_lo":     round(sr_lo, 4),
            "sharpe_arith_hi":     round(sr_hi, 4),
            "sortino":       round(so, 4),
            "sortino_arith_lo":    round(so_lo, 4),
            "sortino_arith_hi":    round(so_hi, 4),
            "max_drawdown":  round(mdd, 4),
            "calmar":        round(cal, 4),
            "omega":         round(om, 4),
            "tail_ratio":    round(tr, 4),
            "skewness":      round(skew, 4),
            "excess_kurtosis": round(kurt, 4),
            "hist_var_95":   round(var95, 4),
            "hist_es_95":    round(es95, 4),
            "hist_var_99":   round(var99, 4),
            "hist_es_99":    round(es99, 4),
        }

        if weights is not None and len(weights) > 1:
            metrics["avg_turnover"] = round(average_turnover(weights), 4)
            metrics["sspw"]         = round(sspw(weights), 6)
        else:
            metrics["avg_turnover"] = np.nan
            metrics["sspw"]         = np.nan

        return metrics

    # ── Tabelas de saída ─────────────────────────────────────────────────────

    # DataFrame de métricas (estratégia × métrica).
    def table(
        self,
        metrics: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        if not self._metrics:
            raise RuntimeError("Adicione estratégias com add() primeiro.")

        default = [
            "annual_return", "annual_vol", "sharpe", "sortino",
            "max_drawdown", "calmar", "omega", "hist_var_99", "hist_es_99",
            "avg_turnover", "sspw", "skewness", "excess_kurtosis", "n_obs",
        ]
        cols = metrics or default

        rows = {}
        for name, m in self._metrics.items():
            rows[name] = {c: m.get(c, np.nan) for c in cols}

        return pd.DataFrame(rows).T[cols]

    # DataFrame com estimativas pontuais e ICs de Sharpe e Sortino.
    def bootstrap_ci_table(self) -> pd.DataFrame:
        rows = {}
        for name, m in self._metrics.items():
            rows[name] = {
                "sharpe":            m.get("sharpe", np.nan),
                "sharpe_arith_lo":   m.get("sharpe_arith_lo", np.nan),
                "sharpe_arith_hi":   m.get("sharpe_arith_hi", np.nan),
                "sortino":           m.get("sortino", np.nan),
                "sortino_arith_lo":  m.get("sortino_arith_lo", np.nan),
                "sortino_arith_hi":  m.get("sortino_arith_hi", np.nan),
            }
        return pd.DataFrame(rows).T

    # ── Testes estatísticos ──────────────────────────────────────────────────

    # Matriz de p-valores Ledoit-Wolf (2008) pairwise para todos os pares.
    def ledoit_wolf_matrix(self, kernel: str = "bartlett") -> pd.DataFrame:
        if len(self._returns) < 2:
            raise RuntimeError("São necessárias ao menos 2 estratégias.")
        lw = LedoitWolfSharpeTest(kernel=kernel)
        return lw.matrix(self._returns, risk_free=self.rf, periods=self.periods)

    # Teste LW para um par específico de estratégias.
    def ledoit_wolf_pairwise(
        self,
        name1: str,
        name2: str,
        kernel: str = "bartlett",
    ) -> Dict:
        lw = LedoitWolfSharpeTest(kernel=kernel)
        return lw.test(
            self._returns[name1],
            self._returns[name2],
            risk_free=self.rf,
            periods=self.periods,
            name1=name1,
            name2=name2,
        )

    # Testa cada estratégia contra o benchmark (Ledoit-Wolf).
    def significance_vs_benchmark(
        self,
        benchmark_name: str,
        kernel: str = "bartlett",
    ) -> pd.DataFrame:
        if benchmark_name not in self._returns:
            raise KeyError(f"Benchmark '{benchmark_name}' não encontrado.")

        lw   = LedoitWolfSharpeTest(kernel=kernel)
        rows = []
        for name in self._returns:
            if name == benchmark_name:
                continue
            res = lw.test(
                self._returns[name],
                self._returns[benchmark_name],
                risk_free=self.rf, periods=self.periods,
                name1=name, name2=benchmark_name,
            )
            rows.append(res)

        return pd.DataFrame(rows)[[
            "name1", "SR1", "SR2", "diff_SR", "z_stat", "pvalue", "sig", "H0_rejected"
        ]].rename(columns={"name1": "estrategia", "SR1": "SR_estrategia", "SR2": "SR_benchmark"})

    # ── Exportação LaTeX ─────────────────────────────────────────────────────

    # Exporta tabela LaTeX completa.
    def to_latex(
        self,
        path: str,
        caption: str = "Métricas de performance dos portfólios (período OOS).",
        label: str = "tab:performance",
        metrics: Optional[List[str]] = None,
        include_bootstrap: bool = True,
        include_lw: bool = True,
    ) -> str:
        df = self.table(metrics)

        col_map = {
            "annual_return": "Ret. a.a. (\\%)",
            "annual_vol":    "Vol. a.a. (\\%)",
            "sharpe":        "Sharpe",
            "sortino":       "Sortino",
            "max_drawdown":  "Max DD (\\%)",
            "calmar":        "Calmar",
            "omega":         "Omega",
            "hist_var_99":   "VaR 99\\% (\\%)",
            "hist_es_99":    "ES 99\\% (\\%)",
            "avg_turnover":  "Turnover",
            "sspw":          "SSPW",
            "skewness":      "Skew",
            "excess_kurtosis": "Ex. Kurt.",
            "n_obs":         "$T$",
        }

        pct_cols = {"annual_return", "annual_vol", "max_drawdown", "hist_var_99", "hist_es_99"}
        df_tex = df.copy()
        for c in pct_cols:
            if c in df_tex.columns:
                df_tex[c] = (df_tex[c] * 100).round(2)

        df_tex = df_tex.rename(columns={c: v for c, v in col_map.items() if c in df_tex.columns})
        df_tex.index.name = "Estratégia"

        lines = [
            "\\begin{table}[htbp]",
            "  \\centering",
            f"  \\caption{{{caption}}}",
            f"  \\label{{{label}}}",
            "  \\footnotesize",
            "  \\begin{tabular}{l" + "r" * len(df_tex.columns) + "}",
            "  \\toprule",
        ]
        header = "  Estratégia & " + " & ".join(df_tex.columns) + " \\\\"
        lines.append(header)
        lines.append("  \\midrule")
        for idx, row in df_tex.iterrows():
            vals = []
            for v in row:
                if isinstance(v, float):
                    vals.append(f"{v:.4f}" if not np.isnan(v) else "--")
                else:
                    vals.append(str(v))
            lines.append(f"  {idx} & " + " & ".join(vals) + " \\\\")
        lines.append("  \\bottomrule")

        if include_bootstrap:
            lines.append("  \\midrule")
            lines.append("  \\multicolumn{" + str(len(df_tex.columns) + 1) + "}{l}"
                         "{\\textit{Bootstrap IC " + f"{int(self.bootstrap_ci*100)}\\%" + " (Sharpe)}}")
            lines.append("  \\\\")
            ci_df = self.bootstrap_ci_table()
            for idx, row in ci_df.iterrows():
                lo = row.get("sharpe_lo", np.nan)
                hi = row.get("sharpe_hi", np.nan)
                ic = f"[{lo:.3f}, {hi:.3f}]" if not (np.isnan(lo) or np.isnan(hi)) else "--"
                lines.append(f"  {idx} & \\multicolumn{{2}}{{c}}{{{ic}}} \\\\")

        lines.append("  \\end{tabular}")
        lines.append("\\end{table}")

        tex = "\n".join(lines)
        from pathlib import Path
        Path(path).write_text(tex, encoding="utf-8")
        logger.info(f"Tabela LaTeX salva em {path}")

        if include_lw and len(self._returns) >= 2:
            lw_path = path.replace(".tex", "_lw_pvalues.tex")
            lw_mat  = self.ledoit_wolf_matrix()
            lw_lines = [
                "\\begin{table}[htbp]",
                "  \\centering",
                "  \\caption{P-valores Ledoit-Wolf (2008) — H$_0$: SR$_i$ = SR$_j$.}",
                "  \\label{tab:lw_pvalues}",
                "  \\footnotesize",
                "  \\begin{tabular}{l" + "r" * len(lw_mat.columns) + "}",
                "  \\toprule",
                "  & " + " & ".join(lw_mat.columns) + " \\\\",
                "  \\midrule",
            ]
            for idx, row in lw_mat.iterrows():
                vals = []
                for v in row:
                    if np.isnan(v):
                        vals.append("--")
                    elif v == 1.0:
                        vals.append("1.000")
                    else:
                        stars = _sig_stars(v)
                        vals.append(f"{v:.3f}{stars}")
                lw_lines.append(f"  {idx} & " + " & ".join(vals) + " \\\\")
            lw_lines += ["  \\bottomrule", "  \\end{tabular}", "\\end{table}"]
            Path(lw_path).write_text("\n".join(lw_lines), encoding="utf-8")
            logger.info(f"Matriz LW salva em {lw_path}")

        return tex

    # ── repr ─────────────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        return (
            f"PerformanceMetrics(strategies={list(self._metrics.keys())}, "
            f"rf={self.rf:.4f})"
        )


# Wrapper de alto nível

# Calcula métricas e matriz LW para um conjunto de estratégias.
def compute_all_metrics(
    returns_dict: Dict[str, Union[np.ndarray, pd.Series]],
    weights_dict: Optional[Dict[str, pd.DataFrame]] = None,
    risk_free: float = 0.1075,
    periods: int = 252,
    n_bootstrap: int = 1_000,
    block_size: float = 20.0,
) -> Tuple["PerformanceMetrics", pd.DataFrame, pd.DataFrame]:
    pm = PerformanceMetrics(
        risk_free=risk_free,
        periods=periods,
        n_bootstrap=n_bootstrap,
        block_size=block_size,
    )
    weights_dict = weights_dict or {}
    for name, r in returns_dict.items():
        pm.add(name, r, weights_dict.get(name))

    table = pm.table()
    lw    = pm.ledoit_wolf_matrix() if len(returns_dict) >= 2 else pd.DataFrame()
    return pm, table, lw


if __name__ == "__main__":
    import argparse as _argparse
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    _SCRIPT_DIR = Path(__file__).resolve().parent
    _PROJECT_ROOT_CANDIDATES = [
        _SCRIPT_DIR.parent.parent,
        _SCRIPT_DIR.parent,
        Path.cwd().parent.parent,
        Path.cwd().parent,
        Path.cwd(),
    ]
    _STOCKS_SUBDIR = "raw_b3_ibrx50"

    def _valid_root(p: Path) -> bool:
        stocks = p / "data" / _STOCKS_SUBDIR
        ext    = p / "data" / "external"
        return (stocks.exists() and any(stocks.glob("*.xlsx")) and
                (ext / "selic_daily.csv").exists())

    _diag_lines = [
        f"[DIAG] __file__       : {Path(__file__).resolve()}",
        f"[DIAG] cwd            : {Path.cwd()}",
        f"[DIAG] Candidatos de raiz testados:",
    ]
    _AUTO_DATA_DIR = None
    for _candidate in _PROJECT_ROOT_CANDIDATES:
        _stocks_dir = _candidate / "data" / _STOCKS_SUBDIR
        _n_xlsx     = len(list(_stocks_dir.glob("*.xlsx"))) if _stocks_dir.exists() else 0
        _raw_ok     = _n_xlsx > 0
        _ext_path   = _candidate / "data" / "external"
        _selic_ok   = (_ext_path / "selic_daily.csv").exists()
        if not _selic_ok and _ext_path.exists():
            _found = list(_ext_path.iterdir())
            _diag_lines.append(f"    [data/external existe mas selic_daily.csv nao achou. Arquivos: {[f.name for f in _found]}]")
        _status   = "OK" if (_raw_ok and _selic_ok) else f"FALHOU (n_xlsx={_n_xlsx}, selic={_selic_ok})"
        _diag_lines.append(f"  {_candidate}  ->  {_status}")
        if _raw_ok and _selic_ok and _AUTO_DATA_DIR is None:
            _AUTO_DATA_DIR = str(_candidate)
    _diag_lines.append(f"[DIAG] Raiz escolhida  : {_AUTO_DATA_DIR or 'NENHUMA (vai para demo)'}")
    print("\n".join(_diag_lines))

    _cli = _argparse.ArgumentParser(
        description="Metricas de performance -- dados reais ou demo sintetico"
    )
    _cli.add_argument(
        "--results-dir", default=None, type=str,
        help="Diretorio com returns_panel.csv de run_comparison.py "
             "(ex: C:/garch_copula/results/comparison). "
             "Se omitido, tenta --data-dir; se ambos ausentes, roda demo sintetico.",
    )
    _cli.add_argument(
        "--data-dir", default=None, type=str,
        help="Raiz do projeto (ex: C:/garch_copula). "
             "Espera data/raw_b3_ibrx50/*.xlsx (universo de 81 acoes do "
             "IBrX-50, descoberto dinamicamente) e "
             "data/external/selic_daily.csv. "
             "Constroi portfolios EW, RP, MV e CVaR-EVT automaticamente.",
    )
    _cli.add_argument("--window",      default=252,   type=int,
                      help="Janela de estimacao em dias uteis (default: 252).")
    _cli.add_argument("--rebal",       default=21,    type=int,
                      help="Frequencia de rebalanceamento em dias uteis (default: 21).")
    _cli.add_argument("--risk-free",   default=0.1075, type=float)
    _cli.add_argument("--n-bootstrap", default=1_000,  type=int)
    _cli.add_argument("--benchmark",   default="Equal Weight", type=str,
                      help="Estrategia benchmark no teste de Sharpe.")
    _cli_args = _cli.parse_args()
    rf = _cli_args.risk_free

    _loaded_real = False

    if _cli_args.data_dir is None and _cli_args.results_dir is None and _AUTO_DATA_DIR:
        print(f"[Auto] Raiz detectada: {_AUTO_DATA_DIR}")
        _cli_args.data_dir = _AUTO_DATA_DIR

    if not _loaded_real and _cli_args.data_dir is not None:
        _droot      = Path(_cli_args.data_dir)
        _stocks_dir = _droot / "data" / _STOCKS_SUBDIR
        _ext_dir    = _droot / "data" / "external"

        _SELIC_FILE = _ext_dir / "selic_daily.csv"
        _n_xlsx     = len(list(_stocks_dir.glob("*.xlsx"))) if _stocks_dir.exists() else 0

        if _n_xlsx == 0 or not _SELIC_FILE.exists():
            print(f"[AVISO] Dados nao encontrados em --data-dir:\n  "
                  f"  {_stocks_dir} (xlsx encontrados: {_n_xlsx})\n"
                  f"  {_SELIC_FILE} (existe: {_SELIC_FILE.exists()})")
            print("Tentando --results-dir ou demo sintetico.")
        else:
            print(f"\nCarregando universo de acoes (Economatica): {_stocks_dir}")

            import sys as _sys_pm, pathlib as _pathlib_pm
            _data_mod_dir = str(_pathlib_pm.Path(__file__).resolve().parent.parent / "data")
            if _data_mod_dir not in _sys_pm.path:
                _sys_pm.path.insert(0, _data_mod_dir)
            from split_adjustment import build_adjusted_panel

            _raw_panel, _adj_panel, _events_df, _unconf_df = build_adjusted_panel(str(_stocks_dir))
            print(f"  {_adj_panel.shape[1]} ativos carregados "
                  f"(correcoes de split: {len(_events_df)} eventos).")

            _price_df = _adj_panel.sort_index()
            _ASSETS   = list(_price_df.columns)
            _ret_df   = np.log(_price_df / _price_df.shift(1)).iloc[1:]

            _selic_df    = pd.read_csv(_SELIC_FILE, parse_dates=["Date"], index_col="Date")
            _selic_col   = next((c for c in ["Daily_Return", "Rate_Daily_Selic", "Rate"]
                                 if c in _selic_df.columns), _selic_df.columns[0])
            _selic_daily = _selic_df[_selic_col].sort_index()

            _selic_period = _selic_daily.reindex(_ret_df.index).ffill()
            rf = float(((1 + _selic_period.mean()) ** 252 - 1))
            print(f"  Ativos      : {len(_ASSETS)} ({_ASSETS[:5]}...)")
            print(f"  Periodo     : {_ret_df.index[0].date()} -> {_ret_df.index[-1].date()}")
            print(f"  Obs         : {len(_ret_df)}")
            print(f"  Selic (rf)  : {rf:.4%}")

            import sys as _sys_du_pm, pathlib as _pathlib_du_pm
            _utils_dir_pm = str(_pathlib_du_pm.Path(__file__).resolve().parent.parent / "utils")
            if _utils_dir_pm not in _sys_du_pm.path:
                _sys_du_pm.path.insert(0, _utils_dir_pm)
            from dynamic_universe import eligible_assets, expand_weights, nan_safe_portfolio_return
            _MIN_COVERAGE = 0.95

            _WINDOW = _cli_args.window
            _REBAL  = _cli_args.rebal
            _N      = len(_ASSETS)
            _dates  = _ret_df.index
            _rebal_dates = _dates[_WINDOW::_REBAL]

            _ew_w, _rp_w, _mv_w, _cv_w = {}, {}, {}, {}
            print(f"  Calculando pesos ({len(_rebal_dates)} rebalanceamentos, "
                  f"universo dinamico min_coverage={_MIN_COVERAGE})...")

            for _rd in _rebal_dates:
                _idx    = _dates.get_loc(_rd)
                _window = _ret_df.iloc[_idx - _WINDOW : _idx]
                _names_t = eligible_assets(_window, min_coverage=_MIN_COVERAGE)
                if len(_names_t) == 0:
                    continue
                _hist = _window[_names_t].values
                _cov  = np.cov(_hist.T)
                _n_t  = len(_names_t)
                _w0   = np.ones(_n_t) / _n_t
                _bnds = [(0.01, 1.0)] * _n_t
                _cons = {"type": "eq", "fun": lambda w: w.sum() - 1}

                _ew_w[_rd] = expand_weights(_w0, _names_t, _ASSETS)

                # Risk Parity
                def _rp_obj(w, cov=_cov):
                    w = np.maximum(w, 1e-8)
                    vol = np.sqrt(w @ cov @ w)
                    rc  = w * (cov @ w) / vol
                    return np.sum((rc[:, None] - rc[None, :]) ** 2)
                _res = minimize(_rp_obj, _w0, method="SLSQP", bounds=_bnds, constraints=_cons)
                _w_sub = _res.x if _res.success else _w0.copy()
                _rp_w[_rd] = expand_weights(_w_sub, _names_t, _ASSETS)

                # Min Variance
                def _mv_obj(w, cov=_cov):
                    return w @ cov @ w
                _res = minimize(_mv_obj, _w0, method="SLSQP", bounds=_bnds, constraints=_cons)
                _w_sub = _res.x if _res.success else _w0.copy()
                _mv_w[_rd] = expand_weights(_w_sub, _names_t, _ASSETS)

                # CVaR (minimiza CVaR 95% historico)
                def _cv_obj(w, hist=_hist):
                    pr  = hist @ w
                    var = np.percentile(pr, 5)
                    return -pr[pr <= var].mean()
                _res = minimize(_cv_obj, _w0, method="SLSQP", bounds=_bnds, constraints=_cons)
                _w_sub = _res.x if _res.success else _w0.copy()
                _cv_w[_rd] = expand_weights(_w_sub, _names_t, _ASSETS)

            # ── Converte pesos em series de retorno e DataFrame de pesos ─────
            # Usa nan_safe_portfolio_return: pesos ja estao expandidos para
            # o universo global (81), com 0 nos ativos nao elegiveis nesse
            # rebalanceamento, entao NaN fora do periodo real de cada ativo
            # nao contamina o retorno do dia (0 * NaN = NaN em IEEE-754).
            def _build_returns(
                weights_dict: dict,
                ret_df: pd.DataFrame,
            ):
                all_d  = ret_df.index
                rets, dates_out, w_rows = [], [], {}
                rb_list = sorted(weights_dict.keys())
                for i, rb in enumerate(rb_list):
                    rb_i   = all_d.get_loc(rb)
                    nxt_i  = all_d.get_loc(rb_list[i + 1]) if i + 1 < len(rb_list) else len(all_d)
                    w      = weights_dict[rb]
                    for d, row in ret_df.iloc[rb_i:nxt_i].iterrows():
                        rets.append(nan_safe_portfolio_return(row.values, w))
                        dates_out.append(d)
                        w_rows[d] = w
                ret_s = pd.Series(rets, index=dates_out)
                w_df  = pd.DataFrame(w_rows, index=_ASSETS).T
                w_df.index = pd.to_datetime(w_df.index)
                w_df.columns = _ASSETS
                return ret_s, w_df

            _ew_ret, _ew_wdf = _build_returns(_ew_w, _ret_df)
            _rp_ret, _rp_wdf = _build_returns(_rp_w, _ret_df)
            _mv_ret, _mv_wdf = _build_returns(_mv_w, _ret_df)
            _cv_ret, _cv_wdf = _build_returns(_cv_w, _ret_df)

            pm = PerformanceMetrics(
                risk_free    = rf,
                n_bootstrap  = _cli_args.n_bootstrap,
                block_size   = 20.0,
                bootstrap_ci = 0.95,
                random_state = 42,
            )
            pm.add("Equal Weight", _ew_ret, _ew_wdf)
            pm.add("Risk Parity",  _rp_ret, _rp_wdf)
            pm.add("Min Variance", _mv_ret, _mv_wdf)
            pm.add("CVaR-EVT",     _cv_ret, _cv_wdf)
            _loaded_real = True
    if _cli_args.results_dir is not None:
        _rdir = Path(_cli_args.results_dir)
        _panel_path = _rdir / "returns_panel.csv"
        if _panel_path.exists():
            print(f"\nCarregando retornos reais de: {_panel_path}")
            _panel = pd.read_csv(_panel_path, index_col=0, parse_dates=True)
            print(f"  Estrategias : {list(_panel.columns)}")
            print(f"  Periodo     : {_panel.index[0].date()} -> {_panel.index[-1].date()}")
            print(f"  Obs         : {len(_panel)}")

            _selic_path = _rdir / "selic_rate.csv"
            if _selic_path.exists():
                try:
                    rf = float(pd.read_csv(_selic_path).iloc[0, 0])
                    print(f"  Selic (rf)  : {rf:.4%}")
                except Exception:
                    pass

            pm = PerformanceMetrics(
                risk_free    = rf,
                n_bootstrap  = _cli_args.n_bootstrap,
                block_size   = 20.0,
                bootstrap_ci = 0.95,
                random_state = 42,
            )
            for _col in _panel.columns:
                pm.add(_col, _panel[_col].dropna().values)

            _loaded_real = True
        else:
            print(f"[AVISO] returns_panel.csv nao encontrado em {_rdir}. "
                  "Rodando demo sintetico.")

    if not _loaded_real:
        print("\n[Demo sintetico -- passe --results-dir para usar dados reais]")
        np.random.seed(42)
        T  = 500
        rf = _cli_args.risk_free
        strategies = {
            "Equal Weight": np.random.normal(0.0003, 0.012, T),
            "Risk Parity":  np.random.normal(0.0004, 0.010, T),
            "Min Variance": np.random.normal(0.00035, 0.009, T),
            "CVaR-EVT":     np.random.normal(0.0005, 0.011, T),
        }
        weights_demo = {
            name: pd.DataFrame(
                np.random.dirichlet(np.ones(6), T),
                columns=[f"A{i}" for i in range(6)],
            )
            for name in strategies
        }
        pm = PerformanceMetrics(risk_free=rf, n_bootstrap=500)
        for name, r in strategies.items():
            pm.add(name, r, weights_demo[name])

    print("\n── Metricas de Performance ──")
    print(pm.table().to_string())

    print("\n── ICs Bootstrap (Sharpe/Sortino) ──")
    print(pm.bootstrap_ci_table().to_string())

    print("\n── Matriz Ledoit-Wolf (p-valores) ──")
    print(pm.ledoit_wolf_matrix().round(3).to_string())

    _benchmark = _cli_args.benchmark
    print(f"\n── Teste vs {_benchmark} ──")
    print(pm.significance_vs_benchmark("Equal Weight").to_string(index=False))
