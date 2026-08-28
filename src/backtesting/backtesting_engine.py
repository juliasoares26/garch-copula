
from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import chi2, norm

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


# Dataclasses de resultado

# Resultado de um único período de rebalanceamento no walk-forward.
@dataclass
class PeriodResult:
    date: pd.Timestamp
    weights: np.ndarray
    realized_return: float
    var_95: float
    var_99: float
    es_95: float
    es_99: float
    violation_95: bool
    violation_99: bool
    copula_ll: float = np.nan
    regime: int = -1
    n_train_obs: int = 0
    metadata: Dict = field(default_factory=dict)


# Resultado completo do backtesting walk-forward.
@dataclass
class BacktestResult:
    strategy_name: str
    period_results: List[PeriodResult]
    returns_series: pd.Series
    weights_df: pd.DataFrame
    var_95_series: pd.Series
    var_99_series: pd.Series
    es_95_series: pd.Series
    es_99_series: pd.Series
    performance_metrics: Dict
    var_backtest_95: Dict
    var_backtest_99: Dict
    regime_series: Optional[pd.Series] = None

    @property
    def total_return(self) -> float:
        return float((1 + self.returns_series).prod() - 1)

    @property
    def annualized_return(self) -> float:
        T = len(self.returns_series)
        return float((1 + self.total_return) ** (252 / T) - 1)

    @property
    def annualized_vol(self) -> float:
        return float(self.returns_series.std() * np.sqrt(252))

    @property
    def sharpe(self) -> float:
        rf = self.performance_metrics.get("risk_free_rate", 0.0)
        excess = self.annualized_return - rf
        return excess / max(self.annualized_vol, 1e-8)


# VaR Backtesting (Kupiec + Christoffersen)

# Implementa os testes de backtesting de VaR mais utilizados na literatura:
class VaRBacktestEngine:

    # alpha: nível de significância dos testes (não do VaR).
    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha

    # Executa bateria completa de testes para uma série de VaR.
    def run(
        self,
        realized_returns: np.ndarray,
        var_forecasts: np.ndarray,
        confidence_level: float = 0.99,
        name: str = "",
    ) -> Dict:
        T = len(realized_returns)
        alpha_var = 1 - confidence_level

        hits = (realized_returns < -var_forecasts).astype(int)
        n_hits = int(hits.sum())
        pct_hits = n_hits / T

        result = {
            "name": name,
            "confidence_level": confidence_level,
            "T": T,
            "n_violations": n_hits,
            "violation_rate": round(pct_hits, 5),
            "expected_rate": round(alpha_var, 5),
        }

        kupiec = self._kupiec_pof(hits, T, alpha_var)
        result.update({f"kupiec_{k}": v for k, v in kupiec.items()})

        christ_ind = self._christoffersen_independence(hits)
        result.update({f"cc_ind_{k}": v for k, v in christ_ind.items()})

        lr_cc = kupiec["lr_stat"] + christ_ind["lr_stat"]
        p_cc  = 1 - chi2.cdf(lr_cc, df=2) if np.isfinite(lr_cc) else np.nan
        result["cc_lr_stat"]   = round(float(lr_cc), 4)
        result["cc_pvalue"]    = round(float(p_cc), 4) if np.isfinite(p_cc) else np.nan
        result["cc_adequate"]  = float(p_cc) > self.alpha if np.isfinite(p_cc) else None

        result["lopez_loss"] = self._lopez_loss(realized_returns, var_forecasts)

        dq = self._dynamic_quantile_test(hits, var_forecasts, lags=4)
        result.update({f"dq_{k}": v for k, v in dq.items()})

        result["model_adequate"] = (
            result["kupiec_pvalue"] > self.alpha
            and result["cc_ind_pvalue"] > self.alpha
            and result["cc_adequate"]
        )

        logger.info(
            f"VaR Backtest [{name}] {confidence_level:.0%}: "
            f"violations={n_hits}/{T} ({pct_hits:.2%}) | "
            f"Kupiec p={result['kupiec_pvalue']:.3f} | "
            f"CC p={result['cc_pvalue']:.3f} | "
            f"adequate={'✓' if result['model_adequate'] else '✗'}"
        )
        return result

    # ── Kupiec (1995) ─────────────────────────────────────────────────────────

    def _kupiec_pof(self, hits: np.ndarray, T: int, alpha_var: float) -> Dict:
        x = int(hits.sum())
        p = x / T if T > 0 else 0.0
        if x == 0:
            lr = -2 * T * np.log(1 - alpha_var)
        elif x == T:
            lr = -2 * T * np.log(alpha_var)
        else:
            ll_h0 = x * np.log(alpha_var) + (T - x) * np.log(1 - alpha_var)
            ll_h1 = x * np.log(p) + (T - x) * np.log(1 - p)
            lr = -2 * (ll_h0 - ll_h1)
        pval = 1 - chi2.cdf(lr, df=1) if np.isfinite(lr) else np.nan
        return {
            "lr_stat":  round(float(lr), 4),
            "pvalue":   round(float(pval), 4) if np.isfinite(pval) else np.nan,
            "adequate": float(pval) > self.alpha if np.isfinite(pval) else None,
        }

    # ── Christoffersen (1998) ─────────────────────────────────────────────────

    def _christoffersen_independence(self, hits: np.ndarray) -> Dict:
        n00 = n01 = n10 = n11 = 0
        for t in range(1, len(hits)):
            if hits[t-1] == 0 and hits[t] == 0: n00 += 1
            elif hits[t-1] == 0 and hits[t] == 1: n01 += 1
            elif hits[t-1] == 1 and hits[t] == 0: n10 += 1
            else: n11 += 1
        pi_01 = n01 / max(n00 + n01, 1)
        pi_11 = n11 / max(n10 + n11, 1)
        pi    = (n01 + n11) / max(n00 + n01 + n10 + n11, 1)
        try:
            ll_ind = (
                (n00 + n10) * np.log(max(1 - pi, 1e-10))
                + (n01 + n11) * np.log(max(pi, 1e-10))
            )
            ll_dep = (
                n00 * np.log(max(1 - pi_01, 1e-10)) + n01 * np.log(max(pi_01, 1e-10))
                + n10 * np.log(max(1 - pi_11, 1e-10)) + n11 * np.log(max(pi_11, 1e-10))
            )
            lr  = -2 * (ll_ind - ll_dep)
            pval = 1 - chi2.cdf(lr, df=1)
        except Exception:
            lr, pval = np.nan, np.nan
        return {
            "lr_stat":       round(float(lr), 4) if np.isfinite(lr) else np.nan,
            "pvalue":        round(float(pval), 4) if np.isfinite(pval) else np.nan,
            "adequate":      float(pval) > self.alpha if np.isfinite(pval) else None,
            "pi_01":         round(pi_01, 4),
            "pi_11":         round(pi_11, 4),
            "clustering":    bool(abs(pi_11 - pi_01) > 0.05),
        }

    # ── Lopez (1998) ──────────────────────────────────────────────────────────

    def _lopez_loss(
        self, returns: np.ndarray, var_forecasts: np.ndarray
    ) -> float:
        losses = []
        for r, v in zip(returns, var_forecasts):
            if r < -v:
                losses.append(1 + (r + v) ** 2)
            else:
                losses.append(0.0)
        return round(float(np.mean(losses)), 6)

    # ── Dynamic Quantile Test ─────────────────────────────────────────────────

    # Engle & Manganelli (2004) — simplificado via OLS.
    def _dynamic_quantile_test(
        self, hits: np.ndarray, var_forecasts: np.ndarray, lags: int = 4
    ) -> Dict:
        T = len(hits)
        alpha_var = hits.mean()
        hit_centered = hits - alpha_var

        if T < lags + 5:
            return {"stat": np.nan, "pvalue": np.nan, "adequate": None}

        X_list = [np.ones(T - lags)]
        X_list.append(var_forecasts[lags:])
        for lag in range(1, lags + 1):
            X_list.append(hit_centered[lags - lag: T - lag])
        X = np.column_stack(X_list)
        y = hit_centered[lags:]

        try:
            beta, res, _, _ = np.linalg.lstsq(X, y, rcond=None)
            y_hat = X @ beta
            ssr_r = float(((y - y.mean()) ** 2).sum())
            ssr_u = float((y - y_hat).T @ (y - y_hat))
            k = X.shape[1]
            n = len(y)
            F = ((ssr_r - ssr_u) / k) / ((ssr_u / (n - k)) + 1e-10)
            pval = 1 - stats.f.cdf(F, k, n - k)
            return {
                "stat":     round(float(F), 4),
                "pvalue":   round(float(pval), 4),
                "adequate": float(pval) > self.alpha,
            }
        except Exception:
            return {"stat": np.nan, "pvalue": np.nan, "adequate": None}


# Performance metrics

# Calcula métricas de performance de portfólio padrão da indústria e academia.
class PerformanceEngine:

    def __init__(self, risk_free_rate: float = 0.0, periods_per_year: int = 252):
        self.rf  = risk_free_rate / periods_per_year
        self.n   = periods_per_year

    # Calcula todas as métricas para uma série de retornos.
    def compute(
        self,
        returns: pd.Series,
        benchmark_returns: Optional[pd.Series] = None,
        weights_df: Optional[pd.DataFrame] = None,
    ) -> Dict:
        r = returns.dropna().values
        T = len(r)
        if T < 5:
            return {}

        result = {}

        ann_ret = float(np.mean(r) * self.n)
        ann_vol = float(np.std(r, ddof=1) * np.sqrt(self.n))
        result["annual_return"]   = round(ann_ret, 6)
        result["annual_vol"]      = round(ann_vol, 6)
        result["total_return"]    = round(float((1 + r).prod() - 1), 6)

        excess = np.mean(r) - self.rf
        result["sharpe_ratio"]  = round(excess * self.n / max(ann_vol, 1e-8), 4)

        downside = r[r < self.rf] - self.rf
        downside_vol = np.sqrt(np.mean(downside ** 2)) * np.sqrt(self.n) if len(downside) > 0 else 1e-8
        result["sortino_ratio"] = round(excess * self.n / max(downside_vol, 1e-8), 4)

        cum = (1 + r).cumprod()
        roll_max = np.maximum.accumulate(cum)
        drawdown = (cum - roll_max) / (roll_max + 1e-10)
        max_dd = float(drawdown.min())
        result["max_drawdown"]    = round(max_dd, 6)
        result["avg_drawdown"]    = round(float(drawdown.mean()), 6)
        result["calmar_ratio"]    = round(ann_ret / max(abs(max_dd), 1e-8), 4)

        for cl, label in [(0.95, "95"), (0.99, "99")]:
            var = float(-np.quantile(r, 1 - cl))
            tail = r[r <= -var]
            es = float(-tail.mean()) if len(tail) > 0 else var
            result[f"hist_var_{label}"]  = round(var, 6)
            result[f"hist_es_{label}"]   = round(es, 6)

        result["skewness"]       = round(float(stats.skew(r)), 4)
        result["excess_kurtosis"] = round(float(stats.kurtosis(r)), 4)

        result["hit_rate"]       = round(float(np.mean(r > 0)), 4)

        gains  = r[r > 0].sum()
        losses = abs(r[r < 0].sum())
        result["profit_factor"]  = round(float(gains / max(losses, 1e-10)), 4)

        thr = self.rf
        omega_num = np.sum(np.maximum(r - thr, 0))
        omega_den = np.sum(np.maximum(thr - r, 0))
        result["omega_ratio"]    = round(float(omega_num / max(omega_den, 1e-10)), 4)

        p95 = abs(np.percentile(r, 95))
        p05 = abs(np.percentile(r, 5))
        result["tail_ratio"]     = round(float(p95 / max(p05, 1e-10)), 4)

        if benchmark_returns is not None:
            bm = benchmark_returns.reindex(returns.index).dropna().values
            n_common = min(len(r), len(bm))
            r2, bm2  = r[-n_common:], bm[-n_common:]
            excess_r = r2 - bm2
            te       = float(np.std(excess_r, ddof=1) * np.sqrt(self.n))
            ir       = float(np.mean(excess_r) * self.n / max(te, 1e-8))
            result["tracking_error"]    = round(te, 6)
            result["information_ratio"] = round(ir, 4)
            result["beta"]              = round(float(np.cov(r2, bm2)[0, 1] / max(np.var(bm2), 1e-10)), 4)
            result["alpha_annual"]      = round(float((np.mean(r2) - result["beta"] * np.mean(bm2)) * self.n), 6)

        if weights_df is not None and len(weights_df) > 1:
            turnover = float(np.abs(weights_df.diff().dropna()).sum(axis=1).mean() / 2)
            result["avg_daily_turnover"] = round(turnover, 6)
            result["annual_turnover"]    = round(turnover * self.n, 4)

        return result


# Motor principal: Walk-Forward Backtest

# Backtesting walk-forward (expanding ou rolling window) para
class WalkForwardBacktest:

    def __init__(
        self,
        estimation_window: int = 252,
        rebalancing_frequency: int = 21,
        min_estimation_window: int = 126,
        window_type: str = "rolling",
        confidence_levels: List[float] = None,
        optimizer_fn: Optional[Callable] = None,
        risk_fn: Optional[Callable] = None,
        risk_free_rate: float = 0.1075,
        verbose: bool = True,
    ):
        self.estimation_window     = estimation_window
        self.rebalancing_frequency = rebalancing_frequency
        self.min_estimation_window = min_estimation_window
        self.window_type           = window_type
        self.confidence_levels     = confidence_levels or [0.95, 0.99]
        self.optimizer_fn          = optimizer_fn
        self.risk_fn               = risk_fn
        self.rf                    = risk_free_rate
        self.verbose               = verbose

        self._results: List[PeriodResult] = []

    # Executa o backtesting walk-forward completo.
    def run(
        self,
        returns_df: pd.DataFrame,
        strategy_name: str = "copula_evt",
        initial_weights: Optional[np.ndarray] = None,
        benchmark_returns: Optional[pd.Series] = None,
    ) -> BacktestResult:
        T, d = returns_df.shape
        dates = returns_df.index
        n_assets = d

        if initial_weights is None:
            initial_weights = np.ones(d) / d

        w_current = initial_weights.copy()
        self._results = []

        rebal_indices = list(range(
            self.estimation_window,
            T - 1,
            self.rebalancing_frequency
        ))

        logger.info(
            f"WalkForward [{strategy_name}]  "
            f"T={T}  window={self.window_type}  "
            f"est_window={self.estimation_window}  "
            f"rebal_freq={self.rebalancing_frequency}  "
            f"n_rebal={len(rebal_indices)}"
        )

        for t_idx in rebal_indices:
            if self.window_type == "rolling":
                t_start = t_idx - self.estimation_window
            else:
                t_start = 0
            t_end = t_idx

            train_df = returns_df.iloc[t_start:t_end]

            try:
                if self.optimizer_fn is not None and len(train_df) >= self.min_estimation_window:
                    w_new = self.optimizer_fn(train_df)
                    if w_new is not None and len(w_new) == d:
                        w_current = np.clip(w_new, 0, 1)
                        w_current /= w_current.sum()
            except Exception as e:
                logger.warning(f"  t={t_idx}: optimizer falhou ({e}), mantendo pesos")

            t_oos_end = min(t_idx + self.rebalancing_frequency, T)
            oos_df    = returns_df.iloc[t_idx:t_oos_end]

            for t_oos in range(len(oos_df)):
                date = oos_df.index[t_oos]
                ret_t = float(oos_df.iloc[t_oos].values @ w_current)

                risk_metrics = {"var_95": np.nan, "var_99": np.nan,
                                "es_95": np.nan,  "es_99": np.nan}
                if self.risk_fn is not None:
                    try:
                        risk_metrics = self.risk_fn(train_df, w_current)
                    except Exception as e:
                        logger.debug(f"  risk_fn falhou em {date}: {e}")

                period = PeriodResult(
                    date=date,
                    weights=w_current.copy(),
                    realized_return=ret_t,
                    var_95=float(risk_metrics.get("var_95", np.nan)),
                    var_99=float(risk_metrics.get("var_99", np.nan)),
                    es_95=float(risk_metrics.get("es_95", np.nan)),
                    es_99=float(risk_metrics.get("es_99", np.nan)),
                    violation_95=(ret_t < -risk_metrics.get("var_95", np.inf)
                                  if np.isfinite(risk_metrics.get("var_95", np.nan)) else False),
                    violation_99=(ret_t < -risk_metrics.get("var_99", np.inf)
                                  if np.isfinite(risk_metrics.get("var_99", np.nan)) else False),
                    n_train_obs=len(train_df),
                )
                self._results.append(period)

        if self.verbose:
            logger.info(f"  Períodos OOS gerados: {len(self._results)}")

        return self._compile_results(strategy_name, benchmark_returns)

    # Consolida lista de PeriodResult em BacktestResult.
    def _compile_results(
        self,
        strategy_name: str,
        benchmark_returns: Optional[pd.Series],
    ) -> BacktestResult:
        if not self._results:
            raise RuntimeError("Nenhum resultado gerado. Verifique os parâmetros.")

        dates   = pd.DatetimeIndex([r.date for r in self._results])
        returns = pd.Series([r.realized_return for r in self._results], index=dates)
        var_95  = pd.Series([r.var_95  for r in self._results], index=dates)
        var_99  = pd.Series([r.var_99  for r in self._results], index=dates)
        es_95   = pd.Series([r.es_95   for r in self._results], index=dates)
        es_99   = pd.Series([r.es_99   for r in self._results], index=dates)

        n_assets = len(self._results[0].weights)
        weights_df = pd.DataFrame(
            [r.weights for r in self._results],
            index=dates,
        )

        var_engine = VaRBacktestEngine(alpha=0.05)
        r_arr = returns.dropna().values

        def _backtest_var(var_series, cl, name):
            v = var_series.dropna()
            common = returns.index.intersection(v.index)
            if len(common) < 10:
                return {}
            return var_engine.run(
                returns.loc[common].values,
                v.loc[common].values,
                confidence_level=cl,
                name=f"{name} {cl:.0%}",
            )

        bt_95 = _backtest_var(var_95, 0.95, strategy_name)
        bt_99 = _backtest_var(var_99, 0.99, strategy_name)

        perf_engine = PerformanceEngine(
            risk_free_rate=self.rf, periods_per_year=252
        )
        perf = perf_engine.compute(returns, benchmark_returns, weights_df)
        perf["risk_free_rate"] = self.rf

        return BacktestResult(
            strategy_name=strategy_name,
            period_results=self._results,
            returns_series=returns,
            weights_df=weights_df,
            var_95_series=var_95,
            var_99_series=var_99,
            es_95_series=es_95,
            es_99_series=es_99,
            performance_metrics=perf,
            var_backtest_95=bt_95,
            var_backtest_99=bt_99,
        )


# Comparação de estratégias: Diebold-Mariano

# Teste de Diebold & Mariano (1995) para comparar modelos de previsão.
class DieboldMarianoTest:

    # h: horizonte de previsão (default: 1 dia).
    def __init__(self, h: int = 1):
        self.h = h

    # Compara dois modelos de VaR.
    def test(
        self,
        realized: np.ndarray,
        var_1: np.ndarray,
        var_2: np.ndarray,
        loss_fn: str = "lopez",
    ) -> Dict:
        d1 = self._loss(realized, var_1, loss_fn)
        d2 = self._loss(realized, var_2, loss_fn)

        delta = d1 - d2
        T = len(delta)

        gamma0 = np.var(delta, ddof=1)
        nw_var = gamma0
        for lag in range(1, self.h):
            gamma = np.cov(delta[lag:], delta[:-lag])[0, 1]
            nw_var += 2 * (1 - lag / self.h) * gamma
        nw_var = max(nw_var, 1e-10)

        dm_stat = np.mean(delta) / np.sqrt(nw_var / T)
        pval    = 2 * (1 - norm.cdf(abs(dm_stat)))

        return {
            "dm_stat": round(float(dm_stat), 4),
            "pvalue":  round(float(pval), 4),
            "reject_H0": float(pval) < 0.05,
            "better_model": "model_1" if dm_stat > 0 else "model_2",
            "mean_loss_1": round(float(d1.mean()), 6),
            "mean_loss_2": round(float(d2.mean()), 6),
        }

    @staticmethod
    def _loss(r: np.ndarray, var: np.ndarray, fn: str) -> np.ndarray:
        if fn == "lopez":
            return np.array([
                1 + (ri + vi) ** 2 if ri < -vi else 0.0
                for ri, vi in zip(r, var)
            ])
        elif fn == "mse":
            return (r + var) ** 2
        else:
            raise ValueError(f"loss_fn desconhecido: {fn}")


# Backtest de múltiplas estratégias

# Executa e compara múltiplas estratégias de portfólio em paralelo.
class MultiStrategyBacktest:

    def __init__(
        self,
        estimation_window: int = 252,
        rebalancing_frequency: int = 21,
        risk_free_rate: float = 0.1075,
        verbose: bool = True,
    ):
        self.estimation_window     = estimation_window
        self.rebalancing_frequency = rebalancing_frequency
        self.rf                    = risk_free_rate
        self.verbose               = verbose
        self._results: Dict[str, BacktestResult] = {}

    # Registra uma estratégia para comparação.
    def add_strategy(
        self,
        name: str,
        optimizer_fn: Callable,
        risk_fn: Optional[Callable] = None,
    ) -> "MultiStrategyBacktest":
        self._strategies = getattr(self, "_strategies", {})
        self._strategies[name] = {"optimizer": optimizer_fn, "risk": risk_fn}
        return self

    # Executa todas as estratégias registradas.
    def run_all(
        self,
        returns_df: pd.DataFrame,
        benchmark_returns: Optional[pd.Series] = None,
    ) -> Dict[str, BacktestResult]:
        strategies = getattr(self, "_strategies", {})
        if not strategies:
            raise RuntimeError("Nenhuma estratégia registrada. Use add_strategy().")

        for name, fns in strategies.items():
            logger.info(f"\n{'='*55}")
            logger.info(f"Executando estratégia: {name}")
            logger.info(f"{'='*55}")
            wf = WalkForwardBacktest(
                estimation_window=self.estimation_window,
                rebalancing_frequency=self.rebalancing_frequency,
                optimizer_fn=fns["optimizer"],
                risk_fn=fns.get("risk"),
                risk_free_rate=self.rf,
                verbose=self.verbose,
            )
            self._results[name] = wf.run(
                returns_df,
                strategy_name=name,
                benchmark_returns=benchmark_returns,
            )

        return self._results

    # Tabela comparativa de todas as estratégias.
    def compare(self) -> pd.DataFrame:
        rows = []
        for name, res in self._results.items():
            perf = res.performance_metrics
            bt99 = res.var_backtest_99
            row = {
                "Estratégia":       name,
                "Ret. Anual (%)":   round(perf.get("annual_return", np.nan) * 100, 2),
                "Vol. Anual (%)":   round(perf.get("annual_vol", np.nan) * 100, 2),
                "Sharpe":           round(perf.get("sharpe_ratio", np.nan), 3),
                "Sortino":          round(perf.get("sortino_ratio", np.nan), 3),
                "Max DD (%)":       round(perf.get("max_drawdown", np.nan) * 100, 2),
                "Calmar":           round(perf.get("calmar_ratio", np.nan), 3),
                "CVaR 99% (%)":     round(perf.get("hist_es_99", np.nan) * 100, 2),
                "Violações 99%":    bt99.get("n_violations", np.nan),
                "Kupiec p":         bt99.get("kupiec_pvalue", np.nan),
                "CC adequado":      bt99.get("model_adequate", None),
            }
            rows.append(row)
        return pd.DataFrame(rows).set_index("Estratégia")

    # Matriz de testes DM entre todos os pares de estratégias.
    def diebold_mariano_matrix(
        self, confidence_level: float = 0.99
    ) -> pd.DataFrame:
        names = list(self._results.keys())
        n = len(names)
        pval_matrix = pd.DataFrame(index=names, columns=names, dtype=float)
        dm = DieboldMarianoTest(h=1)

        var_key = f"var_{int(confidence_level*100)}_series"
        for i, n1 in enumerate(names):
            for j, n2 in enumerate(names):
                if i == j:
                    pval_matrix.loc[n1, n2] = 1.0
                    continue
                r1 = self._results[n1]
                r2 = self._results[n2]
                common = r1.returns_series.index.intersection(r2.returns_series.index)
                if len(common) < 20:
                    pval_matrix.loc[n1, n2] = np.nan
                    continue
                var1 = getattr(r1, var_key).reindex(common).fillna(0).values
                var2 = getattr(r2, var_key).reindex(common).fillna(0).values
                ret  = r1.returns_series.reindex(common).values
                result = dm.test(ret, var1, var2)
                pval_matrix.loc[n1, n2] = result["pvalue"]

        return pval_matrix

    # Salva resultados em CSV.
    def save(self, path: Union[str, Path]):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        comparison = self.compare()
        comparison.to_csv(path / "strategy_comparison.csv")
        for name, res in self._results.items():
            safe = name.replace(" ", "_").replace("/", "_")
            res.returns_series.to_csv(path / f"returns_{safe}.csv")
            res.weights_df.to_csv(path / f"weights_{safe}.csv")
        logger.info(f"Resultados salvos em {path}")


# Benchmarks naive (para comparação)

# Portfólio equi-ponderado.
def equal_weight_optimizer(returns_df: pd.DataFrame) -> np.ndarray:
    d = returns_df.shape[1]
    return np.ones(d) / d


# Mínima variância com shrinkage Ledoit-Wolf.
def min_variance_optimizer(returns_df: pd.DataFrame) -> np.ndarray:
    from scipy.optimize import minimize as _minimize
    d = returns_df.shape[1]
    try:
        from sklearn.covariance import LedoitWolf
        cov = LedoitWolf().fit(returns_df.values).covariance_
    except ImportError:
        S = returns_df.cov().values
        mu_var = np.trace(S) / d
        delta = min(0.2, d / max(len(returns_df), d + 1))
        cov = (1 - delta) * S + delta * mu_var * np.eye(d)
    if np.linalg.cond(cov) > 1e12:
        cov += 1e-6 * np.eye(d)
    def obj(w): return float(w @ cov @ w)
    def grad(w): return 2 * cov @ w
    best_w, best_var = np.ones(d) / d, np.inf
    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
    for trial in range(5):
        w0 = np.random.dirichlet(np.ones(d)) if trial > 0 else np.ones(d) / d
        res = _minimize(obj, w0, jac=grad, method="SLSQP",
                        bounds=[(0.0, 1.0)] * d, constraints=cons,
                        options={"ftol": 1e-12, "maxiter": 2000})
        if res.success and res.fun < best_var:
            best_var = res.fun
            best_w = res.x.copy()
    return best_w


# Pesos inversamente proporcionais à volatilidade realizada.
def inverse_vol_optimizer(returns_df: pd.DataFrame) -> np.ndarray:
    vols = returns_df.std().values
    vols = np.maximum(vols, 1e-8)
    w = 1 / vols
    return w / w.sum()


# Fábrica de risk_fn compatível com WalkForwardBacktest usando
def make_garch_evt_risk_fn(
    garch_model: str = "gjr",
    threshold_quantile: float = 0.85,
    min_exceedances: int = 25,
):
    import sys
    from pathlib import Path
    _src = Path(__file__).resolve().parent.parent
    if str(_src) not in sys.path:
        sys.path.insert(0, str(_src))

    _diagnostics: Dict = {"sigma_t": [], "port_std": [], "garch_ok": []}

    def _risk_fn(train_df: pd.DataFrame, weights: np.ndarray) -> Dict:
        try:
            from marginals.garch import GARCHFitter
            from marginals.evt_gpd import GPD
        except ImportError:
            try:
                from garch import GARCHFitter
                from evt_gpd import GPD
            except ImportError:
                GARCHFitter = GPD = None

        port_returns = train_df.values @ weights

        def _gauss(sigma, returns=port_returns):
            try:
                kurt_exc = float(stats.kurtosis(returns, fisher=True))
                nu = float(np.clip(6.0 / max(kurt_exc, 0.1) + 4.0, 3.0, 30.0))
            except Exception:
                nu = 6.0
            scale_t = sigma * float(np.sqrt((nu - 2.0) / nu)) if nu > 2 else sigma
            q95 = stats.t.ppf(0.05, df=nu)
            q99 = stats.t.ppf(0.01, df=nu)
            t_pdf_q95 = stats.t.pdf(q95, df=nu)
            t_pdf_q99 = stats.t.pdf(q99, df=nu)
            return {
                "var_95": float(abs(q95) * scale_t),
                "var_99": float(abs(q99) * scale_t),
                "es_95":  float(scale_t * t_pdf_q95 / 0.05 * (nu + q95**2) / (nu - 1)),
                "es_99":  float(scale_t * t_pdf_q99 / 0.01 * (nu + q99**2) / (nu - 1)),
            }
        port_series  = pd.Series(port_returns, index=train_df.index, name="portfolio")
        port_std     = float(port_series.std())
        sigma_t      = port_std
        std_resid    = port_returns / max(port_std, 1e-8)
        _garch_ok    = False

        if GARCHFitter is not None:
            try:
                fitter    = GARCHFitter(model_type=garch_model, dist="normal")
                gres      = fitter.fit_single(port_series, ticker="portfolio")
                std_resid = gres.std_residuals.values
                sigma_t   = float(gres.conditional_vol.iloc[-1])
                _garch_ok = True
            except Exception as _garch_exc:
                logger.warning(
                    f"GARCHFitter falhou — usando fallback sigma_t=port_std={port_std:.6f}. "
                    f"Razão: {type(_garch_exc).__name__}: {_garch_exc}"
                )

        if not _garch_ok:
            logger.warning(
                f"Fallback ativo: sigma_t=port_std={sigma_t:.6f} (vol incondicional da janela). "
                f"VaR pode ser inconsistente com janelas onde GARCH convergiu — "
                f"inspecione risk_fn.diagnostics['garch_ok'] para identificar os períodos afetados."
            )

        if not np.isfinite(sigma_t) or sigma_t <= 0:
            _diagnostics["sigma_t"].append(port_std)
            _diagnostics["port_std"].append(port_std)
            _diagnostics["garch_ok"].append(False)
            return _gauss(port_std)

        _diagnostics["sigma_t"].append(sigma_t)
        _diagnostics["port_std"].append(port_std)
        _diagnostics["garch_ok"].append(_garch_ok)

        logger.info(
            f"sigma_t={sigma_t:.6f} | port_std={port_std:.6f} | "
            f"garch_ok={_garch_ok} | ratio={sigma_t/max(port_std,1e-8):.4f}"
        )

        losses = -std_resid
        n_obs  = len(losses)

        if GPD is not None:
            gpd = GPD()

            u = float(np.quantile(losses, threshold_quantile))

            max_exc_frac = 0.15
            u_floor = float(np.quantile(losses, max(threshold_quantile, 1.0 - max_exc_frac)))
            u_floor = max(u_floor, 0.0)
            u = max(u, u_floor)

            n_exc_check = int((losses > u).sum())
            if n_exc_check < min_exceedances:
                logger.warning(
                    f"GPD: threshold u={u:.4f} deixa apenas {n_exc_check} exceedances "
                    f"(mínimo={min_exceedances}). Fallback gaussiano."
                )
                return _gauss(sigma_t)

            try:
                u_stab = gpd.select_threshold(
                    losses,
                    method="stability",
                    min_exceedances=min_exceedances,
                    plot=False,
                )
                n_exc_stab = int((losses > u_stab).sum())
                u_cap = float(np.quantile(losses, 0.92))
                if n_exc_stab >= min_exceedances and u_stab <= u_cap and u_stab > 0:
                    u = u_stab
            except Exception:
                pass

            try:
                gpd.fit(losses, threshold=u, method="mle", return_std_errors=False)

                logger.info(
                    f"GPD: xi={gpd.xi:.4f} sigma={gpd.sigma:.4f} "
                    f"n_exc={gpd.n_exceedances} u={u:.4f} converged={gpd.converged}"
                )

                if gpd.converged and gpd.n_exceedances >= min_exceedances:
                    var_95 = float(gpd.var(0.05, n_obs) * sigma_t)
                    var_99 = float(gpd.var(0.01, n_obs) * sigma_t)
                    es_95  = float(gpd.expected_shortfall(0.05, n_obs) * sigma_t)
                    es_99  = float(gpd.expected_shortfall(0.01, n_obs) * sigma_t)

                    if all(np.isfinite([var_95, var_99, es_95, es_99])) and \
                       0 < var_95 < var_99 < 0.30 and es_99 > var_99:
                        return {
                            "var_95": var_95,
                            "var_99": var_99,
                            "es_95":  es_95,
                            "es_99":  es_99,
                        }
                    else:
                        logger.warning(
                            f"GPD produziu valores fora do range esperado "
                            f"(var_95={var_95:.4f}, var_99={var_99:.4f}, "
                            f"es_95={es_95:.4f}, es_99={es_99:.4f}). "
                            "Fallback gaussiano."
                        )
            except Exception as exc:
                logger.warning(f"GPD falhou: {exc}. Fallback gaussiano.")

        return _gauss(sigma_t)

    _risk_fn.diagnostics = _diagnostics
    return _risk_fn


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    np.random.seed(42)

    T, d = 800, 5
    tickers = [f"ATIVO{i+1}" for i in range(d)]
    dates   = pd.bdate_range("2020-01-02", periods=T)

    R = 0.35 * np.ones((d, d)) + 0.65 * np.eye(d)
    L = np.linalg.cholesky(R)
    vols = np.array([0.025, 0.022, 0.018, 0.015, 0.020])
    rets = (np.random.standard_normal((T, d)) @ L.T) * vols
    returns_df = pd.DataFrame(rets, index=dates, columns=tickers)

    print(f"returns_df: {returns_df.shape}")

    print("\n VaRBacktestEngine ")
    var_engine = VaRBacktestEngine(alpha=0.05)
    port_rets  = returns_df.mean(axis=1).values
    var_fcast  = np.full(T, 0.025)

    for cl in [0.95, 0.99]:
        r = var_engine.run(port_rets, var_fcast, confidence_level=cl, name="naive")
        print(f"  VaR {cl:.0%}: violations={r['n_violations']}/{T} "
              f"({r['violation_rate']:.2%}) | Kupiec p={r['kupiec_pvalue']} "
              f"| adequate={r['model_adequate']}")

    print("\n PerformanceEngine ")
    perf = PerformanceEngine(risk_free_rate=0.1075)
    port_series = pd.Series(port_rets, index=dates)
    metrics = perf.compute(port_series)
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    print("\n WalkForwardBacktest ")

    def naive_risk_fn(train_df, weights):
        cov = train_df.cov().values
        sigma = np.sqrt(weights @ cov @ weights)
        return {
            "var_95": stats.norm.ppf(0.95) * sigma,
            "var_99": stats.norm.ppf(0.99) * sigma,
            "es_95":  stats.norm.pdf(stats.norm.ppf(0.05)) / 0.05 * sigma,
            "es_99":  stats.norm.pdf(stats.norm.ppf(0.01)) / 0.01 * sigma,
        }

    wf = WalkForwardBacktest(
        estimation_window=252,
        rebalancing_frequency=21,
        optimizer_fn=equal_weight_optimizer,
        risk_fn=naive_risk_fn,
        risk_free_rate=0.1075,
        verbose=True,
    )
    result = wf.run(returns_df, strategy_name="equal_weight")
    print(f"  Total return: {result.total_return:.2%}")
    print(f"  Sharpe: {result.sharpe:.3f}")
    print(f"  Violations 99%: {result.var_backtest_99.get('n_violations', 'N/A')}")

    print("\n MultiStrategyBacktest ")
    garch_evt_risk = make_garch_evt_risk_fn(garch_model="gjr")
    ms = MultiStrategyBacktest(estimation_window=252, rebalancing_frequency=21)
    ms.add_strategy("Equal Weight",     equal_weight_optimizer, naive_risk_fn)
    ms.add_strategy("Min Variance",     min_variance_optimizer, naive_risk_fn)
    ms.add_strategy("Inverse Vol",      inverse_vol_optimizer,  naive_risk_fn)
    ms.add_strategy("GARCH-EVT EW",     equal_weight_optimizer, garch_evt_risk)
    ms.add_strategy("GARCH-EVT MinVar", min_variance_optimizer, garch_evt_risk)
    ms.run_all(returns_df)
    print(ms.compare().to_string())

    diag     = garch_evt_risk.diagnostics
    sigma_t  = np.array(diag["sigma_t"])
    port_std = np.array(diag["port_std"])
    garch_ok = np.array(diag["garch_ok"])

    n_total   = len(sigma_t)
    n_ok      = int(garch_ok.sum())
    n_fallback = n_total - n_ok

    print(f"\n Diagnóstico GARCH (n_janelas={n_total}) ")
    print(f"  GARCH convergiu : {n_ok}/{n_total} ({n_ok/max(n_total,1):.1%})")
    print(f"  Fallback port_std: {n_fallback}/{n_total} ({n_fallback/max(n_total,1):.1%})")

    if n_total > 0:
        print(f"  sigma_t  — mean={sigma_t.mean():.4f} std={sigma_t.std():.4f} "
              f"min={sigma_t.min():.4f} max={sigma_t.max():.4f}")
        print(f"  port_std — mean={port_std.mean():.4f} std={port_std.std():.4f} "
              f"min={port_std.min():.4f} max={port_std.max():.4f}")
        if n_total > 1:
            corr = np.corrcoef(sigma_t, port_std)[0, 1]
            print(f"  correlação sigma_t vs port_std: {corr:.4f}")
            if corr > 0.99 and sigma_t.std() < port_std.std() * 0.05:
                print("  ⚠️  ALERTA: GARCH provavelmente colapsando — "
                      "sigma_t ≈ port_std em todas as janelas.")
            else:
                print("  ✓  GARCH gerando vol condicional variável (sem colapso).")

    print("\n DM Test ")
    dm_mat = ms.diebold_mariano_matrix()
    print(dm_mat.to_string())

    print("\n Todos os testes concluídos.")
