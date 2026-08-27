
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


# Dataclasses de resultado  (de backtesting_engine.py)

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


# VaR Backtesting  (de backtesting_engine.py)

# Testes de backtesting de VaR: Kupiec (1995), Christoffersen (1998),
class VaRBacktestEngine:

    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha

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
        result["cc_lr_stat"]  = round(float(lr_cc), 4)
        result["cc_pvalue"]   = round(float(p_cc), 4) if np.isfinite(p_cc) else np.nan
        result["cc_adequate"] = float(p_cc) > self.alpha if np.isfinite(p_cc) else None

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

    def _christoffersen_independence(self, hits: np.ndarray) -> Dict:
        n00 = n01 = n10 = n11 = 0
        for t in range(1, len(hits)):
            if hits[t-1] == 0 and hits[t] == 0:   n00 += 1
            elif hits[t-1] == 0 and hits[t] == 1:  n01 += 1
            elif hits[t-1] == 1 and hits[t] == 0:  n10 += 1
            else:                                    n11 += 1
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
            lr   = -2 * (ll_ind - ll_dep)
            pval = 1 - chi2.cdf(lr, df=1)
        except Exception:
            lr, pval = np.nan, np.nan
        return {
            "lr_stat":    round(float(lr), 4) if np.isfinite(lr) else np.nan,
            "pvalue":     round(float(pval), 4) if np.isfinite(pval) else np.nan,
            "adequate":   float(pval) > self.alpha if np.isfinite(pval) else None,
            "pi_01":      round(pi_01, 4),
            "pi_11":      round(pi_11, 4),
            "clustering": bool(abs(pi_11 - pi_01) > 0.05),
        }

    def _lopez_loss(self, returns: np.ndarray, var_forecasts: np.ndarray) -> float:
        losses = [1 + (r + v) ** 2 if r < -v else 0.0 for r, v in zip(returns, var_forecasts)]
        return round(float(np.mean(losses)), 6)

    def _dynamic_quantile_test(
        self, hits: np.ndarray, var_forecasts: np.ndarray, lags: int = 4
    ) -> Dict:
        T = len(hits)
        alpha_var = hits.mean()
        hit_centered = hits - alpha_var

        if T < lags + 5:
            return {"stat": np.nan, "pvalue": np.nan, "adequate": None}

        X_list = [np.ones(T - lags), var_forecasts[lags:]]
        for lag in range(1, lags + 1):
            X_list.append(hit_centered[lags - lag: T - lag])
        X = np.column_stack(X_list)
        y = hit_centered[lags:]

        try:
            beta, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
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


# Performance metrics  (de backtesting_engine.py)

# Métricas de performance de portfólio: Sharpe, Sortino, Calmar,
class PerformanceEngine:

    def __init__(self, risk_free_rate: float = 0.0, periods_per_year: int = 252):
        self.rf = risk_free_rate / periods_per_year
        self.n  = periods_per_year

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
        result["annual_return"] = round(ann_ret, 6)
        result["annual_vol"]    = round(ann_vol, 6)
        result["total_return"]  = round(float((1 + r).prod() - 1), 6)

        excess = np.mean(r) - self.rf
        result["sharpe_ratio"] = round(excess * self.n / max(ann_vol, 1e-8), 4)

        downside_dev = np.minimum(r - self.rf, 0)
        downside_vol = np.sqrt(np.mean(downside_dev ** 2)) * np.sqrt(self.n)
        result["downside_vol"] = round(float(downside_vol), 6)
        result["sortino_ratio"] = round(excess * self.n / max(downside_vol, 1e-8), 4)

        cum      = (1 + r).cumprod()
        roll_max = np.maximum.accumulate(cum)
        drawdown = (cum - roll_max) / (roll_max + 1e-10)
        max_dd   = float(drawdown.min())
        result["max_drawdown"] = round(max_dd, 6)
        result["avg_drawdown"] = round(float(drawdown.mean()), 6)
        result["calmar_ratio"] = round(ann_ret / max(abs(max_dd), 1e-8), 4)

        for cl, label in [(0.95, "95"), (0.99, "99")]:
            var  = float(-np.quantile(r, 1 - cl))
            tail = r[r <= -var]
            es   = float(-tail.mean()) if len(tail) > 0 else var
            result[f"hist_var_{label}"] = round(var, 6)
            result[f"hist_es_{label}"]  = round(es, 6)

        result["skewness"]        = round(float(stats.skew(r)), 4)
        result["excess_kurtosis"] = round(float(stats.kurtosis(r)), 4)
        result["hit_rate"]        = round(float(np.mean(r > 0)), 4)

        gains  = r[r > 0].sum()
        losses = abs(r[r < 0].sum())
        result["profit_factor"] = round(float(gains / max(losses, 1e-10)), 4)

        thr       = self.rf
        omega_num = np.sum(np.maximum(r - thr, 0))
        omega_den = np.sum(np.maximum(thr - r, 0))
        result["omega_ratio"] = round(float(omega_num / max(omega_den, 1e-10)), 4)

        p95 = abs(np.percentile(r, 95))
        p05 = abs(np.percentile(r, 5))
        result["tail_ratio"] = round(float(p95 / max(p05, 1e-10)), 4)

        if benchmark_returns is not None:
            bm       = benchmark_returns.reindex(returns.index).dropna().values
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


# Walk-Forward Backtest  (de backtesting_engine.py)

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

    def run(
        self,
        returns_df: pd.DataFrame,
        strategy_name: str = "copula_evt",
        initial_weights: Optional[np.ndarray] = None,
        benchmark_returns: Optional[pd.Series] = None,
    ) -> BacktestResult:
        T, d = returns_df.shape
        dates = returns_df.index
        asset_names = list(returns_df.columns)

        if initial_weights is None:
            initial_weights = np.ones(d) / d

        w_current = initial_weights.copy()
        self._results = []

        rebal_indices = list(range(
            self.estimation_window,
            T - 1,
            self.rebalancing_frequency,
        ))

        logger.info(
            f"WalkForward [{strategy_name}]  "
            f"T={T}  window={self.window_type}  "
            f"est_window={self.estimation_window}  "
            f"rebal_freq={self.rebalancing_frequency}  "
            f"n_rebal={len(rebal_indices)}"
        )

        import sys as _sys_du, pathlib as _pathlib_du
        _utils_dir = str(_pathlib_du.Path(__file__).resolve().parent.parent / "utils")
        if _utils_dir not in _sys_du.path:
            _sys_du.path.insert(0, _utils_dir)
        from dynamic_universe import eligible_assets, expand_weights, nan_safe_portfolio_return
        MIN_COVERAGE = 0.95

        for t_idx in rebal_indices:
            t_start = t_idx - self.estimation_window if self.window_type == "rolling" else 0
            window_slice = returns_df.iloc[t_start:t_idx]
            asset_names_t = eligible_assets(window_slice, min_coverage=MIN_COVERAGE)
            train_df = window_slice[asset_names_t]
            d_t = len(asset_names_t)

            try:
                if (self.optimizer_fn is not None and d_t > 0
                        and len(train_df) >= self.min_estimation_window):
                    train_df_cov = train_df.dropna()
                    w_new = self.optimizer_fn(train_df_cov)
                    if w_new is not None and len(w_new) == d_t:
                        w_sub = np.clip(w_new, 0, 1)
                        w_sub = w_sub / w_sub.sum()
                        w_current = expand_weights(w_sub, asset_names_t, asset_names)
            except Exception as e:
                logger.warning(f"  t={t_idx}: optimizer falhou ({e}), mantendo pesos")

            t_oos_end = min(t_idx + self.rebalancing_frequency, T)
            oos_df    = returns_df.iloc[t_idx:t_oos_end]

            for t_oos in range(len(oos_df)):
                date  = oos_df.index[t_oos]
                ret_t = nan_safe_portfolio_return(oos_df.iloc[t_oos].values, w_current)

                risk_metrics = {"var_95": np.nan, "var_99": np.nan,
                                "es_95": np.nan,  "es_99": np.nan}
                if self.risk_fn is not None:
                    try:
                        w_for_risk = pd.Series(w_current, index=asset_names).reindex(train_df.columns).values
                        risk_metrics = self.risk_fn(train_df, w_for_risk)
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

    def _compile_results(
        self,
        strategy_name: str,
        benchmark_returns: Optional[pd.Series],
    ) -> BacktestResult:
        if not self._results:
            raise RuntimeError("Nenhum resultado gerado. Verifique os parâmetros.")

        dates   = pd.DatetimeIndex([r.date for r in self._results])
        returns = pd.Series([r.realized_return for r in self._results], index=dates)
        var_95  = pd.Series([r.var_95 for r in self._results], index=dates)
        var_99  = pd.Series([r.var_99 for r in self._results], index=dates)
        es_95   = pd.Series([r.es_95  for r in self._results], index=dates)
        es_99   = pd.Series([r.es_99  for r in self._results], index=dates)

        weights_df = pd.DataFrame(
            [r.weights for r in self._results], index=dates
        )

        var_engine = VaRBacktestEngine(alpha=0.05)

        def _backtest_var(var_series, cl, name):
            v      = var_series.dropna()
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

        perf_engine = PerformanceEngine(risk_free_rate=self.rf, periods_per_year=252)
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


# Diebold-Mariano  (de backtesting_engine.py)

# Teste de Diebold & Mariano (1995) para comparar a acurácia preditiva
class DieboldMarianoTest:

    def __init__(self, h: int = 1):
        self.h = h

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
            gamma  = np.cov(delta[lag:], delta[:-lag])[0, 1]
            nw_var += 2 * (1 - lag / self.h) * gamma
        nw_var = max(nw_var, 1e-10)

        dm_stat = np.mean(delta) / np.sqrt(nw_var / T)
        pval    = 2 * (1 - norm.cdf(abs(dm_stat)))

        return {
            "dm_stat":      round(float(dm_stat), 4),
            "pvalue":       round(float(pval), 4),
            "reject_H0":    float(pval) < 0.05,
            "better_model": "model_1" if dm_stat > 0 else "model_2",
            "mean_loss_1":  round(float(d1.mean()), 6),
            "mean_loss_2":  round(float(d2.mean()), 6),
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


# MultiStrategyBacktest  (de backtesting_engine.py)

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

    def add_strategy(
        self,
        name: str,
        optimizer_fn: Callable,
        risk_fn: Optional[Callable] = None,
    ) -> "MultiStrategyBacktest":
        self._strategies = getattr(self, "_strategies", {})
        self._strategies[name] = {"optimizer": optimizer_fn, "risk": risk_fn}
        return self

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

    def compare(self) -> pd.DataFrame:
        rows = []
        for name, res in self._results.items():
            perf = res.performance_metrics
            bt99 = res.var_backtest_99
            rows.append({
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
            })
        return pd.DataFrame(rows).set_index("Estratégia")

    def diebold_mariano_matrix(self, confidence_level: float = 0.99) -> pd.DataFrame:
        names      = list(self._results.keys())
        pval_matrix = pd.DataFrame(index=names, columns=names, dtype=float)
        dm         = DieboldMarianoTest(h=1)
        var_key    = f"var_{int(confidence_level*100)}_series"

        for n1 in names:
            for n2 in names:
                if n1 == n2:
                    pval_matrix.loc[n1, n2] = 1.0
                    continue
                r1 = self._results[n1]
                r2 = self._results[n2]
                common = r1.returns_series.index.intersection(r2.returns_series.index)
                if len(common) < 20:
                    pval_matrix.loc[n1, n2] = np.nan
                    continue
                var1   = getattr(r1, var_key).reindex(common).fillna(0).values
                var2   = getattr(r2, var_key).reindex(common).fillna(0).values
                ret    = r1.returns_series.reindex(common).values
                result = dm.test(ret, var1, var2)
                pval_matrix.loc[n1, n2] = result["pvalue"]

        return pval_matrix

    def save(self, path: Union[str, Path]):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.compare().to_csv(path / "strategy_comparison.csv")
        for name, res in self._results.items():
            safe = name.replace(" ", "_").replace("/", "_")
            res.returns_series.to_csv(path / f"returns_{safe}.csv")
            res.weights_df.to_csv(path / f"weights_{safe}.csv")
        logger.info(f"Resultados salvos em {path}")

try:
    from preprocessor import DataPreprocessor
except ImportError:
    DataPreprocessor = None

try:
    from risk_based_portfolios import (
        RiskBasedPortfolioSuite,
        EqualWeightOptimizer,
        MinVarianceOptimizer,
        RiskParityOptimizer,
        MaxDiversificationOptimizer,
        MaxDecorrelationOptimizer,
    )
except ImportError:
    raise ImportError(
        "risk_based_portfolios.py não encontrado. "
        "Certifique-se de que está no PYTHONPATH."
    )


# Funções auxiliares de risco (risk_fn para WalkForwardBacktest)

# Retorna risk_fn baseada em VaR/ES histórico simples.
def historical_var_risk_fn(
    confidence_levels: Tuple[float, float] = (0.95, 0.99),
    min_obs: int = 60,
) -> Callable:
    def _fn(train_df: pd.DataFrame, weights: np.ndarray) -> Dict:
        r = train_df.values @ weights
        result = {}
        for cl in confidence_levels:
            label = "95" if cl == 0.95 else "99"
            if len(r) < min_obs:
                result[f"var_{label}"] = np.nan
                result[f"es_{label}"]  = np.nan
                continue
            var = float(-np.quantile(r, 1 - cl))
            tail = r[r <= -var]
            es   = float(-tail.mean()) if len(tail) > 0 else var
            result[f"var_{label}"] = var
            result[f"es_{label}"]  = es
        return result

    return _fn


# VaR via EWMA (RiskMetrics) — assume normalidade com volatilidade EWMA.
def ewma_var_risk_fn(
    lam: float = 0.94,
    confidence_levels: Tuple[float, float] = (0.95, 0.99),
    min_obs: int = 30,
) -> Callable:
    from scipy.stats import norm as _norm

    def _fn(train_df: pd.DataFrame, weights: np.ndarray) -> Dict:
        r = train_df.values @ weights
        T = len(r)
        if T < min_obs:
            return {f"var_{l}": np.nan for l in ["95", "99"]} | \
                   {f"es_{l}":  np.nan for l in ["95", "99"]}

        var_ewma = float(r[-1] ** 2)
        for t in range(T - 2, -1, -1):
            var_ewma = lam * var_ewma + (1 - lam) * float(r[t] ** 2)
        sigma = np.sqrt(var_ewma)

        result = {}
        for cl in confidence_levels:
            label = "95" if cl == 0.95 else "99"
            z     = _norm.ppf(cl)
            var   = float(sigma * z)
            es    = float(sigma * _norm.pdf(z) / (1 - cl))
            result[f"var_{label}"] = var
            result[f"es_{label}"]  = es
        return result

    return _fn


# Carregador de retornos (wrapper ao redor de DataPreprocessor)

# Pré-processa preços e retorna DataFrame de retornos limpos.
def load_returns(
    prices: pd.DataFrame,
    method: str = "log",
    handle_missing: str = "forward_fill",
    winsorize: bool = True,
    winsorize_bounds: Tuple[float, float] = (0.01, 0.99),
) -> pd.DataFrame:
    if DataPreprocessor is not None:
        proc = DataPreprocessor(prices)
        proc.handle_missing_data(method=handle_missing)
        proc.compute_returns(method=method)
        if winsorize:
            returns = proc.winsorize_outliers(*winsorize_bounds)
        else:
            returns = proc.returns
        return returns.dropna()

    clean = prices.ffill().bfill()
    if method == "log":
        returns = np.log(clean / clean.shift(1)).iloc[1:]
    else:
        returns = clean.pct_change().iloc[1:]
    if winsorize:
        lo, hi = winsorize_bounds
        for col in returns.columns:
            s = returns[col]
            returns[col] = s.clip(s.quantile(lo), s.quantile(hi))
    return returns.dropna()


# RollingBacktestRunner

# Executa walk-forward backtest para múltiplas estratégias risk-based
class RollingBacktestRunner:

    STRATEGY_LABELS = {
        "ew":  "Equal Weight",
        "mv":  "Minimum Variance",
        "rp":  "Risk Parity",
        "md":  "Maximum Diversification",
        "mde": "Maximum Decorrelation",
    }

    def __init__(
        self,
        estimation_window: int = 252,
        rebalancing_frequency: int = 21,
        window_type: str = "rolling",
        max_weight: float = 0.40,
        min_weight: float = 0.0,
        cov_estimator: str = "ledoit_wolf",
        risk_fn_type: str = "historical",
        risk_free_rate: float = 0.1075,
        strategies: Optional[List[str]] = None,
        n_restarts: int = 5,
        verbose: bool = True,
    ):
        self.estimation_window     = estimation_window
        self.rebalancing_frequency = rebalancing_frequency
        self.window_type           = window_type
        self.max_weight            = max_weight
        self.min_weight            = min_weight
        self.cov_estimator         = cov_estimator
        self.risk_fn_type          = risk_fn_type
        self.rf                    = risk_free_rate
        self.strategies            = strategies or ["ew", "mv", "rp", "md", "mde"]
        self.n_restarts            = n_restarts
        self.verbose               = verbose

        self._suite  = RiskBasedPortfolioSuite(
            strategies    = self.strategies,
            max_weight    = max_weight,
            min_weight    = min_weight,
            cov_estimator = cov_estimator,
            n_restarts    = n_restarts,
        )
        self._backtest_results: Dict[str, BacktestResult] = {}

    # ── Risk function factory ────────────────────────────────────────────────

    def _make_risk_fn(self) -> Callable:
        if self.risk_fn_type == "ewma":
            return ewma_var_risk_fn()
        return historical_var_risk_fn()

    # ── Optimizer factories ──────────────────────────────────────────────────

    # Retorna callable(returns_df) → np.ndarray de pesos.
    def _make_optimizer_fn(self, strategy: str) -> Callable[[pd.DataFrame], np.ndarray]:
        kwargs = dict(
            max_weight    = self.max_weight,
            min_weight    = self.min_weight,
            cov_estimator = self.cov_estimator,
            n_restarts    = self.n_restarts,
        )
        optimizers = {
            "ew":  EqualWeightOptimizer(**kwargs),
            "mv":  MinVarianceOptimizer(**kwargs),
            "rp":  RiskParityOptimizer(**kwargs),
            "md":  MaxDiversificationOptimizer(**kwargs),
            "mde": MaxDecorrelationOptimizer(**kwargs),
        }
        opt = optimizers[strategy]
        return opt.__call__

    # ── Walk-forward por estratégia ──────────────────────────────────────────

    def _run_strategy(
        self,
        strategy: str,
        returns_df: pd.DataFrame,
        benchmark_returns: Optional[pd.Series],
    ) -> BacktestResult:
        label    = self.STRATEGY_LABELS.get(strategy, strategy)
        risk_fn  = self._make_risk_fn()
        opt_fn   = self._make_optimizer_fn(strategy)

        backtester = WalkForwardBacktest(
            estimation_window    = self.estimation_window,
            rebalancing_frequency= self.rebalancing_frequency,
            min_estimation_window= max(60, self.estimation_window // 2),
            window_type          = self.window_type,
            confidence_levels    = [0.95, 0.99],
            optimizer_fn         = opt_fn,
            risk_fn              = risk_fn,
            risk_free_rate       = self.rf,
            verbose              = self.verbose,
        )

        logger.info(f"\n── Executando backtest: {label} ──")
        return backtester.run(
            returns_df        = returns_df,
            strategy_name     = label,
            benchmark_returns = benchmark_returns,
        )

    # ── Ponto de entrada principal ───────────────────────────────────────────

    # Executa walk-forward para todas as estratégias selecionadas.
    def run_all(
        self,
        returns_df: pd.DataFrame,
        benchmark_returns: Optional[pd.Series] = None,
        cvar_result: Optional[BacktestResult] = None,
    ) -> Dict[str, BacktestResult]:
        T, d = returns_df.shape
        logger.info(
            f"\nRollingBacktestRunner — {len(self.strategies)} estratégias  "
            f"T={T}  d={d}  window={self.estimation_window}  "
            f"rebal={self.rebalancing_frequency}"
        )

        self._backtest_results = {}

        for strategy in self.strategies:
            try:
                res = self._run_strategy(strategy, returns_df, benchmark_returns)
                self._backtest_results[strategy] = res
            except Exception as exc:
                logger.error(f"Backtest '{strategy}' falhou: {exc}")

        if cvar_result is not None:
            self._backtest_results["cvar_evt"] = cvar_result

        logger.info(f"\nBacktests concluídos: {list(self._backtest_results.keys())}")
        return self._backtest_results

    # ── Adição de estratégia customizada ─────────────────────────────────────

    # Adiciona e executa uma estratégia com optimizer_fn externo.
    def add_custom_strategy(
        self,
        key: str,
        optimizer_fn: Callable[[pd.DataFrame], np.ndarray],
        returns_df: pd.DataFrame,
        benchmark_returns: Optional[pd.Series] = None,
        label: Optional[str] = None,
    ) -> BacktestResult:
        risk_fn = self._make_risk_fn()
        backtester = WalkForwardBacktest(
            estimation_window    = self.estimation_window,
            rebalancing_frequency= self.rebalancing_frequency,
            min_estimation_window= max(60, self.estimation_window // 2),
            window_type          = self.window_type,
            confidence_levels    = [0.95, 0.99],
            optimizer_fn         = optimizer_fn,
            risk_fn              = risk_fn,
            risk_free_rate       = self.rf,
            verbose              = self.verbose,
        )
        name = label or key
        logger.info(f"\n── Executando backtest customizado: {name} ──")
        result = backtester.run(
            returns_df        = returns_df,
            strategy_name     = name,
            benchmark_returns = benchmark_returns,
        )
        self._backtest_results[key] = result
        return result

    # ── Relatório / comparação ───────────────────────────────────────────────

    # DataFrame comparativo com métricas de performance para todas as
    def summary(self, risk_free_rate: Optional[float] = None) -> pd.DataFrame:
        if not self._backtest_results:
            raise RuntimeError("Execute run_all() primeiro.")

        rf = risk_free_rate if risk_free_rate is not None else self.rf
        engine = PerformanceEngine(risk_free_rate=rf)

        rows = []
        for key, res in self._backtest_results.items():
            r = res.returns_series
            perf = engine.compute(r, weights_df=res.weights_df)

            excess_cagr    = res.annualized_return - rf
            downside_vol   = perf.get("downside_vol", np.nan)
            sortino_cagr   = (
                excess_cagr / max(downside_vol, 1e-8)
                if np.isfinite(downside_vol) else np.nan
            )
            mdd            = perf.get("max_drawdown", np.nan)
            calmar_cagr    = (
                res.annualized_return / max(abs(mdd), 1e-8)
                if np.isfinite(mdd) else np.nan
            )

            var_bt_99 = res.var_backtest_99 if hasattr(res, "var_backtest_99") else {}
            rows.append({
                "estrategia":       res.strategy_name,
                "annual_return":    round(res.annualized_return, 4),
                "annual_vol":       round(res.annualized_vol, 4),
                "sharpe":           round(res.sharpe, 4),
                "sortino":          round(sortino_cagr, 4) if np.isfinite(sortino_cagr) else np.nan,
                "max_drawdown":     mdd,
                "calmar":           round(calmar_cagr, 4) if np.isfinite(calmar_cagr) else np.nan,
                "hist_var_99":      perf.get("hist_var_99", np.nan),
                "hist_es_99":       perf.get("hist_es_99", np.nan),
                "kupiec_p_99":      var_bt_99.get("kupiec_pvalue", np.nan),
                "cc_p_99":          var_bt_99.get("cc_pvalue", np.nan),
                "var_adequate_99":  var_bt_99.get("model_adequate", np.nan),
                "avg_turnover":     perf.get("avg_daily_turnover", np.nan),
                "n_obs":            len(r),
            })

        df = pd.DataFrame(rows).sort_values("sharpe", ascending=False).reset_index(drop=True)
        return df

    # DataFrame (data × estratégia) com retornos diários OOS de cada estratégia.
    def returns_panel(self) -> pd.DataFrame:
        if not self._backtest_results:
            raise RuntimeError("Execute run_all() primeiro.")
        panel = pd.DataFrame({
            res.strategy_name: res.returns_series
            for res in self._backtest_results.values()
        })
        return panel

    # DataFrame de pesos históricos para uma estratégia específica.
    def weights_panel(self, strategy_key: str) -> pd.DataFrame:
        if strategy_key not in self._backtest_results:
            raise KeyError(f"Estratégia '{strategy_key}' não encontrada.")
        return self._backtest_results[strategy_key].weights_df

    # Matriz de p-valores Diebold-Mariano entre pares de estratégias.
    def diebold_mariano_matrix(
        self,
        loss_fn: str = "se",
        h: int = 1,
    ) -> pd.DataFrame:
        if not self._backtest_results:
            raise RuntimeError("Execute run_all() primeiro.")

        dm = DieboldMarianoTest(h=h)
        keys   = list(self._backtest_results.keys())
        labels = [self._backtest_results[k].strategy_name for k in keys]
        n      = len(keys)
        matrix = pd.DataFrame(np.nan, index=labels, columns=labels)

        for i, ki in enumerate(keys):
            ri = self._backtest_results[ki].returns_series
            vi = self._backtest_results[ki].var_99_series
            if vi is None or vi.isna().all():
                continue
            for j, kj in enumerate(keys):
                if i == j:
                    continue
                rj = self._backtest_results[kj].returns_series
                vj = self._backtest_results[kj].var_99_series
                if vj is None or vj.isna().all():
                    continue
                common = ri.index.intersection(rj.index)
                if len(common) < 30:
                    continue
                try:
                    res = dm.test(
                        realized_returns=ri.loc[common].values,
                        var1=vi.reindex(common).values,
                        var2=vj.reindex(common).values,
                        loss_fn=loss_fn,
                    )
                    matrix.loc[labels[i], labels[j]] = res.get("pvalue", np.nan)
                except Exception:
                    pass

        return matrix

    # Salva returns_panel, summary e pesos de cada estratégia em CSV.
    def save(self, output_dir: Union[str, Path]) -> None:
        if not self._backtest_results:
            raise RuntimeError("Execute run_all() primeiro.")
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)

        self.summary().to_csv(path / "backtest_summary.csv", index=False)
        self.returns_panel().to_csv(path / "backtest_returns.csv")

        for key, res in self._backtest_results.items():
            if res.weights_df is not None:
                res.weights_df.to_csv(path / f"weights_{key}.csv")

        logger.info(f"Resultados salvos em {path}")


# ComparisonBacktest — wrapper de alto nível sobre MultiStrategyBacktest

# Interface simplificada que:
class ComparisonBacktest:

    def __init__(self, runner_kwargs: Optional[Dict] = None):
        self._runner_kwargs = runner_kwargs or {}
        self._runner: Optional[RollingBacktestRunner] = None
        self._multi: Optional[MultiStrategyBacktest]  = None
        self._external: Dict[str, BacktestResult]     = {}

    def fit(
        self,
        returns_df: pd.DataFrame,
        benchmark_returns: Optional[pd.Series] = None,
    ) -> "ComparisonBacktest":
        self._runner = RollingBacktestRunner(**self._runner_kwargs)
        self._runner.run_all(returns_df, benchmark_returns)
        return self

    # Incorpora resultado externo (ex.: CVaR do pipeline principal).
    def add_pipeline_result(
        self,
        key: str,
        result: BacktestResult,
    ) -> "ComparisonBacktest":
        if self._runner is None:
            raise RuntimeError("Execute fit() primeiro.")
        self._runner._backtest_results[key] = result
        self._external[key] = result
        return self

    def report(self) -> pd.DataFrame:
        if self._runner is None:
            raise RuntimeError("Execute fit() primeiro.")
        return self._runner.summary()

    def returns_panel(self) -> pd.DataFrame:
        return self._runner.returns_panel()

    def save(self, output_dir: Union[str, Path]) -> None:
        self._runner.save(output_dir)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    np.random.seed(42)

    T, d = 800, 6
    tickers = [f"ATIVO{i+1}" for i in range(d)]
    vols    = np.random.uniform(0.012, 0.025, d)
    corr    = 0.35 * np.ones((d, d)) + 0.65 * np.eye(d)
    cov     = np.outer(vols, vols) * corr
    raw     = np.random.multivariate_normal(np.zeros(d), cov, T)
    returns = pd.DataFrame(raw, columns=tickers,
                           index=pd.date_range("2018-01-01", periods=T, freq="B"))

    benchmark = pd.Series(np.random.normal(0.0003, 0.012, T),
                          index=returns.index, name="IBOV")

    runner = RollingBacktestRunner(
        estimation_window    = 252,
        rebalancing_frequency= 21,
        window_type          = "rolling",
        max_weight           = 0.40,
        cov_estimator        = "ledoit_wolf",
        risk_fn_type         = "historical",
        strategies           = ["ew", "mv", "rp", "md", "mde"],
        n_restarts           = 3,
    )

    runner.run_all(returns, benchmark_returns=benchmark)

    print("\n── Resumo de Performance ──")
    print(runner.summary().to_string(index=False))

    panel = runner.returns_panel()
    print(f"\nReturns panel: {panel.shape}")
    print(f"Período OOS: {panel.index[0].date()} → {panel.index[-1].date()}")
