
from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import chi2, norm

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# Backtesting do Expected Shortfall (CVaR).
class ESBacktest:

    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha

    # Executa bateria de testes de ES.
    def run(
        self,
        realized_returns: np.ndarray,
        var_forecasts: np.ndarray,
        es_forecasts: np.ndarray,
        confidence_level: float = 0.99,
        n_bootstrap: int = 1000,
        seed: int = 42,
    ) -> Dict:
        result = {}
        alpha_var = 1 - confidence_level
        hits  = realized_returns < -var_forecasts
        n_hit = int(hits.sum())
        T = len(realized_returns)

        result["T"] = T
        result["n_violations"] = n_hit
        result["expected_violations"] = round(T * alpha_var, 1)

        if n_hit < 5:
            logger.warning(f"ES backtest: apenas {n_hit} excedências. Resultados instáveis.")
            result["warning"] = f"apenas {n_hit} excedências — amostra insuficiente"

        mf = self._mcneil_frey(realized_returns, var_forecasts, es_forecasts, hits)
        result.update({f"mf_{k}": v for k, v in mf.items()})

        az1 = self._acerbi_szekely_z1(realized_returns, es_forecasts, hits)
        result.update({f"az1_{k}": v for k, v in az1.items()})

        az2 = self._acerbi_szekely_z2(realized_returns, es_forecasts, alpha_var)
        result.update({f"az2_{k}": v for k, v in az2.items()})

        nz = self._nolde_ziegel_score(realized_returns, var_forecasts, es_forecasts)
        result["nz_score"] = nz

        boot = self._bootstrap_es(realized_returns, alpha_var, n_bootstrap, seed)
        result.update({f"boot_{k}": v for k, v in boot.items()})

        tests_adequate = [
            result.get("mf_pvalue", 1.0) > self.alpha,
            result.get("az1_pvalue", 1.0) > self.alpha,
            result.get("az2_pvalue", 1.0) > self.alpha,
        ]
        n_adequate = sum(t for t in tests_adequate if t is not None)
        result["es_model_adequate"] = n_adequate >= 2

        logger.info(
            f"ES Backtest {confidence_level:.0%}: "
            f"violations={n_hit}/{T} | "
            f"MF p={result.get('mf_pvalue', 'N/A')} | "
            f"AZ1 p={result.get('az1_pvalue', 'N/A')} | "
            f"adequate={result['es_model_adequate']}"
        )
        return result

    # McNeil & Frey (2000): excesso de perda padronizado.
    def _mcneil_frey(
        self,
        returns: np.ndarray,
        var_f: np.ndarray,
        es_f: np.ndarray,
        hits: np.ndarray,
    ) -> Dict:
        excess = returns[hits] + var_f[hits]
        es_hit = es_f[hits]
        if len(excess) < 3:
            return {"stat": np.nan, "pvalue": np.nan}

        standardized = excess / np.maximum(np.abs(es_hit), 1e-10)

        t_stat, pval_two = stats.ttest_1samp(standardized, 0.0)
        pval = float(pval_two / 2) if t_stat < 0 else float(1.0 - pval_two / 2)

        return {
            "stat":        round(float(t_stat), 4),
            "pvalue":      round(float(pval), 4),
            "mean_excess": round(float(standardized.mean()), 6),
            "std_excess":  round(float(standardized.std()), 6),
        }

    # Acerbi & Szekely (2014), Statistic Z1.
    def _acerbi_szekely_z1(
        self,
        returns: np.ndarray,
        es_f: np.ndarray,
        hits: np.ndarray,
    ) -> Dict:
        T = len(returns)
        n_hit = hits.sum()
        if n_hit == 0:
            return {"stat": np.nan, "pvalue": 1.0}
        alpha_est = n_hit / T
        z1 = (np.sum(-returns[hits] / np.maximum(es_f[hits], 1e-10))
              / (T * max(alpha_est, 1e-6))) - 1.0

        rng = np.random.default_rng(42)
        z1_boot = []
        alpha_target = hits.mean()

        for _ in range(500):
            r_b = rng.choice(returns, size=T, replace=True)
            q_boot = np.quantile(r_b, 1.0 - alpha_target)
            hits_b = r_b < q_boot
            if hits_b.sum() == 0:
                continue
            es_b = -r_b[hits_b].mean()
            if es_b <= 0:
                continue
            z1_b = (np.sum(-r_b[hits_b] / es_b)
                    / (T * max(hits_b.mean(), 1e-6))) - 1.0
            z1_boot.append(z1_b)

        z1_boot = np.array(z1_boot)
        es_f_scalar = max(float(np.mean(np.abs(es_f[hits]))) if hits.sum() > 0 else 1.0, 1e-10)
        z1_boot_corrected = []
        rng2 = np.random.default_rng(123)
        for _ in range(500):
            r_b    = rng2.choice(returns, size=T, replace=True)
            q_boot = np.quantile(r_b, 1.0 - alpha_target)
            hits_b = r_b < q_boot
            if hits_b.sum() == 0:
                continue
            z1_b = (np.sum(-r_b[hits_b] / es_f_scalar)
                    / (T * max(hits_b.mean(), 1e-6))) - 1.0
            z1_boot_corrected.append(z1_b)
        z1_boot_corrected = np.array(z1_boot_corrected)
        pval = float(np.mean(z1_boot_corrected <= z1)) if len(z1_boot_corrected) > 0 else np.nan

        return {
            "stat":   round(float(z1), 6),
            "pvalue": round(float(pval), 4),
        }

    # Acerbi & Szekely (2014), Statistic Z2.
    def _acerbi_szekely_z2(
        self,
        returns: np.ndarray,
        es_f: np.ndarray,
        alpha_var: float,
    ) -> Dict:
        T = len(returns)
        exceed_es = (returns < -es_f).sum()
        expected  = T * alpha_var ** 2
        if expected < 1:
            return {"stat": np.nan, "pvalue": np.nan}
        z2 = exceed_es / max(expected, 1e-6) - 1.0
        pval = 1 - chi2.cdf(max(0, z2 * T), df=1) if np.isfinite(z2) else np.nan
        return {
            "stat":           round(float(z2), 6),
            "pvalue":         round(float(pval), 4) if np.isfinite(pval) else np.nan,
            "exceed_es_count": int(exceed_es),
        }

    # Score de Nolde & Ziegel (2017) para avaliação conjunta (VaR, ES).
    def _nolde_ziegel_score(
        self,
        returns: np.ndarray,
        var_f: np.ndarray,
        es_f: np.ndarray,
    ) -> float:
        alpha = 0.01
        hit = (returns < -var_f).astype(float)
        score = (hit - alpha) / np.maximum(es_f, 1e-10) + hit * returns / np.maximum(es_f ** 2, 1e-10)
        return round(float(score.mean()), 6)

    # Bootstrap CI para o ES histórico realizado.
    def _bootstrap_es(
        self,
        returns: np.ndarray,
        alpha_var: float,
        n_bootstrap: int = 1000,
        seed: int = 42,
    ) -> Dict:
        np.random.seed(seed)
        T = len(returns)
        var_hist = -np.quantile(returns, alpha_var)
        es_hist  = -returns[returns < -var_hist].mean() if (returns < -var_hist).any() else var_hist

        boot_es = []
        for _ in range(n_bootstrap):
            r_b    = np.random.choice(returns, T, replace=True)
            var_b  = -np.quantile(r_b, alpha_var)
            tail_b = r_b[r_b < -var_b]
            if len(tail_b) > 0:
                boot_es.append(-tail_b.mean())

        boot_es = np.array(boot_es)
        return {
            "es_hist":    round(float(es_hist), 6),
            "es_ci_low":  round(float(np.percentile(boot_es, 2.5)), 6) if len(boot_es) > 0 else np.nan,
            "es_ci_high": round(float(np.percentile(boot_es, 97.5)), 6) if len(boot_es) > 0 else np.nan,
        }


# EVT Walk-Forward Validator

# Walk-forward especializado para o pipeline EVT: re-estima GARCH-EVT
class EVTWalkForward:

    def __init__(
        self,
        estimation_window: int = 252,
        rebalancing_frequency: int = 5,
        copula_type: str = "cvine",
        confidence_levels: List[float] = None,
        max_trees: int = 3,
        min_tau: float = 0.05,
        n_simulations: int = 5000,
        risk_free_rate: float = 0.1075,
        seed: int = 42,
        n_jobs: int = -1,
    ):
        self.estimation_window     = estimation_window
        self.rebalancing_frequency = rebalancing_frequency

        self.copula_type           = copula_type
        self.confidence_levels     = confidence_levels or [0.95, 0.99]
        self.max_trees             = max_trees
        self.min_tau               = min_tau
        self.n_simulations         = n_simulations
        self.rf                    = risk_free_rate
        self.seed                  = seed
        self.n_jobs                = n_jobs

        self._period_metrics: List[Dict] = []

    # Executa a validação walk-forward EVT completa.
    def run(
        self,
        returns_df: pd.DataFrame,
        initial_weights: Optional[np.ndarray] = None,
    ) -> pd.DataFrame:
        _col_stds = returns_df.std()
        _median_std = float(_col_stds.median())
        if _median_std > 0.10:
            logger.warning(
                f"EVTWalkForward.run(): retornos detectados em escala percentual "
                f"(median std={_median_std:.4f} > 0.10). Dividindo por 100 automaticamente. "
                f"Para suprimir este aviso, passe retornos decimais diretamente."
            )
            returns_df = returns_df / 100.0
        elif _median_std < 1e-4:
            logger.warning(
                f"EVTWalkForward.run(): retornos com std muito baixo "
                f"(median std={_median_std:.6f}). Verifique a escala dos dados."
            )

        T, d = returns_df.shape
        dates = returns_df.index
        w = initial_weights if initial_weights is not None else np.ones(d) / d
        self._period_metrics = []

        regime_labels = self._detect_regime(returns_df)

        rebal_indices = list(range(
            self.estimation_window,
            T - 1,
            self.rebalancing_frequency,
        ))

        logger.info(
            f"EVTWalkForward | T={T} d={d} | "
            f"copula={self.copula_type} | "
            f"window={self.estimation_window} | "
            f"rebal={self.rebalancing_frequency} | "
            f"n_rebal={len(rebal_indices)} | "
            f"n_jobs={self.n_jobs}"
        )

        # ── Função de uma janela (serializável para joblib) ──────────────────
        def _process_one(idx, t_idx):
            t_start = t_idx - self.estimation_window
            train   = returns_df.iloc[t_start:t_idx]
            current_regime = int(regime_labels.iloc[t_idx]) if t_idx < len(regime_labels) else 0
            risk_metrics   = self._estimate_and_forecast(train, w, regime=current_regime)

            t_oos_end = min(t_idx + self.rebalancing_frequency, T)
            oos = returns_df.iloc[t_idx:t_oos_end]
            rows = []
            for t_oos in range(len(oos)):
                date  = oos.index[t_oos]
                ret_t = float(oos.iloc[t_oos].values @ w)
                row = {
                    "date":    date,
                    "return":  ret_t,
                    "weights": w.copy(),
                    "n_train": len(train),
                }
                for cl in self.confidence_levels:
                    cl_str = f"{int(cl*100)}"
                    var_t  = risk_metrics.get(f"var_{cl_str}", np.nan)
                    es_t   = risk_metrics.get(f"es_{cl_str}",  np.nan)
                    row[f"var_{cl_str}"] = var_t
                    row[f"es_{cl_str}"]  = es_t
                    row[f"hit_{cl_str}"] = (ret_t < -var_t if np.isfinite(var_t) else False)
                rows.append(row)

            if (idx + 1) % 5 == 0:
                logger.info(f"  Rebalanceamento {idx+1}/{len(rebal_indices)} em {dates[t_idx].date()}")
            return rows

        from joblib import Parallel, delayed as _delayed
        all_rows = Parallel(n_jobs=self.n_jobs, backend="loky", verbose=0)(
            _delayed(_process_one)(idx, t_idx)
            for idx, t_idx in enumerate(rebal_indices)
        )

        for rows in all_rows:
            self._period_metrics.extend(rows)

        return self._build_results_df()

    # Detecta regime de mercado por volatilidade rolling.
    def _detect_regime(self, returns_df: pd.DataFrame) -> pd.Series:
        port = returns_df.mean(axis=1)
        vol_short = port.rolling(21).std()
        vol_long  = port.rolling(63).std()
        regime = (vol_short > vol_long).astype(int).fillna(0)
        logger.debug(
            f"Regimes detectados: 0={int((regime==0).sum())}d, 1={int((regime==1).sum())}d"
        )
        return regime

    # Estima GARCH-EVT + vine copula na janela de treinamento e
    def _estimate_and_forecast(
        self, train: pd.DataFrame, weights: np.ndarray, regime: int = 0
    ) -> Dict:
        risk_metrics = {}
        try:
            import sys, os
            from pathlib import Path
            _backtesting_dir = Path(__file__).resolve().parent.parent
           
            from src.risk.semi_parametric import SemiParametricGARCH_EVT
            from src.copulas.vine_copulas    import CVineCopula, DVineCopula, RVineCopula
            from src.risk.copula_var_es   import CopulaEVTRisk, OptimizedCopulaRisk
            from src.copulas.estimation      import PITTransformer

            marginal_models = {}
            pit_data = []
            for col in train.columns:
                model = SemiParametricGARCH_EVT()
                try:
                    model.fit(train[col], run_diagnostics=False)
                    u = model.probability_integral_transform()
                    marginal_models[col] = model
                    pit_data.append(u)
                except Exception as _fit_err:
                    logger.warning(
                        f"    GARCH-EVT falhou para {col}: {_fit_err}. "
                        f"Usando ECDF empírica como marginal."
                    )
                    from scipy.stats import rankdata as _rankdata

                    # Marginal mínima: ECDF pura, compatível com CopulaEVTRisk.
                    class _EmpiricalMarginal:
                        def __init__(self, ret_arr):
                            self.returns = ret_arr
                            self.std_residuals = None
                            self._hybrid_cdf = None

                    r_arr = train[col].values.astype(float)
                    model = _EmpiricalMarginal(r_arr)
                    u = _rankdata(r_arr) / (len(r_arr) + 1)
                    marginal_models[col] = model
                    pit_data.append(u)

            pseudo_obs = np.clip(np.column_stack(pit_data), 1e-6, 1 - 1e-6)

            CopulaClass = {"cvine": CVineCopula, "dvine": DVineCopula,
                           "rvine": RVineCopula}.get(self.copula_type, CVineCopula)
            copula = CopulaClass(n_dim=train.shape[1])
            copula.fit(pseudo_obs, max_trees=self.max_trees, min_tau=self.min_tau)

            risk = CopulaEVTRisk()
            risk.fit(train, marginal_models, copula, list(train.columns))
            opt_risk = OptimizedCopulaRisk(risk)
            var_es_df = opt_risk.portfolio_var_es_batch(
                weights,
                confidence_levels=self.confidence_levels,
                n_simulations=self.n_simulations,
                seed=self.seed,
            )
            for _, row in var_es_df.iterrows():
                cl_str = f"{int(row['Confidence']*100)}"
                risk_metrics[f"var_{cl_str}"] = float(abs(row["VaR"]))
                risk_metrics[f"es_{cl_str}"]  = float(abs(row["ES"]))

            _var_values = [v for k, v in risk_metrics.items()
                           if k.startswith("var_") and np.isfinite(v)]
            if _var_values and max(_var_values) > 0.50:
                logger.warning(
                    f"  Pipeline EVT retornou VaR={max(_var_values):.4f} > 0.50 "
                    f"— escala de resíduos detectada. Descartando e usando fallback."
                )
                risk_metrics = {}
                raise ValueError(
                    f"Escala EVT inválida: VaR={max(_var_values):.4f}. "
                    f"Verifique model.returns vs model.std_residuals em "
                    f"SemiParametricGARCH_EVT."
                )

        except Exception as e:
            import traceback as _tb
            logger.warning(
                f"  Pipeline EVT falhou — {type(e).__name__}: {e}\n"
                + _tb.format_exc(limit=6)
                + "  → Usando t-Student + EWMA vol como fallback."
            )
            port_ret = (train * weights).sum(axis=1).values
            mu_p = port_ret.mean()

            _pstd = port_ret.std()
            if _pstd > 0.5:
                logger.error(
                    f"  Fallback t-Student: port_ret.std()={_pstd:.4f} > 0.5 — "
                    f"os retornos recebidos parecem estar na escala errada. "
                    f"Verifique returns_df passado para EVTWalkForward.run()."
                )

            lam = 0.94
            sq_demeaned = (port_ret - mu_p) ** 2
            weights_ewma = np.array(
                [(1 - lam) * lam ** i for i in range(len(sq_demeaned) - 1, -1, -1)]
            )
            weights_ewma /= weights_ewma.sum()
            sig_ewma = float(np.sqrt(np.dot(weights_ewma, sq_demeaned)))

            from scipy.stats import kurtosis as _kurtosis
            kurt_exc = float(_kurtosis(port_ret, fisher=True))
            if kurt_exc > 0.1:
                nu_est = np.clip(6.0 / kurt_exc + 4.0, 3.0, 30.0)
            else:
                nu_est = 30.0

            scale_t = sig_ewma * np.sqrt((nu_est - 2.0) / nu_est) if nu_est > 2 else sig_ewma

            logger.debug(
                f"    Fallback params: mu={mu_p:.6f} sig_ewma={sig_ewma:.6f} "
                f"nu={nu_est:.1f} scale_t={scale_t:.6f}"
            )

            for cl in self.confidence_levels:
                cl_str = f"{int(cl * 100)}"
                alpha  = 1.0 - cl
                q_t    = stats.t.ppf(alpha, df=nu_est)
                var_t  = float(abs(mu_p + q_t * scale_t))

                t_pdf_q = stats.t.pdf(q_t, df=nu_est)
                es_t = float(abs(
                    mu_p - scale_t * t_pdf_q / alpha * (nu_est + q_t ** 2) / (nu_est - 1)
                ))
                es_t = max(es_t, var_t * 1.01)

                risk_metrics[f"var_{cl_str}"] = var_t
                risk_metrics[f"es_{cl_str}"]  = es_t

        if regime == 1:
            stress_factor = 1.15
            logger.debug(f"  Regime estresse detectado → escalando VaR/ES por {stress_factor:.2f}")
            for key in list(risk_metrics.keys()):
                risk_metrics[key] = risk_metrics[key] * stress_factor

        return risk_metrics

    def _build_results_df(self) -> pd.DataFrame:
        if not self._period_metrics:
            return pd.DataFrame()
        rows = []
        for m in self._period_metrics:
            row = {
                "date":    m["date"],
                "return":  m["return"],
                "n_train": m["n_train"],
            }
            for k, v in m.items():
                if k not in ("date", "return", "n_train", "weights"):
                    row[k] = v
            rows.append(row)
        df = pd.DataFrame(rows).set_index("date")
        logger.info(f"EVTWalkForward concluído: {len(df)} observações OOS")
        return df

    # Métricas resumidas do walk-forward EVT.
    def summary(self, results_df: pd.DataFrame) -> Dict:
        out = {}
        for cl in self.confidence_levels:
            cl_str = f"{int(cl*100)}"
            hit_col = f"hit_{cl_str}"
            var_col = f"var_{cl_str}"
            es_col  = f"es_{cl_str}"
            if hit_col not in results_df.columns:
                continue
            n_hits = int(results_df[hit_col].sum())
            T      = len(results_df.dropna(subset=[hit_col]))
            out[f"violations_{cl_str}"]      = n_hits
            out[f"violation_rate_{cl_str}"]   = round(n_hits / max(T, 1), 4)
            out[f"expected_rate_{cl_str}"]    = round(1 - cl, 4)
            out[f"avg_var_{cl_str}"]          = round(float(results_df[var_col].dropna().mean()), 6)
            out[f"avg_es_{cl_str}"]           = round(float(results_df[es_col].dropna().mean()), 6)

            alpha_expected = 1 - cl
            from scipy.stats import chi2 as _chi2
            if n_hits > 0 and n_hits < T:
                lr_stat = 2 * (
                    n_hits * np.log(n_hits / (T * alpha_expected + 1e-10)) +
                    (T - n_hits) * np.log((T - n_hits) / (T * (1 - alpha_expected) + 1e-10))
                )
                kupiec_pval = round(float(1 - _chi2.cdf(lr_stat, df=1)), 4)
            else:
                kupiec_pval = 0.0
            out[f"kupiec_pvalue_{cl_str}"] = kupiec_pval
            out[f"kupiec_adequate_{cl_str}"] = kupiec_pval > 0.05
        out["total_return"] = round(float((1 + results_df["return"]).prod() - 1), 4)
        out["annual_vol"]   = round(float(results_df["return"].std() * np.sqrt(252)), 4)
        return out


# Copula Stability Validator

# Testa se os parâmetros estimados da vine copula são estáveis ao longo do
class CopulaStabilityValidator:

    def __init__(self, window: int = 63):
        self.window = window

    # Kendall τ rolling para todos os pares de ativos.
    def rolling_kendall_tau(
        self,
        returns_df: pd.DataFrame,
        step: int = 5,
    ) -> pd.DataFrame:
        if returns_df.std().median() > 0.10:
            returns_df = returns_df / 100.0

        T, d = returns_df.shape
        cols = returns_df.columns.tolist()
        pairs = [(i, j, f"{cols[i]}_{cols[j]}")
                 for i in range(d) for j in range(i+1, d)]

        records = []
        indices = []
        for t in range(self.window, T, step):
            block = returns_df.iloc[t - self.window: t].values
            row = {}
            for i, j, name in pairs:
                from scipy.stats import kendalltau
                tau, _ = kendalltau(block[:, i], block[:, j])
                row[name] = float(tau)
            records.append(row)
            indices.append(returns_df.index[t])

        df = pd.DataFrame(records, index=pd.DatetimeIndex(indices))
        logger.info(f"Rolling τ: {df.shape} (window={self.window}, step={step})")
        return df

    # CUSUM test para detectar quebra estrutural em séries de parâmetros.
    def cusum_test(
        self,
        parameter_series: pd.Series,
        significance: float = 0.05,
    ) -> Dict:
        x = parameter_series.dropna().values
        T = len(x)
        if T < 10:
            return {"break_detected": False, "break_date": None}

        mu = x.mean()
        s  = x.std() + 1e-10

        cusum     = np.cumsum((x - mu) / s)
        cusum_abs = np.abs(cusum)
        max_cusum = float(cusum_abs.max())

        cv = {0.10: 1.224, 0.05: 1.358, 0.01: 1.628}.get(significance, 1.358)
        normalized = max_cusum / np.sqrt(T)
        break_detected = normalized > cv

        break_idx = int(cusum_abs.argmax())
        break_date = parameter_series.dropna().index[break_idx] if break_detected else None

        return {
            "break_detected":  break_detected,
            "max_cusum_stat":  round(float(normalized), 4),
            "critical_value":  cv,
            "significance":    significance,
            "break_idx":       break_idx if break_detected else None,
            "break_date":      break_date,
        }

    # Testa autocorrelação de Ljung-Box nos parâmetros rolling.
    def parameter_autocorrelation(
        self,
        rolling_tau_df: pd.DataFrame,
        lags: int = 5,
    ) -> pd.DataFrame:
        try:
            from statsmodels.stats.diagnostic import acorr_ljungbox
        except ImportError:
            logger.warning("statsmodels não disponível para LB test.")
            return pd.DataFrame()

        rows = []
        for col in rolling_tau_df.columns:
            series = rolling_tau_df[col].dropna()
            if len(series) < 20:
                continue
            try:
                lb = acorr_ljungbox(series.values, lags=lags, return_df=True)
                pval_last = float(lb["lb_pvalue"].iloc[-1])
                rows.append({
                    "pair": col,
                    "lb_stat": round(float(lb["lb_stat"].iloc[-1]), 4),
                    "lb_pvalue": round(pval_last, 4),
                    "has_dynamics": pval_last < 0.05,
                    "mean_tau": round(float(series.mean()), 4),
                    "std_tau":  round(float(series.std()), 4),
                })
            except Exception:
                pass

        df = pd.DataFrame(rows)
        if len(df) > 0:
            n_dyn = df["has_dynamics"].sum()
            logger.info(
                f"LB autocorrelação dos τ: {n_dyn}/{len(df)} pares têm dinâmica temporal"
            )
        return df


# Regime Switching Validator

# Avalia se a detecção de regime de mercado agrega valor ao portfólio.
class RegimeSwitchingValidator:

    def __init__(self, risk_free_rate: float = 0.1075):
        self.rf = risk_free_rate

    # Compara performance do portfólio estático vs regime-switching.
    def compare_regime_vs_static(
        self,
        returns_df: pd.DataFrame,
        regime_labels: pd.Series,
        weights_static: pd.DataFrame,
        weights_regime: pd.DataFrame,
    ) -> Dict:
        common = returns_df.index

        def _port_returns(weights_df):
            common_idx = weights_df.index.intersection(common)
            w = weights_df.reindex(common_idx).ffill()
            r = returns_df.reindex(common_idx)
            return (r * w).sum(axis=1)

        ret_static = _port_returns(weights_static)
        ret_regime = _port_returns(weights_regime)

        from scipy.stats import ttest_rel
        n = min(len(ret_static), len(ret_regime))
        t_stat, pval = ttest_rel(ret_regime.values[-n:], ret_static.values[-n:])

        def _metrics(r):
            ann_ret = r.mean() * 252
            ann_vol = r.std() * np.sqrt(252)
            cum = (1 + r.values).cumprod()
            dd  = ((cum - np.maximum.accumulate(cum)) / np.maximum.accumulate(cum)).min()
            return {
                "annual_return": round(float(ann_ret), 4),
                "annual_vol":    round(float(ann_vol), 4),
                "sharpe":        round(float((ann_ret - self.rf) / max(ann_vol, 1e-8)), 4),
                "max_drawdown":  round(float(dd), 4),
            }

        static_m = _metrics(ret_static)
        regime_m = _metrics(ret_regime)

        return {
            "static":        static_m,
            "regime":        regime_m,
            "sharpe_lift":   round(float(regime_m["sharpe"] - static_m["sharpe"]), 4),
            "return_lift":   round(float(regime_m["annual_return"] - static_m["annual_return"]), 4),
            "t_stat":        round(float(t_stat), 4),
            "pvalue":        round(float(pval), 4),
            "regime_adds_value": pval < 0.05 and regime_m["sharpe"] > static_m["sharpe"],
        }

    # Estatísticas condicionais ao regime (vol, correlação, skewness).
    def regime_conditional_statistics(
        self,
        returns_df: pd.DataFrame,
        regime_labels: pd.Series,
    ) -> pd.DataFrame:
        if returns_df.std().median() > 0.10:
            returns_df = returns_df / 100.0
        port = returns_df.mean(axis=1)
        common = port.index.intersection(regime_labels.dropna().index)
        regimes = sorted(regime_labels.dropna().astype(int).unique())
        rows = []
        for r in regimes:
            mask = regime_labels.loc[common] == r
            ret  = port.loc[common][mask]
            corr_vals = []
            for i in range(returns_df.shape[1]):
                for j in range(i+1, returns_df.shape[1]):
                    sub = returns_df.loc[common][mask]
                    if len(sub) > 5:
                        c = float(sub.iloc[:, i].corr(sub.iloc[:, j]))
                        corr_vals.append(c)
            rows.append({
                "regime":         r,
                "n_days":         int(mask.sum()),
                "mean_ret":       round(float(ret.mean() * 252), 4),
                "vol":            round(float(ret.std() * np.sqrt(252)), 4),
                "skewness":       round(float(stats.skew(ret.values)), 4),
                "kurtosis":       round(float(stats.kurtosis(ret.values)), 4),
                "avg_corr":       round(float(np.mean(corr_vals)), 4) if corr_vals else np.nan,
            })
        return pd.DataFrame(rows).set_index("regime")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    np.random.seed(42)

    T, d = 600, 4
    tickers = [f"ATIVO{i+1}" for i in range(d)]
    dates   = pd.bdate_range("2021-01-04", periods=T)
    vols    = np.array([0.025, 0.022, 0.018, 0.015])
    R       = 0.35 * np.ones((d, d)) + 0.65 * np.eye(d)
    L       = np.linalg.cholesky(R)
    rets    = (np.random.standard_normal((T, d)) @ L.T) * vols
    returns_df = pd.DataFrame(rets, index=dates, columns=tickers)

    print(f"returns_df: {returns_df.shape}")

    print("\n ESBacktest ")
    port_ret = returns_df.mean(axis=1).values
    var_f    = np.full(T, 0.022)
    es_f     = np.full(T, 0.030)
    es_bt    = ESBacktest(alpha=0.05)
    res      = es_bt.run(port_ret, var_f, es_f, confidence_level=0.99, n_bootstrap=200)
    print(f"  MF stat={res['mf_stat']:.3f} p={res['mf_pvalue']:.3f}")
    print(f"  AZ1 stat={res['az1_stat']:.5f} p={res['az1_pvalue']:.3f}")
    print(f"  ES adequate: {res['es_model_adequate']}")
    print(f"  Boot ES hist: {res['boot_es_hist']:.4f} CI [{res['boot_es_ci_low']:.4f}, {res['boot_es_ci_high']:.4f}]")

    print("\n EVTWalkForward ")
    wf = EVTWalkForward(estimation_window=252, rebalancing_frequency=5)
    results_df = wf.run(returns_df)
    print(f"  Resultado OOS: {results_df.shape}")
    print(f"  Colunas: {list(results_df.columns)}")
    summary = wf.summary(results_df)
    print(f"  Summary: {summary}")
    for cl in [95, 99]:
        kp = summary.get(f"kupiec_pvalue_{cl}", "N/A")
        ok = summary.get(f"kupiec_adequate_{cl}", "N/A")
        print(f"  Kupiec {cl}%: p={kp} → {'✓ adequado' if ok else '✗ mal-calibrado'}")

    print("\n CopulaStabilityValidator ")
    csv = CopulaStabilityValidator(window=63)
    rolling_tau = csv.rolling_kendall_tau(returns_df, step=5)
    print(f"  Rolling τ: {rolling_tau.shape}")
    cusum_res = csv.cusum_test(rolling_tau.iloc[:, 0], significance=0.05)
    print(f"  CUSUM par 1: break={cusum_res['break_detected']} stat={cusum_res['max_cusum_stat']:.3f}")
    acf_df = csv.parameter_autocorrelation(rolling_tau)
    if not acf_df.empty:
        print(f"  LB dynamics: {acf_df['has_dynamics'].sum()}/{len(acf_df)} pares")

    print("\n RegimeSwitchingValidator ")
    regime_labels = pd.Series(
        (returns_df.mean(axis=1).rolling(21).std() > returns_df.mean(axis=1).rolling(252).std()).astype(int).values,
        index=dates,
    ).fillna(0)
    rv = RegimeSwitchingValidator(risk_free_rate=0.1075)
    stats_df = rv.regime_conditional_statistics(returns_df, regime_labels)
    print(stats_df.to_string())

    print("\n Todos os testes concluídos.")
