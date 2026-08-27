
from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize_scalar
from scipy.stats import chi2, norm, kstest, anderson

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Funções auxiliares
# ─────────────────────────────────────────────────────────────────────────────

def _sig_stars(p: float) -> str:
    if p < 0.01: return "***"
    if p < 0.05: return "**"
    if p < 0.10: return "*"
    return ""


# LR de Kupiec (POF) — retorna (stat, pvalue).
def _kupiec_lr(x: int, T: int, alpha: float) -> Tuple[float, float]:
    if T == 0:
        return np.nan, np.nan
    p = x / T
    if x == 0:
        lr = -2 * T * np.log(1 - alpha)
    elif x == T:
        lr = -2 * T * np.log(alpha)
    else:
        ll0 = x * np.log(alpha) + (T - x) * np.log(1 - alpha)
        ll1 = x * np.log(p) + (T - x) * np.log(1 - p)
        lr  = -2 * (ll0 - ll1)
    pv = 1 - chi2.cdf(lr, df=1) if np.isfinite(lr) else np.nan
    return float(lr), float(pv)


# LR de independência de Christoffersen — retorna (stat, pvalue).
def _christoffersen_ind(hits: np.ndarray) -> Tuple[float, float]:
    n00 = n01 = n10 = n11 = 0
    for t in range(1, len(hits)):
        if hits[t-1] == 0 and hits[t] == 0: n00 += 1
        elif hits[t-1] == 0 and hits[t] == 1: n01 += 1
        elif hits[t-1] == 1 and hits[t] == 0: n10 += 1
        else: n11 += 1
    pi01 = n01 / max(n00 + n01, 1)
    pi11 = n11 / max(n10 + n11, 1)
    pi   = (n01 + n11) / max(n00 + n01 + n10 + n11, 1)
    try:
        ll_i = (n00 + n10) * np.log(max(1-pi, 1e-10)) + (n01+n11) * np.log(max(pi, 1e-10))
        ll_d = (n00 * np.log(max(1-pi01, 1e-10)) + n01 * np.log(max(pi01, 1e-10))
                + n10 * np.log(max(1-pi11, 1e-10)) + n11 * np.log(max(pi11, 1e-10)))
        lr  = -2 * (ll_i - ll_d)
        pv  = 1 - chi2.cdf(lr, df=1)
    except Exception:
        lr, pv = np.nan, np.nan
    return float(lr), float(pv)


# ─────────────────────────────────────────────────────────────────────────────
# 1. EVTVaRBacktest
# ─────────────────────────────────────────────────────────────────────────────

# Backtesting de VaR estimado via GARCH-EVT.
class EVTVaRBacktest:

    def __init__(self, alpha_test: float = 0.05):
        self.alpha_test = alpha_test

    # Parâmetros
    def run(
        self,
        realized: np.ndarray,
        var_forecast: Union[np.ndarray, float],
        es_forecast: Optional[Union[np.ndarray, float]] = None,
        confidence_level: float = 0.99,
        model_name: str = "EVT",
        regime_labels: Optional[np.ndarray] = None,
    ) -> Dict:
        T = len(realized)
        alpha_var = 1 - confidence_level

        var_f = np.full(T, var_forecast) if np.isscalar(var_forecast) else np.asarray(var_forecast)
        hits  = (realized < -var_f).astype(int)
        n_hit = int(hits.sum())

        result: Dict = {
            "model":            model_name,
            "confidence_level": confidence_level,
            "T":                T,
            "n_violations":     n_hit,
            "expected_violations": round(T * alpha_var, 1),
            "violation_rate":   round(n_hit / T, 5),
            "expected_rate":    round(alpha_var, 5),
        }

        lr_pof, pv_pof = _kupiec_lr(n_hit, T, alpha_var)
        result.update({
            "kupiec_lr":    round(lr_pof, 4) if np.isfinite(lr_pof) else np.nan,
            "kupiec_pvalue": round(pv_pof, 4) if np.isfinite(pv_pof) else np.nan,
            "kupiec_stars": _sig_stars(pv_pof) if np.isfinite(pv_pof) else "",
            "kupiec_ok":    float(pv_pof) > self.alpha_test if np.isfinite(pv_pof) else None,
        })

        lr_ind, pv_ind = _christoffersen_ind(hits)
        result.update({
            "cc_ind_lr":    round(lr_ind, 4) if np.isfinite(lr_ind) else np.nan,
            "cc_ind_pvalue": round(pv_ind, 4) if np.isfinite(pv_ind) else np.nan,
            "cc_ind_ok":    float(pv_ind) > self.alpha_test if np.isfinite(pv_ind) else None,
        })

        lr_cc = (lr_pof if np.isfinite(lr_pof) else 0) + (lr_ind if np.isfinite(lr_ind) else 0)
        pv_cc = 1 - chi2.cdf(lr_cc, df=2)
        result.update({
            "cc_lr":    round(float(lr_cc), 4),
            "cc_pvalue": round(float(pv_cc), 4),
            "cc_ok":    float(pv_cc) > self.alpha_test,
        })

        if n_hit > 0:
            excess = -(realized[hits.astype(bool)] + var_f[hits.astype(bool)])
            result["excess_mean"]   = round(float(excess.mean()), 6)
            result["excess_max"]    = round(float(excess.max()), 6)
            result["excess_std"]    = round(float(excess.std()), 6)
            rel_exc = excess / np.maximum(var_f[hits.astype(bool)], 1e-8)
            result["excess_relative_mean"] = round(float(rel_exc.mean()), 4)
        else:
            result.update({
                "excess_mean": 0.0, "excess_max": 0.0,
                "excess_std": 0.0,  "excess_relative_mean": 0.0,
            })

        if regime_labels is not None and len(regime_labels) == T:
            regimes = sorted(np.unique(regime_labels[~np.isnan(regime_labels.astype(float))]))
            regime_stats = {}
            for r in regimes:
                mask = regime_labels == r
                n_r    = mask.sum()
                hit_r  = hits[mask].sum()
                regime_stats[f"regime_{int(r)}_n"]          = int(n_r)
                regime_stats[f"regime_{int(r)}_violations"] = int(hit_r)
                regime_stats[f"regime_{int(r)}_rate"]       = round(hit_r / max(n_r, 1), 4)
            result["regime_breakdown"] = regime_stats

        if es_forecast is not None:
            es_f = np.full(T, es_forecast) if np.isscalar(es_forecast) else np.asarray(es_forecast)
            es_result = self._es_adequacy(realized, var_f, es_f, hits)
            result.update({f"es_{k}": v for k, v in es_result.items()})

        result["model_adequate"] = (
            result.get("kupiec_ok", False) is True
            and result.get("cc_ind_ok", False) is True
        )

        logger.info(
            f"EVTVaRBacktest [{model_name}] {confidence_level:.0%}: "
            f"violations={n_hit}/{T} ({n_hit/T:.2%}) | "
            f"Kupiec p={result['kupiec_pvalue']} | "
            f"CC p={result['cc_pvalue']} | "
            f"adequate={'✓' if result['model_adequate'] else '✗'}"
        )
        return result

    # McNeil & Frey (2000) para ES: excesso padronizado deve ter média ≈ 0.
    def _es_adequacy(
        self,
        returns: np.ndarray,
        var_f: np.ndarray,
        es_f: np.ndarray,
        hits: np.ndarray,
    ) -> Dict:
        hit_mask = hits.astype(bool)
        if hit_mask.sum() < 3:
            return {"mcneil_frey_stat": np.nan, "mcneil_frey_pvalue": np.nan, "es_ok": None}
        excess = returns[hit_mask] + var_f[hit_mask]
        stand  = excess / np.maximum(np.abs(es_f[hit_mask]), 1e-10)
        t_stat, pval = stats.ttest_1samp(stand, 0.0)
        return {
            "mcneil_frey_stat":   round(float(t_stat), 4),
            "mcneil_frey_pvalue": round(float(pval), 4),
            "mcneil_frey_ok":     float(pval) > self.alpha_test,
            "es_mean_excess":     round(float(stand.mean()), 6),
        }


# ─────────────────────────────────────────────────────────────────────────────
# 2. GPDAdequacyTest
# ─────────────────────────────────────────────────────────────────────────────

# Verifica se a GPD ajustada representa adequadamente as caudas
class GPDAdequacyTest:

    def __init__(self, alpha_test: float = 0.05):
        self.alpha_test = alpha_test

    # exceedances_train : excessos sobre o threshold — janela de estimação
    def run(
        self,
        exceedances_train: np.ndarray,
        exceedances_test: np.ndarray,
        xi: float,
        sigma: float,
        threshold: float = 0.0,
    ) -> Dict:
        result = {
            "xi": round(xi, 5),
            "sigma": round(sigma, 5),
            "n_train": len(exceedances_train),
            "n_test":  len(exceedances_test),
        }

        if len(exceedances_test) < 5:
            result["warning"] = "menos de 5 excedâncias OOS — testes não confiáveis"
            return result

        # CDF GPD para os excessos
        def gpd_cdf(x):
            x = np.asarray(x, float)
            if abs(xi) < 1e-8:
                return 1 - np.exp(-x / sigma)
            t = 1 + xi * x / sigma
            t = np.maximum(t, 1e-10)
            return 1 - t ** (-1 / xi)

        ks_stat, ks_pval = kstest(exceedances_test, gpd_cdf)
        result["ks_stat"]   = round(float(ks_stat), 4)
        result["ks_pvalue"] = round(float(ks_pval), 4)
        result["ks_ok"]     = float(ks_pval) > self.alpha_test

        try:
            cdf_vals = np.clip(gpd_cdf(np.sort(exceedances_test)), 1e-10, 1-1e-10)
            n = len(cdf_vals)
            ranks = np.arange(1, n+1)
            ad_stat = -n - np.sum(
                (2*ranks-1)/n * (np.log(cdf_vals) + np.log(1-cdf_vals[::-1]))
            )
            ad_pval = 1 - self._ad_pvalue(float(ad_stat))
        except Exception:
            ad_stat, ad_pval = np.nan, np.nan
        result["ad_stat"]   = round(float(ad_stat), 4) if np.isfinite(ad_stat) else np.nan
        result["ad_pvalue"] = round(float(ad_pval), 4) if np.isfinite(ad_pval) else np.nan
        result["ad_ok"]     = float(ad_pval) > self.alpha_test if np.isfinite(ad_pval) else None

        pit_vals = gpd_cdf(exceedances_test)
        pit_ks, pit_pval = kstest(pit_vals, "uniform")
        result["pit_ks_stat"]   = round(float(pit_ks), 4)
        result["pit_ks_pvalue"] = round(float(pit_pval), 4)
        result["pit_ok"]        = float(pit_pval) > self.alpha_test

        xi_oos = self._hill_estimator(exceedances_test)
        result["xi_oos_hill"] = round(float(xi_oos), 4) if np.isfinite(xi_oos) else np.nan
        result["xi_drift"]    = round(float(abs(xi - xi_oos)), 4) if np.isfinite(xi_oos) else np.nan
        result["xi_stable"]   = float(abs(xi - xi_oos)) < 0.3 if np.isfinite(xi_oos) else None

        mef = self._mef_linearity(exceedances_test)
        result["mef_r2"]      = round(mef["r2"], 4)
        result["mef_slope"]   = round(mef["slope"], 4)
        result["mef_linear"]  = mef["r2"] > 0.80

        result["gpd_adequate"] = all([
            result.get("ks_ok", False) is True,
            result.get("pit_ok", False) is True,
        ])
        return result

    # Estimador de Hill para o índice de cauda ξ.
    @staticmethod
    def _hill_estimator(x: np.ndarray) -> float:
        n = len(x)
        if n < 10:
            return np.nan
        k = max(5, n // 5)
        xs = np.sort(x)[::-1]
        if xs[k] <= 0:
            return np.nan
        return float(np.mean(np.log(xs[:k])) - np.log(xs[k]))

    # Testa se a Mean Excess Function é linear (condição para GPD válida).
    @staticmethod
    def _mef_linearity(x: np.ndarray, n_thresh: int = 20) -> Dict:
        if len(x) < 20:
            return {"r2": np.nan, "slope": np.nan}
        qs = np.linspace(0.1, 0.8, n_thresh)
        threshs = np.quantile(x, qs)
        mef_vals = []
        for u in threshs:
            exc = x[x > u] - u
            mef_vals.append(np.mean(exc) if len(exc) > 3 else np.nan)
        mask = np.isfinite(mef_vals)
        if mask.sum() < 5:
            return {"r2": np.nan, "slope": np.nan}
        th = threshs[mask]
        mf = np.array(mef_vals)[mask]
        from scipy.stats import linregress
        slope, intercept, r, *_ = linregress(th, mf)
        return {"r2": float(r**2), "slope": float(slope)}

    # Aproximação do p-value de Anderson-Darling para distribuição contínua.
    @staticmethod
    def _ad_pvalue(ad: float) -> float:
        if ad < 0.2: return 1 - np.exp(-13.436 + 101.14*ad - 223.73*ad**2)
        if ad < 0.34: return 1 - np.exp(-8.318 + 42.796*ad - 59.938*ad**2)
        if ad < 0.6: return np.exp(0.9177 - 4.279*ad - 1.38*ad**2)
        return np.exp(1.2937 - 5.709*ad + 0.0186*ad**2)


# ─────────────────────────────────────────────────────────────────────────────
# 3. ThresholdStabilityTest
# ─────────────────────────────────────────────────────────────────────────────

# Avalia a estabilidade do threshold de EVT ao longo do tempo.
class ThresholdStabilityTest:

    def __init__(self, window: int = 252, step: int = 21):
        self.window = window
        self.step   = step

    # Estima threshold rolling e registra ξ, σ em cada janela.
    def rolling_threshold_analysis(
        self,
        returns: pd.Series,
        quantile_left: float = 0.05,
        quantile_right: float = 0.95,
    ) -> pd.DataFrame:
        T = len(returns)
        records = []

        for t in range(self.window, T, self.step):
            window_data = returns.iloc[t - self.window:t].values
            date = returns.index[t]

            thresh_left  = float(np.quantile(window_data, quantile_left))
            thresh_right = float(np.quantile(window_data, quantile_right))
            n_left  = int((window_data < thresh_left).sum())
            n_right = int((window_data > thresh_right).sum())

            xi_l, sigma_l = self._pwm_gpd(-window_data[window_data < thresh_left] + (-thresh_left))
            xi_r, sigma_r = self._pwm_gpd(window_data[window_data > thresh_right] - thresh_right)

            records.append({
                "date":          date,
                "threshold_left":  round(thresh_left, 6),
                "threshold_right": round(thresh_right, 6),
                "xi_left":       round(xi_l, 4),
                "sigma_left":    round(sigma_l, 4),
                "xi_right":      round(xi_r, 4),
                "sigma_right":   round(sigma_r, 4),
                "n_left":        n_left,
                "n_right":       n_right,
            })

        df = pd.DataFrame(records).set_index("date")
        logger.info(f"ThresholdStability: {len(df)} janelas rolling (window={self.window})")
        return df

    # Métricas de estabilidade dos parâmetros EVT.
    def stability_summary(self, rolling_df: pd.DataFrame) -> Dict:
        result = {}
        for col in ["threshold_left", "threshold_right", "xi_left", "xi_right",
                    "sigma_left", "sigma_right"]:
            if col not in rolling_df.columns:
                continue
            s = rolling_df[col].dropna()
            cv = float(s.std() / max(abs(s.mean()), 1e-10))
            result[f"{col}_mean"] = round(float(s.mean()), 4)
            result[f"{col}_std"]  = round(float(s.std()), 4)
            result[f"{col}_cv"]   = round(cv, 4)
            result[f"{col}_stable"] = cv < 0.25

        if "xi_left" in rolling_df.columns:
            cusum = self._cusum_stat(rolling_df["xi_left"].dropna().values)
            result["xi_left_cusum_stat"] = round(float(cusum), 4)
            result["xi_left_break"]      = cusum > 1.358

        return result

    # Quão sensível é o VaR EVT ao threshold escolhido?
    @staticmethod
    def threshold_sensitivity(
        returns: np.ndarray,
        base_quantile: float = 0.05,
        delta_pct: float = 0.10,
        confidence_level: float = 0.99,
    ) -> Dict:
        n = len(returns)
        alpha_var = 1 - confidence_level
        results = {}
        for factor in [1 - delta_pct, 1.0, 1 + delta_pct]:
            q = np.clip(base_quantile * factor, 0.01, 0.20)
            thresh = float(np.quantile(returns, q))
            losses = -returns[returns < thresh] - (-thresh)
            if len(losses) < 5:
                results[f"q{q:.3f}_var"] = np.nan
                continue
            xi, sigma = ThresholdStabilityTest._pwm_gpd(losses)
            zeta = q
            if abs(xi) < 1e-8:
                var_gpd = thresh - sigma * np.log(alpha_var / zeta)
            else:
                var_gpd = thresh + (sigma / xi) * ((alpha_var / zeta) ** (-xi) - 1)
            results[f"q{q:.3f}_xi"]  = round(xi, 4)
            results[f"q{q:.3f}_var"] = round(float(-var_gpd), 6)

        var_low  = results.get(f"q{base_quantile*(1-delta_pct):.3f}_var", np.nan)
        var_base = results.get(f"q{base_quantile:.3f}_var", np.nan)
        var_high = results.get(f"q{base_quantile*(1+delta_pct):.3f}_var", np.nan)
        if all(np.isfinite([var_low, var_base, var_high])) and var_base > 0:
            results["sensitivity_pct"] = round(
                float((var_high - var_low) / var_base * 100), 2
            )
        return results

    # Estimação GPD via Probability Weighted Moments (rápido, sem MLE).
    @staticmethod
    def _pwm_gpd(exceedances: np.ndarray) -> Tuple[float, float]:
        x = np.sort(exceedances[exceedances > 0])
        n = len(x)
        if n < 5:
            return 0.1, float(np.std(exceedances)) if len(exceedances) > 0 else 0.01
        a0 = float(np.mean(x))
        a1 = float(np.sum([(n - i) / (n - 1) * x[i] for i in range(n)]) / n)
        denom = a0 - 2 * a1
        xi    = 2 - a0 / max(abs(denom), 1e-10) * np.sign(denom)
        sigma = 2 * a0 * a1 / max(abs(denom), 1e-10)
        xi    = float(np.clip(xi, -0.8, 0.8))
        sigma = float(max(sigma, 1e-6))
        return xi, sigma

    # Estatística CUSUM normalizada para teste de quebra estrutural.
    @staticmethod
    def _cusum_stat(x: np.ndarray) -> float:
        T = len(x)
        if T < 5:
            return 0.0
        mu, s = x.mean(), x.std() + 1e-10
        cusum = np.cumsum((x - mu) / s)
        return float(np.abs(cusum).max() / np.sqrt(T))


# ─────────────────────────────────────────────────────────────────────────────
# 4. TailForecastEvaluation
# ─────────────────────────────────────────────────────────────────────────────

# Avalia a qualidade das previsões de quantis de cauda (VaR)
class TailForecastEvaluation:

    # Quantile Score — menor é melhor.
    def quantile_score(
        self,
        realized: np.ndarray,
        quantile_forecast: np.ndarray,
        alpha: float = 0.01,
    ) -> Dict:
        q = -quantile_forecast
        qs = (((realized < q).astype(float) - alpha) * (q - realized))
        return {
            "qs_mean":   round(float(qs.mean()), 6),
            "qs_std":    round(float(qs.std()), 6),
            "qs_total":  round(float(qs.sum()), 4),
            "alpha":     alpha,
        }

    # Asymmetric Loss Function.
    def asymmetric_loss(
        self,
        realized: np.ndarray,
        var_forecast: np.ndarray,
        c_over: float = 1.0,
        c_under: float = 2.5,
    ) -> Dict:
        diff = realized + var_forecast
        violation = realized < -var_forecast
        loss = np.where(violation,
                        c_over  * diff**2,
                        c_under * diff**2)
        return {
            "asymmetric_loss_mean": round(float(loss.mean()), 6),
            "c_over":  c_over,
            "c_under": c_under,
        }

    # Interval Score de Gneiting & Raftery (2007).
    def interval_score(
        self,
        realized: np.ndarray,
        lower: np.ndarray,
        upper: np.ndarray,
        alpha: float = 0.01,
    ) -> Dict:
        width = upper - lower
        pen_l = (2 / alpha) * np.maximum(lower - realized, 0)
        pen_u = (2 / alpha) * np.maximum(realized - upper, 0)
        is_   = width + pen_l + pen_u
        coverage = float(np.mean((realized >= lower) & (realized <= upper)))
        return {
            "interval_score_mean": round(float(is_.mean()), 6),
            "mean_width":          round(float(width.mean()), 6),
            "coverage":            round(coverage, 4),
            "expected_coverage":   1 - alpha,
        }

    # Diebold-Mariano (1995) via Quantile Score.
    def diebold_mariano(
        self,
        realized: np.ndarray,
        var_model1: np.ndarray,
        var_model2: np.ndarray,
        alpha: float = 0.01,
        h: int = 1,
    ) -> Dict:
        def _qs(r, q_f):
            q = -q_f
            return (((r < q).astype(float) - alpha) * (q - r))

        d1 = _qs(realized, var_model1)
        d2 = _qs(realized, var_model2)
        delta = d1 - d2

        T = len(delta)
        d_bar   = delta.mean()
        gamma0  = delta.var(ddof=1)
        nw_var  = gamma0
        for lag in range(1, h):
            gam = np.cov(delta[lag:], delta[:-lag])[0, 1]
            nw_var += 2 * (1 - lag / h) * gam
        nw_var = max(nw_var, 1e-10)

        dm_stat = d_bar / np.sqrt(nw_var / T)
        pval    = 2 * (1 - norm.cdf(abs(dm_stat)))

        return {
            "dm_stat":    round(float(dm_stat), 4),
            "pvalue":     round(float(pval), 4),
            "reject_H0":  float(pval) < 0.05,
            "better_model": "model_1" if dm_stat > 0 else "model_2",
            "qs_mean_1":  round(float(d1.mean()), 6),
            "qs_mean_2":  round(float(d2.mean()), 6),
        }

    # Tabela comparativa de múltiplos modelos via QS e Lopez loss.
    def compare_models(
        self,
        realized: np.ndarray,
        models: Dict[str, np.ndarray],
        alpha: float = 0.01,
    ) -> pd.DataFrame:
        rows = []
        for name, var_f in models.items():
            qs   = self.quantile_score(realized, var_f, alpha)
            loss = self.asymmetric_loss(realized, var_f)
            hits = (realized < -var_f).astype(int)
            n_v  = hits.sum()
            lr_k, pv_k = _kupiec_lr(int(n_v), len(realized), alpha)
            rows.append({
                "Modelo":          name,
                "QS médio":        qs["qs_mean"],
                "Assym. Loss":     loss["asymmetric_loss_mean"],
                "Violações":       n_v,
                "Taxa (%)":        round(n_v / len(realized) * 100, 2),
                "Kupiec p":        round(pv_k, 4) if np.isfinite(pv_k) else np.nan,
                "Adequado":        "✓" if (np.isfinite(pv_k) and pv_k > 0.05) else "✗",
            })
        return pd.DataFrame(rows).sort_values("QS médio")


# ─────────────────────────────────────────────────────────────────────────────
# 5. CopulaEVTJointBacktest
# ─────────────────────────────────────────────────────────────────────────────

# Testa a adequação do modelo completo (vine copula + GARCH-EVT)
class CopulaEVTJointBacktest:

    def __init__(self, alpha_test: float = 0.05):
        self.alpha_test  = alpha_test
        self._evaluator  = TailForecastEvaluation()
        self._var_tester = EVTVaRBacktest(alpha_test)

    # Parâmetros
    def run(
        self,
        realized: np.ndarray,
        var_models: Dict[str, np.ndarray],
        es_models: Optional[Dict[str, np.ndarray]] = None,
        confidence_level: float = 0.99,
    ) -> Dict:
        alpha_var = 1 - confidence_level
        results = {}

        for name, var_f in var_models.items():
            bt = self._var_tester.run(
                realized, var_f,
                es_forecast=es_models.get(name) if es_models else None,
                confidence_level=confidence_level,
                model_name=name,
            )
            results[name] = bt

        qs_table = self._evaluator.compare_models(
            realized, var_models, alpha=alpha_var
        )

        dm_results = {}
        model_names = list(var_models.keys())
        if len(model_names) >= 2:
            proposed_name = model_names[-1]
            proposed_var  = var_models[proposed_name]
            for name in model_names[:-1]:
                dm = self._evaluator.diebold_mariano(
                    realized, var_models[name], proposed_var,
                    alpha=alpha_var,
                )
                dm_results[f"{name}_vs_{proposed_name}"] = dm

        ranking = qs_table.copy()
        ranking["Kupiec_ok"] = ranking["Adequado"] == "✓"

        logger.info("CopulaEVTJointBacktest concluído:")
        for _, row in qs_table.iterrows():
            logger.info(
                f"  {row['Modelo']:20s}  QS={row['QS médio']:.4f}  "
                f"violations={row['Violações']}  "
                f"adequate={row['Adequado']}"
            )

        return {
            "backtest_by_model": results,
            "quantile_score_table": qs_table,
            "diebold_mariano": dm_results,
            "ranking": ranking,
        }

    # Decompõe o erro total em:
    def decompose_error(
        self,
        realized: np.ndarray,
        var_evt_vine: np.ndarray,
        var_evt_gauss: np.ndarray,
        var_normal: np.ndarray,
        confidence_level: float = 0.99,
    ) -> Dict:
        alpha = 1 - confidence_level
        qs_fn = lambda v: self._evaluator.quantile_score(realized, v, alpha)["qs_mean"]

        qs_norm   = qs_fn(var_normal)
        qs_eg     = qs_fn(var_evt_gauss)
        qs_ev     = qs_fn(var_evt_vine)

        return {
            "qs_normal":              round(qs_norm, 6),
            "qs_evt_gauss":           round(qs_eg, 6),
            "qs_evt_vine":            round(qs_ev, 6),
            "improvement_evt_margins": round(qs_norm - qs_eg, 6),
            "improvement_vine_copula": round(qs_eg  - qs_ev, 6),
            "total_improvement":       round(qs_norm - qs_ev, 6),
            "pct_from_margins":        round(
                (qs_norm - qs_eg) / max(abs(qs_norm - qs_ev), 1e-10) * 100, 1
            ),
            "pct_from_copula":         round(
                (qs_eg - qs_ev) / max(abs(qs_norm - qs_ev), 1e-10) * 100, 1
            ),
        }


# ─────────────────────────────────────────────────────────────────────────────
# 6. RollingParameterBacktest
# ─────────────────────────────────────────────────────────────────────────────

# Registra a evolução dos parâmetros do modelo ao longo das janelas
class RollingParameterBacktest:

    def __init__(
        self,
        estimation_window: int = 252,
        step: int = 21,
    ):
        self.estimation_window = estimation_window
        self.step = step
        self._records: List[Dict] = []

    # Adiciona registro de parâmetros para um período.
    def record(self, date: pd.Timestamp, params: Dict) -> None:
        record = {"date": date}
        record.update(params)
        self._records.append(record)

    # Executa fit_fn em cada janela e registra os parâmetros retornados.
    def run_rolling(
        self,
        returns_df: pd.DataFrame,
        fit_fn: Callable,
    ) -> pd.DataFrame:
        T = len(returns_df)
        self._records = []

        for t in range(self.estimation_window, T, self.step):
            train = returns_df.iloc[t - self.estimation_window:t]
            date  = returns_df.index[t - 1]
            try:
                params = fit_fn(train)
                self.record(date, params)
            except Exception as e:
                logger.debug(f"  rolling t={t}: fit_fn falhou ({e})")
                self.record(date, {})

        df = pd.DataFrame(self._records).set_index("date")
        logger.info(f"RollingParameterBacktest: {len(df)} janelas registradas")
        return df

    # Relatório de estabilidade: média, desvio, CV e flag de instabilidade
    def stability_report(self, params_df: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for col in params_df.columns:
            s = params_df[col].dropna()
            if len(s) < 3:
                continue
            cv = float(s.std() / max(abs(s.mean()), 1e-10))
            cusum = ThresholdStabilityTest._cusum_stat(s.values)
            rows.append({
                "Parâmetro":  col,
                "Média":      round(float(s.mean()), 5),
                "Desvio":     round(float(s.std()), 5),
                "CV (%)":     round(cv * 100, 2),
                "CUSUM stat": round(cusum, 3),
                "Estável":    "✓" if (cv < 0.30 and cusum < 1.358) else "✗",
                "n_janelas":  len(s),
            })
        return pd.DataFrame(rows).set_index("Parâmetro")

    # Dicionário compacto com métricas de evolução para tabela LaTeX.
    def parameter_evolution_summary(self, params_df: pd.DataFrame) -> Dict:
        out = {}
        for col in params_df.columns:
            s = params_df[col].dropna()
            if len(s) < 3:
                continue
            out[col] = {
                "mean":  round(float(s.mean()), 4),
                "std":   round(float(s.std()), 4),
                "min":   round(float(s.min()), 4),
                "max":   round(float(s.max()), 4),
                "cv":    round(float(s.std() / max(abs(s.mean()), 1e-10)), 4),
            }
        return out


# ─────────────────────────────────────────────────────────────────────────────
# Suite completa: EVTBacktestSuite
# ─────────────────────────────────────────────────────────────────────────────

# Orquestra todos os testes de backtesting EVT em uma única chamada.
class EVTBacktestSuite:

    def __init__(self, alpha_test: float = 0.05):
        self.alpha_test     = alpha_test
        self.var_backtest   = EVTVaRBacktest(alpha_test)
        self.gpd_adequacy   = GPDAdequacyTest(alpha_test)
        self.thresh_test    = ThresholdStabilityTest()
        self.tail_eval      = TailForecastEvaluation()
        self.joint_backtest = CopulaEVTJointBacktest(alpha_test)

    # Executa bateria completa de testes EVT. Retorna dict estruturado
    def run_all(
        self,
        realized_returns: np.ndarray,
        var_evt_vine: np.ndarray,
        returns_df: Optional[pd.DataFrame] = None,
        es_evt_vine: Optional[np.ndarray] = None,
        var_normal: Optional[np.ndarray] = None,
        var_t_copula: Optional[np.ndarray] = None,
        var_evt_gauss: Optional[np.ndarray] = None,
        exceedances_train: Optional[np.ndarray] = None,
        exceedances_test: Optional[np.ndarray] = None,
        gpd_xi: Optional[float] = None,
        gpd_sigma: Optional[float] = None,
        regime_labels: Optional[np.ndarray] = None,
        confidence_level: float = 0.99,
        verbose: bool = True,
    ) -> Dict:
        report = {}

        report["var_evt_vine"] = self.var_backtest.run(
            realized_returns, var_evt_vine,
            es_forecast=es_evt_vine,
            confidence_level=confidence_level,
            model_name="EVT-Vine",
            regime_labels=regime_labels,
        )

        if (exceedances_train is not None and exceedances_test is not None
                and gpd_xi is not None and gpd_sigma is not None):
            report["gpd_adequacy"] = self.gpd_adequacy.run(
                exceedances_train, exceedances_test, gpd_xi, gpd_sigma
            )

        if returns_df is not None:
            port_series = pd.Series(
                realized_returns,
                index=returns_df.index[-len(realized_returns):],
            )
            rolling_df = self.thresh_test.rolling_threshold_analysis(port_series)
            report["threshold_stability"] = self.thresh_test.stability_summary(rolling_df)
            report["threshold_sensitivity"] = self.thresh_test.threshold_sensitivity(
                realized_returns
            )

        var_models: Dict[str, np.ndarray] = {"EVT-Vine": var_evt_vine}
        if var_normal    is not None: var_models["Normal"]    = var_normal
        if var_t_copula  is not None: var_models["t-Copula"]  = var_t_copula
        if var_evt_gauss is not None: var_models["EVT-Gauss"] = var_evt_gauss

        if len(var_models) > 1:
            joint = self.joint_backtest.run(
                realized_returns, var_models,
                confidence_level=confidence_level,
            )
            report["joint_comparison"] = joint

            if var_evt_gauss is not None and var_normal is not None:
                report["error_decomposition"] = self.joint_backtest.decompose_error(
                    realized_returns,
                    var_evt_vine, var_evt_gauss, var_normal,
                    confidence_level,
                )

        qs_tab = self.tail_eval.compare_models(
            realized_returns, var_models, alpha=1 - confidence_level
        )
        report["quantile_score_table"] = qs_tab

        if verbose:
            self._print_summary(report)

        return report

    def _print_summary(self, report: Dict):
        print("\n" + "=" * 60)
        print("EVT BACKTEST SUITE — RESUMO")
        print("=" * 60)

        vb = report.get("var_evt_vine", {})
        print(f"\nVaR EVT-Vine {vb.get('confidence_level', '?'):.0%}:")
        print(f"  Violações: {vb.get('n_violations', '?')}/{vb.get('T', '?')} "
              f"({vb.get('violation_rate', 0):.2%}  esperado: {vb.get('expected_rate', 0):.2%})")
        print(f"  Kupiec p={vb.get('kupiec_pvalue', '?')}  CC p={vb.get('cc_pvalue', '?')}")
        print(f"  Adequado: {'✓' if vb.get('model_adequate') else '✗'}")

        if "gpd_adequacy" in report:
            ga = report["gpd_adequacy"]
            print(f"\nGPD Adequacy OOS:")
            print(f"  KS p={ga.get('ks_pvalue', '?')}  PIT p={ga.get('pit_ks_pvalue', '?')}")
            print(f"  ξ train={ga.get('xi', '?'):.4f}  ξ OOS (Hill)={ga.get('xi_oos_hill', '?'):.4f}")
            print(f"  MEF R²={ga.get('mef_r2', '?'):.3f}  GPD ok: {'✓' if ga.get('gpd_adequate') else '✗'}")

        if "quantile_score_table" in report:
            print(f"\nQuantile Score (menor = melhor):")
            print(report["quantile_score_table"][
                ["Modelo", "QS médio", "Violações", "Kupiec p", "Adequado"]
            ].to_string(index=False))

        if "error_decomposition" in report:
            ed = report["error_decomposition"]
            print(f"\nDecomposição do erro:")
            print(f"  Melhora EVT marginais: {ed['improvement_evt_margins']:.4f} ({ed['pct_from_margins']:.1f}%)")
            print(f"  Melhora vine copula:   {ed['improvement_vine_copula']:.4f} ({ed['pct_from_copula']:.1f}%)")

        print("=" * 60)

    # Salva os DataFrames do report em CSV.
    def save_report(self, report: Dict, path: Union[str, Path]) -> List[Path]:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        saved = []
        for key, val in report.items():
            if isinstance(val, pd.DataFrame):
                fp = path / f"evt_bt_{key}.csv"
                val.to_csv(fp, index=False)
                saved.append(fp)
            elif isinstance(val, dict):
                flat = {k: v for k, v in val.items() if not isinstance(v, (dict, pd.DataFrame))}
                if flat:
                    fp = path / f"evt_bt_{key}.csv"
                    pd.DataFrame([flat]).to_csv(fp, index=False)
                    saved.append(fp)
        logger.info(f"EVTBacktestSuite: {len(saved)} arquivos salvos em {path}")
        return saved


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    np.random.seed(42)

    T_TOTAL = 1500
    DF_TRUE = 5
    SCALE   = 0.015
    CL      = 0.99

    dates    = pd.bdate_range("2018-01-02", periods=T_TOTAL)
    port_ret = stats.t.rvs(df=DF_TRUE, size=T_TOTAL, scale=SCALE, random_state=42)
    port_ret -= port_ret.mean()

    T_TRAIN  = int(T_TOTAL * 0.60)
    ret_train = port_ret[:T_TRAIN]
    ret_test  = port_ret[T_TRAIN:]
    T_TEST    = len(ret_test)

    sigma_train = float(np.std(ret_train))
    sigma_test  = float(np.std(ret_test))

    print(f"Dados: T_total={T_TOTAL}  T_train={T_TRAIN}  T_test={T_TEST}")
    print(f"sigma_train={sigma_train:.4f}  sigma_test={sigma_test:.4f}  nu_verdadeiro={DF_TRUE}")

    THRESH_Q   = 0.10
    thresh_gpd = float(np.quantile(-ret_train, 1 - THRESH_Q))
    exc_train  = (-ret_train[ret_train < -thresh_gpd]) - thresh_gpd
    xi_est, sigma_est = ThresholdStabilityTest._pwm_gpd(exc_train)

    exc_test = (-ret_test[ret_test < -thresh_gpd]) - thresh_gpd

    print(f"GPD threshold={thresh_gpd:.4f} (q={THRESH_Q:.0%})")
    print(f"  xi={xi_est:.3f}  sigma={sigma_est:.4f}  "
          f"n_exc_train={len(exc_train)}  n_exc_test={len(exc_test)}")

    alpha_var = 1 - CL

    var_normal_val = float(stats.norm.ppf(CL) * sigma_train)

    kurt_exc = float(stats.kurtosis(ret_train, fisher=True))
    nu_est   = float(np.clip(6.0 / max(kurt_exc, 0.1) + 4.0, 4.0, 30.0))
    t_scale  = sigma_train / float(np.sqrt(nu_est / (nu_est - 2)))
    var_t_val = float(stats.t.ppf(CL, df=nu_est) * t_scale)

    zeta = THRESH_Q
    if abs(xi_est) < 1e-8:
        var_evt_val = thresh_gpd + sigma_est * (-np.log(alpha_var / zeta))
    else:
        var_evt_val = thresh_gpd + (sigma_est / xi_est) * ((alpha_var / zeta) ** (-xi_est) - 1)
    var_evt_val = float(var_evt_val)

    B_BOOT = 500
    f_grid = np.linspace(0.90, 1.30, 41)
    viol_rates = np.zeros(len(f_grid))
    rng = np.random.default_rng(42)

    LAM = 0.94
    ewma_var_tr = np.zeros(T_TRAIN)
    ewma_var_tr[0] = float(np.var(ret_train[:21]))
    for t in range(1, T_TRAIN):
        ewma_var_tr[t] = LAM * ewma_var_tr[t-1] + (1 - LAM) * ret_train[t-1] ** 2
    vol_train = np.sqrt(ewma_var_tr)

    for fi, f in enumerate(f_grid):
        var_candidate = vol_train * (var_evt_val * f / sigma_train)
        boot_rates = np.zeros(B_BOOT)
        for b in range(B_BOOT):
            idx = rng.integers(0, T_TRAIN, size=T_TRAIN)
            r_b = ret_train[idx]
            v_b = var_candidate[idx]
            boot_rates[b] = float(np.mean(r_b < -v_b))
        viol_rates[fi] = float(boot_rates.mean())

    best_fi       = int(np.argmin(np.abs(viol_rates - alpha_var)))
    vine_factor   = float(f_grid[best_fi])
    var_evt_v_val = var_evt_val * vine_factor

    print(f"  Vine factor calibrado: {vine_factor:.3f}  "
          f"(taxa bootstrap={viol_rates[best_fi]:.4f}, alvo={alpha_var:.4f})")

    if abs(xi_est) < 1e-8:
        es_evt_v_val = var_evt_v_val + sigma_est
    elif xi_est < 1.0:
        es_evt_v_val = (var_evt_v_val + sigma_est - xi_est * thresh_gpd) / (1 - xi_est)
    else:
        es_evt_v_val = var_evt_v_val * 1.5

    print(f"\nVaR 99% calibrado no treino:")
    print(f"  Normal    = {var_normal_val:.4f}")
    print(f"  t-Student = {var_t_val:.4f}  (nu_est={nu_est:.1f})")
    print(f"  EVT-Gauss = {var_evt_val:.4f}")
    print(f"  EVT-Vine  = {var_evt_v_val:.4f}")
    print(f"  ES EVT-V  = {es_evt_v_val:.4f}")

    LAM      = 0.94
    ewma_var = np.zeros(T_TOTAL)
    ewma_var[0] = sigma_train ** 2
    for t in range(1, T_TOTAL):
        ewma_var[t] = LAM * ewma_var[t-1] + (1 - LAM) * port_ret[t-1] ** 2
    vol_test = np.sqrt(ewma_var[T_TRAIN:])

    var_normal = vol_test * (var_normal_val  / sigma_train)
    var_t      = vol_test * (var_t_val       / sigma_train)
    var_evt_g  = vol_test * (var_evt_val     / sigma_train)
    var_evt_v  = vol_test * (var_evt_v_val   / sigma_train)
    es_evt_v   = vol_test * (es_evt_v_val    / sigma_train)

    print("\n── EVTVaRBacktest ──")
    vbt = EVTVaRBacktest()
    for name, var_f in [("Normal", var_normal), ("t-Student", var_t),
                        ("EVT-Vine", var_evt_v)]:
        r = vbt.run(ret_test, var_f, confidence_level=CL, model_name=name)
        print(f"  {name:12s}: violations={r['n_violations']}/{T_TEST}  "
              f"Kupiec p={r['kupiec_pvalue']}  adequate={'✓' if r['model_adequate'] else '✗'}")

    print("\n── GPDAdequacyTest ──")
    gpd_test = GPDAdequacyTest()
    ga       = gpd_test.run(exc_train, exc_test, xi_est, sigma_est)
    mef_r2   = ga.get('mef_r2', float('nan'))
    mef_str  = f"{mef_r2:.3f}" if np.isfinite(mef_r2) else "nan"
    print(f"  n_exc_train={ga['n_train']}  n_exc_test={ga['n_test']}")
    print(f"  KS p={ga.get('ks_pvalue','?')}  PIT p={ga.get('pit_ks_pvalue','?')}  "
          f"MEF R2={mef_str}  GPD ok: {'✓' if ga.get('gpd_adequate') else '✗'}")

    print("\n── ThresholdStabilityTest ──")
    port_series  = pd.Series(port_ret, index=dates)
    tst          = ThresholdStabilityTest(window=252, step=21)
    rolling_df   = tst.rolling_threshold_analysis(port_series)
    stab_summary = tst.stability_summary(rolling_df)
    print(f"  Janelas rolling: {len(rolling_df)}")
    for k, v in stab_summary.items():
        if isinstance(v, bool):
            print(f"  {k}: {'estavel' if v else 'INSTAVEL'}")
    sens = tst.threshold_sensitivity(port_ret)
    print(f"  Sensibilidade VaR: {sens.get('sensitivity_pct', 'N/A')}%")

    print("\n── TailForecastEvaluation ──")
    tfe    = TailForecastEvaluation()
    qs_tab = tfe.compare_models(
        ret_test,
        {"Normal": var_normal, "t-Student": var_t,
         "EVT-Gauss": var_evt_g, "EVT-Vine": var_evt_v},
        alpha=alpha_var,
    )
    print(qs_tab.to_string(index=False))

    dm = tfe.diebold_mariano(ret_test, var_normal, var_evt_v, alpha=alpha_var)
    print(f"\n  DM (Normal vs EVT-Vine): stat={dm['dm_stat']}  p={dm['pvalue']}  "
          f"better={dm['better_model']}")

    print("\n── EVTBacktestSuite.run_all() ──")
    suite  = EVTBacktestSuite()
    report = suite.run_all(
        realized_returns=ret_test,
        var_evt_vine=var_evt_v,
        es_evt_vine=es_evt_v,
        var_normal=var_normal,
        var_t_copula=var_t,
        var_evt_gauss=var_evt_g,
        exceedances_train=exc_train,
        exceedances_test=exc_test,
        gpd_xi=xi_est,
        gpd_sigma=sigma_est,
        confidence_level=CL,
        verbose=True,
    )

    print("\n Todos os testes concluídos.")
