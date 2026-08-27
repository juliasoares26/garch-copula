import sys
from pathlib import Path

MARGINALS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(MARGINALS_DIR))

import pandas as pd
import numpy as np
from typing import Optional, Dict, Tuple
import logging
from arch import arch_model
import warnings
from scipy import stats
import matplotlib.pyplot as plt
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy.stats import skew

from evt_gpd import GPD

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SemiParametricGARCH_EVT:
  def __init__(self):
    self.garch_model = None
    self.garch_result = None
    self.residuals = None
    self.standardized_residuals = None

    self.gpd_left = None
    self.gpd_right = None

    self.threshold_left = None
    self.threshold_right = None

    self.diagnostics = {}
    self.tail_properties = {}

    self.fitted = False


  # Valida parâmetros GPD para evitar valores extremos
  def _validate_gpd_params(self, xi, sigma, side='left'):
    
    if xi < -0.9:
        logger.warning(f"ξ {side} muito negativo ({xi:.4f}), clipping para -0.9")
        xi = -0.9
    
    if xi > 0.5:
        logger.warning(f"ξ {side} muito positivo ({xi:.4f}), clipping para 0.5")
        xi = 0.5
    
    if sigma <= 0:
        raise ValueError(f"σ {side} deve ser positivo, obtido: {sigma}")
    
    if xi < 0:
        upper_endpoint = -sigma / xi
        if upper_endpoint < 2.0:
            logger.warning(f"Upper endpoint {side} muito baixo: {upper_endpoint:.4f}")
    
    return xi, sigma

  def fit(
    self,
    returns: pd.Series,
    garch_spec: str = 'GARCH',
    garch_p: int = 1,
    garch_q: int = 1,
    dist: str = 'skewt',
    threshold_method: str = 'quantile',
    left_quantile: float = 0.10,
    right_quantile: float = 0.90,
    run_diagnostics: bool = True,
    plot_diagnostics: bool = False,
) -> Dict:

    logger.info("Ajustando modelo semi-paramétrico GARCH-EVT")
    
    self._original_returns = returns.copy()
    self.returns = returns.copy()

    if len(returns) < 500:
        logger.warning(
            f"Amostra pequena: {len(returns)} obs (mínimo recomendado 500). "
            f"GPD menos confiável — prosseguindo."
        )

    if returns.std() < 0.001:
        logger.warning(
            f"Volatilidade muito baixa: {returns.std():.6f}. "
            f"GPD pode não convergir — CDF empírica usada como fallback."
        )

    if np.abs(returns.skew()) > 20:
        logger.warning(f"Skewness extrema detectada: {returns.skew():.2f}")
        logger.warning("Aplicando transformação robusta...")
        q01 = returns.quantile(0.001)
        q99 = returns.quantile(0.999)
        returns = returns.clip(q01, q99)
        self.returns = returns.copy()
    
    self.quantile_left = left_quantile
    self.quantile_right = right_quantile
    
    logger.info("Ajustando GARCH...")
    self._fit_garch_adaptive(returns, dist)
    
    logger.info("Extraindo resíduos padronizados...")
    self._extract_residuals()
    
    if run_diagnostics:
        logger.info("Executando diagnósticos GARCH...")
        self.diagnose_garch()
    
    if run_diagnostics:
        logger.info("Analisando propriedades das caudas...")
        self.diagnose_tails(plot=plot_diagnostics)
    
    logger.info("Selecionando thresholds...")
    if threshold_method == 'hill':
        self._select_thresholds_hill()
    else:
        self._select_thresholds_auto(left_quantile, right_quantile)
    
    logger.info("Ajustando GPD para ambas as caudas...")
    self._fit_tail_gpd()
    
    self._fitted = True
    self.fitted = True
    
    logger.info("Modelo ajustado com sucesso!")
    
    return self.get_summary()

  # Calcula a Cumulative Distribution Function para um valor x usando o modelo semi-paramét…
  def cdf(self, x: float) -> float:
    if not self._fitted:
        raise ValueError("Modelo não foi ajustado. Chame fit() primeiro.")
    
    z = self._standardize_return(x)
    
    return self._hybrid_cdf(z)

  # Inversa da CDF híbrida (quantile function): dado u em (0,1), retorna o
  def ppf(self, u, t: int = None):
    if not self._fitted:
        raise ValueError("Modelo não foi ajustado. Chame fit() primeiro.")

    u_arr = np.atleast_1d(np.asarray(u, dtype=float))
    u_arr = np.clip(u_arr, 1e-10, 1 - 1e-10)

    z_out = np.empty_like(u_arr)

    zeta_l = self.quantile_left
    zeta_u = 1.0 - self.quantile_right
    u_left  = self.threshold_left  if self.threshold_left  is not None else -np.inf
    u_right = self.threshold_right if self.threshold_right is not None else  np.inf

    mask_left   = (u_arr < zeta_l) & (self.gpd_left is not None)
    mask_right  = (u_arr > (1.0 - zeta_u)) & (self.gpd_right is not None)
    mask_center = ~mask_left & ~mask_right

    if np.any(mask_left):
        gpd_sf = np.clip(u_arr[mask_left] / zeta_l, 1e-12, 1.0)
        excess = np.array([self.gpd_left.quantile(1.0 - sf) for sf in gpd_sf])
        z_out[mask_left] = u_left - excess

    if np.any(mask_right):
        gpd_sf = np.clip((1.0 - u_arr[mask_right]) / zeta_u, 1e-12, 1.0)
        excess = np.array([self.gpd_right.quantile(1.0 - sf) for sf in gpd_sf])
        z_out[mask_right] = u_right + excess

    if np.any(mask_center):
        resid = self.std_residuals
        center_mask_resid = (resid >= u_left) & (resid <= u_right)
        center_vals = np.sort(resid[center_mask_resid])
        if len(center_vals) == 0:
            z_out[mask_center] = stats.norm.ppf(u_arr[mask_center])
        else:
            emp_p = (u_arr[mask_center] - zeta_l) / (1.0 - zeta_l - zeta_u)
            emp_p = np.clip(emp_p, 0.0, 1.0)
            z_out[mask_center] = np.quantile(center_vals, emp_p)

    mean = self.returns.mean()
    if t is None:
        vol = self.conditional_vol.mean()
    else:
        vol = self.conditional_vol.iloc[t]

    x_out = mean + vol * z_out

    return float(x_out[0]) if np.isscalar(u) or np.ndim(u) == 0 else x_out

  # Converte retorno em resíduo padronizado usando volatilidade condicional
  def _standardize_return(self, x: float, t: int = None) -> float:
    mean = self.returns.mean()
    if t is None:
        vol = self.conditional_vol.mean()
    else:
        vol = self.conditional_vol.iloc[t]

    z = (x - mean) / vol
    return z

  # CDF híbrida: usa distribuição GARCH no centro e GPD nas caudas
  def _hybrid_cdf(self, z: float) -> float:
    u_left  = self.threshold_left  if self.threshold_left  is not None else -np.inf
    u_right = self.threshold_right if self.threshold_right is not None else  np.inf
    zeta_l = self.quantile_left
    zeta_u = 1.0 - self.quantile_right

    if z <= u_left and self.gpd_left is not None:
        excess = u_left - z
        xi  = self.gpd_left.xi
        sig = self.gpd_left.sigma
        if xi < 0:
            max_excess = -sig / xi
            excess = min(excess, max_excess * 0.9999)
        if abs(xi) < 1e-8:
            gpd_sf = np.exp(-excess / sig)
        else:
            gpd_sf = max((1.0 + xi * excess / sig) ** (-1.0 / xi), 0.0)
        return float(np.clip(zeta_l * gpd_sf, 0.0, 1.0))

    elif z >= u_right and self.gpd_right is not None:
        excess = z - u_right
        xi  = self.gpd_right.xi
        sig = self.gpd_right.sigma
        if xi < 0:
            max_excess = -sig / xi
            excess = min(excess, max_excess * 0.9999)
        if abs(xi) < 1e-8:
            gpd_sf = np.exp(-excess / sig)
        else:
            gpd_sf = max((1.0 + xi * excess / sig) ** (-1.0 / xi), 0.0)
        return float(np.clip(1.0 - zeta_u * gpd_sf, 0.0, 1.0))

    else:
        resid = self.std_residuals
        center_mask = (resid >= u_left) & (resid <= u_right)
        center_vals  = resid[center_mask]
        if len(center_vals) == 0:
            return float(np.clip(stats.norm.cdf(z), 0.0, 1.0))
        emp_cdf = np.mean(center_vals <= z)
        return float(np.clip(zeta_l + (1.0 - zeta_l - zeta_u) * emp_cdf, 0.0, 1.0))
    
  def probability_integral_transform(self, returns=None):
    if not hasattr(self, '_fitted') or not self._fitted:
        raise RuntimeError("Modelo precisa ser ajustado antes de usar PIT.")

    resid = self.std_residuals
    n = len(resid)

    ranks = np.argsort(np.argsort(resid))
    u = (ranks + 1) / (n + 1)

    return np.clip(u, 1e-6, 1 - 1e-6)
  # CDF semi-paramétrica: GARCH no centro + GPD nas caudas
  def _semi_parametric_cdf(self, z: float) -> float:
    if z < self.threshold_left:
        p_left = self.left_quantile
        
        excess = self.threshold_left - z
        
        if self.gpd_left_xi != 0:
            gpd_cdf = 1 - (1 + self.gpd_left_xi * excess / self.gpd_left_sigma) ** (-1/self.gpd_left_xi)
        else:
            gpd_cdf = 1 - np.exp(-excess / self.gpd_left_sigma)
        
        return p_left * (1 - gpd_cdf)
    
    elif z > self.threshold_right:
        p_right = 1 - self.right_quantile
        
        excess = z - self.threshold_right
        
        if self.gpd_right_xi != 0:
            gpd_cdf = 1 - (1 + self.gpd_right_xi * excess / self.gpd_right_sigma) ** (-1/self.gpd_right_xi)
        else:
            gpd_cdf = 1 - np.exp(-excess / self.gpd_right_sigma)
        
        return self.right_quantile + p_right * gpd_cdf
    
    else:
        empirical_cdf = np.mean(self.std_residuals <= z)
        
        return (self.left_quantile + 
                (self.right_quantile - self.left_quantile) * empirical_cdf)
                
  def _fit_garch(self, returns: pd.Series, spec: str, p: int, q: int, dist: str):
    try:
      if spec.upper() == 'GARCH':
        vol = 'GARCH'
      elif spec.upper() == 'EGARCH':
        vol = 'EGARCH'
      elif spec.upper() in ['GJR', 'GJR-GARCH']:
        vol = 'GARCH'
        o = 1 
      else:
        raise ValueError(f"spec desconhecido: {spec}")
      
      if spec.upper() in ['GJR', 'GJR-GARCH']:
        self.garch_model = arch_model(
          returns,
          vol=vol,
          p=p,
          o=1,
          q=q,
          dist=dist,
          rescale=True
        )
      else:
        self.garch_model = arch_model(
          returns,
          vol=vol,
          p=p,
          q=q,
          dist=dist,
          rescale=True
      )

      with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        self.garch_result = self.garch_model.fit(disp='off',
                                                  show_warning=False,
                                                  options={'ftol': 1e-10})
  
      logger.info(f"Garch({p}, {q}) ajustado com dist={dist}")
      logger.info(f"AIC: {self.garch_result.aic:.2f}")
      logger.info(f"BIC: {self.garch_result.bic:.2f}")

    except Exception as e:
      logger.error(f"Erro ao ajustar GARCH: {str(e)}")
      raise
    
  def _fit_garch_adaptive(self, returns: pd.Series, dist: str = 'skewt'):
    
      specs_to_test = [
        ('GARCH', 1, 0, 1),
        ('GARCH', 2, 0, 1),
        ('GARCH', 1, 1, 1),
        ('GARCH', 2, 1, 1),
        ('EGARCH', 1, 0, 1),
    ]
    
      best_bic = np.inf
      best_result = None
      best_spec = None
    
      for vol_type, p, o, q in specs_to_test:
        try:
            if vol_type == 'GARCH' and o > 0:
                model = arch_model(returns, vol='GARCH', p=p, o=o, q=q, 
                                   dist=dist, rescale=True)
            else:
                model = arch_model(returns, vol=vol_type, p=p, q=q, 
                                   dist=dist, rescale=True)
            
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = model.fit(disp='off', show_warning=False, 
                                   options={'ftol': 1e-10})
            
            resid_std = result.resid / result.conditional_volatility
            resid_sq = resid_std ** 2
            lb_result = acorr_ljungbox(resid_sq, lags=10, return_df=True)
            n_sig = (lb_result['lb_pvalue'] < 0.05).sum()
            
            bic_adjusted = result.bic + (n_sig * 50)
            
            spec_label = (f"GJR-GARCH({p},{q})" if o > 0 and vol_type == 'GARCH'
                          else f"{vol_type}({p},{o},{q})")
            logger.info(f"  {spec_label}: BIC={result.bic:.2f}, "
                       f"autocorr_lags={n_sig}/10")
            
            if bic_adjusted < best_bic:
                best_bic = bic_adjusted
                best_result = result
                best_spec = (vol_type, p, o, q)
                
        except Exception as e:
            logger.warning(f"  {vol_type}({p},{o},{q}) falhou: {e}")
    
      if best_result is None:
        raise ValueError("Todas especificações GARCH falharam")
    
      vol_type, p, o, q = best_spec
      best_label = (f"GJR-GARCH({p},{q})" if o > 0 and vol_type == 'GARCH'
                    else f"{vol_type}({p},{o},{q})")
      self.garch_result = best_result
      _scale = float(getattr(best_result.model, '_scale', 1.0) or 1.0)
      _cv_raw = best_result.conditional_volatility.values
      _cv_corrected = _cv_raw / _scale
      _ret_std = float(np.std(np.asarray(self.returns).ravel(), ddof=1))
      _cv_mean = float(np.mean(_cv_corrected))
      if _cv_mean > 5.0 * _ret_std or _cv_mean < 0.01 * _ret_std:
          logger.debug(
              f"conditional_vol após _scale={_scale:.1f}: mean={_cv_mean:.6f}, "
              f"ret_std={_ret_std:.6f}. Usando std(returns) como sigma_bar."
          )
          _cv_corrected = np.full_like(_cv_raw, _ret_std)
      self.conditional_vol = pd.Series(
          _cv_corrected,
          index=best_result.conditional_volatility.index,
      )
      logger.info(f"✓ Melhor modelo: {best_label}")
      logger.info(f"  BIC={best_result.bic:.2f}, AIC={best_result.aic:.2f}")

      return best_result
  
  def _extract_residuals(self) -> None:
    self._extract_standardized_residuals()
  
  def _extract_standardized_residuals(self) -> None:
    resid = self.garch_result.resid / self.garch_result.conditional_volatility
    
    resid_mean = np.mean(resid)
    resid_std = np.std(resid)
    resid = (resid - resid_mean) / resid_std
    
    p1, p99 = np.percentile(resid, [1, 99])
    iqr_robust = p99 - p1
    lower_bound = p1 - 3 * iqr_robust
    upper_bound = p99 + 3 * iqr_robust
    
    n_outliers = np.sum((resid < lower_bound) | (resid > upper_bound))
    pct_outliers = 100 * n_outliers / len(resid)
    
    if pct_outliers > 5:
        logger.warning(f"Clipping {n_outliers} outliers extremos ({pct_outliers:.1f}%)")
        lower_bound = np.percentile(resid, 0.1)
        upper_bound = np.percentile(resid, 99.9)
        resid = np.clip(resid, lower_bound, upper_bound)
        
        resid = (resid - np.mean(resid)) / np.std(resid)
    elif n_outliers > 0:
        logger.info(f"Clipping {n_outliers} outliers extremos")
        resid = np.clip(resid, lower_bound, upper_bound)
        resid = (resid - np.mean(resid)) / np.std(resid)
    
    final_std = np.std(resid)
    final_skew = abs(skew(resid))
    
    if final_std < 0.9 or final_std > 1.1:
        logger.warning(f"Std final fora do ideal: {final_std:.4f} (esperado ~1.0)")
    
    _skew_iter = 0
    _clip_pct = 0.5
    while abs(skew(resid)) > 3 and _skew_iter < 3 and _clip_pct <= 1.5:
        clip_lo = np.percentile(resid, _clip_pct)
        clip_hi = np.percentile(resid, 100 - _clip_pct)
        resid = np.clip(resid, clip_lo, clip_hi)
        resid = (resid - np.mean(resid)) / np.std(resid)
        _clip_pct += 0.5
        _skew_iter += 1
    
    if abs(skew(resid)) > 3:
        logger.warning(f"Skewness alta após padronização: {abs(skew(resid)):.4f}")
    
    self.standardized_residuals = pd.Series(resid, index=self.garch_result.resid.index)
    self.std_residuals = self.standardized_residuals
    logger.info(f"Resíduos extraídos: média={np.mean(resid):.4f}, std={np.std(resid):.4f}")

  def _select_thresholds_auto(self, left_quantile: float = 0.05, right_quantile: float = 0.95):
    resid = self.standardized_residuals.values
    
    self.threshold_left = np.quantile(resid, left_quantile)
    self.threshold_right = np.quantile(resid, right_quantile)
    
    n_left = np.sum(resid < self.threshold_left)
    n_right = np.sum(resid > self.threshold_right)
    
    logger.info(f"threshold left (q={left_quantile}): {self.threshold_left:.4f} ({n_left} obs)")
    logger.info(f"threshold right (q={right_quantile}): {self.threshold_right:.4f} ({n_right} obs)")

  def _select_thresholds_hill(self):
    resid = self.standardized_residuals.values

    left_data = -resid[resid < 0]
    left_sorted = np.sort(left_data)[::-1]
    n_left = len(left_sorted)

    k_min, k_max = max(10, int(0.05 * n_left)), int(0.20 * n_left)
    k_values = np.arange(k_min, k_max)
    hill_estimates = []

    for k in k_values:
      log_ratios = np.log(left_sorted[:k]) - np.log(left_sorted[k])
      hill_estimates.append(np.mean(log_ratios))

    best_k_idx = np.argmin(np.abs(np.array(hill_estimates) - 0.2))
    optimal_k_left = k_values[best_k_idx]
    self.threshold_left = left_sorted[optimal_k_left]

    right_data = resid[resid > 0]
    right_sorted = np.sort(right_data)[::-1]
    n_right = len(right_sorted)

    k_min, k_max = max(10, int(0.05 * n_right)), int(0.20 * n_right)
    k_values = np.arange(k_min, k_max)
    hill_estimates = []

    for k in k_values:
      log_ratios = np.log(right_sorted[:k]) - np.log(right_sorted[k])
      hill_estimates.append(np.mean(log_ratios))

    best_k_idx = np.argmin(np.abs(np.array(hill_estimates) - 0.2))
    optimal_k_right = k_values[best_k_idx]
    self.threshold_right = right_sorted[optimal_k_right]

    logger.info(f"Hill threshold left: {self.threshold_left:.4f} (k={optimal_k_left})")
    logger.info(f"Hill threshold right: {self.threshold_right:.4f} (k={optimal_k_right})")

  def _fit_tail_gpd(self):
    resid = self.standardized_residuals.values

    left_mask = resid < self.threshold_left
    n_left = np.sum(left_mask)

    if n_left < 30:
        logger.warning(f"Poucas excedências na cauda esquerda ({n_left}), usando empírico")
        self.gpd_left = None
    else:
        left_exceedances = np.abs(resid[left_mask] - self.threshold_left)
        
        if len(np.unique(left_exceedances)) < 10:
            logger.warning("Excedências esquerdas com pouca variabilidade, usando empírico")
            self.gpd_left = None
        else:
            self.gpd_left = GPD()
            try:
                result_left = self.gpd_left.fit(
                    left_exceedances,
                    threshold=0,
                    method='mle',
                    return_std_errors=True
                )

                xi_left = result_left['xi']
                sigma_left = result_left['sigma']
                
                if abs(xi_left - (-0.5)) < 0.001:
                    logger.debug(
                        f"ξ esquerda saturou no bound MLE ({xi_left:.4f}). "
                        f"Aceitando -0.5 — cauda Weibull na borda da região identificável."
                    )
                
                if xi_left is not None:
                    if abs(xi_left) > 1.0:
                        logger.warning(f"ξ esquerda extremo ({xi_left:.4f}), usando empírico")
                        self.gpd_left = None
                    elif sigma_left <= 0 or sigma_left > 100:
                        logger.warning(f"σ esquerda inválido ({sigma_left:.4f}), usando empírico")
                        self.gpd_left = None
                    else:
                        logger.info("GPD cauda esquerda:")
                        xi_se = result_left.get('xi_se')
                        sigma_se = result_left.get('sigma_se')
                        
                        if xi_se is not None:
                            logger.info(f"ξ = {xi_left:.4f} ± {xi_se:.4f}")
                        else:
                            logger.info(f"ξ = {xi_left:.4f}")
                        
                        if sigma_se is not None:
                            logger.info(f"σ = {sigma_left:.4f} ± {sigma_se:.4f}")
                        else:
                            logger.info(f"σ = {sigma_left:.4f}")
                        
                        logger.info(f"  Exceedances: {n_left}")
                        
            except Exception as e:
                logger.error(f"Erro GPD esquerda: {e}, usando empírico")
                self.gpd_left = None

    right_mask = resid > self.threshold_right
    n_right = np.sum(right_mask)

    if n_right < 30:
        logger.warning(f"Poucas excedências na cauda direita ({n_right}), usando empírico")
        self.gpd_right = None
    else:
        right_exceedances = resid[right_mask] - self.threshold_right
        
        if len(np.unique(right_exceedances)) < 10:
            logger.warning("Excedências direitas com pouca variabilidade, usando empírico")
            self.gpd_right = None
        else:
            self.gpd_right = GPD()
            try:
                result_right = self.gpd_right.fit(
                    right_exceedances,
                    threshold=0,
                    method='mle',
                    return_std_errors=True
                )

                xi_right = result_right['xi']
                sigma_right = result_right['sigma']
                
                if abs(xi_right - (-0.5)) < 0.001:
                    logger.debug(
                        f"ξ direita saturou no bound MLE ({xi_right:.4f}). "
                        f"Aceitando -0.5 — cauda Weibull na borda da região identificável."
                    )

                if xi_right is not None:
                    if abs(xi_right) > 1.0:
                        logger.warning(f"ξ direita extremo ({xi_right:.4f}), usando empírico")
                        self.gpd_right = None
                    elif sigma_right <= 0 or sigma_right > 100:
                        logger.warning(f"σ direita inválido ({sigma_right:.4f}), usando empírico")
                        self.gpd_right = None
                    else:
                        logger.info("GPD cauda direita:")
                        xi_se = result_right.get('xi_se')
                        sigma_se = result_right.get('sigma_se')
                        
                        if xi_se is not None:
                            logger.info(f"  ξ = {xi_right:.4f} ± {xi_se:.4f}")
                        else:
                            logger.info(f"  ξ = {xi_right:.4f}")
                        
                        if sigma_se is not None:
                            logger.info(f"  σ = {sigma_right:.4f} ± {sigma_se:.4f}")
                        else:
                            logger.info(f"  σ = {sigma_right:.4f}")
                        
                        logger.info(f"  Exceedances: {n_right}")
                        
            except Exception as e:
                logger.error(f"Erro GPD direita: {e}, usando empírico")
                self.gpd_right = None

    MIN_SIGMA = 0.05

    if self.gpd_left is not None:
      if self.gpd_left.sigma < MIN_SIGMA or self.gpd_left.xi > 0.9:
        logger.warning(
            f"GPD esquerda problemática (ξ={self.gpd_left.xi:.4f}, "
            f"σ={self.gpd_left.sigma:.4f}), usando empírico"
        )
        self.gpd_left = None

    if self.gpd_right is not None:
      if self.gpd_right.sigma < MIN_SIGMA or self.gpd_right.xi > 0.9:
        logger.warning(
            f"GPD direita problemática (ξ={self.gpd_right.xi:.4f}, "
            f"σ={self.gpd_right.sigma:.4f}), usando empírico"
        )
        self.gpd_right = None
    if self.gpd_left is not None:
        self.gpd_left_xi = self.gpd_left.xi
        self.gpd_left_sigma = self.gpd_left.sigma
    else:
        self.gpd_left_xi = None
        self.gpd_left_sigma = None
    
    if self.gpd_right is not None:
        self.gpd_right_xi = self.gpd_right.xi
        self.gpd_right_sigma = self.gpd_right.sigma
    else:
        self.gpd_right_xi = None
        self.gpd_right_sigma = None
    
    self.std_residuals = resid


  # Valida e corrige parâmetros GPD extremos
  def _validate_and_fix_gpd(self):
    import logging
    logger = logging.getLogger(__name__)
    
    if self.gpd_left is not None:
        xi_left = self.gpd_left.xi
        sigma_left = self.gpd_left.sigma
        
        if sigma_left < 0.1:
            logger.warning(f"GPD Left: σ muito pequeno ({sigma_left:.4f}), ajustando para 0.5")
            sigma_left = 0.5
        
        if xi_left < -0.9:
            logger.warning(f"GPD Left: ξ muito negativo ({xi_left:.4f}), clipping para -0.8")
            xi_left = -0.8
        
        if xi_left > 0.5:
            logger.warning(f"GPD Left: ξ muito positivo ({xi_left:.4f}), clipping para 0.5")
            xi_left = 0.5
        
        if xi_left < 0:
            endpoint = -sigma_left / xi_left
            if endpoint < 1.5:
                logger.warning(f"GPD Left: endpoint muito baixo ({endpoint:.2f}), reajustando")
                xi_left = -sigma_left / 3.0
        
        if abs(xi_left - self.gpd_left.xi) > 1e-6 or abs(sigma_left - self.gpd_left.sigma) > 1e-6:
            logger.info(f"Recriando GPD esquerda: ξ={xi_left:.4f}, σ={sigma_left:.4f}")
            from evt_gpd import GPD
            self.gpd_left = GPD()
            self.gpd_left.xi = xi_left
            self.gpd_left.sigma = sigma_left
            self.gpd_left.threshold = self.threshold_left
            self.gpd_left.n_exceedances = len(self.returns[self.returns < self.threshold_left])
    
    if self.gpd_right is not None:
        xi_right = self.gpd_right.xi
        sigma_right = self.gpd_right.sigma
        
        if sigma_right < 0.1:
            logger.warning(f"GPD Right: σ muito pequeno ({sigma_right:.4f}), ajustando para 0.5")
            sigma_right = 0.5
        
        if xi_right < -0.9:
            logger.warning(f"GPD Right: ξ muito negativo ({xi_right:.4f}), clipping para -0.8")
            xi_right = -0.8
        
        if xi_right > 0.5:
            logger.warning(f"GPD Right: ξ muito positivo ({xi_right:.4f}), clipping para 0.5")
            xi_right = 0.5
        
        if xi_right < 0:
            endpoint = -sigma_right / xi_right
            if endpoint < 1.5:
                logger.warning(f"GPD Right: endpoint muito baixo ({endpoint:.2f}), reajustando")
                xi_right = -sigma_right / 3.0
        
        if abs(xi_right - self.gpd_right.xi) > 1e-6 or abs(sigma_right - self.gpd_right.sigma) > 1e-6:
            logger.info(f"Recriando GPD direita: ξ={xi_right:.4f}, σ={sigma_right:.4f}")
            from evt_gpd import GPD
            self.gpd_right = GPD()
            self.gpd_right.xi = xi_right
            self.gpd_right.sigma = sigma_right
            self.gpd_right.threshold = self.threshold_right
            self.gpd_right.n_exceedances = len(self.returns[self.returns > self.threshold_right])
            
  def diagnose_tails(self, plot: bool = True) -> Dict:
    if self.standardized_residuals is None:
      raise ValueError("Execute _extract_residuals primeiro")
    
    resid = self.standardized_residuals.values

    diagnostics = {}

    _, p_shapiro = stats.shapiro(resid[:5000] if len(resid) > 5000 else resid)
    diagnostics['shapiro_pvalue'] = p_shapiro

    diagnostics['kurtosis_excess'] = stats.kurtosis(resid)
    diagnostics['skewness'] = stats.skew(resid)

    left_data = -resid[resid < 0]
    right_data = resid[resid > 0]

    if len(left_data) > 50:
      left_sorted = np.sort(left_data)[::-1]
      k = min(100, len(left_sorted) // 4)
      hill_left = np.mean(np.log(left_sorted[:k]) - np.log(left_sorted[k])) 
      diagnostics['hill_xi_left'] = hill_left

    if len(right_data) > 50:
      right_sorted = np.sort(right_data)[::-1]
      k = min(100, len(right_sorted) // 4)
      hill_right = np.mean(np.log(right_sorted[:k]) - np.log(right_sorted[k])) 
      diagnostics['hill_xi_right'] = hill_right

    if 'hill_xi_left' in diagnostics:
      if diagnostics['hill_xi_left'] > 0.5:
        tail_type_left = 'muito pesada - fréchet'
      elif diagnostics['hill_xi_left'] > 0.1:
        tail_type_left = "pesada - pareto"
      elif diagnostics['hill_xi_left'] > -0.1:
          tail_type_left = "exponencial - Gumbel"
      else:
          tail_type_left = "limitada - Weibull"
      diagnostics['tail_type_left'] = tail_type_left

    if 'hill_xi_right' in diagnostics:
      if diagnostics['hill_xi_right'] > 0.5:
        tail_type_right = 'muito pesada - fréchet'
      elif diagnostics['hill_xi_right'] > 0.1:
        tail_type_right = "pesada - pareto"
      elif diagnostics['hill_xi_right'] > -0.1:
          tail_type_right = "exponencial - Gumbel"
      else:
          tail_type_right = "limitada - Weibull"
      diagnostics['tail_type_right'] = tail_type_right

    xi_left = diagnostics.get('hill_xi_left', 0)
    xi_right = diagnostics.get('hill_xi_right', 0)

    if xi_left > 0.2 and xi_right > 0.2:
      copula_suggestion = 't-student - tail-dependence simetrica'
    elif xi_left > 0.2 and xi_right < 0.1:
      copula_suggestion = "clayton - lower tail dependence"
    elif xi_left < 0.1 and xi_right > 0.2:
      copula_suggestion = "gumbel ou joe - upper tail dependence"
    else:
      copula_suggestion = "gaussian - sem tail dependence"
    
    diagnostics['copula_suggestion'] = copula_suggestion

    self.tail_properties = diagnostics

    if plot:
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        stats.probplot(resid, dist="norm", plot=axes[0, 0])
        axes[0, 0].set_title('QQ-Plot (Normal)')
        
        axes[0, 1].hist(resid, bins=50, density=True, alpha=0.7, edgecolor='black')
        x = np.linspace(resid.min(), resid.max(), 100)
        axes[0, 1].plot(x, stats.norm.pdf(x, resid.mean(), resid.std()), 'r-', lw=2)
        axes[0, 1].set_title('Distribuição dos Resíduos')
        axes[0, 1].set_xlabel('Resíduos Padronizados')
        
        if len(left_data) > 50:
            left_sorted = np.sort(left_data)[::-1]
            k_range = range(10, min(200, len(left_sorted) // 2))
            hill_estimates = [np.mean(np.log(left_sorted[:k]) - np.log(left_sorted[k])) 
                            for k in k_range]
            axes[0, 2].plot(k_range, hill_estimates)
            axes[0, 2].axhline(y=diagnostics['hill_xi_left'], color='r', linestyle='--')
            axes[0, 2].set_title('Hill Plot - Cauda Esquerda')
            axes[0, 2].set_xlabel('k')
            axes[0, 2].set_ylabel('ξ estimado')
        
        if len(right_data) > 50:
            right_sorted = np.sort(right_data)[::-1]
            k_range = range(10, min(200, len(right_sorted) // 2))
            hill_estimates = [np.mean(np.log(right_sorted[:k]) - np.log(right_sorted[k])) 
                            for k in k_range]
            axes[1, 0].plot(k_range, hill_estimates)
            axes[1, 0].axhline(y=diagnostics['hill_xi_right'], color='r', linestyle='--')
            axes[1, 0].set_title('Hill Plot - Cauda Direita')
            axes[1, 0].set_xlabel('k')
            axes[1, 0].set_ylabel('ξ estimado')
        
        if len(left_data) > 50:
            thresholds = np.percentile(left_data, np.linspace(80, 98, 20))
            mean_excess = [np.mean(left_data[left_data > u] - u) for u in thresholds]
            axes[1, 1].plot(thresholds, mean_excess, 'o-')
            axes[1, 1].set_title('Mean Excess Plot - Esquerda')
            axes[1, 1].set_xlabel('Threshold u')
            axes[1, 1].set_ylabel('Mean Excess')
        
        if len(right_data) > 50:
            thresholds = np.percentile(right_data, np.linspace(80, 98, 20))
            mean_excess = [np.mean(right_data[right_data > u] - u) for u in thresholds]
            axes[1, 2].plot(thresholds, mean_excess, 'o-')
            axes[1, 2].set_title('Mean Excess Plot - Direita')
            axes[1, 2].set_xlabel('Threshold u')
            axes[1, 2].set_ylabel('Mean Excess')
        
        plt.tight_layout()
        plt.savefig('tail_diagnostics.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    logger.info("Diagnóstico de Caudas")
    logger.info(f"Kurtosis excess: {diagnostics['kurtosis_excess']:.4f}")
    logger.info(f"Skewness: {diagnostics['skewness']:.4f}")
    
    if 'hill_xi_left' in diagnostics:
        logger.info(f"Hill ξ esquerda: {diagnostics['hill_xi_left']:.4f}")
    else:
        logger.info("Hill ξ esquerda: N/A")
    
    if 'hill_xi_right' in diagnostics:
        logger.info(f"Hill ξ direita: {diagnostics['hill_xi_right']:.4f}")
    else:
        logger.info("Hill ξ direita: N/A")
    
    logger.info(f"Tipo cauda esquerda: {diagnostics.get('tail_type_left', 'N/A')}")
    logger.info(f"Tipo cauda direita: {diagnostics.get('tail_type_right', 'N/A')}")
    logger.info(f"Sugestão de copula: {diagnostics['copula_suggestion']}")
    
    if 'hill_xi_left' in diagnostics and 'hill_xi_right' in diagnostics:
        left_sorted = np.sort(-resid[resid < 0])[::-1]
        right_sorted = np.sort(resid[resid > 0])[::-1]
        
        k_range = range(30, min(150, len(left_sorted)//3))
        hill_left_estimates = [np.mean(np.log(left_sorted[:k]) - np.log(left_sorted[k])) 
                               for k in k_range]
        hill_right_estimates = [np.mean(np.log(right_sorted[:k]) - np.log(right_sorted[k])) 
                                for k in k_range]
        
        stability_left = np.std(hill_left_estimates[-50:])
        stability_right = np.std(hill_right_estimates[-50:])
        
        diagnostics['hill_stability_left'] = stability_left
        diagnostics['hill_stability_right'] = stability_right
        
        if stability_left > 0.1 or stability_right > 0.1:
            logger.warning("Hill plots NÃO estáveis (σ > 0.1)")
        else:
            logger.info("Hill plots estáveis")
    
    return diagnostics

  def diagnose_garch(self) -> Dict:
    if self.standardized_residuals is None:
      raise ValueError("ajuste o modelo primeiro")
    
    resid_sq = self.standardized_residuals ** 2

    lb_result = acorr_ljungbox(resid_sq, lags=20, return_df=True)

    n_significant = (lb_result['lb_pvalue'] < 0.05).sum()
    autocorr_present = n_significant > 10

    diagnostics = {
      'ljungbox_results': lb_result,
      'autocorr_detected': autocorr_present,
      'n_significant_lags': n_significant,
      'mean_pvalue': lb_result['lb_pvalue'].mean()
    }

    self.diagnostics['garch'] = diagnostics

    if autocorr_present:
      logger.warning("Autocorrelação detectada nos resíduos ao quadrado!")
    else:
        logger.info("sem autocorrelação significativa nos resíduos²")
    
    return diagnostics
  
  def predict_volatility(self, horizon: int = 1) -> np.ndarray:
    if not self.fitted:
      raise ValueError("ajuste o modelo primeiro")
    
    forecast = self.garch_result.forecast(horizon=horizon)
    return np.sqrt(forecast.variance.values[-1, :]) 
  
  def var(self, p: float, horizon: int = 1) -> float:
    if not self.fitted:
        raise ValueError("Ajuste o modelo primeiro")

    sigma_forecast = self.predict_volatility(horizon)[0]
    mu_forecast = self.garch_result.params.get('mu', 0) if 'mu' in self.garch_result.params else 0

    resid = self.standardized_residuals.values
    n = len(resid)
    
    n_left = np.sum(resid < self.threshold_left)
    prob_left_tail = n_left / n

    if (1 - p) <= prob_left_tail and self.gpd_left is not None:
        conditional_prob = (1 - p) / prob_left_tail
        
        xi = self.gpd_left.xi
        sigma_gpd = self.gpd_left.sigma
        
        if abs(xi) < 1e-6:
            gpd_exceedance = sigma_gpd * (-np.log(1 - conditional_prob))
        else:
            gpd_exceedance = (sigma_gpd / xi) * ((1 - conditional_prob) ** (-xi) - 1)
        
        var_residual = self.threshold_left - gpd_exceedance
    else:
        var_residual = np.quantile(resid, 1 - p)
    
    var_return = mu_forecast + sigma_forecast * var_residual
    
    return var_return
  
  def expected_shortfall(self, p: float, horizon: int = 1) -> float:
    if not self.fitted:
        raise ValueError("Ajuste o modelo primeiro")
    
    sigma_forecast = self.predict_volatility(horizon)[0]
    mu_forecast = self.garch_result.params.get('mu', 0) if 'mu' in self.garch_result.params else 0

    resid = self.standardized_residuals.values
    n = len(resid)
    
    n_left = np.sum(resid < self.threshold_left)
    prob_left_tail = n_left / n

    if (1 - p) <= prob_left_tail and self.gpd_left is not None:
        conditional_prob = (1 - p) / prob_left_tail
        
        xi = self.gpd_left.xi
        sigma_gpd = self.gpd_left.sigma
        
        if abs(xi) < 1e-6:
            gpd_quantile = sigma_gpd * (-np.log(1 - conditional_prob))
        else:
            gpd_quantile = (sigma_gpd / xi) * ((1 - conditional_prob) ** (-xi) - 1)
        
        if abs(xi) < 1e-6:
            es_exceedance = gpd_quantile + sigma_gpd
        elif xi < 1:
            es_exceedance = (gpd_quantile + sigma_gpd) / (1 - xi)
        else:
            logger.warning(f"ES não existe para ξ={xi:.4f} >= 1, usando VaR+σ")
            es_exceedance = gpd_quantile + sigma_gpd
        
        es_residual = self.threshold_left - es_exceedance
    else:
        var_threshold = np.quantile(resid, 1 - p)
        es_residual = resid[resid < var_threshold].mean()
    
    es_return = mu_forecast + sigma_forecast * es_residual
    
    return es_return
  
  def get_summary(self) -> Dict:
    summary = {
      'garch': {
        'aic': self.garch_result.aic,
        'bic': self.garch_result.bic,
        'params': self.garch_result.params.to_dict()
      }
    }

    if self.gpd_left is not None:
      summary['gpd_left'] = {
        'xi': self.gpd_left.xi,
        'sigma': self.gpd_left.sigma,
        'threshold': self.threshold_left,
        'n_exceedances': self.gpd_left.n_exceedances
      }
    
    if self.gpd_right is not None:
      summary['gpd_right'] = {
        'xi': self.gpd_right.xi,
        'sigma': self.gpd_right.sigma,
        'threshold': self.threshold_right,
        'n_exceedances': self.gpd_right.n_exceedances
      }

    return summary
