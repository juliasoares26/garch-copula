import pandas as pd
import numpy as np
from typing import Optional, Dict, List, Tuple
import logging
from scipy import stats
from scipy.interpolate import interp1d
from joblib import Parallel, delayed

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import numba
from numba import jit, prange

@jit(nopython=True, cache=True, fastmath=True)
def fast_portfolio_returns(simulated_returns, weights):
    return simulated_returns @ weights

@jit(nopython=True, cache=True, parallel=True)
def fast_quantile_batch(data, quantiles):
    n = len(quantiles)
    result = np.empty(n)
    for i in prange(n):
        result[i] = np.quantile(data, quantiles[i])
    return result

@jit(nopython=True, cache=True)
def fast_es_calculation(returns, var_threshold):
    tail_returns = returns[returns <= var_threshold]
    if len(tail_returns) > 0:
        return tail_returns.mean()
    return var_threshold

class CopulaEVTRisk:
  def __init__(self):
    self.marginal_models = {}
    self.copula_model = None
    self.asset_names = None
    self.fitted = False
    self.marginal_cdfs = {}
    self.marginal_inv_cdfs = {}

  def fit(
      self,
      returns: pd.DataFrame,
      marginal_models: Dict,
      copula_model,
      asset_names: Optional[List[str]] = None
  ):
    self.marginal_models = marginal_models
    self.copula_model = copula_model

    if asset_names is None:
      self.asset_names = list(returns.columns)
    else:
      self.asset_names = asset_names

    self._precompute_marginal_transforms()

    self.fitted = True

    logger.info(f"CopulaEVTRisk configurado: {len(self.asset_names)} ativos")

  # Versão vetorizada de _semi_parametric_cdf — opera sobre um array inteiro
  def _semi_parametric_cdf_vectorized(
      self,
      x_vec: np.ndarray,
      model,
      lower_quantile: float = 0.05,
      upper_quantile: float = 0.95,
  ) -> np.ndarray:
    ret = model.returns
    threshold_lower = np.quantile(ret, lower_quantile)
    threshold_upper = np.quantile(ret, upper_quantile)

    mask_left   = x_vec < threshold_lower
    mask_right  = x_vec > threshold_upper
    mask_center = ~mask_left & ~mask_right

    result = np.empty_like(x_vec, dtype=np.float64)

    if mask_center.any():
      xc = x_vec[mask_center]
      ret_sorted = np.sort(ret)
      n_ret = len(ret_sorted)

      n_lower = np.searchsorted(ret_sorted, threshold_lower, side='right')
      n_upper_thresh = np.searchsorted(ret_sorted, threshold_upper, side='right')
      total_center = (n_upper_thresh - n_lower) / n_ret

      n_le_x = np.searchsorted(ret_sorted, xc, side='right')
      emp_cdf_center = (n_le_x - n_lower) / n_ret

      if total_center > 0:
        result[mask_center] = lower_quantile + (upper_quantile - lower_quantile) * (
            emp_cdf_center / total_center
        )
      else:
        result[mask_center] = np.searchsorted(ret_sorted, xc, side='right') / n_ret

    if mask_left.any():
      xl = x_vec[mask_left]
      if hasattr(model, 'gpd_left') and model.gpd_left is not None:
        excess = threshold_lower - xl
        cdf_excess = model.gpd_left.cdf(excess)
        result[mask_left] = lower_quantile * (1 - cdf_excess)
      else:
        ret_sorted = np.sort(ret)
        result[mask_left] = np.searchsorted(ret_sorted, xl, side='right') / len(ret_sorted)

    if mask_right.any():
      xr = x_vec[mask_right]
      if hasattr(model, 'gpd_right') and model.gpd_right is not None:
        excess = xr - threshold_upper
        cdf_excess = model.gpd_right.cdf(excess)
        result[mask_right] = upper_quantile + (1 - upper_quantile) * cdf_excess
      else:
        ret_sorted = np.sort(ret)
        result[mask_right] = np.searchsorted(ret_sorted, xr, side='right') / len(ret_sorted)

    return result

  # Constrói CDF e inversa (retorno → U e U → retorno) para cada ativo.
  def _precompute_marginal_transforms(self):
    logger.info("pré-computando transformações marginais EVT")

    def _build_one(asset_name, model):
      try:
        has_hybrid = (
            hasattr(model, '_hybrid_cdf') and
            hasattr(model, 'std_residuals') and
            model.std_residuals is not None and
            len(np.asarray(model.std_residuals).ravel()) > 10
        )

        if has_hybrid:
          z_check = np.asarray(model.std_residuals).ravel()
          z_std_check = float(np.std(z_check)) if len(z_check) > 1 else 1.0
          if z_std_check > 5.0 or z_std_check < 0.1:
            logger.debug(
                f"{asset_name}: std(std_residuals)={z_std_check:.3f} fora do "
                f"intervalo esperado [0.1, 5.0]. Usando ECDF."
            )
            return _build_empirical(asset_name, model)
          return _build_semi_parametric(asset_name, model)
        else:
          return _build_empirical(asset_name, model)

      except Exception as e:
        logger.warning(f"_build_one falhou para {asset_name}: {e}. Usando ECDF.")
        return _build_empirical_fallback(asset_name, model)

    # Constrói CDF/inv-CDF via _hybrid_cdf do modelo.
    def _build_semi_parametric(asset_name, model):
      z = np.asarray(model.std_residuals).ravel()
      n_z = len(z)

      ret = np.asarray(model.returns).ravel() if hasattr(model, 'returns') and model.returns is not None else z
      mu_r = float(np.mean(ret))
      if hasattr(model, 'conditional_vol') and model.conditional_vol is not None:
        try:
          _cv = np.asarray(model.conditional_vol).ravel()
          sigma_bar = float(_cv.mean()) if len(_cv) > 0 else float(np.std(ret, ddof=1))
        except Exception:
          sigma_bar = float(np.std(ret, ddof=1)) if len(ret) > 1 else 1.0
      elif hasattr(model, 'garch_result') and model.garch_result is not None:
        try:
          _scale = float(getattr(model.garch_result.model, '_scale', 1.0) or 1.0)
          sigma_bar = float(model.garch_result.conditional_volatility.mean()) * _scale
        except Exception:
          sigma_bar = float(np.std(ret, ddof=1)) if len(ret) > 1 else 1.0
      else:
        sigma_bar = float(np.std(ret, ddof=1)) if len(ret) > 1 else 1.0
      sigma_bar = max(sigma_bar, 1e-8)

      z_min, z_max = z.min(), z.max()
      z_std = max(z.std(), 0.5)

      gpd_left = getattr(model, 'gpd_left', None)
      gpd_right = getattr(model, 'gpd_right', None)
      thresh_l = float(getattr(model, 'threshold_left', np.quantile(z, 0.10)))
      thresh_r = float(getattr(model, 'threshold_right', np.quantile(z, 0.90)))

      if gpd_left is not None and gpd_left.xi < 0:
        max_exc_l = -gpd_left.sigma / gpd_left.xi
        z_ext_min = thresh_l - max_exc_l * 0.9999
      else:
        z_ext_min = z_min - 3 * z_std

      if gpd_right is not None and gpd_right.xi < 0:
        max_exc_r = -gpd_right.sigma / gpd_right.xi
        z_ext_max = thresh_r + max_exc_r * 0.9999
      else:
        z_ext_max = z_max + 3 * z_std

      _Z_CLIP = 15.0
      z_ext_min = max(z_ext_min, -_Z_CLIP)
      z_ext_max = min(z_ext_max,  _Z_CLIP)
      grid_z = np.unique(np.concatenate([
        np.linspace(z_ext_min, thresh_l, 400),
        np.linspace(thresh_l,  thresh_r, 800),
        np.linspace(thresh_r,  z_ext_max, 400),
      ]))

      cdf_z = np.array([model._hybrid_cdf(float(zi)) for zi in grid_z])

      cdf_z = np.clip(cdf_z, 0.0, 1.0)
      unique_mask = np.concatenate([[True], np.diff(cdf_z) > 1e-12])
      gz = grid_z[unique_mask]
      cz = cdf_z[unique_mask]

      if len(gz) < 10:
        logger.warning(f"{asset_name}: grade CDF degenerada, usando ECDF.")
        return _build_empirical(asset_name, model)

      grid_r = mu_r + sigma_bar * gz

      ret_std = float(np.std(ret, ddof=1)) if len(ret) > 1 else 1.0
      ret_std = max(ret_std, 1e-8)
      if sigma_bar > 5.0 * ret_std:
        logger.warning(
            f"{asset_name}: sigma_bar={sigma_bar:.6f} parece estar na escala "
            f"de resíduos (ret_std={ret_std:.6f}). Recalculando via std(returns)."
        )
        sigma_bar = ret_std
        grid_r = mu_r + sigma_bar * gz
        if sigma_bar > 5.0 * ret_std:
          logger.warning(f"{asset_name}: sigma_bar fallback também falhou — usando ECDF.")
          return _build_empirical(asset_name, model)

      cdf_fn = interp1d(grid_r, cz, kind='linear', bounds_error=False,
                        fill_value=(0.0, 1.0))

      inv_fn = interp1d(cz, grid_r, kind='linear', bounds_error=False,
                        fill_value=(grid_r[0], grid_r[-1]))

      logger.debug(
        f"{asset_name}: CDF semi-paramétrica OK  "
        f"mu={mu_r:.5f}  sigma_bar={sigma_bar:.5f}  "
        f"grid=[{grid_r.min():.4f}, {grid_r.max():.4f}]  n={len(gz)}  "
        f"(fonte: {'conditional_vol' if hasattr(model, 'conditional_vol') and model.conditional_vol is not None else 'std(ret)'})"
      )
      return asset_name, cdf_fn, inv_fn

    # ECDF sobre retornos históricos — fallback quando _hybrid_cdf indisponível.
    def _build_empirical(asset_name, model):
      for attr in ("returns", "_original_returns", "std_residuals", "standardized_residuals"):
        val = getattr(model, attr, None)
        if val is not None:
          arr = np.asarray(val).ravel()
          if len(arr) > 10:
            ret_sorted = np.sort(arr)
            n = len(ret_sorted)
            cdf_vals = (np.arange(1, n + 1)) / (n + 1)
            cdf_fn = interp1d(ret_sorted, cdf_vals, kind='linear',
                              bounds_error=False, fill_value=(0.0, 1.0))
            inv_fn = interp1d(cdf_vals, ret_sorted, kind='linear',
                              bounds_error=False, fill_value=(ret_sorted[0], ret_sorted[-1]))
            logger.debug(f"{asset_name}: ECDF sobre '{attr}'  n={n}")
            return asset_name, cdf_fn, inv_fn
      return None

    def _build_empirical_fallback(asset_name, model):
      result = _build_empirical(asset_name, model)
      return result

    from joblib import Parallel, delayed as _delayed
    built = Parallel(n_jobs=-1, backend='threading')(
        _delayed(_build_one)(name, mdl)
        for name, mdl in self.marginal_models.items()
    )
    for result in built:
      if result is None:
        continue
      asset_name, cdf_fn, inv_fn = result
      self.marginal_cdfs[asset_name] = cdf_fn
      self.marginal_inv_cdfs[asset_name] = inv_fn

    n_ok = len(self.marginal_inv_cdfs)
    n_total = len(self.marginal_models)
    if n_ok < n_total:
      logger.warning(f"CDF invertida construída para {n_ok}/{n_total} ativos — "
                     f"restantes usarão ECDF empírica no simulate.")
    logger.info("transformações EVT pré-computadas")

  # cdf semiparametrica, evt empirica no centro e gpd nas caudas
  def _semi_parametric_cdf(
      self,
      x:float,
      model,
      lower_quantile: float = 0.05,
      upper_quantile: float = 0.95
  ) -> float:
    
    threshold_lower = np.quantile(model.returns, lower_quantile)
    threshold_upper = np.quantile(model.returns, upper_quantile)

    if x < threshold_lower:
      if hasattr(model, 'gpd_left') and model.gpd_left is  not None:
        excess = threshold_lower - x
        cdf_excess = model.gpd_left.cdf(excess)
        return lower_quantile * (1 - cdf_excess)
      else:
        return np.mean(model.returns <= x)
      
    elif x > threshold_upper:
      if hasattr(model, 'gpd_right') and model.gpd_right is not None:
        excess = x - threshold_upper
        cdf_excess = model.gpd_right.cdf(excess)
        return upper_quantile + (1 - upper_quantile) * cdf_excess
      else:
        return np.mean(model.returns <= x)
      
    else:
      emp_cdf = np.mean(model.returns <= x)

      emp_cdf_center = np.mean((model.returns >= threshold_lower) & (model.returns <= x))
      total_center = np.mean((model.returns >= threshold_lower) & (model.returns <= threshold_upper))
      
      if total_center > 0:
        return lower_quantile + (upper_quantile - lower_quantile) * (emp_cdf_center / total_center)
      else:
        return emp_cdf
  
  # transforma retorno em pseudo-observações [0,1] usando cdfs semi-parametricas
  def transform_to_uniform(self, returns: pd.DataFrame) -> np.ndarray:
    n_obs = len(returns)
    n_assets = len(self.asset_names)
    
    uniform_data = np.zeros((n_obs, n_assets))

    for i, asset in enumerate(self.asset_names):
      if asset not in self.marginal_cdfs:
        logger.warning(f"cdf não encontrada para {asset}, usando rankdata")
        uniform_data[:, i] = stats.rankdata(returns[asset]) / (len(returns) + 1)
      else:
        cdf_func = self.marginal_cdfs[asset]
        asset_returns = returns[asset].values
        uniform_data[:, i] = cdf_func(asset_returns)

        uniform_data[:, i] = np.clip(uniform_data[:, i], 1e-10, 1 - 1e-10)

    logger.info(f"transformando para uniformes via CDF EVT: {uniform_data.shape}")

    return uniform_data
  
  def simulate_portfolio_returns(
      self, 
      weights: np.ndarray,
      n_simulations: int = 1000,
      horizon: int = 1,
      use_copula: bool = True,
      seed: Optional[int] = None,
      precomputed_scenarios: Optional[np.ndarray] = None,
  ) -> np.ndarray:
    
    if not self.fitted:
      raise ValueError("Execute fit() primeiro")

    if use_copula and precomputed_scenarios is not None:
      uniform_samples = np.clip(precomputed_scenarios, 1e-10, 1 - 1e-10)
    elif use_copula:
      if seed is not None:
        np.random.seed(seed)
      if self.copula_model is None:
        raise ValueError("copula não definida")
      if hasattr(self.copula_model, 'simulate'):
        uniform_samples = self.copula_model.simulate(n_samples=n_simulations, seed=seed)
      elif hasattr(self.copula_model, 'rvs'):
        uniform_samples = self.copula_model.rvs(n_simulations)
      else:
        raise ValueError("copula não tem método simulate() ou rvs()")
      uniform_samples = np.clip(uniform_samples, 1e-10, 1 - 1e-10)
    else:
      rng = np.random.default_rng(seed)
      uniform_samples = rng.uniform(1e-10, 1 - 1e-10,
                                    (n_simulations, len(self.asset_names)))

    simulated_returns = np.zeros((len(uniform_samples), len(self.asset_names)))
    for i, asset in enumerate(self.asset_names):
      u = uniform_samples[:, i]
      if asset in self.marginal_inv_cdfs:
        simulated_returns[:, i] = self.marginal_inv_cdfs[asset](u)
      else:
        logger.warning(f"inverse CDF não encontrada para {asset}, usando ECDF direta")
        u_clipped = np.clip(u, 0.0, 1.0)
        simulated_returns[:, i] = np.quantile(
          self.marginal_models[asset].returns, u_clipped
        )

    if horizon > 1:
      simulated_returns *= np.sqrt(horizon)

    portfolio_returns = simulated_returns @ weights
    return portfolio_returns

  def portfolio_var(
      self,
      weights: np.ndarray,
      confidence_level: float = 0.95,
      n_simulations: int = 1000,
      horizon: int = 1,
      use_copula: bool = True,
      seed: Optional[int] = None,
      precomputed_scenarios: Optional[np.ndarray] = None,
  ) -> float:
    portfolio_returns = self.simulate_portfolio_returns(
      weights, n_simulations, horizon, use_copula, seed, precomputed_scenarios
    )
    return np.quantile(portfolio_returns, 1 - confidence_level)
  
  def portfolio_es(
      self,
      weights: np.ndarray,
      confidence_level: float = 0.95,
      n_simulations: int = 1000,
      horizon: int = 1,
      use_copula: bool = True,
      seed: Optional[int] = None,
      precomputed_scenarios: Optional[np.ndarray] = None,
  ) -> float:
    portfolio_returns = self.simulate_portfolio_returns(
      weights, n_simulations, horizon, use_copula, seed, precomputed_scenarios
    )
    var = np.quantile(portfolio_returns, 1 - confidence_level)
    es = portfolio_returns[portfolio_returns <= var].mean()
    return es
  
  def tail_var_es(
      self,
      weights: np.ndarray,
      tail_quantiles: List[float] = [0.95, 0.99, 0.995, 0.999],
      n_simulations: int = 50000,
      horizon: int = 1,
      seed: Optional[int] = None
  ) -> pd.DataFrame:
    

    logger.info(f"calculando tail VaR/ES para {len(tail_quantiles)} quantis")

    portfolio_returns = self.simulate_portfolio_returns(
      weights, n_simulations, horizon, use_copula=True, seed=seed
    )

    results = []

    for q in tail_quantiles:
      var_q = np.quantile(portfolio_returns, 1 - q)
      es_q = portfolio_returns[portfolio_returns <= var_q].mean()

      n_tail = np.sum(portfolio_returns <= var_q)

      results.append({
        'Confidence_Level': q,
        'VaR': var_q,
        'ES': es_q,
        'ES_VaR_Ratio': es_q / var_q if var_q != 0 else np.nan,
        'N_Tail_Obs': n_tail,
        'Tail_Pct': 100 * n_tail / len(portfolio_returns)
      })

    df = pd.DataFrame(results)
    logger.info(f"Tail VaR/ES calculado")

    return df
  
  def compare_with_without_copula(
      self,
      weights: np.ndarray,
      confidence_level: float = 0.99,
      n_simulations: int = 1000,
      seed: Optional[int] = None,
      precomputed_scenarios: Optional[np.ndarray] = None,
  ) -> pd.DataFrame:
    logger.info("comparando: copula vs independencia")

    var_copula = self.portfolio_var(weights, confidence_level, n_simulations, 1,
                                    True, seed, precomputed_scenarios)
    es_copula  = self.portfolio_es(weights, confidence_level, n_simulations, 1,
                                    True, seed, precomputed_scenarios)

    var_indep = self.portfolio_var(weights, confidence_level, n_simulations, 1,
                                   False, seed)
    es_indep  = self.portfolio_es(weights, confidence_level, n_simulations, 1,
                                   False, seed)

    comparison = pd.DataFrame({
      'Method': ['Copula-EVT', 'Independence'],
      'VaR': [var_copula, var_indep],
      'ES': [es_copula, es_indep],
      'ES_VaR_Ratio': [es_copula / var_copula, es_indep / var_indep]
    })
    comparison['VaR_Diff_%'] = np.where(
        var_indep != 0,
        100 * (comparison['VaR'] - var_indep) / abs(var_indep),
        np.nan
    )
    comparison['ES_Diff_%'] = np.where(
        es_indep != 0,
        100 * (comparison['ES'] - es_indep) / abs(es_indep),
        np.nan
    )
    return comparison

  def scenario_analysis(
      self,
      weights: np.ndarray,
      scenarios: Dict[str, Dict],
      n_simulations: int = 1000,
      seed: Optional[int] = None
  ) -> pd.DataFrame:
    
    logger.info(f"Análise de {len(scenarios)} cenários")

    original_marginals = {}
    for asset, model in self.marginal_models.items():
      original_marginals[asset] = {
        'gpd_left_xi': model.gpd_left.xi if hasattr(model, 'gpd_left') and hasattr(model.gpd_left, 'xi') else None,
        'gpd_right_xi': model.gpd_right.xi if hasattr(model, 'gpd_right') and hasattr(model.gpd_right, 'xi') else None,
        'volatility': model.returns.std()
     }

    if hasattr(self.copula_model, 'corr') or hasattr(self.copula_model, 'tau'):
      original_corr = self._extract_copula_correlation()
    else:
      original_corr = None

    results = []

    for name, params in scenarios.items():
      logger.info(f"  Cenário: {name}")
            
      tail_mult = params.get('tail_multiplier', 1.0)
      corr_mult = params.get('correlation_multiplier', 1.0)
      vol_mult = params.get('volatility_multiplier', 1.0)
            
      self._apply_marginal_stress(tail_mult, vol_mult)
            
      self._apply_copula_stress(corr_mult, original_corr)
            
      self._precompute_marginal_transforms()
            
      var_stressed = self.portfolio_var(
        weights, 0.99, n_simulations, use_copula=True, seed=seed
      )
      es_stressed = self.portfolio_es(
        weights, 0.99, n_simulations, use_copula=True, seed=seed
      )
            
      results.append({
      'Scenario': name,
      'Tail_Multiplier': tail_mult,
      'Corr_Multiplier': corr_mult,
      'Vol_Multiplier': vol_mult,
      'VaR_99': var_stressed,
      'ES_99': es_stressed,
      'ES_VaR_Ratio': es_stressed / var_stressed if var_stressed != 0 else np.nan
      })
            
      self._restore_marginals(original_marginals)
      if original_corr is not None:
        self._restore_copula_correlation(original_corr)
        
      self._precompute_marginal_transforms()

    df = pd.DataFrame(results)
    logger.info("Análise de cenários concluída")

    return df
    
  # Aplica stress aos parâmetros das marginais EVT
  def _apply_marginal_stress(
        self,
        tail_multiplier: float,
        volatility_multiplier: float
    ):
        for asset, model in self.marginal_models.items():
            if hasattr(model, 'gpd_left') and hasattr(model.gpd_left, 'xi'):
                model.gpd_left.xi *= tail_multiplier
            
            if hasattr(model, 'gpd_right') and hasattr(model.gpd_right, 'xi'):
                model.gpd_right.xi *= tail_multiplier
            
            if volatility_multiplier != 1.0:
                mean_return = model.returns.mean()
                model.returns = mean_return + (model.returns - mean_return) * volatility_multiplier
    
  # Aplica stress à estrutura de dependência da copula
  def _apply_copula_stress(
        self,
        correlation_multiplier: float,
        original_corr: Optional[np.ndarray]
    ):
        if correlation_multiplier == 1.0 or original_corr is None:
            return
        
        if hasattr(self.copula_model, 'corr'):
            stressed_corr = original_corr.copy()
            n = len(stressed_corr)
            
            for i in range(n):
                for j in range(i+1, n):
                    stressed_corr[i, j] = np.clip(
                        original_corr[i, j] * correlation_multiplier,
                        -0.99, 0.99
                    )
                    stressed_corr[j, i] = stressed_corr[i, j]
            
            stressed_corr = self._nearest_positive_definite(stressed_corr)
            
            self.copula_model.corr = stressed_corr
        
        elif hasattr(self.copula_model, 'tau'):
            original_tau = self.copula_model.tau
            stressed_tau = np.clip(
                original_tau * correlation_multiplier,
                -0.99, 0.99
            )
            self.copula_model.tau = stressed_tau
    
  # Extrai matriz de correlação/dependência da copula
  def _extract_copula_correlation(self) -> Optional[np.ndarray]:
        if hasattr(self.copula_model, 'corr'):
            return self.copula_model.corr.copy()
        elif hasattr(self.copula_model, 'tau'):
            return self.copula_model.tau
        else:
            return None
    
  # Restaura parâmetros originais das marginais
  def _restore_marginals(self, original_params: Dict):
        for asset, params in original_params.items():
            model = self.marginal_models[asset]
            
            if params['gpd_left_xi'] is not None:
                if hasattr(model, 'gpd_left') and hasattr(model.gpd_left, 'xi'):
                    model.gpd_left.xi = params['gpd_left_xi']
            
            if params['gpd_right_xi'] is not None:
                if hasattr(model, 'gpd_right') and hasattr(model.gpd_right, 'xi'):
                    model.gpd_right.xi = params['gpd_right_xi']
    
  # Restaura correlação original da copula
  def _restore_copula_correlation(self, original_corr):
        if hasattr(self.copula_model, 'corr'):
            self.copula_model.corr = original_corr.copy()
        elif hasattr(self.copula_model, 'tau'):
            self.copula_model.tau = original_corr
    
  # Encontra a matriz positiva definida mais próxima de A
  def _nearest_positive_definite(self, A: np.ndarray) -> np.ndarray:
        B = (A + A.T) / 2
        s, V = np.linalg.eigh(B)
        H = V @ np.diag(s) @ V.T
        A2 = (B + H) / 2
        A3 = (A2 + A2.T) / 2
        
        if self._is_positive_definite(A3):
            return A3
        
        spacing = np.spacing(np.linalg.norm(A))
        I = np.eye(A.shape[0])
        k = 1
        while not self._is_positive_definite(A3):
            mineig = np.min(np.real(np.linalg.eigvals(A3)))
            A3 += I * (-mineig * k**2 + spacing)
            k += 1
        
        return A3
    
  # Verifica se matriz é positiva definida
  def _is_positive_definite(self, A: np.ndarray) -> bool:
        try:
            np.linalg.cholesky(A)
            return True
        except np.linalg.LinAlgError:
            return False
    
  # Contribuição marginal de cada ativo para VaR/ES
  def marginal_contribution_to_risk(
        self,
        weights: np.ndarray,
        confidence_level: float = 0.99,
        n_simulations: int = 1000,
        seed: Optional[int] = None
    ) -> pd.DataFrame:
        logger.info("Calculando contribuições marginais...")
        
        var_base = self.portfolio_var(weights, confidence_level, n_simulations, seed=seed)
        es_base = self.portfolio_es(weights, confidence_level, n_simulations, seed=seed)
        
        epsilon = 0.001
        contributions = []
        
        for i, asset in enumerate(self.asset_names):
            w_perturbed = weights.copy()
            w_perturbed[i] += epsilon
            w_perturbed = w_perturbed / w_perturbed.sum()
            
            var_perturbed = self.portfolio_var(w_perturbed, confidence_level, n_simulations, seed=seed)
            es_perturbed = self.portfolio_es(w_perturbed, confidence_level, n_simulations, seed=seed)
            
            marginal_var = (var_perturbed - var_base) / epsilon
            marginal_es = (es_perturbed - es_base) / epsilon
            
            component_var = weights[i] * marginal_var
            component_es = weights[i] * marginal_es
            
            contributions.append({
                'Asset': asset,
                'Weight': weights[i],
                'Marginal_VaR': marginal_var,
                'Component_VaR': component_var,
                'Contribution_VaR_%': 100 * component_var / var_base,
                'Marginal_ES': marginal_es,
                'Component_ES': component_es,
                'Contribution_ES_%': 100 * component_es / es_base
            })
        
        df = pd.DataFrame(contributions)
        logger.info("Contribuições calculadas")
        
        return df
    
  # Backtesting: conta violações do VaR
  def backtesting_violations(
        self,
        weights: np.ndarray,
        realized_returns: pd.Series,
        confidence_level: float = 0.99,
        rolling_window: int = 252
    ) -> Dict:
        logger.info("Backtesting VaR...")
        
        violations = []
        var_forecasts = []
        
        var = self.portfolio_var(weights, confidence_level, n_simulations=1000)
        
        for t in range(len(realized_returns)):
            var_forecasts.append(var)
            violations.append(1 if realized_returns.iloc[t] < var else 0)
        
        violations = np.array(violations)
        n_violations = violations.sum()
        n_obs = len(violations)
        violation_rate = n_violations / n_obs if n_obs > 0 else 0
        expected_rate = 1 - confidence_level
        
        if n_violations > 0 and violation_rate > 0:
            lr_stat = -2 * (
                n_violations * np.log(expected_rate) + 
                (n_obs - n_violations) * np.log(1 - expected_rate) -
                n_violations * np.log(violation_rate) -
                (n_obs - n_violations) * np.log(1 - violation_rate)
            )
        else:
            lr_stat = np.nan
        
        from scipy.stats import chi2
        p_value = 1 - chi2.cdf(lr_stat, df=1) if not np.isnan(lr_stat) else np.nan
        
        logger.info(f"Backtesting: {n_violations}/{n_obs} violações ({violation_rate:.2%})")
        logger.info(f"Esperado: {expected_rate:.2%}, p-value: {p_value:.4f}")
        
        return {
            'n_violations': n_violations,
            'n_observations': n_obs,
            'violation_rate': violation_rate,
            'expected_rate': expected_rate,
            'kupiec_lr': lr_stat,
            'kupiec_pvalue': p_value,
            'test_passed': p_value > 0.05 if not np.isnan(p_value) else False
        }

class OptimizedCopulaRisk:

    def __init__(self, copula_risk: CopulaEVTRisk):
        self.copula_risk = copula_risk
        self.n_jobs = -1

    def portfolio_var_es_batch(
        self,
        weights: np.ndarray,
        confidence_levels: list = [0.95, 0.99, 0.995],
        n_simulations: int = 10000,
        seed: int = 42,
        precomputed_scenarios: Optional[np.ndarray] = None,
    ):
        logger.info(f"VaR/ES batch (níveis={confidence_levels})...")

        portfolio_returns = self.copula_risk.simulate_portfolio_returns(
            weights, n_simulations, horizon=1, use_copula=True,
            seed=seed, precomputed_scenarios=precomputed_scenarios,
        )

        quantiles = np.array([1 - cl for cl in confidence_levels])
        vars_ = fast_quantile_batch(portfolio_returns, quantiles)

        results = []
        for i, cl in enumerate(confidence_levels):
            var = vars_[i]
            es = fast_es_calculation(portfolio_returns, var)
            results.append({
                'Confidence': cl,
                'VaR': var,
                'ES': es,
                'ES_VaR_Ratio': es / var if var != 0 else np.nan,
            })
        return pd.DataFrame(results)

    # Component VaR via perturbação numérica.
    def component_var_parallel(
        self,
        weights: np.ndarray,
        confidence_level: float = 0.99,
        n_simulations: int = 1000,
        epsilon: float = 0.01,
        seed: int = 42,
        precomputed_scenarios: Optional[np.ndarray] = None,
    ):
        logger.info("Calculando Component VaR (paralelo)...")

        sim_returns = np.zeros((
            len(precomputed_scenarios) if precomputed_scenarios is not None
            else n_simulations,
            len(self.copula_risk.asset_names),
        ))
        _sc = precomputed_scenarios
        uniform_samples = (
            np.clip(_sc, 1e-10, 1 - 1e-10)
            if _sc is not None
            else None
        )
        if uniform_samples is not None:
            for i, asset in enumerate(self.copula_risk.asset_names):
                if asset in self.copula_risk.marginal_inv_cdfs:
                    sim_returns[:, i] = self.copula_risk.marginal_inv_cdfs[asset](
                        uniform_samples[:, i]
                    )
                else:
                    sim_returns[:, i] = np.quantile(
                        self.copula_risk.marginal_models[asset].returns,
                        uniform_samples[:, i],
                    )
        else:
            sim_returns = None

        def _port_var(w):
            if sim_returns is not None:
                pr = sim_returns @ w
            else:
                pr = self.copula_risk.simulate_portfolio_returns(
                    w, n_simulations, seed=seed
                )
            return np.quantile(pr, 1 - confidence_level)

        var_base = _port_var(weights)

        def _perturbed(i):
            w_p = weights.copy()
            w_p[i] += epsilon
            w_p /= w_p.sum()
            var_p = _port_var(w_p)
            mv = (var_p - var_base) / epsilon
            cv = mv * weights[i]
            return {
                'Asset': self.copula_risk.asset_names[i],
                'Weight': weights[i],
                'Marginal_VaR': mv,
                'Component_VaR': cv,
                'Contribution_%': 100 * cv / abs(var_base),
            }

        from joblib import Parallel, delayed as _delayed
        results = Parallel(n_jobs=self.n_jobs, backend='threading')(
            _delayed(_perturbed)(i) for i in range(len(weights))
        )
        return pd.DataFrame(results)


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)

    print("=" * 65)
    print("DIAGNOSTICO: escala dos atributos por modelo marginal")
    print("=" * 65)

    import sys, os
    from pathlib import Path

    RETURNS_PATH  = Path(__file__).parent.parent.parent / "data" / "processed" / "returns.parquet"
    USE_LOG_RETURNS = False

    path = Path(RETURNS_PATH)
    if not path.exists():
        print(f"[ERRO] Arquivo nao encontrado: {RETURNS_PATH}")
        print("       Ajuste RETURNS_PATH no bloco de configuracao.")
        sys.exit(1)

    if path.suffix == ".parquet":
        raw = pd.read_parquet(path)
    elif path.suffix in (".csv", ".tsv"):
        raw = pd.read_csv(path, index_col=0, parse_dates=True)
    else:
        print(f"[ERRO] Formato nao suportado: {path.suffix}  (use .parquet ou .csv)")
        sys.exit(1)

    if USE_LOG_RETURNS:
        returns_df = np.log(raw / raw.shift(1)).dropna()
    else:
        returns_df = raw.dropna()

    print(f"Dados carregados: {returns_df.shape}  "
          f"({returns_df.index[0].date()} → {returns_df.index[-1].date()})")
    print(f"Ativos: {list(returns_df.columns)}\n")

    import importlib.util, pathlib

    _here = pathlib.Path(__file__).resolve().parent
    _sp_candidates = [
        _here / "semi_parametric.py",
        _here.parent / "marginals" / "semi_parametric.py",
        _here.parent / "semi_parametric.py",
        _here.parent.parent / "semi_parametric.py",
    ]
    _sp_path = next((p for p in _sp_candidates if p.exists()), None)

    if _sp_path is None:
        print("[ERRO] semi_parametric.py nao encontrado. Paths tentados:")
        for p in _sp_candidates:
            print(f"  {p}")
        sys.exit(1)

    print(f"[INFO] Carregando semi_parametric de: {_sp_path}")
    try:
        _spec = importlib.util.spec_from_file_location("semi_parametric", _sp_path)
        _mod  = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_mod)
        SemiParametricGARCH_EVT = _mod.SemiParametricGARCH_EVT
    except Exception as _e:
        print(f"[ERRO] Falha ao carregar semi_parametric.py: {_e}")
        import traceback; traceback.print_exc()
        sys.exit(1)

    marginal_models = {}
    n_ativos = len(returns_df.columns)

    for i, col in enumerate(returns_df.columns, 1):
        print(f"[{i}/{n_ativos}] Ajustando {col} ...")
        series = returns_df[col].dropna()
        m = SemiParametricGARCH_EVT()
        try:
            m.fit(
                series,
                dist="skewt",
                threshold_method="quantile",
                left_quantile=0.05,
                right_quantile=0.95,
                run_diagnostics=False,
                plot_diagnostics=False,
            )
        except Exception as e:
            print(f"  [AVISO] fit() falhou para {col}: {e}")
            if not hasattr(m, "returns") or m.returns is None:
                m.returns = series
        marginal_models[col] = m

    print(f"\nModelos ajustados: {len(marginal_models)}/{n_ativos}\n")

    ATTRS = ["returns", "std_residuals", "standardized_residuals",
             "residuals", "data", "_returns", "_data"]

    issues = []

    for name, model in marginal_models.items():
        sep = "-" * 55
        print(f"\n{sep}")
        print(f"  Ativo : {name}  ({type(model).__name__})")
        print(sep)

        for attr in ATTRS:
            val = getattr(model, attr, None)
            if val is None:
                continue
            arr = np.asarray(val).ravel()
            if len(arr) == 0:
                continue
            std = arr.std()
            flag = ""
            if attr in ("std_residuals", "standardized_residuals") and not (0.8 < std < 1.5):
                flag = "  <- ESCALA SUSPEITA (esperado std~1.0)"
            if attr == "returns" and std > 0.5:
                flag = "  <- ESCALA SUSPEITA (esperado std~0.01-0.03)"
            print(f"  {attr:28s}  n={len(arr):5d}  "
                  f"min={arr.min():+.5f}  max={arr.max():+.5f}  std={std:.5f}{flag}")

        chosen_attr, chosen_arr = None, None
        for attr in ("returns", "_original_returns",
                     "residuals", "data", "_returns", "_data",
                     "std_residuals", "standardized_residuals"):
            val = getattr(model, attr, None)
            if val is not None:
                arr = np.asarray(val).ravel()
                if len(arr) > 10:
                    chosen_attr, chosen_arr = attr, arr
                    break

        if chosen_attr is not None:
            std_val = chosen_arr.std()
            print(f"\n  >> _get_returns usara : '{chosen_attr}'  (std={std_val:.5f})")
            if std_val > 0.5:
                mult = std_val / 0.02
                print(f"  !! PROBLEMA: escala de residuos z_t — VaR ~{mult:.0f}x inflado")
                issues.append((name, f"escala errada em '{chosen_attr}' std={std_val:.4f}"))
            else:
                print(f"  OK: escala compativel com retornos decimais")

    print("\n" + "=" * 65)
    if issues:
        print(f"PROBLEMAS ENCONTRADOS ({len(issues)}):")
        for ativo, desc in issues:
            print(f"  {ativo}: {desc}")
        print()
        print("SOLUCAO: no _precompute_marginal_transforms, ajuste _get_returns")
        print("para que o atributo com std~0.01-0.03 (retornos decimais) venha")
        print("antes de std_residuals/standardized_residuals.")
    else:
        print("Nenhum problema de escala detectado.")
    print("=" * 65)
