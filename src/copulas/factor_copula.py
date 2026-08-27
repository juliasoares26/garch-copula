
import numpy as np
import pandas as pd
import logging
import warnings
from typing import Optional, Dict, List, Tuple, Union
from pathlib import Path
from scipy import stats

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

try:
    from copulas.factor_estimation import FactorCopulaEstimator, estimate_factor_copula
except ImportError:
    try:
        from factor_estimation import FactorCopulaEstimator, estimate_factor_copula
    except ImportError:
        raise ImportError(
            "factor_estimation.py não encontrado. "
            "Coloque em src/copulas/ ou no mesmo diretório."
        )


# FactorCopula — classe principal

# Cópula fatorial de Oh e Patton (2018).
class FactorCopula:

    def __init__(
        self,
        n_dim: int,
        n_factors: int = 1,
        nu_factor: Optional[float] = None,
        max_iter: int = 200,
        tol: float = 1e-5,
        n_quad: int = 32,
        random_state: int = 42,
    ):
        self.n_dim = n_dim
        self.n_factors = n_factors
        self.nu_factor = nu_factor
        self.max_iter = max_iter
        self.tol = tol
        self.n_quad = n_quad
        self.random_state = random_state

        self._estimator: Optional[FactorCopulaEstimator] = None
        self._is_fitted: bool = False
        self._U_fit: Optional[np.ndarray] = None
        self._T: int = 0

        self.log_likelihood_: float = -np.inf
        self.aic_: float = np.inf
        self.bic_: float = np.inf
        self.mbicv_: float = np.inf

    # fit — compatível com CVineCopula.fit()

    # Ajusta a factor copula nas pseudo-observações U ∈ (0,1)^{T×d}.
    def fit(
        self,
        U: np.ndarray,
        families: Optional[List[str]] = None,
        auto_select: bool = True,
        tail_bias: Optional[str] = None,
        max_trees: int = 3,
        min_tau: float = 0.05,
        n_jobs: int = 1,
        **kwargs,
    ) -> "FactorCopula":
        U = np.asarray(U, dtype=np.float64)
        U = np.clip(U, 1e-7, 1 - 1e-7)
        self._T, d = U.shape
        self._U_fit = U.copy()

        if d != self.n_dim:
            logger.warning(
                f"n_dim={self.n_dim} mas U tem {d} colunas. Usando {d}."
            )
            self.n_dim = d

        logger.info(
            f"\n FactorCopula.fit()  T={self._T}  d={d}  "
            f"n_factors={self.n_factors}  ν_Z={'auto' if self.nu_factor is None else self.nu_factor}"
        )

        self._estimator = FactorCopulaEstimator(
            n_factors=self.n_factors,
            nu_factor=self.nu_factor,
            max_iter=self.max_iter,
            tol=self.tol,
            n_quad=self.n_quad,
            random_state=self.random_state,
        )
        self._estimator.fit(U)

        self.log_likelihood_ = self._estimator.log_likelihood_
        self.aic_   = self._estimator.aic_
        self.bic_   = self._estimator.bic_
        self.mbicv_ = self._estimator.mbicv_
        self._is_fitted = True

        logger.info(
            f"  ν_Z={self._estimator.nu_factor_:.2f}  "
            f"LL={self.log_likelihood_:.2f}  AIC={self.aic_:.2f}  "
            f"mBICV={self.mbicv_:.2f}  "
            f"converged={self._estimator.converged_}"
        )
        return self

    # simulate — compatível com CopulaEVTRisk.simulate_portfolio_returns()

    # Simula n_sim cenários de pseudo-observações U ∈ (0,1)^{n_sim × d}.
    def simulate(
        self,
        n_sim: int = 10_000,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("Execute fit() primeiro.")
        return self._estimator.simulate(n_sim=n_sim, seed=seed)

    # log_likelihood — compatível com seleção de cópula

    # Retorna a log-verossimilhança marginal estimada no fit.
    def log_likelihood(self, U: Optional[np.ndarray] = None) -> float:
        if not self._is_fitted:
            raise RuntimeError("Execute fit() primeiro.")
        return self.log_likelihood_

    # Informações de dependência

    # Tail dependence analítica entre ativos i e j.
    def tail_dependence(self, i: int = 0, j: int = 1) -> Dict[str, float]:
        if not self._is_fitted:
            raise RuntimeError("Execute fit() primeiro.")
        return self._estimator.tail_dependence(i, j)

    # Matriz completa de tail dependence lower.
    def tail_dependence_matrix(self) -> pd.DataFrame:
        if not self._is_fitted:
            raise RuntimeError("Execute fit() primeiro.")
        return self._estimator.tail_dependence_matrix()

    # Matriz de correlação implícita R_ij = λ_i * λ_j.
    def implied_correlation(self) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("Execute fit() primeiro.")
        return self._estimator.implied_correlation()

    # Loadings do fator (d, n_factors).
    @property
    def lambdas_(self) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("Execute fit() primeiro.")
        return self._estimator.lambdas_

    # Graus de liberdade estimados do fator.
    @property
    def nu_factor_(self) -> float:
        if not self._is_fitted:
            raise RuntimeError("Execute fit() primeiro.")
        return self._estimator.nu_factor_

    # Métodos de diagnóstico

    # Resumo dos loadings, variância explicada e tail dependence.
    def get_summary(self) -> pd.DataFrame:
        if not self._is_fitted:
            raise RuntimeError("Execute fit() primeiro.")
        return self._estimator.get_summary()

    # Tabela comparativa factor copula vs vine (passando AIC/LL/mBICV da vine).
    def compare_with_vine(
        self,
        vine_aic: float,
        vine_ll: float,
        vine_mbicv: Optional[float] = None,
    ) -> pd.DataFrame:
        n_params = self.n_dim * self.n_factors + (1 if self.nu_factor is None else 0)
        rows = [
            {
                "model":    "Factor Copula",
                "LL":       round(self.log_likelihood_, 2),
                "AIC":      round(self.aic_, 2),
                "mBICV":    round(self.mbicv_, 2),
                "n_params": n_params,
                "nu":       round(self.nu_factor_, 2),
            },
            {
                "model":    "Vine Copula",
                "LL":       round(vine_ll, 2),
                "AIC":      round(vine_aic, 2),
                "mBICV":    round(vine_mbicv, 2) if vine_mbicv is not None else "—",
                "n_params": "—",
                "nu":       "—",
            },
        ]
        df = pd.DataFrame(rows).set_index("model")
        return df

    # Retorna dicionário com métricas chave — compatível com
    def get_regime_summary(self) -> Dict:
        if not self._is_fitted:
            raise RuntimeError("Execute fit() primeiro.")
        lam = self.lambdas_[:, 0]
        td_mean = np.mean([
            self.tail_dependence(0, j)["lower_tail"]
            for j in range(1, self.n_dim)
        ])
        return {
            "copula_type":    "FactorCopula",
            "n_factors":      self.n_factors,
            "nu_factor":      round(self.nu_factor_, 3),
            "lambda_mean":    round(float(np.abs(lam).mean()), 4),
            "lambda_max":     round(float(np.abs(lam).max()), 4),
            "tail_dep_mean":  round(float(td_mean), 4),
            "log_likelihood": round(self.log_likelihood_, 2),
            "aic":            round(self.aic_, 2),
            "bic":            round(self.bic_, 2),
            "mbicv":          round(self.mbicv_, 2),
            "converged":      self._estimator.converged_,
            "n_iter":         self._estimator.n_iter_,
        }

    # Integração com CopulaEVTRisk

    # Alias de implied_correlation() para compatibilidade com
    def get_correlation_matrix(self) -> np.ndarray:
        return self.implied_correlation()

    # Save / Load

    def save(self, path: Union[str, Path]) -> None:
        import pickle
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info(f"FactorCopula salvo em {path}")

    @classmethod
    def load(cls, path: Union[str, Path]) -> "FactorCopula":
        import pickle
        with open(path, "rb") as f:
            obj = pickle.load(f)
        logger.info(f"FactorCopula carregado de {path}")
        return obj


# FactorCopulaRisk — wrapper para CopulaEVTRisk com factor copula

# Wrapper que adapta FactorCopula à interface de CopulaEVTRisk,
class FactorCopulaRisk:

    def __init__(self):
        self._copula: Optional[FactorCopula] = None
        self._marginal_models: dict = {}
        self._asset_names: List[str] = []
        self._returns: Optional[pd.DataFrame] = None
        self._is_fitted: bool = False

    def fit(
        self,
        returns: pd.DataFrame,
        marginal_models: dict,
        copula_model: FactorCopula,
        asset_names: List[str],
    ) -> "FactorCopulaRisk":
        self._returns = returns
        self._marginal_models = marginal_models
        self._copula = copula_model
        self._asset_names = asset_names
        self._is_fitted = True
        logger.info(
            f"FactorCopulaRisk ajustado  d={len(asset_names)}  "
            f"ν_Z={copula_model.nu_factor_:.2f}"
        )
        return self

    # Simula retornos do portfólio via factor copula + inversão das marginais.
    def simulate_portfolio_returns(
        self,
        weights: np.ndarray,
        n_simulations: int = 50_000,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("Execute fit() primeiro.")

        U_sim = self._copula.simulate(n_simulations, seed=seed)
        d = len(self._asset_names)
        R_sim = np.zeros_like(U_sim)

        for j, asset in enumerate(self._asset_names):
            model = self._marginal_models.get(asset)
            u_j = U_sim[:, j]
            if model is not None and hasattr(model, 'ppf'):
                try:
                    R_sim[:, j] = model.ppf(u_j)
                    continue
                except Exception:
                    pass
            hist = self._returns[asset].dropna().values
            R_sim[:, j] = np.quantile(hist, u_j)

        port_returns = R_sim @ weights
        return port_returns

    def portfolio_var(
        self,
        weights: np.ndarray,
        confidence_level: float = 0.99,
        n_simulations: int = 50_000,
        seed: Optional[int] = None,
    ) -> float:
        port_returns = self.simulate_portfolio_returns(
            weights, n_simulations, seed
        )
        return float(-np.quantile(port_returns, 1 - confidence_level))

    def portfolio_es(
        self,
        weights: np.ndarray,
        confidence_level: float = 0.99,
        n_simulations: int = 50_000,
        seed: Optional[int] = None,
    ) -> float:
        port_returns = self.simulate_portfolio_returns(
            weights, n_simulations, seed
        )
        var = -np.quantile(port_returns, 1 - confidence_level)
        tail = port_returns[port_returns <= -var]
        return float(-tail.mean()) if len(tail) > 0 else var

    # Tabela VaR/ES para múltiplos níveis de confiança.
    def var_es_summary(
        self,
        weights: np.ndarray,
        confidence_levels: List[float] = [0.95, 0.99, 0.995],
        n_simulations: int = 50_000,
        seed: Optional[int] = None,
    ) -> pd.DataFrame:
        port_returns = self.simulate_portfolio_returns(
            weights, n_simulations, seed
        )
        rows = []
        for cl in confidence_levels:
            var = -np.quantile(port_returns, 1 - cl)
            tail = port_returns[port_returns <= -var]
            es = -tail.mean() if len(tail) > 0 else var
            rows.append({
                "confidence_level": cl,
                "VaR": round(float(var), 6),
                "ES": round(float(es), 6),
                "copula": "FactorCopula",
                "nu_factor": round(self._copula.nu_factor_, 2),
            })
        return pd.DataFrame(rows)


# Função de conveniência para main_pipeline.py

# Wrapper de alto nível para uso em OptimizedPipeline.step4_fit_copula()
def fit_factor_copula(
    pseudo_obs: np.ndarray,
    n_dim: int,
    n_factors: int = 1,
    nu_factor: Optional[float] = None,
    max_iter: int = 200,
    tol: float = 1e-5,
    n_quad: int = 32,
    random_state: int = 42,
    save_path: Optional[str] = None,
) -> FactorCopula:
    model = FactorCopula(
        n_dim=n_dim,
        n_factors=n_factors,
        nu_factor=nu_factor,
        max_iter=max_iter,
        tol=tol,
        n_quad=n_quad,
        random_state=random_state,
    )
    model.fit(pseudo_obs)

    if save_path is not None:
        model.save(save_path)

    summary = model.get_regime_summary()
    logger.info(
        f"FactorCopula ajustada: ν={summary['nu_factor']}  "
        f"λ_mean={summary['lambda_mean']}  "
        f"tail_dep_mean={summary['tail_dep_mean']}  "
        f"AIC={summary['aic']}  mBICV={summary['mbicv']}"
    )
    return model


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    np.random.seed(42)

    T, d = 500, 10
    lam_true = np.array([0.75, 0.70, 0.65, 0.60, 0.55,
                         0.40, 0.35, 0.30, 0.25, 0.20])
    nu_true = 5.0
    Z_factor = stats.t.rvs(df=nu_true, size=T) / np.sqrt(nu_true / (nu_true - 2))
    eps = np.random.standard_normal((T, d))
    sig_e = np.sqrt(1 - lam_true ** 2)
    X = lam_true * Z_factor[:, np.newaxis] + sig_e * eps
    U = np.clip(stats.norm.cdf(X).astype(np.float32), 1e-6, 1 - 1e-6)

    print(f"U: {U.shape}")
    print(f"λ verdadeiros: {lam_true}")

    print("\n FactorCopula.fit() ")
    fc = FactorCopula(n_dim=d, n_factors=1, nu_factor=None, max_iter=80, n_quad=24)
    fc.fit(U)

    print(f"\nλ estimados:  {fc.lambdas_[:, 0].round(3)}")
    print(f"ν estimado:   {fc.nu_factor_:.2f}  (verdadeiro={nu_true})")
    print(f"AIC={fc.aic_:.2f}  mBICV={fc.mbicv_:.2f}  LL={fc.log_likelihood_:.2f}")

    print("\nSummary:")
    print(fc.get_summary().to_string(index=False))

    print("\nTail dependence (seleção de pares):")
    for i, j in [(0, 1), (0, 5), (4, 9)]:
        td = fc.tail_dependence(i, j)
        print(
            f"  ({i},{j}): λ_L={td['lower_tail']:.4f}  "
            f"ρ_ij={td['rho_ij']:.4f}"
        )

    print("\nCorrelação implícita (5x5 bloco):")
    R = fc.implied_correlation()
    print(pd.DataFrame(R[:5, :5]).round(3).to_string())

    U_sim = fc.simulate(5000, seed=1)
    print(f"\nSimulação: {U_sim.shape}  range=[{U_sim.min():.3f}, {U_sim.max():.3f}]")

    print("\n fit_factor_copula() ")
    fc2 = fit_factor_copula(U, n_dim=d, n_factors=1, max_iter=60, n_quad=20)
    print(f"AIC={fc2.aic_:.2f}  mBICV={fc2.mbicv_:.2f}  ν={fc2.nu_factor_:.2f}")

    summary = fc2.get_regime_summary()
    print("\nRegime summary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    print("\n FactorCopulaRisk ")
    asset_names = [f"ATIVO{i+1}" for i in range(d)]
    dates = pd.date_range("2020-01-01", periods=T, freq="B")
    returns_df = pd.DataFrame(X * 0.01, index=dates, columns=asset_names)
    weights = np.ones(d) / d

    fcr = FactorCopulaRisk()
    fcr.fit(returns_df, {}, fc2, asset_names)
    summary_risk = fcr.var_es_summary(weights, n_simulations=5000, seed=42)
    print(summary_risk.to_string(index=False))

    print("\nTodos os testes concluídos.")
