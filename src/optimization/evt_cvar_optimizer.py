
import numpy as np
import pandas as pd
import logging
import warnings
from typing import Optional, Dict, List, Tuple, Union
from scipy.optimize import minimize, LinearConstraint, Bounds
from scipy import stats

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    from semi_parametric import SemiParametricGARCH_EVT
    from vine_copulas import CVineCopula, DVineCopula
except ImportError as e:
    logger.debug(f"Import direto de semi_parametric/vine_copulas falhou: {e}")
    SemiParametricGARCH_EVT = None
    CVineCopula = DVineCopula = None


# 1. Cálculo de CVaR via simulação

# Calcula VaR e CVaR (ES) de uma serie de retornos simulados.
def compute_cvar(
    port_returns: np.ndarray,
    confidence_level: float = 0.99,
) -> Tuple[float, float]:
    r = np.asarray(port_returns, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) == 0:
        return np.nan, np.nan
    var  = float(-np.quantile(r, 1 - confidence_level))
    tail = r[r <= -var]
    cvar = float(-tail.mean()) if len(tail) > 0 else var
    return var, cvar


# Converte cenários uniformes em retornos do portfólio.
def simulate_portfolio_returns(
    weights: np.ndarray,
    U_sim: np.ndarray,
    returns_df: pd.DataFrame,
    marginal_models: Optional[Dict] = None,
    asset_names: Optional[List[str]] = None,
) -> np.ndarray:
    n_sim, d = U_sim.shape
    R_sim    = np.zeros((n_sim, d))
    names    = asset_names or list(returns_df.columns[:d])

    for j, name in enumerate(names):
        u_raw = U_sim[:, j]
        nan_mask = ~np.isfinite(u_raw)
        if nan_mask.any():
            logger.debug(
                f"'{name}': {nan_mask.sum()} NaN em U_sim substituidos por 0.5"
            )
            u_raw = u_raw.copy()
            u_raw[nan_mask] = 0.5
        u_j   = np.clip(u_raw, 1e-6, 1 - 1e-6)
        model = (marginal_models or {}).get(name)

        if model is not None and hasattr(model, "ppf"):
            try:
                R_sim[:, j] = model.ppf(u_j)
                continue
            except Exception as e:
                logger.warning(
                    f"ppf() FALHOU para '{name}' — usando fallback historico. "
                    f"Excecao: {type(e).__name__}: {e}"
                )

        elif model is not None:
            logger.warning(
                f"marginal_models['{name}'] nao tem .ppf(), usando fallback "
                f"historico (perde a modelagem GPD da cauda)."
            )

        hist = returns_df[name].dropna().values
        R_sim[:, j] = np.quantile(hist, u_j)

    return R_sim @ weights


# 2. EVTCVaROptimizer

# Otimização de portfólio via CVaR com marginais GARCH-EVT e cópula.
class EVTCVaROptimizer:

    def __init__(
        self,
        confidence_level: float = 0.99,
        n_simulations: int = 10_000,
        seed: int = 42,
        allow_short: bool = False,
        max_weight: float = 0.40,
    ):
        self.confidence_level = confidence_level
        self.n_simulations    = n_simulations
        self.seed             = seed
        self.allow_short      = allow_short
        self.max_weight       = max_weight

        self._U_sim: Optional[np.ndarray]     = None
        self._returns_df: Optional[pd.DataFrame] = None
        self._marginal_models: Dict            = {}
        self._asset_names: List[str]           = []
        self._is_fitted                        = False
        self.results_: Dict                    = {}

    # Prepara simulações Monte Carlo para otimização.
    def fit(
        self,
        returns_df: pd.DataFrame,
        copula_model,
        marginal_models: Optional[Dict] = None,
        asset_names: Optional[List[str]] = None,
    ) -> "EVTCVaROptimizer":
        self._returns_df    = returns_df
        self._marginal_models = marginal_models or {}
        self._asset_names   = asset_names or list(returns_df.columns)

        logger.info(
            f"EVTCVaROptimizer.fit()  d={len(self._asset_names)}  "
            f"n_sim={self.n_simulations}  alpha={self.confidence_level}"
        )
        min_sim_99 = int(100 / (1 - self.confidence_level))
        if self.n_simulations < min_sim_99:
            logger.warning(
                f"n_simulations={self.n_simulations} insuficiente para "
                f"VaR {self.confidence_level:.0%} estavel — recomendado >= "
                f"{min_sim_99} (100 obs na cauda). CVaR pode ser negativo."
            )

        np.random.seed(self.seed)
        try:
            self._U_sim = copula_model.simulate(self.n_simulations, seed=self.seed)
            logger.info(f"Cenários simulados: {self._U_sim.shape}")
        except Exception as e:
            logger.error(f"Falha ao simular cópula: {e}")
            raise

        self._is_fitted = True
        return self

    # CVaR do portfólio para dados pesos — função objetivo.
    def _portfolio_cvar(self, weights: np.ndarray) -> float:
        port_r = simulate_portfolio_returns(
            weights, self._U_sim, self._returns_df,
            self._marginal_models, self._asset_names,
        )
        _, cvar = compute_cvar(port_r, self.confidence_level)
        return cvar

    def _build_constraints(self, d: int) -> List:
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]
        return constraints

    def _build_bounds(self, d: int) -> Bounds:
        lb = -self.max_weight if self.allow_short else 0.0
        ub = self.max_weight
        return Bounds(lb=[lb]*d, ub=[ub]*d)

    # Minimiza CVaR do portfólio.
    def optimize_min_cvar(
        self,
        initial_weights: Optional[np.ndarray] = None,
        n_restarts: int = 5,
    ) -> Dict:
        if not self._is_fitted:
            raise RuntimeError("Execute fit() primeiro.")

        d     = len(self._asset_names)
        w0    = initial_weights if initial_weights is not None else np.ones(d) / d
        cons  = self._build_constraints(d)
        bnds  = self._build_bounds(d)

        best_w, best_cvar = w0.copy(), np.inf

        for trial in range(n_restarts):
            if trial == 0:
                w_init = w0.copy()
            else:
                w_init = np.random.dirichlet(np.ones(d))

            try:
                res = minimize(
                    self._portfolio_cvar,
                    w_init,
                    method="SLSQP",
                    bounds=list(zip(bnds.lb, bnds.ub)),
                    constraints=cons,
                    options={"ftol": 1e-9, "maxiter": 1000},
                )
                if res.success and res.fun < best_cvar:
                    best_cvar = res.fun
                    best_w    = res.x.copy()
            except Exception as e:
                logger.debug(f"Restart {trial}: {e}")

        port_r = simulate_portfolio_returns(
            best_w, self._U_sim, self._returns_df,
            self._marginal_models, self._asset_names,
        )
        var, cvar = compute_cvar(port_r, self.confidence_level)
        ann_ret   = float(np.mean(port_r) * 252)
        ann_vol   = float(np.std(port_r) * np.sqrt(252))

        result = {
            "strategy":          "min_cvar",
            "weights":           dict(zip(self._asset_names, best_w.round(6))),
            "CVaR":              round(cvar, 6),
            "VaR":               round(var, 6),
            "annual_return":     round(ann_ret, 6),
            "annual_vol":        round(ann_vol, 6),
            "sharpe":            round(ann_ret / ann_vol if ann_vol > 0 else 0, 4),
            "confidence_level":  self.confidence_level,
        }
        self.results_["min_cvar"] = result
        logger.info(f"Min-CVaR: CVaR={cvar:.4f}  VaR={var:.4f}  ret={ann_ret:.4f}")
        return result

    # Fronteira eficiente Mean-CVaR.
    def optimize_mean_cvar(
        self,
        target_return: Optional[float] = None,
        n_points: int = 20,
    ) -> pd.DataFrame:
        if not self._is_fitted:
            raise RuntimeError("Execute fit() primeiro.")

        d    = len(self._asset_names)
        bnds = self._build_bounds(d)

        ret_min = optimize_single_asset_metric(
            self._U_sim, self._returns_df,
            self._marginal_models, self._asset_names, metric="min_return"
        )
        ret_max = optimize_single_asset_metric(
            self._U_sim, self._returns_df,
            self._marginal_models, self._asset_names, metric="max_return"
        )

        targets = np.linspace(ret_min * 1.05, ret_max * 0.95, n_points)
        frontier = []

        for target in targets:
            cons = [
                {"type": "eq",  "fun": lambda w: np.sum(w) - 1.0},
                {"type": "ineq", "fun": lambda w, t=target: self._mean_return(w) - t},
            ]
            w0 = np.ones(d) / d
            try:
                res = minimize(
                    self._portfolio_cvar,
                    w0,
                    method="SLSQP",
                    bounds=list(zip(bnds.lb, bnds.ub)),
                    constraints=cons,
                    options={"ftol": 1e-8, "maxiter": 500},
                )
                if res.success:
                    port_r = simulate_portfolio_returns(
                        res.x, self._U_sim, self._returns_df,
                        self._marginal_models, self._asset_names,
                    )
                    var, cvar = compute_cvar(port_r, self.confidence_level)
                    ann_ret   = float(np.mean(port_r) * 252)
                    frontier.append({
                        "target_return": round(float(target), 6),
                        "annual_return": round(ann_ret, 6),
                        "CVaR":          round(cvar, 6),
                        "VaR":           round(var, 6),
                        "weights":       dict(zip(self._asset_names, res.x.round(4))),
                    })
            except Exception:
                continue

        df = pd.DataFrame(frontier)
        self.results_["mean_cvar_frontier"] = df
        logger.info(f"Mean-CVaR frontier: {len(df)} pontos")
        return df

    def _mean_return(self, weights: np.ndarray) -> float:
        port_r = simulate_portfolio_returns(
            weights, self._U_sim, self._returns_df,
            self._marginal_models, self._asset_names,
        )
        return float(np.mean(port_r) * 252)

    # CVaR Risk Parity: iguala a contribuição marginal de CVaR de cada ativo.
    def optimize_cvar_parity(
        self,
        n_restarts: int = 5,
    ) -> Dict:
        if not self._is_fitted:
            raise RuntimeError("Execute fit() primeiro.")

        d    = len(self._asset_names)
        bnds = self._build_bounds(d)
        cons = self._build_constraints(d)

        def cvar_parity_objective(weights: np.ndarray) -> float:
            port_r   = simulate_portfolio_returns(
                weights, self._U_sim, self._returns_df,
                self._marginal_models, self._asset_names,
            )
            _, cvar  = compute_cvar(port_r, self.confidence_level)
            contribs = []
            eps      = 1e-4
            for i in range(d):
                w_plus        = weights.copy()
                w_plus[i]    += eps
                w_plus       /= w_plus.sum()
                port_plus     = simulate_portfolio_returns(
                    w_plus, self._U_sim, self._returns_df,
                    self._marginal_models, self._asset_names,
                )
                _, cvar_plus = compute_cvar(port_plus, self.confidence_level)
                marginal      = (cvar_plus - cvar) / eps
                contribs.append(weights[i] * marginal)

            contribs  = np.array(contribs)
            target    = cvar / d
            return float(np.sum((contribs - target) ** 2))

        best_w, best_obj = np.ones(d)/d, np.inf
        for trial in range(n_restarts):
            w_init = np.random.dirichlet(np.ones(d)) if trial > 0 else np.ones(d)/d
            try:
                res = minimize(
                    cvar_parity_objective, w_init,
                    method="SLSQP",
                    bounds=list(zip(bnds.lb, bnds.ub)),
                    constraints=cons,
                    options={"ftol": 1e-8, "maxiter": 500},
                )
                if res.success and res.fun < best_obj:
                    best_obj = res.fun
                    best_w   = res.x.copy()
            except Exception:
                continue

        port_r = simulate_portfolio_returns(
            best_w, self._U_sim, self._returns_df,
            self._marginal_models, self._asset_names,
        )
        var, cvar = compute_cvar(port_r, self.confidence_level)
        ann_ret   = float(np.mean(port_r) * 252)
        ann_vol   = float(np.std(port_r) * np.sqrt(252))

        result = {
            "strategy":      "cvar_parity",
            "weights":       dict(zip(self._asset_names, best_w.round(6))),
            "CVaR":          round(cvar, 6),
            "VaR":           round(var, 6),
            "annual_return": round(ann_ret, 6),
            "annual_vol":    round(ann_vol, 6),
            "sharpe":        round(ann_ret / ann_vol if ann_vol > 0 else 0, 4),
        }
        self.results_["cvar_parity"] = result
        logger.info(f"CVaR Parity: CVaR={cvar:.4f}")
        return result

    # Tabela comparativa das estratégias otimizadas.
    def compare_strategies(self) -> pd.DataFrame:
        rows = []
        for name, r in self.results_.items():
            if isinstance(r, dict) and "CVaR" in r:
                rows.append({
                    "estrategia":    r.get("strategy", name),
                    "CVaR":          r["CVaR"],
                    "VaR":           r.get("VaR", np.nan),
                    "annual_return": r.get("annual_return", np.nan),
                    "annual_vol":    r.get("annual_vol", np.nan),
                    "sharpe":        r.get("sharpe", np.nan),
                })
        return pd.DataFrame(rows).sort_values("CVaR")

    # Retorna pesos como DataFrame ordenado.
    def weights_dataframe(self, strategy: str = "min_cvar") -> pd.DataFrame:
        if strategy not in self.results_:
            raise ValueError(f"Estratégia '{strategy}' não calculada.")
        r = self.results_[strategy]
        w = r.get("weights", {})
        df = pd.DataFrame(
            list(w.items()), columns=["ativo", "peso"]
        ).sort_values("peso", ascending=False)
        return df


# Retorna retorno mínimo ou máximo atingível (portfólio 100% em um ativo).
def optimize_single_asset_metric(
    U_sim: np.ndarray,
    returns_df: pd.DataFrame,
    marginal_models: Dict,
    asset_names: List[str],
    metric: str = "min_return",
) -> float:
    returns = []
    for i, name in enumerate(asset_names):
        w = np.zeros(len(asset_names))
        w[i] = 1.0
        port_r = simulate_portfolio_returns(w, U_sim, returns_df, marginal_models, asset_names)
        returns.append(float(np.mean(port_r) * 252))
    return min(returns) if metric == "min_return" else max(returns)


# 3. Wrapper para main_pipeline.py

# Wrapper de alto nível para main_pipeline.py.
def run_evt_cvar_optimization(
    returns_df: pd.DataFrame,
    copula_model,
    marginal_models: Optional[Dict] = None,
    asset_names: Optional[List[str]] = None,
    confidence_level: float = 0.99,
    n_simulations: int = 10_000,
    strategies: Optional[List[str]] = None,
    save_path: Optional[str] = None,
) -> Tuple["EVTCVaROptimizer", pd.DataFrame]:
    optimizer = EVTCVaROptimizer(
        confidence_level=confidence_level,
        n_simulations=n_simulations,
    )
    optimizer.fit(returns_df, copula_model, marginal_models, asset_names)

    strategies = strategies or ["min_cvar", "cvar_parity"]

    if "min_cvar"    in strategies: optimizer.optimize_min_cvar()
    if "cvar_parity" in strategies: optimizer.optimize_cvar_parity()
    if "mean_cvar"   in strategies: optimizer.optimize_mean_cvar()

    comparison = optimizer.compare_strategies()

    if save_path:
        path = Path(save_path)
        path.mkdir(parents=True, exist_ok=True)
        comparison.to_csv(path / "evt_cvar_strategies.csv", index=False)
        for strategy in strategies:
            if strategy in optimizer.results_ and isinstance(optimizer.results_[strategy], dict):
                optimizer.weights_dataframe(strategy).to_csv(
                    path / f"weights_{strategy}.csv", index=False
                )

    return optimizer, comparison


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    np.random.seed(42)

    T, d = 600, 5
    tickers = [f"ATIVO{i+1}" for i in range(d)]

    vols   = np.array([0.025, 0.022, 0.018, 0.015, 0.020])
    corr   = 0.4 * np.ones((d, d)) + 0.6 * np.eye(d)
    cov    = np.outer(vols, vols) * corr
    rets   = np.random.multivariate_normal(np.zeros(d), cov, T)
    returns_df = pd.DataFrame(rets, columns=tickers)

    try:
        from vine_copulas import CVineCopula
        from scipy.stats import t as t_dist
        nu  = 5
        Z   = np.random.multivariate_normal(np.zeros(d), corr, T)
        chi2_rv = np.random.chisquare(nu, T) / nu
        Tv  = Z / np.sqrt(chi2_rv[:, None])
        U   = np.clip(t_dist.cdf(Tv, df=nu), 1e-6, 1 - 1e-6).astype(np.float32)

        copula = CVineCopula(n_dim=d)
        copula.fit(U, max_trees=2, min_tau=0.05)

        print(f"dados: {returns_df.shape}")

        optimizer = EVTCVaROptimizer(
            confidence_level=0.99,
            n_simulations=5_000,
            seed=42,
        )
        optimizer.fit(returns_df, copula, asset_names=tickers)

        print("\n Min-CVaR ")
        r1 = optimizer.optimize_min_cvar(n_restarts=3)
        print(f"CVaR={r1['CVaR']:.4f}  VaR={r1['VaR']:.4f}  ret={r1['annual_return']:.4f}")

        print("\n CVaR Parity ")
        r2 = optimizer.optimize_cvar_parity(n_restarts=3)
        print(f"CVaR={r2['CVaR']:.4f}  VaR={r2['VaR']:.4f}")

        print("\n Comparação ")
        print(optimizer.compare_strategies().to_string(index=False))

        print("\n Pesos Min-CVaR ")
        print(optimizer.weights_dataframe("min_cvar").to_string(index=False))

    except ImportError as e:
        print(f"Dependência ausente: {e}")

    print("\n Todos os testes concluídos.")
