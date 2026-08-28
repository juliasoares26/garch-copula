
from __future__ import annotations

import logging
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize, Bounds

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


# Estimadores de covariância

# Covariância amostral com regularização diagonal mínima.
def sample_cov(returns: np.ndarray, min_periods: int = 20) -> np.ndarray:
    T, d = returns.shape
    if T < min_periods:
        return np.eye(d) * 1e-4
    cov = np.cov(returns, rowvar=False)
    cov += np.eye(d) * 1e-8
    return cov


# Shrinkage de Ledoit-Wolf (2004) para covariância.
def ledoit_wolf_cov(returns: np.ndarray) -> np.ndarray:
    T, d = returns.shape
    if T < d + 5:
        return sample_cov(returns)

    try:
        from sklearn.covariance import LedoitWolf
        lw = LedoitWolf(assume_centered=False).fit(returns)
        return lw.covariance_
    except ImportError:
        S = np.cov(returns, rowvar=False)
        mu = np.trace(S) / d
        delta_sq = np.sum((S - mu * np.eye(d)) ** 2)
        beta_bar = np.sum(
            [np.sum((np.outer(r, r) - S) ** 2) for r in returns]
        ) / (T ** 2)
        beta = min(beta_bar / delta_sq, 1.0) if delta_sq > 0 else 0.0
        return (1 - beta) * S + beta * mu * np.eye(d)


# Garante que a matriz de covariância é positiva definida.
def _ensure_pd(cov: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    eigvals = np.linalg.eigvalsh(cov)
    if eigvals.min() < eps:
        cov = cov + (eps - eigvals.min()) * np.eye(len(cov))
    return cov


# Funções de portfólio

def portfolio_vol(w: np.ndarray, cov: np.ndarray) -> float:
    return float(np.sqrt(np.maximum(w @ cov @ w, 0.0)))


# Contribuição marginal de risco: RC_i = w_i * (Cov @ w)_i / sigma_p.
def risk_contributions(w: np.ndarray, cov: np.ndarray) -> np.ndarray:
    sigma = portfolio_vol(w, cov)
    if sigma < 1e-10:
        return np.ones(len(w)) / len(w)
    mrc = cov @ w
    return w * mrc / sigma


# DR = (Σ w_i σ_i) / σ_p — Choueifaty & Coignard (2008).
def diversification_ratio(w: np.ndarray, cov: np.ndarray) -> float:
    asset_vols = np.sqrt(np.diag(cov))
    weighted_vol = float(w @ asset_vols)
    port_vol = portfolio_vol(w, cov)
    return weighted_vol / max(port_vol, 1e-8)


# Base optimizer

# Interface comum para todos os otimizadores desta família.
class _BaseOptimizer:

    def __init__(
        self,
        max_weight: float = 0.40,
        min_weight: float = 0.0,
        cov_estimator: str = "sample",
        n_restarts: int = 3,
        random_state: int = 42,
    ):
        self.max_weight    = max_weight
        self.min_weight    = min_weight
        self.cov_estimator = cov_estimator
        self.n_restarts    = n_restarts
        self.random_state  = random_state
        self._rng           = np.random.default_rng(random_state)

    def _estimate_cov(self, returns: np.ndarray) -> np.ndarray:
        if self.cov_estimator == "ledoit_wolf":
            cov = ledoit_wolf_cov(returns)
        else:
            cov = sample_cov(returns)
        return _ensure_pd(cov)

    def _bounds(self, d: int) -> Bounds:
        return Bounds(lb=self.min_weight, ub=self.max_weight)

    def _sum_one_constraint(self) -> List[Dict]:
        return [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

    def _solve(
        self,
        objective,
        d: int,
        extra_constraints: Optional[List[Dict]] = None,
    ) -> Tuple[np.ndarray, bool]:
        constraints = self._sum_one_constraint()
        if extra_constraints:
            constraints += extra_constraints
        bounds = self._bounds(d)

        # Os restarts são independentes. Executá-los em paralelo reduz o
        # tempo total sem alterar a função objetivo, restrições, inicializações
        # ou critérios de aceitação.
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import os

        def _solve_trial(trial):
            rng = np.random.default_rng(self.random_state + int(trial))
            w0 = rng.dirichlet(np.ones(d)) if trial > 0 else np.ones(d) / d
            w0 = np.clip(w0, self.min_weight, self.max_weight)
            w0 /= w0.sum()
            try:
                res = minimize(
                    objective, w0, method="SLSQP",
                    bounds=bounds,
                    constraints=constraints,
                    options={"ftol": 1e-10, "maxiter": 1000},
                )
                return trial, res
            except Exception as exc:
                logger.debug(f"  restart {trial}: {exc}")
                return trial, None

        best_w, best_val = np.ones(d) / d, np.inf
        succeeded = False
        n_trials = max(int(self.n_restarts), 1)
        max_workers = min(n_trials, max(1, (os.cpu_count() or 2) - 1), 5)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_solve_trial, trial) for trial in range(n_trials)]
            for future in as_completed(futures):
                trial, res = future.result()
                if res is not None and res.success and res.fun < best_val:
                    best_val = res.fun
                    best_w = res.x.copy()
                    succeeded = True

        best_w = np.clip(best_w, 0, 1)
        if best_w.sum() > 1e-8:
            best_w /= best_w.sum()
        return best_w, succeeded

    def _build_result(
        self,
        strategy: str,
        weights: np.ndarray,
        returns: np.ndarray,
        cov: np.ndarray,
        asset_names: List[str],
    ) -> Dict:
        r_port = returns @ weights
        ann_ret = float(np.mean(r_port) * 252)
        ann_vol = float(np.std(r_port, ddof=1) * np.sqrt(252))
        sharpe  = ann_ret / max(ann_vol, 1e-8)
        dr      = diversification_ratio(weights, cov)
        rc      = risk_contributions(weights, cov)

        return {
            "strategy":       strategy,
            "weights":        dict(zip(asset_names, weights.round(6))),
            "weights_array":  weights,
            "annual_return":  round(ann_ret, 6),
            "annual_vol":     round(ann_vol, 6),
            "sharpe":         round(sharpe, 4),
            "diversif_ratio": round(dr, 4),
            "risk_contribs":  dict(zip(asset_names, rc.round(6))),
            "herfindahl":     round(float(np.sum(weights ** 2)), 6),
        }


# 1. Equal Weight (EW)

# 1/N — referência naive.
class EqualWeightOptimizer(_BaseOptimizer):

    def optimize(
        self,
        returns: pd.DataFrame,
        asset_names: Optional[List[str]] = None,
    ) -> Dict:
        names  = asset_names or list(returns.columns)
        d      = len(names)
        w      = np.ones(d) / d
        r_arr  = returns[names].values
        cov    = self._estimate_cov(r_arr)
        result = self._build_result("equal_weight", w, r_arr, cov, names)
        logger.info(f"EW: vol={result['annual_vol']:.4f}  DR={result['diversif_ratio']:.3f}")
        return result

    # Interface para WalkForwardBacktest.optimizer_fn.
    def __call__(self, returns: pd.DataFrame) -> np.ndarray:
        d = returns.shape[1]
        return np.ones(d) / d


# 2. Minimum Variance (MV)

# Markowitz (1952) — minimiza variância do portfólio sem target de retorno.
class MinVarianceOptimizer(_BaseOptimizer):

    def optimize(
        self,
        returns: pd.DataFrame,
        asset_names: Optional[List[str]] = None,
    ) -> Dict:
        names = asset_names or list(returns.columns)
        r_arr = returns[names].values
        cov   = self._estimate_cov(r_arr)
        d     = len(names)

        objective = lambda w: float(w @ cov @ w)
        w, ok = self._solve(objective, d)

        if not ok:
            logger.warning("MV não convergiu; usando 1/N fallback.")
            w = np.ones(d) / d

        result = self._build_result("min_variance", w, r_arr, cov, names)
        logger.info(
            f"MV: vol={result['annual_vol']:.4f}  "
            f"DR={result['diversif_ratio']:.3f}  converged={ok}"
        )
        return result

    def __call__(self, returns: pd.DataFrame) -> np.ndarray:
        return self.optimize(returns)["weights_array"]


# 3. Risk Parity / Equal Risk Contribution (RP)

# Maillard, Roncalli & Teïletche (2010) — iguala contribuições de risco.
class RiskParityOptimizer(_BaseOptimizer):

    # Parameters
    def optimize(
        self,
        returns: pd.DataFrame,
        asset_names: Optional[List[str]] = None,
        target_risk_budget: Optional[np.ndarray] = None,
    ) -> Dict:
        names = asset_names or list(returns.columns)
        r_arr = returns[names].values
        cov   = self._estimate_cov(r_arr)
        d     = len(names)

        budget = (
            np.asarray(target_risk_budget, dtype=float)
            if target_risk_budget is not None
            else np.ones(d) / d
        )
        budget /= budget.sum()

        def objective(w: np.ndarray) -> float:
            sigma = portfolio_vol(w, cov)
            if sigma < 1e-10:
                return 0.0
            mrc = cov @ w
            rc  = w * mrc / sigma
            rc_pct = rc / sigma
            return float(np.sum((rc_pct - budget) ** 2))

        orig_min = self.min_weight
        self.min_weight = max(self.min_weight, 1e-5)
        w, ok = self._solve(objective, d)
        self.min_weight = orig_min

        if not ok:
            logger.warning("RP não convergiu; usando 1/N fallback.")
            w = np.ones(d) / d

        result = self._build_result("risk_parity", w, r_arr, cov, names)
        result["risk_budget"] = dict(zip(names, budget.round(4)))
        logger.info(
            f"RP: vol={result['annual_vol']:.4f}  "
            f"DR={result['diversif_ratio']:.3f}  converged={ok}"
        )
        return result

    def __call__(self, returns: pd.DataFrame) -> np.ndarray:
        return self.optimize(returns)["weights_array"]


# 4. Maximum Diversification (MD)

# Choueifaty & Coignard (2008) — maximiza Diversification Ratio.
class MaxDiversificationOptimizer(_BaseOptimizer):

    def optimize(
        self,
        returns: pd.DataFrame,
        asset_names: Optional[List[str]] = None,
    ) -> Dict:
        names     = asset_names or list(returns.columns)
        r_arr     = returns[names].values
        cov       = self._estimate_cov(r_arr)
        d         = len(names)
        asset_vol = np.sqrt(np.diag(cov))

        def objective(w: np.ndarray) -> float:
            return -diversification_ratio(w, cov)

        w, ok = self._solve(objective, d)

        if not ok:
            w = 1.0 / asset_vol
            w /= w.sum()
            logger.warning("MD não convergiu; usando inverse-vol fallback.")

        result = self._build_result("max_diversification", w, r_arr, cov, names)
        logger.info(
            f"MD: vol={result['annual_vol']:.4f}  "
            f"DR={result['diversif_ratio']:.3f}  converged={ok}"
        )
        return result

    def __call__(self, returns: pd.DataFrame) -> np.ndarray:
        return self.optimize(returns)["weights_array"]


# 5. Maximum Decorrelation (MDE)

# Christoffersen et al. (2012) — minimiza variância do portfólio na
class MaxDecorrelationOptimizer(_BaseOptimizer):

    def optimize(
        self,
        returns: pd.DataFrame,
        asset_names: Optional[List[str]] = None,
    ) -> Dict:
        names     = asset_names or list(returns.columns)
        r_arr     = returns[names].values
        cov       = self._estimate_cov(r_arr)
        d         = len(names)

        std_diag  = np.sqrt(np.diag(cov))
        corr      = cov / np.outer(std_diag, std_diag)
        corr      = _ensure_pd(corr)

        objective = lambda w: float(w @ corr @ w)
        w, ok = self._solve(objective, d)

        if not ok:
            logger.warning("MDE não convergiu; usando 1/N fallback.")
            w = np.ones(d) / d

        result = self._build_result("max_decorrelation", w, r_arr, cov, names)
        result["port_correlation_var"] = round(float(w @ corr @ w), 6)
        logger.info(
            f"MDE: vol={result['annual_vol']:.4f}  "
            f"DR={result['diversif_ratio']:.3f}  converged={ok}"
        )
        return result

    def __call__(self, returns: pd.DataFrame) -> np.ndarray:
        return self.optimize(returns)["weights_array"]


# RiskBasedPortfolioSuite — interface unificada

# Executa todas as estratégias risk-based sobre a mesma janela de retornos
class RiskBasedPortfolioSuite:

    STRATEGY_MAP = {
        "ew":  EqualWeightOptimizer,
        "mv":  MinVarianceOptimizer,
        "rp":  RiskParityOptimizer,
        "md":  MaxDiversificationOptimizer,
        "mde": MaxDecorrelationOptimizer,
    }

    def __init__(
        self,
        strategies: Optional[List[str]] = None,
        max_weight: float = 0.40,
        min_weight: float = 0.0,
        cov_estimator: str = "ledoit_wolf",
        n_restarts: int = 5,
    ):
        self.strategies    = strategies or list(self.STRATEGY_MAP.keys())
        self.max_weight    = max_weight
        self.min_weight    = min_weight
        self.cov_estimator = cov_estimator
        self.n_restarts    = n_restarts
        self.results_: Dict[str, Dict] = {}

    def _make_optimizer(self, key: str) -> _BaseOptimizer:
        cls = self.STRATEGY_MAP[key]
        return cls(
            max_weight    = self.max_weight,
            min_weight    = self.min_weight,
            cov_estimator = self.cov_estimator,
            n_restarts    = self.n_restarts,
        )

    # Executa todas as estratégias selecionadas.
    def run(self, returns: pd.DataFrame) -> Dict[str, Dict]:
        self.results_ = {}
        for key in self.strategies:
            if key not in self.STRATEGY_MAP:
                logger.warning(f"Estratégia desconhecida: '{key}'. Pulando.")
                continue
            try:
                opt = self._make_optimizer(key)
                self.results_[key] = opt.optimize(returns)
            except Exception as exc:
                logger.error(f"Erro em estratégia '{key}': {exc}")
        return self.results_

    # DataFrame comparativo de todas as estratégias calculadas.
    def compare(self) -> pd.DataFrame:
        if not self.results_:
            raise RuntimeError("Execute run() primeiro.")
        rows = []
        for key, res in self.results_.items():
            rows.append({
                "estrategia":     res.get("strategy", key),
                "annual_return":  res.get("annual_return", np.nan),
                "annual_vol":     res.get("annual_vol", np.nan),
                "sharpe":         res.get("sharpe", np.nan),
                "diversif_ratio": res.get("diversif_ratio", np.nan),
                "herfindahl":     res.get("herfindahl", np.nan),
            })
        return pd.DataFrame(rows).sort_values("sharpe", ascending=False).reset_index(drop=True)

    # DataFrame (estratégia × ativo) com os pesos de cada portfólio.
    def weights_dataframe(self) -> pd.DataFrame:
        if not self.results_:
            raise RuntimeError("Execute run() primeiro.")
        return pd.DataFrame(
            {key: res["weights"] for key, res in self.results_.items()}
        ).T

    # Retorna callable compatível com WalkForwardBacktest.optimizer_fn.
    def optimizer_fn(self, strategy: str):
        opt = self._make_optimizer(strategy)
        return opt.__call__


# Standalone helpers (compatíveis com backtesting_engine.py)

# Drop-in replacement para equal_weight_optimizer de backtesting_engine.
def equal_weight_optimizer(returns_df: pd.DataFrame) -> np.ndarray:
    d = returns_df.shape[1]
    return np.ones(d) / d


# Drop-in replacement para min_variance_optimizer de backtesting_engine.
def min_variance_optimizer(
    returns_df: pd.DataFrame,
    max_weight: float = 0.40,
    cov_estimator: str = "ledoit_wolf",
) -> np.ndarray:
    opt = MinVarianceOptimizer(max_weight=max_weight, cov_estimator=cov_estimator)
    return opt(returns_df)


def risk_parity_optimizer(
    returns_df: pd.DataFrame,
    max_weight: float = 0.40,
) -> np.ndarray:
    opt = RiskParityOptimizer(max_weight=max_weight)
    return opt(returns_df)


def max_diversification_optimizer(
    returns_df: pd.DataFrame,
    max_weight: float = 0.40,
    cov_estimator: str = "ledoit_wolf",
) -> np.ndarray:
    opt = MaxDiversificationOptimizer(max_weight=max_weight, cov_estimator=cov_estimator)
    return opt(returns_df)


def max_decorrelation_optimizer(
    returns_df: pd.DataFrame,
    max_weight: float = 0.40,
    cov_estimator: str = "ledoit_wolf",
) -> np.ndarray:
    opt = MaxDecorrelationOptimizer(max_weight=max_weight, cov_estimator=cov_estimator)
    return opt(returns_df)


if __name__ == "__main__":
    from pathlib import Path
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    _SCRIPT_DIR    = Path(__file__).resolve().parent
    _STOCKS_SUBDIR = "raw_b3_ibrx50"

    def _valid_root(p: Path) -> bool:
        stocks = p / "data" / _STOCKS_SUBDIR
        ext    = p / "data" / "external"
        return (stocks.exists() and any(stocks.glob("*.xlsx")) and
                (ext / "selic_daily.csv").exists())

    _ROOT = None
    for _cand in [_SCRIPT_DIR.parent.parent, _SCRIPT_DIR.parent,
                  Path.cwd().parent.parent, Path.cwd().parent, Path.cwd()]:
        if _valid_root(_cand):
            _ROOT = _cand
            break

    print(f"[DIAG] __file__  : {Path(__file__).resolve()}")
    print(f"[DIAG] Raiz      : {_ROOT or 'NAO ENCONTRADA (rodando demo sintetico)'}")

    if _ROOT is not None:
        _stocks_dir = _ROOT / "data" / _STOCKS_SUBDIR
        _ext_dir    = _ROOT / "data" / "external"

        import sys as _sys_rbp, pathlib as _pathlib_rbp
        _data_mod_dir = str(_pathlib_rbp.Path(__file__).resolve().parent.parent / "data")
        if _data_mod_dir not in _sys_rbp.path:
            _sys_rbp.path.insert(0, _data_mod_dir)
        from split_adjustment import build_adjusted_panel

        _raw_panel, _adj_panel, _events_df, _unconf_df = build_adjusted_panel(str(_stocks_dir))
        print(f"[INFO] {_adj_panel.shape[1]} ativos carregados "
              f"(correcoes de split: {len(_events_df)} eventos).")

        _price_df = _adj_panel.sort_index()
        _ASSETS   = list(_price_df.columns)
        returns   = np.log(_price_df / _price_df.shift(1)).iloc[1:]

        _selic_df  = pd.read_csv(_ext_dir / "selic_daily.csv",
                                  parse_dates=["Date"], index_col="Date")
        _selic_col = next((c for c in ["Daily_Return", "Rate_Daily_Selic", "Rate"]
                           if c in _selic_df.columns), _selic_df.columns[0])
        _selic     = _selic_df[_selic_col].sort_index()
        _rf_annual = float(((1 + _selic.mean()) ** 252 - 1))

        print(f"[INFO] Ativos   : {len(_ASSETS)} ({_ASSETS[:5]}...)")
        print(f"[INFO] Período  : {returns.index[0].date()} -> {returns.index[-1].date()}")
        print(f"[INFO] Obs      : {len(returns)}")
        print(f"[INFO] Selic rf : {_rf_annual:.4%}")

        _WINDOW = 252
        _REBAL  = 21

    else:
        print("[Demo sintetico -- coloque os dados em data/raw e data/external]")
        np.random.seed(42)
        T, d    = 500, 8
        tickers = [f"ATIVO{i+1}" for i in range(d)]
        mu      = np.random.uniform(-0.0002, 0.0008, d)
        corr    = 0.3 * np.ones((d, d)) + 0.7 * np.eye(d)
        vols    = np.random.uniform(0.012, 0.028, d)
        cov_m   = np.outer(vols, vols) * corr
        raw     = np.random.multivariate_normal(mu, cov_m, T)
        returns = pd.DataFrame(raw, columns=tickers)
        _rf_annual = 0.1075
        _WINDOW = 252
        _REBAL  = 21

    print(f"\n[INFO] Rolling OOS: window={_WINDOW}  rebal={_REBAL}")
    _dates      = returns.index
    _rebal_idx  = list(range(_WINDOW, len(_dates), _REBAL))
    _strategies = ["ew", "mv", "rp", "md", "mde"]

    _oos_returns: dict[str, list] = {s: [] for s in _strategies}
    _oos_dates:   list = []
    _last_weights: dict[str, np.ndarray] = {}

    _suite = RiskBasedPortfolioSuite(
        strategies=_strategies,
        max_weight=0.40,
        cov_estimator="ledoit_wolf",
        n_restarts=5,
    )

    for _i, _t in enumerate(_rebal_idx):
        _train = returns.iloc[_t - _WINDOW : _t]
        _end   = _rebal_idx[_i + 1] if _i + 1 < len(_rebal_idx) else len(_dates)
        _test  = returns.iloc[_t : _end]
        if len(_test) == 0:
            continue

        _results = _suite.run(_train)
        for _s in _strategies:
            _last_weights[_s] = _results[_s]["weights_array"]

        for _d, _row in _test.iterrows():
            for _s in _strategies:
                _oos_returns[_s].append(float(_last_weights[_s] @ _row.values))
            _oos_dates.append(_d)

    _oos_df = pd.DataFrame(_oos_returns, index=_oos_dates)
    _oos_df.index = pd.to_datetime(_oos_df.index)

    print(f"[INFO] Período OOS: {_oos_df.index[0].date()} -> {_oos_df.index[-1].date()}")
    print(f"[INFO] Obs OOS    : {len(_oos_df)}")

    # ── Métricas OOS ─────────────────────────────────────────────────────────
    def _oos_metrics(ret: pd.Series, rf_ann: float) -> dict:
        r   = ret.values
        ann = float(np.mean(r) * 252)
        vol = float(np.std(r, ddof=1) * np.sqrt(252))
        sr  = (ann - rf_ann) / max(vol, 1e-8)
        cum = np.cumprod(1 + r)
        dd  = (cum - np.maximum.accumulate(cum)) / np.maximum.accumulate(cum)
        mdd = float(dd.min())
        cal = ann / max(abs(mdd), 1e-8)
        return dict(annual_return=round(ann,4), annual_vol=round(vol,4),
                    sharpe_rf=round(sr,4), max_drawdown=round(mdd,4),
                    calmar=round(cal,4))

    _name_map = {"ew":"Equal Weight","mv":"Min Variance","rp":"Risk Parity",
                 "md":"Max Diversification","mde":"Max Decorrelation"}

    _rows = []
    for _s in _strategies:
        _m = _oos_metrics(_oos_df[_s], _rf_annual)
        _m["estrategia"] = _name_map[_s]
        _rows.append(_m)

    _cmp = (pd.DataFrame(_rows)
            .set_index("estrategia")
            [["annual_return","annual_vol","sharpe_rf","max_drawdown","calmar"]]
            .sort_values("sharpe_rf", ascending=False))

    print("\n── Comparação OOS (sem lookahead bias) ──")
    print(_cmp.to_string())

    _suite_last = RiskBasedPortfolioSuite(
        strategies=_strategies, max_weight=0.40,
        cov_estimator="ledoit_wolf", n_restarts=5,
    )
    _suite_last.run(returns.iloc[-_WINDOW:])
    print("\n── Pesos atuais (último rebalanceamento) ──")
    print(_suite_last.weights_dataframe().round(4).to_string())
