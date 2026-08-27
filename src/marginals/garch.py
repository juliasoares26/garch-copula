
from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


try:
    from numba import njit as _njit

    @_njit(cache=True)
    def _garch11_var(r: np.ndarray, omega: float, alpha: float,
                     beta: float) -> np.ndarray:
        T = len(r)
        s2 = np.empty(T)
        s2[0] = np.var(r)
        for t in range(1, T):
            s2[t] = omega + alpha * r[t-1]**2 + beta * s2[t-1]
            if s2[t] < 1e-12:
                s2[t] = 1e-12
        return s2

    @_njit(cache=True)
    def _gjr11_var(r: np.ndarray, omega: float, alpha: float,
                   gamma: float, beta: float) -> np.ndarray:
        T = len(r)
        s2 = np.empty(T)
        s2[0] = np.var(r)
        for t in range(1, T):
            ind = 1.0 if r[t-1] < 0.0 else 0.0
            s2[t] = omega + (alpha + gamma * ind) * r[t-1]**2 + beta * s2[t-1]
            if s2[t] < 1e-12:
                s2[t] = 1e-12
        return s2

    NUMBA_AVAILABLE = True

except ImportError:
    NUMBA_AVAILABLE = False

    def _garch11_var(r, omega, alpha, beta):
        T = len(r)
        s2 = np.empty(T)
        s2[0] = np.var(r)
        for t in range(1, T):
            s2[t] = max(omega + alpha * r[t-1]**2 + beta * s2[t-1], 1e-12)
        return s2

    def _gjr11_var(r, omega, alpha, gamma, beta):
        T = len(r)
        s2 = np.empty(T)
        s2[0] = np.var(r)
        for t in range(1, T):
            ind = 1.0 if r[t-1] < 0.0 else 0.0
            s2[t] = max(omega + (alpha + gamma * ind) * r[t-1]**2 + beta * s2[t-1], 1e-12)
        return s2

# Dependência opcional: biblioteca arch

# Import lazy da arch. Retorna arch_model ou None.
def _get_arch_model():
    try:
        from arch import arch_model as _am
        return _am
    except ImportError:
        return None

ARCH_AVAILABLE: bool = _get_arch_model() is not None
if not ARCH_AVAILABLE:
    logger.warning(
        "Biblioteca arch nao encontrada. "
        "Usando implementacao propria GARCH(1,1) / GJR-GARCH(1,1)."
    )

# Dataclass de resultado

# Resultado do ajuste GARCH para um único ativo.
@dataclass
class GARCHResult:

    ticker: str
    model_type: str
    params: Dict[str, float]
    conditional_vol: pd.Series
    std_residuals: pd.Series
    log_likelihood: float
    aic: float
    bic: float
    converged: bool
    n_obs: int
    dist: str = "normal"

    ljung_box_p: Optional[float] = None
    jarque_bera_p: Optional[float] = None

    def summary(self) -> str:
        lines = [
            f"{'='*52}",
            f"Modelo : {self.model_type}   Ticker : {self.ticker}",
            f"Dist.  : {self.dist}",
            f"{'='*52}",
            "Parâmetros:",
        ]
        for k, v in self.params.items():
            lines.append(f"  {k:14s} = {v:.8f}")
        lines += [
            f"Log-likelihood : {self.log_likelihood:.4f}",
            f"AIC            : {self.aic:.4f}",
            f"BIC            : {self.bic:.4f}",
            f"Convergiu      : {self.converged}",
            f"Observações    : {self.n_obs}",
        ]
        if self.ljung_box_p is not None:
            lines.append(f"LB(10) z²  p   : {self.ljung_box_p:.4f}")
        if self.jarque_bera_p is not None:
            lines.append(f"Jarque-Bera p  : {self.jarque_bera_p:.4f}")
        return "\n".join(lines)


# Implementações próprias (fallback sem arch)

# GARCH(1,1) via MLE com scipy (distribuição Normal).
class _GARCH11:

    def __init__(self):
        self.params_: Optional[np.ndarray] = None
        self.sigma2_: Optional[np.ndarray] = None
        self._converged: bool = False
        self._ll: float = -np.inf

    # ---- variância condicional ----

    def _compute_variance(self, params: np.ndarray, r: np.ndarray) -> np.ndarray:
        omega, alpha, beta = params
        return _garch11_var(r, omega, alpha, beta)

    # ---- função objetivo ----

    def _neg_ll(self, params: np.ndarray, r: np.ndarray) -> float:
        omega, alpha, beta = params
        if omega <= 0 or alpha < 0 or beta < 0 or alpha + beta >= 1:
            return 1e10
        sigma2 = self._compute_variance(params, r)
        return 0.5 * np.sum(np.log(2 * np.pi * sigma2) + r ** 2 / sigma2)

    # ---- ajuste ----

    def fit(self, r: np.ndarray) -> "_GARCH11":
        var = np.var(r)
        x0 = [var * 0.05, 0.08, 0.88]
        bounds = [(1e-10, None), (1e-6, 0.5), (1e-6, 0.9995)]
        res = minimize(self._neg_ll, x0, args=(r,), method="L-BFGS-B",
                       bounds=bounds, options={"maxiter": 3000, "ftol": 1e-10})
        self.params_ = res.x
        self.sigma2_ = self._compute_variance(self.params_, r)
        self._converged = res.success
        self._ll = -res.fun
        return self

    def get_result(self, ticker: str, index: pd.Index, r: np.ndarray) -> GARCHResult:
        omega, alpha, beta = self.params_
        sigma = np.sqrt(self.sigma2_)
        z = r / sigma
        T, k = len(r), 3
        return GARCHResult(
            ticker=ticker, model_type="GARCH(1,1)",
            params={"omega": omega, "alpha": alpha, "beta": beta,
                    "persistence": alpha + beta},
            conditional_vol=pd.Series(sigma, index=index, name=f"{ticker}_vol"),
            std_residuals=pd.Series(z, index=index, name=f"{ticker}_z"),
            log_likelihood=self._ll,
            aic=-2 * self._ll + 2 * k,
            bic=-2 * self._ll + k * np.log(T),
            converged=self._converged, n_obs=T,
        )


# GJR-GARCH(1,1) via MLE com scipy (distribuição Normal).
class _GJRGARCH11:

    def __init__(self):
        self.params_: Optional[np.ndarray] = None
        self.sigma2_: Optional[np.ndarray] = None
        self._converged: bool = False
        self._ll: float = -np.inf

    def _compute_variance(self, params: np.ndarray, r: np.ndarray) -> np.ndarray:
        omega, alpha, gamma, beta = params
        return _gjr11_var(r, omega, alpha, gamma, beta)

    def _neg_ll(self, params: np.ndarray, r: np.ndarray) -> float:
        omega, alpha, gamma, beta = params
        if omega <= 0 or alpha < 0 or gamma < 0 or beta < 0:
            return 1e10
        if alpha + 0.5 * gamma + beta >= 1:
            return 1e10
        sigma2 = self._compute_variance(params, r)
        return 0.5 * np.sum(np.log(2 * np.pi * sigma2) + r ** 2 / sigma2)

    def fit(self, r: np.ndarray) -> "_GJRGARCH11":
        var = np.var(r)
        x0 = [var * 0.05, 0.05, 0.08, 0.85]
        bounds = [(1e-10, None), (1e-6, 0.5), (1e-6, 0.5), (1e-6, 0.9995)]
        res = minimize(self._neg_ll, x0, args=(r,), method="L-BFGS-B",
                       bounds=bounds, options={"maxiter": 3000, "ftol": 1e-10})
        self.params_ = res.x
        self.sigma2_ = self._compute_variance(self.params_, res.x * 0 + r)
        self.sigma2_ = self._compute_variance(self.params_, r)
        self._converged = res.success
        self._ll = -res.fun
        return self

    def get_result(self, ticker: str, index: pd.Index, r: np.ndarray) -> GARCHResult:
        omega, alpha, gamma, beta = self.params_
        sigma = np.sqrt(self.sigma2_)
        z = r / sigma
        T, k = len(r), 4
        return GARCHResult(
            ticker=ticker, model_type="GJR-GARCH(1,1)",
            params={"omega": omega, "alpha": alpha, "gamma": gamma,
                    "beta": beta, "persistence": alpha + 0.5 * gamma + beta},
            conditional_vol=pd.Series(sigma, index=index, name=f"{ticker}_vol"),
            std_residuals=pd.Series(z, index=index, name=f"{ticker}_z"),
            log_likelihood=self._ll,
            aic=-2 * self._ll + 2 * k,
            bic=-2 * self._ll + k * np.log(T),
            converged=self._converged, n_obs=T,
        )


# GARCHFitter  –  interface principal

# Ajusta modelos GARCH para todos os ativos de um portfólio.
class GARCHFitter:

    def __init__(
        self,
        model_type: str = "gjr",
        dist: str = "t",
        p: int = 1,
        q: int = 1,
        run_diagnostics: bool = True,
        returns_are_decimal: bool = True,
    ):
        self.model_type = model_type.lower()
        self.dist = dist
        self.p = p
        self.q = q
        self.run_diagnostics = run_diagnostics
        self.returns_are_decimal = returns_are_decimal

        self.results_: Dict[str, GARCHResult] = {}
        self._is_fitted = False

    # API pública

    # Ajusta GARCH para uma única série de retornos.
    def fit_single(
        self,
        returns: pd.Series,
        ticker: Optional[str] = None,
    ) -> GARCHResult:
        ticker = ticker or str(returns.name or "ativo")
        series = returns.dropna()
        index = series.index

        r_pct = series.values * (100.0 if self.returns_are_decimal else 1.0)

        if ARCH_AVAILABLE:
            result = self._fit_arch(r_pct, ticker, index)
        else:
            result = self._fit_own(r_pct, ticker, index)

        if self.run_diagnostics:
            self._add_diagnostics(result)

        if (
            ARCH_AVAILABLE
            and self.run_diagnostics
            and result.ljung_box_p is not None
            and result.ljung_box_p < 0.05
            and self.q == 1
        ):
            logger.info(
                f"{ticker}: LB(10) p={result.ljung_box_p:.3f} < 0.05 — "
                f"tentando {self.model_type.upper()}({self.p},{self.q + 1})"
            )
            try:
                fitter2 = GARCHFitter(
                    model_type=self.model_type, dist=self.dist,
                    p=self.p, q=self.q + 1,
                    run_diagnostics=True,
                    returns_are_decimal=self.returns_are_decimal,
                )
                result2 = fitter2.fit_single(series, ticker)
                if result2.aic < result.aic:
                    logger.info(
                        f"{ticker}: {result2.model_type} venceu por AIC "
                        f"({result2.aic:.2f} < {result.aic:.2f})"
                    )
                    result = result2
                else:
                    logger.info(
                        f"{ticker}: {result2.model_type} não melhorou AIC "
                        f"({result2.aic:.2f} >= {result.aic:.2f}), mantendo original"
                    )
            except Exception as exc:
                logger.warning(f"{ticker}: refit com q+1 falhou ({exc})")

        _alpha = result.params.get("alpha", np.nan)
        _beta = result.params.get("beta", np.nan)
        _gamma = result.params.get("gamma", None)
        _g_str = f" γ={_gamma:.4f}" if _gamma is not None else ""
        logger.info(
            f"{ticker:16s}  {result.model_type:<18s}"
            f"α={_alpha:.4f}{_g_str} β={_beta:.4f} "
            f"persist={result.params.get('persistence', np.nan):.4f}  "
            f"AIC={result.aic:.2f}  converged={result.converged}"
        )
        return result

    # Ajusta GARCH para todos os ativos do DataFrame.
    def fit_all(
        self,
        returns_df: pd.DataFrame,
        select_best: bool = False,
    ) -> Dict[str, GARCHResult]:
        logger.info(
            f"GARCHFitter.fit_all  modelo={self.model_type}  "
            f"dist={self.dist}  ativos={list(returns_df.columns)}"
        )

        for ticker in returns_df.columns:
            series = returns_df[ticker].dropna()
            if len(series) < 50:
                logger.warning(
                    f"{ticker}: observações insuficientes ({len(series)}). Pulando."
                )
                continue

            if self.model_type == "auto" and select_best:
                result = self._auto_select(series, ticker)
            else:
                result = self.fit_single(series, ticker)

            self.results_[ticker] = result

        self._is_fitted = True
        logger.info(
            f"Ajuste concluído: {len(self.results_)}/{len(returns_df.columns)} ativos."
        )
        return self.results_

    # Outputs para integração com o pipeline

    # DataFrame (T × N) de resíduos padronizados z_t (adimensional).
    def get_std_residuals(self) -> pd.DataFrame:
        self._assert_fitted()
        return pd.DataFrame({t: r.std_residuals for t, r in self.results_.items()})

    # DataFrame (T × N) de volatilidades condicionais σ_t em decimal.
    def get_conditional_vol(self) -> pd.DataFrame:
        self._assert_fitted()
        return pd.DataFrame({t: r.conditional_vol for t, r in self.results_.items()})

    # DataFrame com parâmetros estimados e diagnósticos por ativo.
    def get_params_df(self) -> pd.DataFrame:
        self._assert_fitted()
        rows = []
        for ticker, r in self.results_.items():
            row = {
                "ticker": ticker,
                "model": r.model_type,
                "dist": r.dist,
                "aic": r.aic,
                "bic": r.bic,
                "ll": r.log_likelihood,
                "converged": r.converged,
                "n_obs": r.n_obs,
                "lb10_z2_p": r.ljung_box_p,
                "jb_p": r.jarque_bera_p,
            }
            row.update(r.params)
            rows.append(row)
        return pd.DataFrame(rows).set_index("ticker")

    # Persistência (α + β ou α + γ/2 + β) por ativo.
    def get_persistence(self) -> pd.Series:
        self._assert_fitted()
        return pd.Series(
            {t: r.params.get("persistence", np.nan) for t, r in self.results_.items()},
            name="persistence",
        )

    # Volatilidade condicional anualizada (σ_t × √trading_days).
    def get_annualized_vol(self, trading_days: int = 252) -> pd.DataFrame:
        return self.get_conditional_vol() * np.sqrt(trading_days)

    # Internos – ajuste

    # Usa a biblioteca arch para ajustar o modelo.
    def _fit_arch(
        self, r_pct: np.ndarray, ticker: str, index: pd.Index
    ) -> GARCHResult:
        arch_model = _get_arch_model()
        if arch_model is None:
            logger.warning(f"{ticker}: arch indisponivel no worker — usando fallback.")
            return self._fit_own(r_pct, ticker, index)

        mtype = self.model_type if self.model_type != "auto" else "gjr"

        vol_map     = {"garch": "GARCH", "gjr": "GARCH", "egarch": "EGARCH"}
        label_map   = {"garch": "GARCH", "gjr": "GJR-GARCH", "egarch": "EGARCH"}
        vol_model   = vol_map.get(mtype, "GARCH")
        label_model = label_map.get(mtype, "GARCH")
        o = 1 if mtype == "gjr" else 0
        logger.info(f"{ticker}: _fit_arch  mtype={mtype}  vol={vol_model}  o={o}  dist={self.dist}")

        try:
            am = arch_model(
                r_pct, vol=vol_model, p=self.p, o=o, q=self.q,
                dist=self.dist, power=2.0, rescale=False,
            )
            if mtype == "gjr" and hasattr(am.volatility, "bounds"):
                import types as _types
                _orig_bounds = am.volatility.bounds.__func__

                def _gjr_bounds(self, resids):
                    bds = list(_orig_bounds(self, resids))
                    for idx, pname in enumerate(self.parameter_names()):
                        if pname == "gamma[1]":
                            lo, hi = bds[idx]
                            bds[idx] = (max(lo, 0.0), hi)
                    return bds

                am.volatility.bounds = _types.MethodType(_gjr_bounds, am.volatility)

            fit = am.fit(disp="off", show_warning=False,
                         options={"maxiter": 500, "ftol": 1e-9})

            sigma_dec = fit.conditional_volatility / 100.0
            z = r_pct / fit.conditional_volatility

            p = dict(fit.params)
            for old, new in [("alpha[1]", "alpha"), ("beta[1]", "beta"),
                              ("gamma[1]", "gamma"), ("eta[1]", "eta")]:
                if old in p:
                    p[new] = p.pop(old)

            p["persistence"] = (
                p.get("alpha", 0.0)
                + p.get("beta", 0.0)
                + 0.5 * p.get("gamma", 0.0)
            )

            T, k = len(r_pct), len(fit.params)
            ll = fit.loglikelihood

            return GARCHResult(
                ticker=ticker,
                model_type=f"{label_model}({self.p},{self.q})",
                params=p,
                conditional_vol=pd.Series(sigma_dec, index=index,
                                          name=f"{ticker}_vol"),
                std_residuals=pd.Series(z, index=index,
                                        name=f"{ticker}_z"),
                log_likelihood=ll,
                aic=-2 * ll + 2 * k,
                bic=-2 * ll + k * np.log(T),
                converged=True,
                n_obs=T,
                dist=self.dist,
            )

        except Exception as exc:
            import traceback as _tb
            logger.error(
                f"{ticker}: arch falhou -- usando fallback proprio.\n"
                f"  Excecao: {type(exc).__name__}: {exc}\n"
                + "".join(f"    {l}" for l in _tb.format_exc().splitlines(keepends=True))
            )
            return self._fit_own(r_pct, ticker, index)

    # Implementação própria (fallback) – distribução Normal.
    def _fit_own(
        self, r_pct: np.ndarray, ticker: str, index: pd.Index
    ) -> GARCHResult:
        mtype = self.model_type
        r_dec = r_pct / 100.0

        if mtype in ("gjr", "auto", "egarch"):
            g: Union[_GJRGARCH11, _GARCH11] = _GJRGARCH11()
        else:
            g = _GARCH11()

        g.fit(r_dec)
        return g.get_result(ticker, index, r_dec)

    # Internos – seleção automática

    # Compara GARCH e GJR-GARCH pelo AIC e retorna o melhor.
    def _auto_select(self, series: pd.Series, ticker: str) -> GARCHResult:
        r_garch = GARCHFitter(
            model_type="garch", dist=self.dist,
            run_diagnostics=False,
            returns_are_decimal=self.returns_are_decimal,
        ).fit_single(series, ticker)

        r_gjr = GARCHFitter(
            model_type="gjr", dist=self.dist,
            run_diagnostics=False,
            returns_are_decimal=self.returns_are_decimal,
        ).fit_single(series, ticker)

        best = r_garch if r_garch.aic <= r_gjr.aic else r_gjr
        logger.info(
            f"{ticker}: auto-select → {best.model_type}  "
            f"(GARCH AIC={r_garch.aic:.2f}  GJR AIC={r_gjr.aic:.2f})"
        )
        return best

    # Internos – diagnósticos

    # Adiciona Ljung-Box(10) nos z²_t e Jarque-Bera nos z_t.
    def _add_diagnostics(self, result: GARCHResult) -> None:
        z = result.std_residuals.dropna().values

        try:
            from statsmodels.stats.diagnostic import acorr_ljungbox
            lb = acorr_ljungbox(z ** 2, lags=[10], return_df=True)
            result.ljung_box_p = float(lb["lb_pvalue"].iloc[0])
        except Exception:
            result.ljung_box_p = None

        try:
            _, jb_p = stats.jarque_bera(z)
            result.jarque_bera_p = float(jb_p)
        except Exception:
            result.jarque_bera_p = None

    # Internos – helpers

    def _assert_fitted(self) -> None:
        if not self._is_fitted:
            raise RuntimeError("Execute fit_all() antes de acessar os resultados.")


# Wrapper para o pipeline principal

# Wrapper de alto nível para main_pipeline.py.
def fit_garch_all(
    returns_df: pd.DataFrame,
    model_type: str = "gjr",
    dist: str = "t",
    save_path: Optional[str] = None,
    returns_are_decimal: bool = True,
    run_diagnostics: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, GARCHResult]]:
    fitter = GARCHFitter(
        model_type=model_type,
        dist=dist,
        run_diagnostics=run_diagnostics,
        returns_are_decimal=returns_are_decimal,
    )
    results = fitter.fit_all(returns_df, select_best=(model_type == "auto"))

    std_resid = fitter.get_std_residuals()
    cond_vol = fitter.get_conditional_vol()

    if save_path:
        p = Path(save_path)
        p.mkdir(parents=True, exist_ok=True)
        std_resid.to_parquet(p / "garch_std_residuals.parquet")
        cond_vol.to_parquet(p / "garch_conditional_vol.parquet")
        fitter.get_params_df().to_csv(p / "garch_params.csv")
        logger.info(f"Outputs GARCH salvos em {save_path}")

    return std_resid, cond_vol, results


# Integração direta com src/data/data_loader.DataLoader

# Carrega preços via DataLoader (src/data/data_loader) e ajusta GARCH.
def fit_from_loader(
    data_dir: Optional[str | Path] = None,
    universe_tickers: Optional[List[str]] = None,
    benchmark_tickers: Optional[List[str]] = None,
    start_date: Optional[str] = "2020-01-01",
    end_date: Optional[str] = None,
    model_type: str = "gjr",
    dist: str = "t",
    save_path: Optional[str] = None,
    return_method: str = "log",
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, GARCHResult]]:
    import sys as _sys, importlib as _il

    DataLoader = None
    for _attempt in ('src.data.data_loader', 'data.data_loader'):
        try:
            DataLoader = _il.import_module(_attempt).DataLoader
            break
        except ImportError:
            pass

    if DataLoader is None:
        _here = Path(__file__).resolve().parent
        for _p in (_here.parent, _here.parent.parent):
            if str(_p) not in _sys.path:
                _sys.path.insert(0, str(_p))
        for _attempt in ('src.data.data_loader', 'data.data_loader'):
            try:
                DataLoader = _il.import_module(_attempt).DataLoader
                break
            except ImportError:
                pass

    if DataLoader is None:
        raise ImportError(
            'Não foi possível importar DataLoader. '
            'Execute a partir da raiz do projeto ou com '
            '`python -m src.marginals.garch`.'
        )

    loader = DataLoader(data_dir=data_dir)
    loader.load_data(tickers=universe_tickers, start_date=start_date, end_date=end_date)
    loader.load_benchmarks(tickers=benchmark_tickers, start_date=start_date, end_date=end_date)

    frames = []
    if loader.prices is not None:
        frames.append(loader.prices)
    if loader.benchmarks is not None:
        frames.append(loader.benchmarks)

    if not frames:
        raise ValueError("DataLoader não carregou nenhum dado. Verifique data_dir.")

    prices = pd.concat(frames, axis=1).sort_index()

    if return_method == "log":
        returns = np.log(prices / prices.shift(1)).dropna(how="all")
    else:
        returns = prices.pct_change().dropna(how="all")

    logger.info(
        f"fit_from_loader: {returns.shape[1]} ativos  "
        f"{returns.index[0].date()} → {returns.index[-1].date()}"
    )

    return fit_garch_all(
        returns_df=returns,
        model_type=model_type,
        dist=dist,
        save_path=save_path,
        returns_are_decimal=True,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    print("=" * 65)
    print("TESTE 1: dados sintéticos (GJR-GARCH, 5 ativos, 800 obs)")
    print("=" * 65)

    np.random.seed(42)
    T, n = 800, 5
    dates = pd.date_range("2020-01-01", periods=T, freq="B")
    omega, alpha, gamma, beta = 3e-6, 0.05, 0.08, 0.88
    sim = np.zeros((T, n))
    s2 = np.full(n, omega / (1 - alpha - 0.5 * gamma - beta))
    for t in range(T):
        sig = np.sqrt(s2)
        r = sig * np.random.standard_normal(n)
        sim[t] = r
        ind = (r < 0).astype(float)
        s2 = omega + (alpha + gamma * ind) * r ** 2 + beta * s2

    returns_df = pd.DataFrame(sim, index=dates,
                               columns=[f"Ativo{i+1}" for i in range(n)])

    std_resid, cond_vol, garch_res = fit_garch_all(
        returns_df, model_type="gjr", returns_are_decimal=True
    )
    print(f"\nResíduos padronizados : {std_resid.shape}")
    print(f"Volatilidade cond.    : {cond_vol.shape}")

    fitter = GARCHFitter(model_type="gjr")
    fitter.fit_all(returns_df)
    params = fitter.get_params_df()[["model", "aic", "persistence",
                                     "converged", "lb10_z2_p"]].round(4)
    print(f"\n{params.to_string()}")

    print("\n" + "=" * 65)
    print("TESTE 2: dados reais (carregados via DataLoader)")
    print("=" * 65)

    try:
        std2, vol2, res2 = fit_from_loader(
            start_date="2020-01-01",
            model_type="gjr",
            dist="normal",
        )
        fitter2 = GARCHFitter(model_type="gjr")
        fitter2.results_ = res2
        fitter2._is_fitted = True
        print(fitter2.get_params_df()[["model", "aic", "persistence",
                                       "lb10_z2_p", "jb_p"]].round(4).to_string())
    except Exception as e:
        print(f"[AVISO] Dados reais não encontrados: {e}")
        print("  Verifique se data/raw/ contém os .xlsx do Economatica.")

    print("\nTeste concluído.")
