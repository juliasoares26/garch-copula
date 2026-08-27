
import numpy as np
import pandas as pd
import logging
import warnings
from typing import Optional, Dict, Tuple, List, Union
from scipy import stats
from scipy.optimize import minimize
from scipy.special import gammaln

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


# Utilitários numéricos

# Projeta matriz no cone PSD mais próximo (Higham 2002).
def _nearest_psd(A: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    A = (A + A.T) / 2
    eigvals, eigvecs = np.linalg.eigh(A)
    eigvals = np.maximum(eigvals, eps)
    B = eigvecs @ np.diag(eigvals) @ eigvecs.T
    d = np.maximum(np.sqrt(np.diag(B)), 1e-10)
    return B / np.outer(d, d)


# Converte matriz de quasi-correlação Q → R (correlação verdadeira).
def _corr_from_cov(Q: np.ndarray) -> np.ndarray:
    d = np.sqrt(np.maximum(np.diag(Q), 1e-10))
    R = Q / np.outer(d, d)
    np.fill_diagonal(R, 1.0)
    return R


# Log-densidade t-Student multivariada padronizada para cada linha de X.
def _t_logpdf_mv(X: np.ndarray, R: np.ndarray, nu: float) -> np.ndarray:
    T, d = X.shape
    sign, logdet = np.linalg.slogdet(R)
    if sign <= 0:
        R = _nearest_psd(R)
        sign, logdet = np.linalg.slogdet(R)

    try:
        R_inv = np.linalg.inv(R)
    except np.linalg.LinAlgError:
        R_inv = np.linalg.pinv(R)

    quad = np.einsum("ti,ij,tj->t", X, R_inv, X)

    c = (gammaln((nu + d) / 2)
         - gammaln(nu / 2)
         - 0.5 * d * np.log(nu * np.pi)
         - 0.5 * logdet)
    ll = c - 0.5 * (nu + d) * np.log(1 + quad / nu)
    return ll


# Log-densidade Normal multivariada padronizada para cada linha de X.
def _norm_logpdf_mv(X: np.ndarray, R: np.ndarray) -> np.ndarray:
    T, d = X.shape
    sign, logdet = np.linalg.slogdet(R)
    if sign <= 0:
        R = _nearest_psd(R)
        sign, logdet = np.linalg.slogdet(R)
    try:
        R_inv = np.linalg.inv(R)
    except np.linalg.LinAlgError:
        R_inv = np.linalg.pinv(R)
    quad = np.einsum("ti,ij,tj->t", X, R_inv, X)
    c = -0.5 * (d * np.log(2 * np.pi) + logdet)
    return c - 0.5 * quad


# DCCCopula

# DCC (Dynamic Conditional Correlation) de Engle (2002).
class DCCCopula:

    def __init__(
        self,
        marginal: str = "normal",
        nu: Optional[float] = None,
        a_init: float = 0.05,
        b_init: float = 0.90,
        max_iter: int = 500,
        random_state: int = 42,
    ):
        if marginal not in ("normal", "t"):
            raise ValueError("marginal deve ser 'normal' ou 't'.")
        self.marginal = marginal
        self.nu = nu
        self.a_init = a_init
        self.b_init = b_init
        self.max_iter = max_iter
        self.random_state = random_state

        self.a_: float = a_init
        self.b_: float = b_init
        self.nu_: Optional[float] = nu
        self.Q_bar_: Optional[np.ndarray] = None
        self.R_series_: Optional[np.ndarray] = None
        self.Q_series_: Optional[np.ndarray] = None
        self.z_scores_: Optional[np.ndarray] = None
        self.log_likelihood_: float = -np.inf
        self.aic_: float = np.inf
        self.bic_: float = np.inf
        self.mbicv_: float = np.inf
        self.converged_: bool = False
        self._is_fitted: bool = False
        self._T: int = 0
        self._d: int = 0

    # Fit

    # Estima os parâmetros DCC nas pseudo-observações U ∈ (0,1)^{T×d}.
    def fit(self, U: np.ndarray) -> "DCCCopula":
        np.random.seed(self.random_state)
        U = np.asarray(U, dtype=np.float64)
        U = np.clip(U, 1e-7, 1 - 1e-7)
        self._T, self._d = U.shape
        T, d = self._T, self._d

        logger.info(
            f"DCCCopula.fit()  T={T}  d={d}  "
            f"marginal={self.marginal}  ν={'auto' if self.nu is None else self.nu}"
        )

        if self.marginal == "normal":
            z = stats.norm.ppf(U)
            self.nu_ = None
        else:
            if self.nu is None:
                self.nu_ = self._estimate_nu(U)
            else:
                self.nu_ = float(self.nu)
            z = stats.t.ppf(U, df=self.nu_)

        z = (z - z.mean(axis=0)) / np.maximum(z.std(axis=0), 1e-10)
        self.z_scores_ = z

        self.Q_bar_ = np.corrcoef(z.T)
        self.Q_bar_ = _nearest_psd(self.Q_bar_)

        a_opt, b_opt, converged = self._estimate_dcc_params(z)
        self.a_ = a_opt
        self.b_ = b_opt
        self.converged_ = converged

        R_series, Q_series = self._filter_dcc(z, a_opt, b_opt)
        self.R_series_ = R_series
        self.Q_series_ = Q_series

        ll = self._log_likelihood(z, R_series)
        self.log_likelihood_ = float(ll)
        n_params = 2 + (1 if (self.marginal == "t" and self.nu is None) else 0)
        self.aic_ = -2 * ll + 2 * n_params
        self.bic_ = -2 * ll + np.log(T) * n_params
        self.mbicv_ = -2 * ll + n_params * np.log(T) * 1.0
        self._is_fitted = True

        logger.info(
            f"  a={a_opt:.4f}  b={b_opt:.4f}  "
            f"{'ν=' + str(round(self.nu_, 2)) + '  ' if self.nu_ else ''}"
            f"LL={ll:.2f}  AIC={self.aic_:.2f}  mBICV={self.mbicv_:.2f}  "
            f"converged={converged}"
        )
        return self

    # Estimação de ν

    # Estima ν por MLE nas pseudo-observações (média cross-sectional).
    def _estimate_nu(self, U: np.ndarray) -> float:
        proxy = stats.norm.ppf(U).mean(axis=1)
        proxy = (proxy - proxy.mean()) / np.maximum(proxy.std(), 1e-10)

        def neg_ll(log_nu):
            nu = np.exp(log_nu)
            if nu <= 2.1:
                return 1e10
            scale = np.sqrt(nu / (nu - 2))
            return -float(np.sum(stats.t.logpdf(proxy * scale, df=nu) + np.log(scale)))

        res = minimize(neg_ll, x0=[np.log(5.0)], method="Nelder-Mead",
                       options={"xatol": 0.01, "maxiter": 200})
        nu_est = float(np.exp(res.x[0])) if res.success else 5.0
        nu_est = float(np.clip(nu_est, 2.5, 50.0))
        logger.debug(f"  ν estimado: {nu_est:.2f}")
        return nu_est

    # Filtro DCC

    # Aplica o filtro DCC recursivo:
    def _filter_dcc(
        self, z: np.ndarray, a: float, b: float
    ) -> Tuple[np.ndarray, np.ndarray]:
        T, d = z.shape
        Q_bar = self.Q_bar_
        c = 1.0 - a - b

        R_series = np.zeros((T, d, d))
        Q_series = np.zeros((T, d, d))

        Q_t = Q_bar.copy()

        for t in range(T):
            Q_series[t] = Q_t
            R_series[t] = _corr_from_cov(Q_t)

            outer = np.outer(z[t], z[t])
            Q_next = c * Q_bar + a * outer + b * Q_t
            Q_t = _nearest_psd(Q_next)

        return R_series, Q_series

    # Log-verossimilhança DCC

    # LL = Σ_t log p(z_t | R_t)
    def _log_likelihood(
        self, z: np.ndarray, R_series: np.ndarray
    ) -> float:
        T, d = z.shape
        ll = 0.0
        for t in range(T):
            R_t = R_series[t]
            x_t = z[t : t + 1]
            if self.marginal == "normal":
                ll += float(_norm_logpdf_mv(x_t, R_t)[0])
            else:
                nu = self.nu_ if self.nu_ else 5.0
                ll += float(_t_logpdf_mv(x_t, R_t, nu)[0])
        return ll

    # Estimação dos parâmetros DCC (a, b)

    # MLE de (a, b) via L-BFGS-B com restrições a≥0, b≥0, a+b<1.
    def _estimate_dcc_params(
        self, z: np.ndarray
    ) -> Tuple[float, float, bool]:
        T, d = z.shape
        Q_bar = self.Q_bar_

        def neg_ll(params):
            a, b = params
            if a <= 0 or b <= 0 or a + b >= 0.9999:
                return 1e10
            R_s, _ = self._filter_dcc(z, a, b)
            return -self._log_likelihood(z, R_s)

        bounds = [(1e-4, 0.4), (1e-4, 0.9999)]
        constraints = [{"type": "ineq", "fun": lambda p: 0.9999 - p[0] - p[1]}]
        x0 = [self.a_init, self.b_init]

        res = minimize(
            neg_ll,
            x0=x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": self.max_iter, "ftol": 1e-8},
        )

        if res.success:
            a_opt, b_opt = float(res.x[0]), float(res.x[1])
            converged = True
        else:
            logger.warning(
                "  Otimizador DCC não convergiu; executando grid search."
            )
            a_opt, b_opt, converged = self._grid_search(z)

        a_opt = float(np.clip(a_opt, 1e-4, 0.4))
        b_opt = float(np.clip(b_opt, 1e-4, 0.9999 - a_opt))
        return a_opt, b_opt, converged

    # Grid search grosseiro sobre (a, b) como fallback.
    def _grid_search(
        self, z: np.ndarray
    ) -> Tuple[float, float, bool]:
        best_ll = -np.inf
        best_a, best_b = self.a_init, self.b_init
        for a in [0.02, 0.05, 0.10, 0.15]:
            for b in [0.70, 0.80, 0.85, 0.90, 0.92, 0.95]:
                if a + b >= 0.9999:
                    continue
                R_s, _ = self._filter_dcc(z, a, b)
                ll = self._log_likelihood(z, R_s)
                if ll > best_ll:
                    best_ll = ll
                    best_a, best_b = a, b
        logger.info(f"  Grid search: a={best_a:.3f}  b={best_b:.3f}  LL={best_ll:.2f}")
        return best_a, best_b, False

    # Correlação condicional corrente

    # Retorna R_T (última correlação estimada), shape (d, d).
    def get_current_correlation(self) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("Execute fit() primeiro.")
        return self.R_series_[-1].copy()

    # Retorna R_t para índice t, shape (d, d).
    def get_correlation_at(self, t: int) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("Execute fit() primeiro.")
        return self.R_series_[t].copy()

    # Simulação — interface compatível com CopulaEVTRisk

    # Simula pseudo-observações U ∈ (0,1)^{n_sim × d} a partir da
    def simulate(
        self,
        n_sim: int = 10_000,
        seed: Optional[int] = None,
        use_current_R: bool = True,
    ) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("Execute fit() primeiro.")
        if seed is not None:
            np.random.seed(seed)

        d = self._d

        if use_current_R:
            R = self.get_current_correlation()
        else:
            R = self.R_series_.mean(axis=0)
            R = _nearest_psd(R)

        try:
            L = np.linalg.cholesky(R)
        except np.linalg.LinAlgError:
            R = _nearest_psd(R)
            L = np.linalg.cholesky(R)

        eps = np.random.standard_normal((n_sim, d))
        Z_sim = eps @ L.T

        if self.marginal == "normal":
            U = stats.norm.cdf(Z_sim)
        else:
            nu = self.nu_ if self.nu_ else 5.0
            U = stats.t.cdf(Z_sim, df=nu)

        U = np.clip(U, 1e-7, 1 - 1e-7).astype(np.float32)
        return U

    # Extração de parâmetros para LSTM — interface compatível com
    # fit_predict_copula_params() de lstm_predictor.py

    # Retorna série temporal de parâmetros DCC como DataFrame,
    def get_params_df(
        self,
        index: Optional[pd.Index] = None,
        include_off_diagonal: bool = True,
        max_pairs: int = 50,
    ) -> pd.DataFrame:
        if not self._is_fitted:
            raise RuntimeError("Execute fit() primeiro.")

        T, d = self._T, self._d
        if index is None:
            index = pd.RangeIndex(T)
        elif len(index) != T:
            raise ValueError(f"index tem {len(index)} elementos, esperado {T}.")

        rows = []
        for t in range(T):
            R_t = self.R_series_[t]
            off_diag = R_t[np.triu_indices(d, k=1)]
            row = {
                "dcc_a": self.a_,
                "dcc_b": self.b_,
                "avg_corr": float(off_diag.mean()),
                "avg_abs_corr": float(np.abs(off_diag).mean()),
            }
            if self.nu_ is not None:
                row["dcc_nu"] = self.nu_

            if include_off_diagonal:
                pairs = list(zip(*np.triu_indices(d, k=1)))
                for idx_pair, (i, j) in enumerate(pairs):
                    if idx_pair >= max_pairs:
                        break
                    row[f"corr_{i}_{j}"] = float(R_t[i, j])

            rows.append(row)

        df = pd.DataFrame(rows, index=index)
        logger.debug(
            f"  get_params_df: shape={df.shape}  "
            f"cols={list(df.columns[:6])}..."
        )
        return df

    # Tail dependence (via correlações DCC)

    # Coeficientes de tail dependence para R_T corrente.
    def tail_dependence_current(self) -> pd.DataFrame:
        if not self._is_fitted:
            raise RuntimeError("Execute fit() primeiro.")

        d = self._d
        R = self.get_current_correlation()
        mat = np.zeros((d, d))

        if self.marginal == "t" and self.nu_ is not None:
            nu = self.nu_
            for i in range(d):
                for j in range(i + 1, d):
                    rho = float(R[i, j])
                    if abs(rho) >= 0.9999 or nu >= 50:
                        td = 0.0
                    else:
                        t_arg = -np.sqrt(
                            (nu + 1) * (1 - rho) / max(1 + rho, 1e-6)
                        )
                        td = float(np.clip(2 * stats.t.cdf(t_arg, df=nu + 1), 0, 1))
                    mat[i, j] = td
                    mat[j, i] = td

        np.fill_diagonal(mat, 1.0)
        return pd.DataFrame(mat)

    # Correlação média e métricas resumo

    # Série temporal da correlação média off-diagonal.
    def average_correlation_series(self) -> pd.Series:
        if not self._is_fitted:
            raise RuntimeError("Execute fit() primeiro.")
        T, d = self._T, self._d
        avgs = []
        for t in range(T):
            R_t = self.R_series_[t]
            off = R_t[np.triu_indices(d, k=1)]
            avgs.append(float(off.mean()))
        return pd.Series(avgs, name="avg_correlation")

    # Dicionário com métricas resumo do modelo ajustado.
    def get_summary(self) -> Dict:
        if not self._is_fitted:
            raise RuntimeError("Execute fit() primeiro.")
        R_T = self.get_current_correlation()
        off_diag = R_T[np.triu_indices(self._d, k=1)]
        avg_corr_series = self.average_correlation_series()
        return {
            "copula_type": "DCCCopula",
            "marginal": self.marginal,
            "dcc_a": round(self.a_, 5),
            "dcc_b": round(self.b_, 5),
            "dcc_persistence": round(self.a_ + self.b_, 5),
            "nu": round(self.nu_, 3) if self.nu_ else None,
            "avg_corr_current": round(float(off_diag.mean()), 4),
            "avg_corr_historical_mean": round(float(avg_corr_series.mean()), 4),
            "avg_corr_historical_std": round(float(avg_corr_series.std()), 4),
            "log_likelihood": round(self.log_likelihood_, 3),
            "aic": round(self.aic_, 3),
            "bic": round(self.bic_, 3),
            "mbicv": round(self.mbicv_, 3),
            "converged": self.converged_,
            "T": self._T,
            "d": self._d,
        }

    # Save / Load

    def save(self, path) -> None:
        import pickle
        from pathlib import Path
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info(f"DCCCopula salvo em {path}")

    @classmethod
    def load(cls, path) -> "DCCCopula":
        import pickle
        with open(path, "rb") as f:
            obj = pickle.load(f)
        logger.info(f"DCCCopula carregado de {path}")
        return obj


# Wrapper de alto nível

# Wrapper de alto nível compatível com o padrão do pipeline.
def fit_dcc_copula(
    U: np.ndarray,
    marginal: str = "normal",
    nu: Optional[float] = None,
    a_init: float = 0.05,
    b_init: float = 0.90,
    max_iter: int = 500,
    random_state: int = 42,
    index: Optional[pd.Index] = None,
) -> Tuple["DCCCopula", pd.DataFrame]:
    dcc = DCCCopula(
        marginal=marginal,
        nu=nu,
        a_init=a_init,
        b_init=b_init,
        max_iter=max_iter,
        random_state=random_state,
    )
    dcc.fit(U)
    params_df = dcc.get_params_df(index=index)
    summary = dcc.get_summary()
    logger.info(
        f"DCC ajustado: a={summary['dcc_a']}  b={summary['dcc_b']}  "
        f"persistência={summary['dcc_persistence']}  "
        f"ρ̄_atual={summary['avg_corr_current']}  "
        f"AIC={summary['aic']}  mBICV={summary['mbicv']}"
    )
    return dcc, params_df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    np.random.seed(0)

    T, d = 500, 6
    print(f"\n{'='*55}")
    print(f"  Teste DCCCopula  T={T}  d={d}")
    print(f"{'='*55}")

    rho_base = 0.4
    R_true = np.full((d, d), rho_base)
    np.fill_diagonal(R_true, 1.0)
    L = np.linalg.cholesky(R_true)

    U_list = []
    for t in range(T):
        rho_t = rho_base + 0.25 * np.sin(2 * np.pi * t / T)
        R_t = np.full((d, d), rho_t)
        np.fill_diagonal(R_t, 1.0)
        R_t = _nearest_psd(R_t)
        L_t = np.linalg.cholesky(R_t)
        z = L_t @ np.random.standard_normal(d)
        U_list.append(stats.norm.cdf(z))
    U = np.array(U_list, dtype=np.float32)
    print(f"U: {U.shape}  range=[{U.min():.3f}, {U.max():.3f}]")

    print("\n[marginal=normal]")
    dcc_n = DCCCopula(marginal="normal", a_init=0.05, b_init=0.90)
    dcc_n.fit(U)
    s = dcc_n.get_summary()
    print(f"  a={s['dcc_a']}  b={s['dcc_b']}  "
          f"persistência={s['dcc_persistence']}  "
          f"ρ̄={s['avg_corr_current']}  LL={s['log_likelihood']}  "
          f"AIC={s['aic']}  mBICV={s['mbicv']}")

    U_sim = dcc_n.simulate(5000, seed=1)
    print(f"  Simulação: {U_sim.shape}  range=[{U_sim.min():.3f}, {U_sim.max():.3f}]")

    print("\n[marginal=t]")
    dcc_t = DCCCopula(marginal="t", nu=None, a_init=0.05, b_init=0.90)
    dcc_t.fit(U)
    s = dcc_t.get_summary()
    print(f"  a={s['dcc_a']}  b={s['dcc_b']}  ν={s['nu']}  "
          f"persistência={s['dcc_persistence']}  "
          f"ρ̄={s['avg_corr_current']}  LL={s['log_likelihood']}  "
          f"AIC={s['aic']}  mBICV={s['mbicv']}")

    print("\n[fit_dcc_copula wrapper]")
    idx = pd.date_range("2020-01-01", periods=T, freq="B")
    dcc_w, params_df = fit_dcc_copula(U, marginal="normal", index=idx)
    print(f"  params_df: {params_df.shape}  colunas={list(params_df.columns[:5])}...")
    print(params_df.head(3).round(4).to_string())

    print("\n[tail dependence — marginal t]")
    td_df = dcc_t.tail_dependence_current()
    print(td_df.round(4).to_string())

    avg_s = dcc_n.average_correlation_series()
    print(f"\n[avg_corr_series] min={avg_s.min():.4f}  "
          f"max={avg_s.max():.4f}  mean={avg_s.mean():.4f}")

    print("\nTodos os testes concluídos.")
