
import numpy as np
import pandas as pd
import logging
import warnings
from typing import Optional, Dict, Tuple, List
from scipy import stats
from scipy.optimize import minimize, minimize_scalar
from scipy.special import gammaln, betaln

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


# Funções auxiliares numéricas

# Projeta matriz para o cone PSD mais próximo.
def _nearest_psd(A: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    A = (A + A.T) / 2
    eigvals, eigvecs = np.linalg.eigh(A)
    eigvals = np.maximum(eigvals, eps)
    A2 = eigvecs @ np.diag(eigvals) @ eigvecs.T
    d = np.maximum(np.sqrt(np.diag(A2)), 1e-10)
    return A2 / np.outer(d, d)


# Log-densidade t-Student padronizada (média 0, variância 1) para array x.
def _t_logpdf(x: np.ndarray, nu: float) -> np.ndarray:
    c = gammaln((nu + 1) / 2) - gammaln(nu / 2) - 0.5 * np.log(nu * np.pi)
    scale = np.sqrt(nu / (nu - 2)) if nu > 2 else 1.0
    xs = x * scale
    return c + np.log(scale) - 0.5 * (nu + 1) * np.log(1 + xs ** 2 / nu)


# CDF t-Student padronizada (variância unitária).
def _t_cdf(x: np.ndarray, nu: float) -> np.ndarray:
    scale = np.sqrt(nu / (nu - 2)) if nu > 2 else 1.0
    return stats.t.cdf(x * scale, df=nu)


# Inversa da CDF t-Student padronizada.
def _t_ppf(u: np.ndarray, nu: float) -> np.ndarray:
    scale = np.sqrt(nu / (nu - 2)) if nu > 2 else 1.0
    return stats.t.ppf(u, df=nu) / scale


def _normal_logpdf(x: np.ndarray) -> np.ndarray:
    return -0.5 * np.log(2 * np.pi) - 0.5 * x ** 2


# Gauss-Hermite quadratura para integração sobre o fator latente

# Nós e pesos de Gauss-Hermite para integração ∫ f(x) exp(-x²) dx.
def _gauss_hermite_nodes(n: int = 32) -> Tuple[np.ndarray, np.ndarray]:
    nodes, weights = np.polynomial.hermite.hermgauss(n)
    nodes_std = nodes * np.sqrt(2)
    weights_std = weights / np.sqrt(np.pi)
    return nodes_std, weights_std


# FactorCopulaEstimator

# Estima os parâmetros da factor copula de Oh e Patton (2018):
class FactorCopulaEstimator:

    def __init__(
        self,
        n_factors: int = 1,
        nu_factor: Optional[float] = None,
        nu_idio: Optional[float] = None,
        max_iter: int = 200,
        tol: float = 1e-5,
        n_quad: int = 32,
        random_state: int = 42,
    ):
        self.n_factors = n_factors
        self.nu_factor = nu_factor
        self.nu_idio = nu_idio
        self.max_iter = max_iter
        self.tol = tol
        self.n_quad = n_quad
        self.random_state = random_state

        self.lambdas_: Optional[np.ndarray] = None
        self.nu_factor_: Optional[float] = None
        self.nu_idio_: Optional[float] = None
        self.log_likelihood_: float = -np.inf
        self.aic_: float = np.inf
        self.bic_: float = np.inf
        self.mbicv_: float = np.inf
        self.n_iter_: int = 0
        self.converged_: bool = False
        self.ll_history_: List[float] = []
        self._is_fitted: bool = False
        self._d: int = 0
        self._T: int = 0

    # Fit

    # Estima parâmetros da factor copula nas pseudo-observações U ∈ (0,1)^{T×d}.
    def fit(self, U: np.ndarray) -> "FactorCopulaEstimator":
        np.random.seed(self.random_state)
        U = np.asarray(U, dtype=np.float64)
        U = np.clip(U, 1e-7, 1 - 1e-7)
        self._T, self._d = U.shape
        d, T = self._d, self._T

        logger.info(
            f"FactorCopulaEstimator.fit()  T={T}  d={d}  "
            f"n_factors={self.n_factors}  n_quad={self.n_quad}"
        )

        Z_scores = stats.norm.ppf(U)

        lambdas = self._init_loadings_pca(Z_scores)

        if self.nu_factor is not None:
            nu_z = float(self.nu_factor)
        else:
            nu_z = self._estimate_nu_factor(Z_scores)
        self.nu_factor_ = nu_z

        if self.nu_idio is not None:
            nu_e = float(self.nu_idio)
        else:
            nu_e = np.inf
        self.nu_idio_ = nu_e

        lambdas = self._run_em(Z_scores, lambdas, nu_z, nu_e)
        self.lambdas_ = lambdas

        ll = self._compute_log_likelihood(Z_scores, lambdas, nu_z, nu_e)
        self.log_likelihood_ = float(ll)
        n_params = d * self.n_factors + (1 if self.nu_factor is None else 0)
        self.aic_   = -2 * ll + 2 * n_params
        self.bic_   = -2 * ll + np.log(T) * n_params
        self.mbicv_ = -2 * ll + n_params * np.log(T) * 1.0
        self._is_fitted = True

        logger.info(
            f"  ν_Z={nu_z:.2f}  LL={ll:.2f}  AIC={self.aic_:.2f}  "
            f"BIC={self.bic_:.2f}  mBICV={self.mbicv_:.2f}  iter={self.n_iter_}  "
            f"converged={self.converged_}"
        )
        logger.info(
            f"  loadings mean={np.abs(lambdas).mean():.4f}  "
            f"max={np.abs(lambdas).max():.4f}"
        )
        return self

    # Inicialização PCA

    # Inicializa λ via PCA de Z (scores normais).
    def _init_loadings_pca(self, Z: np.ndarray) -> np.ndarray:
        d = self._d
        k = self.n_factors
        cov = np.cov(Z.T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        idx = np.argsort(eigvals)[::-1]
        eigvecs = eigvecs[:, idx]
        eigvals = eigvals[idx]

        lambdas = np.zeros((d, k))
        for j in range(min(k, d)):
            lam = eigvecs[:, j] * np.sqrt(max(eigvals[j] - 1, 0.01))
            lambdas[:, j] = np.clip(lam, -0.98, 0.98)

        logger.debug(f"  PCA init: lambdas shape={lambdas.shape}")
        return lambdas

    # Estimação de ν_Z

    # Estima ν_Z por máxima verossimilhança marginal sobre a média
    def _estimate_nu_factor(
        self, Z: np.ndarray, nu_grid: np.ndarray = None
    ) -> float:
        factor_proxy = Z.mean(axis=1)
        factor_proxy = (factor_proxy - factor_proxy.mean()) / factor_proxy.std()

        if nu_grid is None:
            nu_grid = np.array([3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 15.0, 20.0, 30.0])

        def neg_ll(nu):
            if nu <= 2.1:
                return 1e10
            return -float(np.sum(_t_logpdf(factor_proxy, nu)))

        ll_grid = np.array([neg_ll(nu) for nu in nu_grid])
        nu_init = nu_grid[np.argmin(ll_grid)]

        res = minimize_scalar(
            neg_ll, bounds=(2.5, 50.0), method="bounded",
            options={"xatol": 0.1}
        )
        nu_est = float(res.x) if res.success else float(nu_init)
        nu_est = np.clip(nu_est, 2.5, 50.0)
        logger.debug(f"  ν_Z estimado: {nu_est:.2f}")
        return nu_est

    # EM principal

    # Algoritmo EM com quadratura de Gauss-Hermite para integrar
    def _run_em(
        self,
        Z: np.ndarray,
        lambdas: np.ndarray,
        nu_z: float,
        nu_e: float,
    ) -> np.ndarray:
        T, d = self._T, self._d
        k = self.n_factors
        nodes, weights = _gauss_hermite_nodes(self.n_quad)

        ll_prev = -np.inf
        self.ll_history_ = []

        for iteration in range(self.max_iter):

            Ez, Ez2 = self._e_step(Z, lambdas, nu_z, nu_e, nodes, weights)

            lambdas_new = self._m_step(Z, Ez, Ez2, nu_e)

            ll = self._compute_log_likelihood(Z, lambdas_new, nu_z, nu_e)
            self.ll_history_.append(float(ll))

            delta = abs(ll - ll_prev) / (abs(ll_prev) + 1e-10)
            if iteration % 20 == 0 or iteration < 5:
                logger.debug(
                    f"  EM iter {iteration:4d}  LL={ll:.4f}  Δ={delta:.2e}"
                )

            lambdas = lambdas_new
            self.n_iter_ = iteration + 1

            if delta < self.tol and iteration > 5:
                self.converged_ = True
                logger.debug(f"  EM convergiu em {iteration+1} iterações.")
                break

            ll_prev = ll

        if np.median(lambdas[:, 0]) < 0:
          lambdas[:, 0] = -lambdas[:, 0]
        return lambdas

    # E-step via quadratura

    # Calcula E[Z_t | X_t] e E[Z_t² | X_t] via quadratura
    def _e_step(
        self,
        Z: np.ndarray,
        lambdas: np.ndarray,
        nu_z: float,
        nu_e: float,
        nodes: np.ndarray,
        weights: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        T, d = self._T, self._d
        k = self.n_factors
        n_q = len(nodes)

        Ez = np.zeros((T, k))
        Ez2 = np.zeros((T, k))

        lam = lambdas[:, 0]
        sig_e = np.sqrt(np.maximum(1 - lam ** 2, 1e-6))

        for t in range(T):
            x_t = Z[t]
            log_w_q = np.zeros(n_q)

            for q, z_q in enumerate(nodes):
                resid = (x_t - lam * z_q) / sig_e
                if np.isinf(nu_e):
                    log_pq = np.sum(_normal_logpdf(resid) - np.log(sig_e))
                else:
                    log_pq = np.sum(
                        _t_logpdf(resid, nu_e) - np.log(sig_e)
                    )
                log_w_q[q] = np.log(max(weights[q], 1e-300)) + log_pq

            log_w_q -= log_w_q.max()
            w_q = np.exp(log_w_q)
            w_q /= w_q.sum() + 1e-300

            Ez[t, 0] = float(np.dot(w_q, nodes))
            Ez2[t, 0] = float(np.dot(w_q, nodes ** 2))

        return Ez, Ez2

    # M-step: WLS por ativo

    # Atualiza λ_i por regressão de X_it sobre E[Z_t | X_t]:
    def _m_step(
        self,
        Z: np.ndarray,
        Ez: np.ndarray,
        Ez2: np.ndarray,
        nu_e: float,
    ) -> np.ndarray:
        T, d = self._T, self._d
        k = self.n_factors
        lambdas_new = np.zeros((d, k))

        for j in range(k):
            ez = Ez[:, j]
            ez2 = Ez2[:, j]
            denom = np.sum(ez2) + 1e-10
            for i in range(d):
                num = np.dot(ez, Z[:, i])
                lambdas_new[i, j] = np.clip(num / denom, -0.99, 0.99)

        return lambdas_new

    # Log-verossimilhança marginal via quadratura

    # LL = Σ_t log ∫ p(X_t | z) p_Z(z) dz
    def _compute_log_likelihood(
        self,
        Z: np.ndarray,
        lambdas: np.ndarray,
        nu_z: float,
        nu_e: float,
    ) -> float:
        T, d = self._T, self._d
        nodes, weights = _gauss_hermite_nodes(self.n_quad)
        lam = lambdas[:, 0]
        sig_e = np.sqrt(np.maximum(1 - lam ** 2, 1e-6))
        ll = 0.0

        for t in range(T):
            x_t = Z[t]
            integrand = np.zeros(len(nodes))
            for q, z_q in enumerate(nodes):
                resid = (x_t - lam * z_q) / sig_e
                if np.isinf(nu_e):
                    log_pq = np.sum(_normal_logpdf(resid) - np.log(sig_e))
                else:
                    log_pq = np.sum(_t_logpdf(resid, nu_e) - np.log(sig_e))
                integrand[q] = weights[q] * np.exp(
                    np.clip(log_pq, -500, 500)
                )
            val = np.sum(integrand)
            ll += np.log(max(val, 1e-300))

        return ll

    # Simulação

    # Simula pseudo-observações U ∈ (0,1)^{n_sim × d} da factor copula.
    def simulate(
        self,
        n_sim: int = 10_000,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("Execute fit() primeiro.")
        if seed is not None:
            np.random.seed(seed)

        d = self._d
        lam = self.lambdas_[:, 0]
        nu_z = self.nu_factor_
        sig_e = np.sqrt(np.maximum(1 - lam ** 2, 1e-6))

        if nu_z >= 50:
            Z_sim = np.random.standard_normal(n_sim)
        else:
            Z_sim = _t_ppf(np.random.uniform(size=n_sim), nu_z)

        eps = np.random.standard_normal((n_sim, d))

        X = lam[np.newaxis, :] * Z_sim[:, np.newaxis] + sig_e[np.newaxis, :] * eps

        U = stats.norm.cdf(X).astype(np.float32)
        U = np.clip(U, 1e-7, 1 - 1e-7)
        return U

    # Tail dependence analítica

    # Coeficiente de tail dependence entre ativos i e j.
    def tail_dependence(self, i: int = 0, j: int = 1) -> Dict[str, float]:
        if not self._is_fitted:
            raise RuntimeError("Execute fit() primeiro.")

        lam_i = float(self.lambdas_[i, 0])
        lam_j = float(self.lambdas_[j, 0])
        rho_ij = lam_i * lam_j
        nu = self.nu_factor_

        if nu >= 50 or abs(rho_ij) >= 0.9999:
            return {"lower_tail": 0.0, "upper_tail": 0.0, "rho_ij": rho_ij}

        t_arg = -np.sqrt((nu + 1) * (1 - rho_ij) / max(1 + rho_ij, 1e-6))
        td = 2 * stats.t.cdf(t_arg, df=nu + 1)
        td = float(np.clip(td, 0.0, 1.0))
        return {"lower_tail": td, "upper_tail": td, "rho_ij": rho_ij}

    # Retorna matriz de tail dependence lower para todos os pares.
    def tail_dependence_matrix(self) -> pd.DataFrame:
        if not self._is_fitted:
            raise RuntimeError("Execute fit() primeiro.")
        d = self._d
        lam = self.lambdas_[:, 0]
        nu = self.nu_factor_
        mat = np.zeros((d, d))
        for i in range(d):
            for j in range(i + 1, d):
                td = self.tail_dependence(i, j)["lower_tail"]
                mat[i, j] = td
                mat[j, i] = td
        np.fill_diagonal(mat, 1.0)
        return pd.DataFrame(mat)

    # Correlação implícita

    # Matriz de correlação implícita: R_ij = λ_i * λ_j para i≠j, 1 para i=j.
    def implied_correlation(self) -> np.ndarray:
        if not self._is_fitted:
            raise RuntimeError("Execute fit() primeiro.")
        lam = self.lambdas_[:, 0]
        R = np.outer(lam, lam)
        np.fill_diagonal(R, 1.0)
        return _nearest_psd(R)

    # Resumo

    # DataFrame com loadings, variância explicada e tail dependence média.
    def get_summary(self) -> pd.DataFrame:
        if not self._is_fitted:
            raise RuntimeError("Execute fit() primeiro.")
        lam = self.lambdas_[:, 0]
        var_explained = lam ** 2
        td_mean = np.mean([
            self.tail_dependence(0, j)["lower_tail"]
            for j in range(1, self._d)
        ])
        rows = []
        for i, (l, v) in enumerate(zip(lam, var_explained)):
            rows.append({
                "asset": i,
                "lambda": round(float(l), 4),
                "var_explained": round(float(v), 4),
                "nu_factor": round(self.nu_factor_, 2),
            })
        df = pd.DataFrame(rows)
        logger.info(
            f"  Factor copula: ν={self.nu_factor_:.2f}  "
            f"λ_mean={np.abs(lam).mean():.4f}  "
            f"tail_dep_mean={td_mean:.4f}  "
            f"LL={self.log_likelihood_:.2f}  AIC={self.aic_:.2f}  "
            f"mBICV={self.mbicv_:.2f}"
        )
        return df

    # Save / Load

    def save(self, path) -> None:
        import pickle
        from pathlib import Path
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info(f"FactorCopulaEstimator salvo em {path}")

    @classmethod
    def load(cls, path) -> "FactorCopulaEstimator":
        import pickle
        with open(path, "rb") as f:
            obj = pickle.load(f)
        logger.info(f"FactorCopulaEstimator carregado de {path}")
        return obj


# Wrapper de alto nível — interface idêntica a estimate_copula() de estimation.py

# Wrapper de alto nível compatível com o padrão de estimation.py.
def estimate_factor_copula(
    U: np.ndarray,
    n_factors: int = 1,
    nu_factor: Optional[float] = None,
    max_iter: int = 200,
    tol: float = 1e-5,
    n_quad: int = 32,
    random_state: int = 42,
) -> Tuple["FactorCopulaEstimator", np.ndarray]:
    estimator = FactorCopulaEstimator(
        n_factors=n_factors,
        nu_factor=nu_factor,
        max_iter=max_iter,
        tol=tol,
        n_quad=n_quad,
        random_state=random_state,
    )
    estimator.fit(U)
    return estimator, U


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
    np.random.seed(42)

    T, d = 600, 8
    lam_true = np.array([0.7, 0.65, 0.6, 0.55, 0.4, 0.35, 0.3, 0.25])
    nu_true = 5.0
    Z_factor = stats.t.rvs(df=nu_true, size=T) / np.sqrt(nu_true / (nu_true - 2))
    eps = np.random.standard_normal((T, d))
    sig_e = np.sqrt(1 - lam_true ** 2)
    X = lam_true[np.newaxis, :] * Z_factor[:, np.newaxis] + sig_e[np.newaxis, :] * eps
    U = stats.norm.cdf(X).astype(np.float32)
    U = np.clip(U, 1e-6, 1 - 1e-6)

    print(f"U: {U.shape}  range=[{U.min():.3f}, {U.max():.3f}]")
    print(f"λ verdadeiros: {lam_true}")
    print(f"ν verdadeiro: {nu_true}")

    est = FactorCopulaEstimator(n_factors=1, nu_factor=None, max_iter=100, n_quad=24)
    est.fit(U)

    print(f"\nλ estimados:  {est.lambdas_[:, 0].round(3)}")
    print(f"ν estimado:   {est.nu_factor_:.2f}  (verdadeiro={nu_true})")
    print(f"LL={est.log_likelihood_:.2f}  AIC={est.aic_:.2f}  mBICV={est.mbicv_:.2f}  iter={est.n_iter_}")

    print("\nSummary:")
    print(est.get_summary().to_string(index=False))

    print("\nTail dependence (pares 0-1, 0-4, 0-7):")
    for j in [1, 4, 7]:
        td = est.tail_dependence(0, j)
        print(f"  (0,{j}): λ_L={td['lower_tail']:.4f}  ρ_ij={td['rho_ij']:.4f}")

    U_sim = est.simulate(5000, seed=0)
    print(f"\nSimulação: {U_sim.shape}  range=[{U_sim.min():.3f}, {U_sim.max():.3f}]")
    print("Todos os testes concluídos.")
