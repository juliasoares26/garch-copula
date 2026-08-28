import numpy as np
import pandas as pd
import logging
import warnings
from typing import Optional, Dict, Tuple, List, Union
from dataclasses import dataclass, field
from scipy import stats
from concurrent.futures import ThreadPoolExecutor, as_completed

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


# Resultado da seleção

# Resultado da seleção de cópula para um par de ativos.
@dataclass
class CopulaSelectionResult:
    pair: Tuple[str, str]
    best_family: str
    best_copula: object
    theta: float
    tau_kendall: float
    log_likelihood: float
    aic: float
    bic: float
    mbicv: float
    tail_lower: float
    tail_upper: float
    selection_criterion: str
    all_results: pd.DataFrame

    def summary(self) -> str:
        lines = [
            f"Par ({self.pair[0]}, {self.pair[1]})",
            f"  Melhor família : {self.best_family}",
            f"  θ              : {self.theta:.4f}",
            f"  τ Kendall      : {self.tau_kendall:.4f}",
            f"  AIC            : {self.aic:.2f}",
            f"  BIC            : {self.bic:.2f}",
            f"  mBICV          : {self.mbicv:.2f}",
            f"  λ_L / λ_U      : {self.tail_lower:.4f} / {self.tail_upper:.4f}",
            f"  Critério usado : {self.selection_criterion.upper()}",
        ]
        return "\n".join(lines)


# Seletor de cópula

# Seleciona a família de cópula ótima para um par de ativos.
class CopulaSelector:

    ALL_FAMILIES = [
        "gaussian",
        "t",
        "clayton",
        "gumbel",
        "frank",
        "joe",
        "clayton180",
    ]

    _N_PARAMS: Dict[str, int] = {
        "gaussian":   1,
        "t":          2,
        "clayton":    1,
        "gumbel":     1,
        "frank":      1,
        "joe":        1,
        "clayton180": 1,
    }

    def __init__(
        self,
        families: Optional[List[str]] = None,
        criterion: str = "aic",
        parallel: bool = True,
        tau_threshold: float = 0.05,
        n_workers: int = 4,
    ):
        self.families = families or self.ALL_FAMILIES
        self.criterion = criterion.lower()
        if self.criterion not in ("aic", "bic", "mbicv"):
            raise ValueError(
                f"criterion deve ser 'aic', 'bic' ou 'mbicv', recebido '{criterion}'"
            )
        self.parallel = parallel
        self.tau_threshold = tau_threshold
        self.n_workers = n_workers

    # Critérios de informação

    # AIC = −2ℓ + 2k
    @staticmethod
    def compute_aic(log_likelihood: float, n_params: int) -> float:
        if not np.isfinite(log_likelihood):
            return np.inf
        return -2.0 * log_likelihood + 2.0 * n_params

    # BIC = −2ℓ + k·log(n)
    @staticmethod
    def compute_bic(log_likelihood: float, n_obs: int, n_params: int) -> float:
        if not np.isfinite(log_likelihood) or n_obs <= 0:
            return np.inf
        return -2.0 * log_likelihood + n_params * np.log(n_obs)

    # mBICV — Nagler et al. (2019).
    @staticmethod
    def compute_mbicv(
        log_likelihood: float,
        n_obs: int,
        n_params: int,
        vine_tree: int = 1,
    ) -> float:
        if not np.isfinite(log_likelihood) or n_obs <= 0:
            return np.inf
        nu_tree = 1.0 / max(vine_tree, 1)
        return -2.0 * log_likelihood + n_params * np.log(n_obs) * nu_tree

    # Seleção para um par

    # Seleciona a melhor cópula para o par (u, v).
    def select(
        self,
        u: np.ndarray,
        v: np.ndarray,
        pair_name: Tuple[str, str] = ("u", "v"),
        vine_tree: int = 1,
    ) -> CopulaSelectionResult:
        u = np.clip(u.ravel(), 1e-9, 1 - 1e-9).astype(np.float64)
        v = np.clip(v.ravel(), 1e-9, 1 - 1e-9).astype(np.float64)
        n_obs = len(u)

        tau, _ = stats.kendalltau(u, v)

        if abs(tau) < self.tau_threshold:
            logger.debug(
                f"Par {pair_name}: |τ|={abs(tau):.3f} < {self.tau_threshold}. Independência."
            )
            return self._independence_result(pair_name, tau)

        results_raw = (
            self._fit_parallel(u, v) if self.parallel else self._fit_sequential(u, v)
        )

        _lam_L_emp, _lam_U_emp = self._empirical_tail_dependence(u, v)

        rows = []
        for fam, copula, error in results_raw:
            if copula is not None and error is None:
                ll  = getattr(copula, "log_likelihood_", np.nan)
                k   = self._N_PARAMS.get(fam, 1)

                aic_val   = self.compute_aic(ll, k)
                bic_val   = self.compute_bic(ll, n_obs, k)
                mbicv_val = self.compute_mbicv(ll, n_obs, k, vine_tree)

                if fam == "t":
                    nu_val = getattr(copula, "nu_", 30.0)
                    if nu_val >= 25.0:
                        penalty = 5.0 * (nu_val / 25.0)
                        aic_val   += penalty
                        bic_val   += penalty
                        mbicv_val += penalty

                row = {
                    "family":         fam,
                    "log_likelihood": ll,
                    "aic":            aic_val,
                    "bic":            bic_val,
                    "mbicv":          mbicv_val,
                    "n_params":       k,
                    "theta":          getattr(copula, "theta_",
                                              getattr(copula, "rho_", np.nan)),
                    "nu":             getattr(copula, "nu_", np.nan),
                    "lower_tail":     _lam_L_emp,
                    "upper_tail":     _lam_U_emp,
                    "converged":      True,
                    "_copula":        copula,
                }
            else:
                row = {
                    "family":         fam,
                    "log_likelihood": -np.inf,
                    "aic":            np.inf,
                    "bic":            np.inf,
                    "mbicv":          np.inf,
                    "n_params":       self._N_PARAMS.get(fam, 1),
                    "theta":          np.nan,
                    "nu":             np.nan,
                    "lower_tail":     np.nan,
                    "upper_tail":     np.nan,
                    "converged":      False,
                    "_copula":        None,
                }
            rows.append(row)

        sel_df = pd.DataFrame(rows)
        sort_col = self.criterion if self.criterion in sel_df.columns else "aic"
        sel_df = sel_df.sort_values(sort_col).reset_index(drop=True)

        _td_low  = float(_lam_L_emp) if np.isfinite(_lam_L_emp) else 1.0
        _td_high = float(_lam_U_emp) if np.isfinite(_lam_U_emp) else 1.0
        _td_threshold = 0.15 + 0.05 * (1 - min(n_obs, 1000) / 1000)

        best_row    = sel_df.iloc[0]
        best_copula = best_row["_copula"]
        best_family = best_row["family"]

        if best_family == "t" and _td_low < _td_threshold and _td_high < _td_threshold:
            gauss_rows = sel_df[sel_df["family"] == "gaussian"]
            if not gauss_rows.empty and gauss_rows.iloc[0]["_copula"] is not None:
                best_row    = gauss_rows.iloc[0]
                best_copula = best_row["_copula"]
                best_family = "gaussian"
                logger.debug(
                    f"Par {pair_name}: t-Student → Gaussiana "
                    f"(λ_L={_td_low:.3f}, λ_U={_td_high:.3f} < {_td_threshold:.2f})"
                )

        display_df = sel_df.drop(columns=["_copula"]).round(4)

        logger.info(
            f"Par {pair_name} | τ={tau:.3f} | n={n_obs} | "
            f"Melhor: {best_family} ({sort_col.upper()}={best_row[sort_col]:.2f})"
        )

        return CopulaSelectionResult(
            pair=pair_name,
            best_family=best_family,
            best_copula=best_copula,
            theta=float(best_row["theta"]),
            tau_kendall=float(tau),
            log_likelihood=float(best_row["log_likelihood"]),
            aic=float(best_row["aic"]),
            bic=float(best_row["bic"]),
            mbicv=float(best_row["mbicv"]),
            tail_lower=float(best_row["lower_tail"]),
            tail_upper=float(best_row["upper_tail"]),
            selection_criterion=self.criterion,
            all_results=display_df,
        )

    # Tail dependence empírica (CFG)

    # Estimador CFG não-paramétrico de dependência de cauda.
    @staticmethod
    def _empirical_tail_dependence(
        u: np.ndarray,
        v: np.ndarray,
        alpha: float = 0.10,
    ) -> Tuple[float, float]:
        try:
            from copulas.elliptical import empirical_tail_dependence
            UV = np.column_stack([u, v])
            td = empirical_tail_dependence(UV, alpha=alpha)
            return float(td["lower_tail"]), float(td["upper_tail"])
        except ImportError:
            pass

        T = len(u)
        k = max(1, int(T * alpha))
        u_lo, v_lo = np.sort(u)[k], np.sort(v)[k]
        lam_L = np.mean((u <= u_lo) & (v <= v_lo)) / (k / T)
        u_hi, v_hi = np.sort(u)[T - k - 1], np.sort(v)[T - k - 1]
        lam_U = np.mean((u >= u_hi) & (v >= v_hi)) / (k / T)
        return float(lam_L), float(lam_U)

    # Ajuste de famílias

    def _fit_one(self, family: str, u: np.ndarray, v: np.ndarray):
        try:
            copula = self._get_copula_instance(family)
            copula.fit(u, v)
            return family, copula, None
        except Exception as e:
            logger.debug(f"  {family}: falhou — {e}")
            return family, None, str(e)

    def _fit_sequential(self, u, v):
        return [self._fit_one(fam, u, v) for fam in self.families]

    def _fit_parallel(self, u, v):
        results = []
        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            futures = {
                executor.submit(self._fit_one, fam, u, v): fam
                for fam in self.families
            }
            for fut in as_completed(futures):
                results.append(fut.result())
        return results

    def _get_copula_instance(self, family: str):
        try:
            from copulas.archimedean import (
                ClaytonCopula, GumbelCopula, FrankCopula, JoeCopula, ClaytonCopula180,
            )
            from copulas.elliptical import BivariateGaussianCopula, BivariateStudentTCopula
        except ImportError:
            from archimedean import (
                ClaytonCopula, GumbelCopula, FrankCopula, JoeCopula, ClaytonCopula180,
            )
            from elliptical import BivariateGaussianCopula, BivariateStudentTCopula

        registry = {
            "gaussian":   BivariateGaussianCopula,
            "t":          BivariateStudentTCopula,
            "clayton":    ClaytonCopula,
            "gumbel":     GumbelCopula,
            "frank":      FrankCopula,
            "joe":        JoeCopula,
            "clayton180": ClaytonCopula180,
        }
        cls = registry.get(family.lower())
        if cls is None:
            raise ValueError(f"Família desconhecida: {family}")
        return cls()

    # Resultado de independência

    def _independence_result(
        self, pair_name: Tuple[str, str], tau: float
    ) -> CopulaSelectionResult:
        try:
            from copulas.archimedean import FrankCopula
        except ImportError:
            from archimedean import FrankCopula

        dummy = FrankCopula()
        dummy.theta_          = 0.001
        dummy.log_likelihood_ = 0.0
        dummy.aic_            = 2.0
        dummy.bic_            = np.log(100)
        dummy._is_fitted       = True

        return CopulaSelectionResult(
            pair=pair_name,
            best_family="independence",
            best_copula=dummy,
            theta=0.0,
            tau_kendall=float(tau),
            log_likelihood=0.0,
            aic=2.0,
            bic=np.log(100),
            mbicv=np.log(100),
            tail_lower=0.0,
            tail_upper=0.0,
            selection_criterion=self.criterion,
            all_results=pd.DataFrame(),
        )

    # Seleção para todos os pares

    # Seleciona a melhor cópula para todos os pares de ativos.
    def select_all_pairs(
        self,
        U: np.ndarray,
        column_names: Optional[List[str]] = None,
        verbose: bool = True,
    ) -> Dict[Tuple[int, int], CopulaSelectionResult]:
        T, d = U.shape
        names = column_names or [f"ativo_{i}" for i in range(d)]
        n_pairs = d * (d - 1) // 2
        logger.info(
            f"Selecionando cópulas para {n_pairs} pares de {d} ativos "
            f"(critério={self.criterion.upper()})"
        )

        results = {}
        for i in range(d):
            for j in range(i + 1, d):
                pair = (names[i], names[j])
                result = self.select(U[:, i], U[:, j], pair_name=pair)
                results[(i, j)] = result
                if verbose:
                    logger.info(
                        f"  ({names[i]}, {names[j]}): {result.best_family} | "
                        f"θ={result.theta:.3f} | τ={result.tau_kendall:.3f} | "
                        f"λ_L={result.tail_lower:.3f} | λ_U={result.tail_upper:.3f}"
                    )
        return results

    def summarize_selection(
        self,
        pair_results: Dict[Tuple[int, int], CopulaSelectionResult],
    ) -> pd.DataFrame:
        rows = []
        for (i, j), res in pair_results.items():
            rows.append({
                "pair":        f"({res.pair[0]}, {res.pair[1]})",
                "i": i, "j": j,
                "best_family": res.best_family,
                "theta":       res.theta,
                "tau_kendall": res.tau_kendall,
                "aic":         res.aic,
                "bic":         res.bic,
                "mbicv":       res.mbicv,
                "lower_tail":  res.tail_lower,
                "upper_tail":  res.tail_upper,
            })
        df = pd.DataFrame(rows)
        logger.info("Distribuição de famílias selecionadas:")
        for fam, cnt in df["best_family"].value_counts().items():
            logger.info(f"  {fam:15s}: {cnt} pares ({cnt/len(df)*100:.1f}%)")
        return df

    def get_family_distribution(
        self,
        pair_results: Dict[Tuple[int, int], CopulaSelectionResult],
    ) -> pd.Series:
        return pd.Series([r.best_family for r in pair_results.values()]).value_counts()


# Teste de dependência de cauda

# Estimadores não-paramétricos de dependência de cauda.
class TailDependenceTest:

    @staticmethod
    def estimate_tail_dependence(
        u: np.ndarray,
        v: np.ndarray,
        method: str = "cfg",
        alpha: float = 0.1,
    ) -> Dict[str, float]:
        u = np.clip(u.ravel(), 1e-9, 1 - 1e-9)
        v = np.clip(v.ravel(), 1e-9, 1 - 1e-9)

        if method == "cfg":
            lam_L, lam_U = TailDependenceTest._cfg_estimator(u, v, alpha)
        elif method == "log":
            lam_L, lam_U = TailDependenceTest._log_estimator(u, v, alpha)
        else:
            raise ValueError(f"Método desconhecido: {method}")

        return {
            "lower":            float(lam_L),
            "upper":            float(lam_U),
            "is_tail_dependent": bool(lam_L > 0.1 or lam_U > 0.1),
            "method":           method,
            "alpha":            alpha,
        }

    @staticmethod
    def _cfg_estimator(u, v, alpha):
        T = len(u)
        k = max(1, int(T * alpha))
        u_lo, v_lo = np.sort(u)[k], np.sort(v)[k]
        lam_L = np.mean((u <= u_lo) & (v <= v_lo)) / (k / T)
        u_hi, v_hi = np.sort(u)[T - k - 1], np.sort(v)[T - k - 1]
        lam_U = np.mean((u >= u_hi) & (v >= v_hi)) / (k / T)
        return float(lam_L), float(lam_U)

    @staticmethod
    def _log_estimator(u, v, alpha):
        T = len(u)
        k = max(1, int(T * alpha))
        ru, rv = stats.rankdata(u), stats.rankdata(v)
        lam_L = np.mean((ru <= k) & (rv <= k)) * T / k
        lam_U = np.mean((ru >= T - k) & (rv >= T - k)) * T / k
        return float(lam_L), float(lam_U)

    @staticmethod
    def recommend_family(
        u: np.ndarray,
        v: np.ndarray,
        tau: Optional[float] = None,
    ) -> List[str]:
        if tau is None:
            tau, _ = stats.kendalltau(u, v)

        if abs(tau) < 0.05:
            return ["independence", "frank"]

        td = TailDependenceTest.estimate_tail_dependence(u, v, method="cfg")

        if td["lower"] > 0.15 and td["upper"] > 0.15:
            recs = ["t", "gaussian"]
        elif td["lower"] > 0.15:
            recs = ["clayton", "t", "gaussian"]
        elif td["upper"] > 0.15:
            recs = ["gumbel", "joe", "t"]
        else:
            recs = ["frank", "gaussian", "t"]

        all_fams = ["gaussian", "t", "clayton", "gumbel", "frank", "joe", "clayton180"]
        for fam in all_fams:
            if fam not in recs:
                recs.append(fam)
        return recs


# API pública — GARCH-Cópula

# Seleciona a melhor cópula para resíduos padronizados do GARCH.
def select_garch_copula(
    u: np.ndarray,
    v: np.ndarray,
    pair_name: Tuple[str, str] = ("i", "j"),
    families: Optional[List[str]] = None,
    use_recommendation: bool = True,
) -> CopulaSelectionResult:
    if use_recommendation and families is None:
        tau, _ = stats.kendalltau(u, v)
        families = TailDependenceTest.recommend_family(u, v, tau)[:5]
        logger.debug(f"Par {pair_name}: famílias recomendadas = {families}")

    selector = CopulaSelector(
        families=families or CopulaSelector.ALL_FAMILIES,
        criterion="aic",
        parallel=True,
    )
    return selector.select(u, v, pair_name=pair_name)


# API pública — vine cópula (mantém mBICV como padrão)

# Seleciona a melhor pair-copula para uso em vine_copulas.py.
def select_pair_copula(
    u: np.ndarray,
    v: np.ndarray,
    pair_name: Tuple[str, str] = ("i", "j"),
    criterion: str = "mbicv",
    families: Optional[List[str]] = None,
    use_recommendation: bool = True,
    vine_tree: int = 1,
) -> CopulaSelectionResult:
    if use_recommendation and families is None:
        tau, _ = stats.kendalltau(u, v)
        families = TailDependenceTest.recommend_family(u, v, tau)[:5]
        logger.debug(f"Par {pair_name}: famílias recomendadas = {families}")

    selector = CopulaSelector(
        families=families or CopulaSelector.ALL_FAMILIES,
        criterion=criterion,
        parallel=True,
    )
    return selector.select(u, v, pair_name=pair_name, vine_tree=vine_tree)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    np.random.seed(42)
    n = 800

    print("\n=== Caso 1: Clayton verdadeiro (θ=2) ===")
    theta_true = 2.0
    _u = np.random.uniform(0, 1, n)
    _w = np.random.uniform(0, 1, n)
    _inner = (_w * _u ** (theta_true + 1)) ** (-theta_true / (theta_true + 1)) + 1 - _u ** (-theta_true)
    u1 = np.clip(_u, 1e-9, 1 - 1e-9)
    v1 = np.clip(np.maximum(_inner, 1e-10) ** (-1 / theta_true), 1e-9, 1 - 1e-9)

    td = TailDependenceTest.estimate_tail_dependence(u1, v1)
    print(f"Tail dep empírica: λ_L={td['lower']:.3f}, λ_U={td['upper']:.3f}")

    result_aic = select_garch_copula(u1, v1, pair_name=("PETR4", "VALE3"))
    print("\n[AIC — GARCH-Cópula]")
    print(result_aic.summary())
    print(result_aic.all_results[["family", "aic", "bic", "mbicv", "theta"]].to_string())

    result_mbicv = select_pair_copula(u1, v1, pair_name=("PETR4", "VALE3"), vine_tree=1)
    print("\n[mBICV — vine cópula]")
    print(result_mbicv.summary())

    print("\n=== Caso 2: Gumbel verdadeiro (θ=2) ===")
    try:
        from copulas.archimedean import GumbelCopula
    except ImportError:
        from archimedean import GumbelCopula
    gum = GumbelCopula()
    gum.theta_    = 2.0
    gum._is_fitted = True
    sim_g = gum.simulate(n, seed=123)
    u2, v2 = sim_g[:, 0], sim_g[:, 1]

    result2 = select_garch_copula(u2, v2, pair_name=("ITUB4", "BBAS3"))
    print(result2.summary())

    print("\n=== Caso 3: select_all_pairs (d=4, AIC) ===")
    R = 0.4 * np.ones((4, 4)) + 0.6 * np.eye(4)
    L = np.linalg.cholesky(R)
    Z = np.random.standard_normal((n, 4)) @ L.T
    U4 = stats.norm.cdf(Z)

    names = ["PETR4", "VALE3", "ITUB4", "BBAS3"]
    selector = CopulaSelector(criterion="aic", parallel=False)
    all_results = selector.select_all_pairs(U4, column_names=names)
    summary_df = selector.summarize_selection(all_results)
    print(
        summary_df[["pair", "best_family", "tau_kendall", "aic", "lower_tail", "upper_tail"]]
        .to_string()
    )
    print("\nDistribuição de famílias:", dict(selector.get_family_distribution(all_results)))
    print("\nTeste concluído.")
