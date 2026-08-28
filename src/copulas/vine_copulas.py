from __future__ import annotations
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from scipy import stats
from scipy.optimize import minimize
from scipy.stats import kendalltau
from joblib import Parallel, delayed

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def _cached_tau(u, v):
    return float(kendalltau(u, v)[0])

def _clear_tau_cache():
    pass


def _empirical_tail(u, v, q=0.05):
    lo_u, lo_v = u < q, v < q
    hi_u, hi_v = u > (1 - q), v > (1 - q)
    n_lo = lo_u.sum(); n_hi = hi_u.sum()
    lam_L = float((lo_u & lo_v).sum() / n_lo) if n_lo > 0 else 0.0
    lam_U = float((hi_u & hi_v).sum() / n_hi) if n_hi > 0 else 0.0
    return lam_L, lam_U


_ROTATION_MAP = {
    'clayton':     'clayton_180',
    'gumbel':      'gumbel_180',
    'clayton_180': 'clayton',
    'gumbel_180':  'gumbel',
    'joe':         'joe_180',
    'joe_180':     'joe',
    'bb1':         'bb1_180',
    'bb1_180':     'bb1',
}


# mBICV de Nagler et al. (2019) — eq. (7).
def _compute_mbicv(log_likelihood: float, n_obs: int, n_params: int, vine_tree: int = 1) -> float:
    if n_obs <= 0 or not np.isfinite(log_likelihood):
        return np.inf
    nu_tree = 1.0 / max(vine_tree, 1)
    return -2.0 * log_likelihood + n_params * np.log(n_obs) * nu_tree


_N_PARAMS_VINE = {
    'gaussian': 1, 't': 2, 'clayton': 1, 'gumbel': 1,
    'frank': 1, 'gumbel_180': 1, 'clayton_180': 1,
    'joe': 1, 'joe_180': 1,
    'bb1': 2, 'bb1_180': 2,
}


# Seleciona família pelo mBICV (Nagler et al., 2019 — eq. 7).
def _select_best_copula(u, v, families, tail_bias=None,
                        aic_threshold=None, tail_err_tol=None,
                        tail_q=0.05, vine_tree=1):
    """Select the pair-copula family strictly by mBICV.

    Pairs with |Kendall tau| below the caller's min_tau are handled before
    this function. No post-selection tail heuristic is allowed to override
    the mBICV winner.
    """
    scores = {}; cops = {}
    n_obs = len(u)
    for fam in families:
        try:
            cop = PairCopula(fam)
            cop.fit(u, v)
            ll = cop.loglikelihood(u, v)
            if not np.isfinite(ll):
                continue
            k = _N_PARAMS_VINE.get(fam, 1)
            scores[fam] = _compute_mbicv(ll, n_obs, k, vine_tree)
            cops[fam] = cop
        except Exception as exc:
            logger.debug(f"mBICV: família {fam} falhou: {exc}")
    if not scores:
        bc = PairCopula('gaussian'); bc.param = 0.0
        return 'gaussian', bc
    best_fam = min(scores, key=scores.__getitem__)
    best_cop = cops[best_fam]
    logger.debug(
        f"mBICV winner={best_fam} score={scores[best_fam]:.4f} "
        f"tree={vine_tree} n={n_obs}"
    )
    return best_fam, best_cop


def _fit_one_pair(u, v, families, tail_bias, aic_threshold, min_tau,
                  tail_err_tol=0.12, tail_q=0.05, vine_tree=1):
    tau = abs(float(kendalltau(u, v)[0]))
    if tau < min_tau:
        cop = PairCopula('gaussian'); cop.param = 0.0
        cop_r = PairCopula('gaussian'); cop_r.param = 0.0
        return 'gaussian', cop, cop_r, tau
    fam, cop   = _select_best_copula(u, v, families, tail_bias,
                                     aic_threshold, tail_err_tol, tail_q,
                                     vine_tree=vine_tree)
    _,   cop_r = _select_best_copula(v, u, families, tail_bias,
                                     aic_threshold, tail_err_tol, tail_q,
                                     vine_tree=vine_tree)
    return fam, cop, cop_r, tau


class PairCopula:
    def __init__(self, family='gaussian'):
        self.family = family.lower(); self.param = None; self.param2 = None

    def fit(self, u, v):
        dispatch = dict(gaussian=self._fit_gaussian, t=self._fit_t,
                        clayton=self._fit_clayton, gumbel=self._fit_gumbel,
                        frank=self._fit_frank, gumbel_180=self._fit_gumbel_180,
                        clayton_180=self._fit_clayton_180,
                        joe=self._fit_joe, joe_180=self._fit_joe_180,
                        bb1=self._fit_bb1, bb1_180=self._fit_bb1_180)
        if self.family not in dispatch: raise ValueError(f"familia desconhecida: {self.family}")
        dispatch[self.family](u, v)
        return {'family': self.family, 'param': self.param, 'param2': self.param2}

    def _fit_gaussian(self, u, v):
        zu = stats.norm.ppf(np.clip(u, 1e-10, 1-1e-10))
        zv = stats.norm.ppf(np.clip(v, 1e-10, 1-1e-10))
        self.param = float(np.clip(np.corrcoef(zu, zv)[0, 1], -0.999999, 0.999999))

    def _fit_t(self, u, v):
        from scipy.special import gammaln
        u_ = np.clip(u, 1e-10, 1-1e-10)
        v_ = np.clip(v, 1e-10, 1-1e-10)

        def neg_ll(params):
            rho, nu = float(params[0]), float(params[1])
            if abs(rho) >= 1 or nu <= 2: return 1e10
            zu = stats.t.ppf(u_, nu)
            zv = stats.t.ppf(v_, nu)
            mask = np.isfinite(zu) & np.isfinite(zv)
            if mask.sum() < 2: return 1e10
            zu, zv = zu[mask], zv[mask]
            q = (zu**2 - 2*rho*zu*zv + zv**2) / (1 - rho**2)
            ll = (gammaln((nu+2)/2) + gammaln(nu/2) - 2*gammaln((nu+1)/2)
                  - 0.5*np.log(1 - rho**2)
                  + (nu+1)*np.log1p(zu**2/nu)
                  + (nu+1)*np.log1p(zv**2/nu)
                  - (nu+2)*np.log1p(q/nu))
            return -float(np.sum(ll))

        zu0 = stats.norm.ppf(u_); zv0 = stats.norm.ppf(v_)
        rho0 = float(np.clip(np.corrcoef(zu0, zv0)[0, 1], -0.95, 0.95))
        k_emp = float(np.mean([
            np.clip(float(np.mean(zu0**4) / max(np.mean(zu0**2)**2, 1e-10) - 3), 0.01, 10),
            np.clip(float(np.mean(zv0**4) / max(np.mean(zv0**2)**2, 1e-10) - 3), 0.01, 10),
        ]))
        nu0 = float(np.clip(6.0 / k_emp + 4.0, 3.0, 20.0))
        best_res = None
        for x0 in [[rho0, nu0], [rho0, max(nu0 * 0.4, 3.0)]]:
            try:
                r = minimize(neg_ll, x0, bounds=[(-0.99, 0.99), (2.5, 30.0)], method='L-BFGS-B')
                if best_res is None or r.fun < best_res.fun:
                    best_res = r
            except Exception:
                pass
        if best_res is not None:
            self.param, self.param2 = float(best_res.x[0]), float(best_res.x[1])
        else:
            self.param, self.param2 = rho0, nu0

    def _fit_clayton(self, u, v):
        tau = self.kendall_tau(u, v); self.param = max(2*tau/(1-tau), 0.01)

    def _fit_gumbel(self, u, v):
        tau = self.kendall_tau(u, v); self.param = max(1/(1-tau), 1.01)

    def _fit_gumbel_180(self, u, v): self._fit_gumbel(1-u, 1-v)
    def _fit_clayton_180(self, u, v): self._fit_clayton(1-u, 1-v)

    def _fit_frank(self, u, v):
        tau = float(self.kendall_tau(u, v))
        if abs(tau) < 0.01: self.param = 0.0; return
        from scipy.integrate import quad
        from scipy.optimize import brentq, minimize_scalar
        # Debye D1(θ) = (1/θ)·∫₀^θ t/(e^t-1) dt — o sinal de θ deve ser
        # preservado no limite de integração (NÃO em abs(θ)): a versão
        # anterior integrava sempre em [0,|θ|], o que torna τ(θ) idêntica
        # para θ e -θ — mas a relação correta é antissimétrica, τ(-θ)=-τ(θ).
        # Esse erro fazia eq(lo) e eq(hi) terem o mesmo sinal sempre que
        # τ<0 (ramo θ<0), quebrando o brentq sistematicamente nesse ramo
        # e empurrando o fallback (sem bounds) para θ≈±1e16.
        def _debye1(th):
            th = float(th)
            if abs(th) < 1e-10: return 1.0
            th_c = np.clip(th, -35.0, 35.0)
            return quad(lambda t: t / (np.expm1(t) if abs(t) > 1e-8 else 1.0),
                        0, th_c, limit=60)[0] / th_c
        def eq(th):
            th = float(th)
            if abs(th) < 1e-10: return tau
            return tau - (1 - 4 / th * (1 - _debye1(th)))
        lo, hi = (0.01, 35.0) if tau > 0 else (-35.0, -0.01)
        try:
            self.param = float(brentq(eq, lo, hi, xtol=1e-6, maxiter=80))
        except Exception:
            # Fallback com bounds idênticos ao do brentq: garante que o
            # parâmetro nunca escapa do range fisicamente válido mesmo
            # quando a relação τ(θ) não tem raiz exata no intervalo
            # (τ no limite da faixa atingível por |θ|≤35).
            def obj(th):
                th = float(np.asarray(th).flat[0])
                return eq(th) ** 2
            res = minimize_scalar(obj, bounds=(lo, hi), method='bounded')
            self.param = float(np.clip(res.x, lo, hi))

    # ── Joe ──────────────────────────────────────────────────────────────────
    # τ(θ) de Joe via série fechada (Joe 1997; mesma usada pelo pacote
    @staticmethod
    def _tau_joe_series(theta, K=2000):
        k = np.arange(1, K + 1, dtype=np.float64)
        term = 1.0 / (k * (k * theta + 2.0) * ((k - 1.0) * theta + 2.0))
        return 1.0 - 4.0 * np.sum(term)

    # MoM exato via inversão numérica (bisseção/brentq) da relação τ(θ)
    def _fit_joe(self, u, v):
        from scipy.optimize import brentq
        tau = float(self.kendall_tau(u, v))
        tau = max(tau, 0.01)
        tau = min(tau, 0.995)

        f = lambda th: self._tau_joe_series(th) - tau
        lo, hi = 1.0 + 1e-6, 200.0
        if f(lo) >= 0:
            self.param = 1.001
            return
        if f(hi) <= 0:
            self.param = 200.0
            return
        self.param = float(np.clip(brentq(f, lo, hi, xtol=1e-8, maxiter=100), 1.001, 200.0))

    def _fit_joe_180(self, u, v): self._fit_joe(1-u, 1-v)

    @staticmethod
    def _joe_cdf(u, v, th):
        eps = 1e-10
        u = np.clip(u, eps, 1-eps); v = np.clip(v, eps, 1-eps)
        A = 1.0 - (1.0-u)**th
        B = 1.0 - (1.0-v)**th
        return 1.0 - (np.maximum((1.0-A) + (1.0-B) - (1.0-A)*(1.0-B), eps))**(1.0/th)

    # PDF analítica c(u,v) = ∂²C/∂u∂v da cópula Joe.
    @staticmethod
    def _joe_pdf(u, v, th):
        eps = 1e-10
        u = np.clip(u, eps, 1-eps); v = np.clip(v, eps, 1-eps)
        pu = (1.0-u)**(th-1); pv = (1.0-v)**(th-1)
        A = 1.0 - (1.0-u)**th;  B = 1.0 - (1.0-v)**th
        S = np.maximum((1.0-A)+(1.0-B)-(1.0-A)*(1.0-B), eps)
        c = S**(1.0/th - 2.0) * pu * pv * ((th-1.0) + S)
        return np.maximum(c, eps)

    # ── BB1 (Clayton + Gumbel — dois parâmetros) ─────────────────────────────
    # MLE: δ e θ via otimização numérica com bounds apertados.
    def _fit_bb1(self, u, v):
        from scipy.optimize import minimize as _min
        tau  = float(self.kendall_tau(u, v))
        lL, lU = _empirical_tail(u, v, q=0.10)
        if lU > 0.01:
            theta0 = np.clip(np.log(2.0) / np.log(2.0 / max(2.0 - lU, 1e-3)), 1.01, 6.0)
        else:
            theta0 = max(1.01, 1.0 + tau)
        if lL > 0.01 and theta0 > 1.001:
            delta0 = np.clip(-np.log(2.0) / (np.log(max(lL, 1e-3)) * theta0), 0.1, 6.0)
        else:
            delta0 = 0.5

        def neg_ll(params):
            th, dl = float(params[0]), float(params[1])
            if th < 1.001 or dl < 0.001: return 1e10
            ll = PairCopula._bb1_loglik(u, v, th, dl)
            return -ll if np.isfinite(ll) else 1e10

        res = _min(neg_ll, [theta0, delta0],
                   bounds=[(1.001, 8.0), (0.01, 8.0)], method='L-BFGS-B')
        self.param, self.param2 = float(np.clip(res.x[0], 1.001, 8.0)), float(np.clip(res.x[1], 0.01, 8.0))

    def _fit_bb1_180(self, u, v): self._fit_bb1(1-u, 1-v)

    # Log-verossimilhança BB1 via PDF analítica (O(n), sem finite differences).
    @staticmethod
    def _bb1_loglik(u, v, theta, delta):
        eps = 1e-10
        u = np.clip(np.asarray(u, float), eps, 1-eps)
        v = np.clip(np.asarray(v, float), eps, 1-eps)
        th, dl = theta, delta
        with np.errstate(all='ignore'):
            x = np.maximum(u**(-dl) - 1.0, eps)
            y = np.maximum(v**(-dl) - 1.0, eps)
            S = np.maximum(x**th + y**th, eps)
            S1t = S**(1.0/th)
            T = 1.0 + S1t

            log_c = (
                -(1.0/dl) * np.log(T)
                - (dl + 1.0) * (np.log(u) + np.log(v))
                + (th - 1.0) * (np.log(x) + np.log(y))
                - np.log(T)
                + (1.0/th - 2.0) * np.log(S)
                + np.log(np.maximum(dl*(th - 1.0) + (dl + 1.0) * S1t / T, eps))
            )
        return float(np.sum(np.where(np.isfinite(log_c), log_c, -50.0)))

    @staticmethod
    def _bb1_cdf(u, v, theta, delta):
        eps = 1e-10
        u = np.clip(u, eps, 1-eps); v = np.clip(v, eps, 1-eps)
        x = (u**(-delta) - 1.0)**theta
        y = (v**(-delta) - 1.0)**theta
        S = np.maximum(x + y, eps)
        return np.maximum((1.0 + S**(1.0/theta))**(-1.0/delta), eps)

    @staticmethod
    def kendall_tau(u, v): return _cached_tau(u, v)

    # H-function da cópula-t em forma fechada (Joe 1997, eq. 6.8).
    @staticmethod
    def _h_t_closed(u, v, rho, nu, deriv):
        eps = 1e-10
        u = np.clip(u, eps, 1 - eps)
        v = np.clip(v, eps, 1 - eps)
        tu = stats.t.ppf(u, nu)
        tv = stats.t.ppf(v, nu)
        if deriv == 1:
            num = tu - rho * tv
        else:
            num = tv - rho * tu
        denom = np.sqrt(np.maximum((nu + tv**2 if deriv == 1 else nu + tu**2)
                                   * (1 - rho**2) / (nu + 1), eps))
        return np.clip(stats.t.cdf(num / denom, nu + 1), eps, 1 - eps)

    # Inversa da h-function da cópula-t em forma fechada.
    @staticmethod
    def _h_t_inv_closed(h, v, rho, nu, deriv):
        eps = 1e-10
        h = np.clip(h, eps, 1 - eps)
        v = np.clip(v, eps, 1 - eps)
        tv = stats.t.ppf(v, nu)
        h_inv = stats.t.ppf(h, nu + 1)
        if deriv == 1:
            scale = np.sqrt(np.maximum((nu + tv**2) * (1 - rho**2) / (nu + 1), eps))
            zu = rho * tv + h_inv * scale
        else:
            tu_ref = tv
            scale = np.sqrt(np.maximum((nu + tu_ref**2) * (1 - rho**2) / (nu + 1), eps))
            zu = rho * tu_ref + h_inv * scale
        return np.clip(stats.t.cdf(zu, nu), eps, 1 - eps)

    def h_function(self, u, v, deriv=1):
        eps = 1e-10
        scalar = (np.ndim(u) == 0 and np.ndim(v) == 0)
        u = np.clip(np.atleast_1d(np.asarray(u, float)), eps, 1-eps)
        v = np.clip(np.atleast_1d(np.asarray(v, float)), eps, 1-eps)
        if self.family in ('gumbel_180', 'clayton_180', 'joe_180', 'bb1_180'):
            base_cop = PairCopula(self.family.replace('_180', ''))
            base_cop.param = self.param; base_cop.param2 = self.param2
            res = np.clip(1 - base_cop.h_function(1-u, 1-v, deriv=deriv), eps, 1-eps)
            return float(res[0]) if scalar else res
        if self.family == 'gaussian':
            rho = self.param; den = np.sqrt(max(1-rho**2, eps))
            if deriv == 1: res = stats.norm.cdf((stats.norm.ppf(u)-rho*stats.norm.ppf(v))/den)
            else:          res = stats.norm.cdf((stats.norm.ppf(v)-rho*stats.norm.ppf(u))/den)
        elif self.family == 't':
            res = self._h_t_closed(u, v, self.param, self.param2, deriv)
        elif self.family == 'clayton':
            th = self.param; term = np.maximum(u**(-th)+v**(-th)-1, eps)
            if deriv == 1: res = np.clip(v**(-th-1)*term**(-1/th-1), eps, 1-eps)
            else:          res = np.clip(u**(-th-1)*term**(-1/th-1), eps, 1-eps)
        elif self.family == 'gumbel':
            th = self.param
            a = (-np.log(np.maximum(u, eps)))**th; b = (-np.log(np.maximum(v, eps)))**th
            s = np.maximum(a+b, eps); sp = s**(1/th); C = np.exp(-sp)
            if deriv == 1: res = np.clip(C*(sp/s)*np.maximum(b,eps)**(1-1/th)/np.maximum(v,eps), eps, 1-eps)
            else:          res = np.clip(C*(sp/s)*np.maximum(a,eps)**(1-1/th)/np.maximum(u,eps), eps, 1-eps)
        elif self.family == 'frank':
            th = self.param
            if abs(th) < eps: res = v.copy() if deriv==1 else u.copy()
            else:
                a = np.clip(-th*u, -700, 700)
                b = np.clip(-th*v, -700, 700)
                en_m1 = np.expm1(np.clip(-th, -700, 700))
                log_en_m1 = np.log(np.maximum(np.abs(en_m1), eps))
                sign_en = np.sign(en_m1)
                log_eu_m1 = np.log(np.maximum(np.abs(np.expm1(a)), eps))
                log_ev_m1 = np.log(np.maximum(np.abs(np.expm1(b)), eps))
                sign_a, sign_b = np.sign(a), np.sign(b)
                log_prod = log_eu_m1 + log_ev_m1
                sign_prod = sign_a * sign_b

                hi = np.maximum(log_en_m1, log_prod)
                lo = np.minimum(log_en_m1, log_prod)
                same = (sign_en == sign_prod) | (sign_en == 0) | (sign_prod == 0)
                log_den_mag = np.where(
                    same, np.logaddexp(log_en_m1, log_prod),
                    hi + np.log1p(-np.exp(np.clip(lo - hi, -700, 0)))
                )
                sign_den = np.where(
                    same, np.where(sign_en != 0, sign_en, sign_prod),
                    np.where(log_en_m1 >= log_prod, sign_en, sign_prod)
                )

                if deriv == 1:
                    log_num_mag = b + log_eu_m1
                    sign_num = sign_a
                else:
                    log_num_mag = a + log_ev_m1
                    sign_num = sign_b

                sign_h = sign_num * sign_den
                res = sign_h * np.exp(np.clip(log_num_mag - log_den_mag, -700, 700))
        else:
            res = np.array([self._h_fd(float(u[i]), float(v[i]), deriv) for i in range(len(u))])
        return float(np.clip(res, eps, 1-eps)[0]) if scalar else np.clip(res, eps, 1-eps)

    def _h_fd(self, u, v, deriv):
        eps = 1e-10
        if deriv == 1:
            d = max(1e-6, min(v,1-v)*1e-4)
            return np.clip((self.cdf(u,min(v+d,1-eps))-self.cdf(u,max(v-d,eps)))/(2*d), eps, 1-eps)
        else:
            d = max(1e-6, min(u,1-u)*1e-4)
            return np.clip((self.cdf(min(u+d,1-eps),v)-self.cdf(max(u-d,eps),v))/(2*d), eps, 1-eps)

    def h_function_inv(self, h, v, deriv=1, max_iter=500, tol=1e-12):
        eps = 1e-12; flat_tol = 1e-6
        scalar = (np.ndim(h) == 0 and np.ndim(v) == 0)
        h = np.clip(np.atleast_1d(np.asarray(h, float)), eps, 1-eps)
        v = np.clip(np.atleast_1d(np.asarray(v, float)), eps, 1-eps)
        if self.family in ('gumbel_180', 'clayton_180', 'joe_180', 'bb1_180'):
            base_cop = PairCopula(self.family.replace('_180', ''))
            base_cop.param = self.param; base_cop.param2 = self.param2
            res = np.clip(1-base_cop.h_function_inv(1-h, 1-v, deriv=deriv), eps, 1-eps)
            return float(res[0]) if scalar else res
        if self.family == 'gaussian':
            rho = self.param; sq = np.sqrt(max(1-rho**2, 1e-14))
            if deriv == 1: zu = sq*stats.norm.ppf(h) + rho*stats.norm.ppf(v)
            else:
                if abs(rho) < 1e-10: zu = stats.norm.ppf(h)
                else: zu = (stats.norm.ppf(v) - sq*stats.norm.ppf(h))/rho
            res = np.clip(stats.norm.cdf(zu), eps, 1-eps)
            return float(res[0]) if scalar else res
        if self.family == 't':
            res = self._h_t_inv_closed(h, v, self.param, self.param2, deriv)
            return float(res[0]) if scalar else res
        lo = np.full_like(h, eps); hi = np.full_like(h, 1-eps)
        h_lo = self.h_function(lo, v, deriv=deriv); h_hi = self.h_function(hi, v, deriv=deriv)
        flat = np.abs(h_hi-h_lo) < flat_tol; result = np.where(flat, h, np.nan)
        h_min = np.minimum(h_lo, h_hi); h_max = np.maximum(h_lo, h_hi)
        clamp_lo = h < h_min; clamp_hi = h > h_max; increasing = h_hi >= h_lo
        result = np.where(~flat & clamp_lo, np.where(increasing, lo, hi), result)
        result = np.where(~flat & clamp_hi, np.where(increasing, hi, lo), result)
        need_bisect = ~flat & ~clamp_lo & ~clamp_hi
        if np.any(need_bisect):
            lo_b = lo.copy(); hi_b = hi.copy()
            for _ in range(max_iter):
                mid = (lo_b+hi_b)/2; fmid = self.h_function(mid, v, deriv=deriv)-h
                flow = self.h_function(lo_b, v, deriv=deriv)-h
                converged = (np.abs(fmid) < tol) | ((hi_b-lo_b) < tol)
                if np.all(converged | ~need_bisect): break
                same_sign = (fmid*flow) >= 0
                lo_b = np.where(need_bisect & same_sign, mid, lo_b)
                hi_b = np.where(need_bisect & ~same_sign, mid, hi_b)
            result = np.where(need_bisect, np.clip((lo_b+hi_b)/2, eps, 1-eps), result)
        return float(np.clip(result, eps, 1-eps)[0]) if scalar else np.clip(result, eps, 1-eps)

    def cdf(self, u, v):
        u = np.clip(u, 1e-10, 1-1e-10); v = np.clip(v, 1e-10, 1-1e-10)
        if self.family in ('gumbel_180', 'clayton_180', 'joe_180', 'bb1_180'):
            base_cop = PairCopula(self.family.replace('_180', ''))
            base_cop.param = self.param; base_cop.param2 = self.param2
            return u+v-1+base_cop.cdf(1-u, 1-v)
        if self.family == 'gaussian':
            rho = np.clip(self.param, -0.999999, 0.999999)
            zu = stats.norm.ppf(u); zv = stats.norm.ppf(v)
            return stats.multivariate_normal.cdf([zu,zv],[0,0],[[1,rho],[rho,1]])
        elif self.family == 'clayton': return (u**(-self.param)+v**(-self.param)-1)**(-1/self.param)
        elif self.family == 'gumbel':
            th = self.param; return np.exp(-((-np.log(u))**th+(-np.log(v))**th)**(1/th))
        elif self.family == 'frank':
            th = self.param
            if abs(th) < 1e-10: return u*v
            return -1/th*np.log(1+(np.exp(-th*u)-1)*(np.exp(-th*v)-1)/(np.exp(-th)-1))
        elif self.family == 't':
            rho = self.param; nu = self.param2
            zu = stats.t.ppf(np.atleast_1d(u), nu)
            zv = stats.t.ppf(np.atleast_1d(v), nu)
            scalar_in = np.ndim(u) == 0
            if scalar_in:
                return stats.multivariate_normal.cdf(
                    [float(zu), float(zv)], [0,0], [[1,rho],[rho,1]]
                )
            cov = [[1, rho], [rho, 1]]
            return np.array([
                stats.multivariate_normal.cdf([zu[i], zv[i]], [0,0], cov)
                for i in range(len(zu))
            ])
        elif self.family == 'joe':
            return self._joe_cdf(u, v, self.param)
        elif self.family == 'bb1':
            return self._bb1_cdf(u, v, self.param, self.param2)

    def pdf(self, u, v):
        eps = 1e-10; u = float(np.clip(u, eps, 1-eps)); v = float(np.clip(v, eps, 1-eps))
        if self.family in ('gumbel_180', 'clayton_180', 'joe_180', 'bb1_180'):
            base_cop = PairCopula(self.family.replace('_180', ''))
            base_cop.param = self.param; base_cop.param2 = self.param2
            return base_cop.pdf(1-u, 1-v)
        if self.family == 'gaussian':
            rho = np.clip(self.param, -0.999999, 0.999999)
            zu = stats.norm.ppf(u); zv = stats.norm.ppf(v)
            r2 = max(1-rho**2, eps)
            return (1/np.sqrt(r2))*np.exp(-0.5*(zu**2+zv**2-2*rho*zu*zv)/r2+0.5*(zu**2+zv**2))
        elif self.family == 'clayton':
            th = self.param; u = max(u,1e-4); v = max(v,1e-4)
            term = u**(-th)+v**(-th)-1
            if term <= 0: return eps
            try:
                r = (1+th)*(u*v)**(-th-1)*term**(-2-1/th)
                return max(float(r), eps) if np.isfinite(r) else eps
            except: return eps
        elif self.family == 'gumbel':
            th = self.param; u = np.clip(u,1e-4,1-1e-4); v = np.clip(v,1e-4,1-1e-4)
            try:
                A = -np.log(u); B = -np.log(v); S = A**th+B**th
                if S <= eps: return eps
                Sp = S**(1/th); C = np.exp(-Sp)
                r = C/(u*v)*(A*B)**(th-1)*S**(1/th-2)*(S+th-1)
                return max(float(r), eps) if np.isfinite(r) else eps
            except: return eps
        elif self.family == 'frank':
            th = self.param
            if abs(th) < eps: return 1.0
            th_u = np.clip(-th * u, -500, 500)
            th_v = np.clip(-th * v, -500, 500)
            th_1 = np.clip(-th,     -500, 500)
            eu = np.exp(th_u); ev = np.exp(th_v); en = np.exp(th_1)
            num = -th * (en - 1) * eu * ev
            den = ((en - 1) + (eu - 1) * (ev - 1)) ** 2
            return max(abs(num / den) if abs(den) > eps else eps, eps)
        elif self.family == 't':
            rho = self.param; nu = self.param2
            zu = stats.t.ppf(u, nu); zv = stats.t.ppf(v, nu)
            from scipy.special import gamma
            dR = 1-rho**2; z = np.array([zu, zv])
            qf = z@np.linalg.inv([[1,rho],[rho,1]])@z
            c = gamma((nu+2)/2)/(gamma(nu/2)*nu*np.pi*np.sqrt(dR))
            f = (1+qf/nu)**(-(nu+2)/2)*c
            return max(f/(stats.t.pdf(zu,nu)*stats.t.pdf(zv,nu)), eps)
        elif self.family == 'joe':
            return float(np.maximum(self._joe_pdf(np.array([u]), np.array([v]), self.param)[0], eps))
        elif self.family == 'bb1':
            h = 1e-5
            c = (self._bb1_cdf(u+h,v+h,self.param,self.param2)-self._bb1_cdf(u+h,v-h,self.param,self.param2)
                 -self._bb1_cdf(u-h,v+h,self.param,self.param2)+self._bb1_cdf(u-h,v-h,self.param,self.param2)) / (4*h**2)
            return max(float(c), eps)
        raise NotImplementedError

    def loglikelihood(self, u, v):
        eps = 1e-10
        u = np.asarray(u, dtype=float); v = np.asarray(v, dtype=float)
        valid = np.isfinite(u) & np.isfinite(v)
        if not np.any(valid):
            return 0.0
        u = u[valid]; v = v[valid]
        if self.family == 'gaussian':
            rho = np.clip(self.param, -0.999999, 0.999999)
            zu = stats.norm.ppf(np.clip(u,eps,1-eps)); zv = stats.norm.ppf(np.clip(v,eps,1-eps))
            r2 = max(1 - rho**2, eps)
            ll = (-0.5*np.log(r2)-(zu**2+zv**2-2*rho*zu*zv)/(2*r2)+0.5*(zu**2+zv**2))
            return float(np.sum(ll))
        if self.family == 'clayton':
            th = self.param; u_ = np.clip(u,eps,1-eps); v_ = np.clip(v,eps,1-eps)
            term = np.maximum(u_**(-th)+v_**(-th)-1, eps)
            ll = (np.log(1+th)-(th+1)*(np.log(u_)+np.log(v_))-(2+1/th)*np.log(term))
            return float(np.sum(ll))
        if self.family == 'gumbel':
            th = self.param; u_ = np.clip(u,eps,1-eps); v_ = np.clip(v,eps,1-eps)
            A = (-np.log(u_))**th; B = (-np.log(v_))**th
            S = np.maximum(A+B, eps); Sp = S**(1/th); C = np.exp(-Sp)
            ll = np.log(np.maximum(C/(u_*v_)*(A*B)**(th-1)*S**(1/th-2)*(S+th-1), eps))
            return float(np.sum(ll))
        if self.family in ('gumbel_180', 'clayton_180', 'joe_180', 'bb1_180'):
            base_cop = PairCopula(self.family.replace('_180',''))
            base_cop.param = self.param; base_cop.param2 = self.param2
            return base_cop.loglikelihood(1-np.asarray(u), 1-np.asarray(v))
        if self.family == 'frank':
            th = self.param
            u_ = np.clip(u, eps, 1-eps); v_ = np.clip(v, eps, 1-eps)
            if abs(th) < eps:
                return 0.0
            absth = abs(th)
            th_c = np.clip(th, -35, 35)
            en_m1 = np.expm1(-th_c)
            log_en_m1 = np.log(np.maximum(np.abs(en_m1), eps))
            sign_en = np.sign(en_m1)
            a = np.clip(-th_c * u_, -500, 500)
            b = np.clip(-th_c * v_, -500, 500)
            log_eu_m1 = np.log(np.maximum(np.abs(np.expm1(a)), eps))
            log_ev_m1 = np.log(np.maximum(np.abs(np.expm1(b)), eps))
            log_prod = log_eu_m1 + log_ev_m1
            sign_prod = np.sign(a) * np.sign(b)
            same_sign = (sign_en == sign_prod) | (sign_en == 0) | (sign_prod == 0)
            log_inner_same = np.logaddexp(log_en_m1, log_prod)
            hi = np.maximum(log_en_m1, log_prod)
            lo = np.minimum(log_en_m1, log_prod)
            diff_arg = np.clip(lo - hi, -700, -1e-12)
            log_inner_diff = hi + np.log1p(-np.exp(diff_arg))
            log_den = np.where(same_sign, log_inner_same, log_inner_diff)
            ll_simple = np.log(absth) + log_en_m1 - th_c*(u_ + v_) - 2*log_den
            return float(np.sum(np.where(np.isfinite(ll_simple), ll_simple, -1e6)))
        if self.family == 'joe':
            return float(np.sum(np.log(np.maximum(self._joe_pdf(
                np.asarray(u, float), np.asarray(v, float), self.param), eps))))
        if self.family == 'bb1':
            return self._bb1_loglik(u, v, self.param, self.param2)
        return sum(np.log(max(self.pdf(float(u[i]),float(v[i])),eps)) for i in range(len(u)))

    def n_params(self):
        if self.family in ('t', 'bb1', 'bb1_180'): return 2
        return 1

    def aic(self, u, v): return 2*self.n_params()-2*self.loglikelihood(u,v)
    def bic(self, u, v, n): return self.n_params()*np.log(n)-2*self.loglikelihood(u,v)

    def tail_dependence(self):
        if self.family == 'gumbel_180': lam=2-2**(1/self.param); return lam,0.0
        if self.family == 'clayton_180': lam=2**(-1/self.param) if self.param>0 else 0; return 0.0,lam
        if self.family == 'joe_180':
            th = self.param; lam = 2 - 2**(1.0/th); return lam, 0.0
        if self.family == 'bb1_180':
            th, dl = self.param, self.param2
            lL = 2**(-1.0/(dl*th)); lU = 2 - 2**(1.0/th)
            return lU, lL
        if self.family == 'clayton': return (2**(-1/self.param) if self.param>0 else 0),0
        elif self.family == 'gumbel': return 0,2-2**(1/self.param)
        elif self.family in ['gaussian','frank']: return 0,0
        elif self.family == 'joe':
            th = self.param; return 0.0, 2-2**(1.0/th)
        elif self.family == 'bb1':
            th, dl = self.param, self.param2
            lL = 2**(-1.0/(dl*th)); lU = 2-2**(1.0/th)
            return lL, lU
        elif self.family == 't':
            rho,nu = self.param,self.param2
            lam = 2*stats.t.cdf(-np.sqrt((nu+1)*(1-rho)/(1+rho)),nu+1)
            return lam,lam
        return 0,0


class CVineCopula:
    def __init__(self, n_dim):
        self.n_dim = n_dim; self.copulas = {}; self.order = None
        self.n_trees_fit = None; self._pair_data = {}

    def fit(self, data, families=None, auto_select=True, order_method='dissmann',
            max_trees=3, min_tau=0.0, tail_bias=None, aic_threshold=None,
            tail_err_tol=None, tail_q=0.05, n_jobs=-1):
        if families is None:
            families = ['gaussian','clayton','gumbel','frank','t','gumbel_180','clayton_180','joe','joe_180','bb1','bb1_180']
        n_trees = min(max_trees, self.n_dim-1) if max_trees else self.n_dim-1
        self.n_trees_fit = n_trees; _clear_tau_cache(); self._select_order(data)
        pseudo = [data[:, self.order].copy()]
        for t in range(n_trees):
            cur = pseudo[t]; n_edge = cur.shape[1]-1
            pairs = [(cur[:,0].copy(), cur[:,e+1].copy()) for e in range(n_edge)]
            results = Parallel(n_jobs=n_jobs, backend='loky', prefer='processes')(
                delayed(_fit_one_pair)(u, v, families, tail_bias, aic_threshold, min_tau,
                                      tail_err_tol, tail_q, vine_tree=t+1)
                for (u, v) in pairs)
            next_cols = []
            for e, (fam, cop, cop_r, tau) in enumerate(results):
                u, v = pairs[e]; key = (t, 0, e+1)
                self.copulas[key] = cop; self._pair_data[key] = (u.copy(), v.copy())
                if t < n_trees-1: next_cols.append(cop.h_function(v, u, deriv=1))
            if t < n_trees-1 and next_cols: pseudo.append(np.column_stack(next_cols))
        return {'n_trees': n_trees, 'order': self.order}

    def _select_order(self, data, alpha=0.6, q_tail=0.05):
        n = data.shape[1]; tau_mat = np.zeros((n,n)); tail_mat = np.zeros((n,n))
        for i in range(n):
            for j in range(i+1, n):
                t = abs(PairCopula.kendall_tau(data[:,i], data[:,j]))
                tau_mat[i,j] = tau_mat[j,i] = t
                hi_i = data[:,i] > (1-q_tail); hi_j = data[:,j] > (1-q_tail)
                n_hi = hi_i.sum()
                lam = float((hi_i & hi_j).sum()/n_hi) if n_hi > 0 else 0.0
                tail_mat[i,j] = tail_mat[j,i] = lam
        score_mat = alpha*tau_mat + (1-alpha)*tail_mat
        best_i, best_j, best_score = 0, 1, -1.0
        for i in range(n):
            for j in range(i+1,n):
                if score_mat[i,j] > best_score: best_score=score_mat[i,j]; best_i,best_j=i,j
        chain = [best_i, best_j]; remaining = [k for k in range(n) if k not in chain]
        while remaining:
            left_end = chain[0]; right_end = chain[-1]
            best_l = max(score_mat[left_end,c] for c in remaining)
            best_r = max(score_mat[right_end,c] for c in remaining)
            if best_l >= best_r:
                best_c = max(remaining, key=lambda c: score_mat[left_end,c]); chain.insert(0, best_c)
            else:
                best_c = max(remaining, key=lambda c: score_mat[right_end,c]); chain.append(best_c)
            remaining.remove(best_c)
        self.order = np.array(chain)

    def loglikelihood(self): return sum(self.copulas[k].loglikelihood(u,v) for k,(u,v) in self._pair_data.items())
    def n_params(self): return sum(c.n_params() for c in self.copulas.values())
    def aic(self): return 2*self.n_params()-2*self.loglikelihood()
    def bic(self, n): return self.n_params()*np.log(n)-2*self.loglikelihood()

    def simulate(self, n_samples, seed=None):
        if seed is not None: np.random.seed(seed)
        d = self.n_dim; T = self.n_trees_fit or d-1
        W = np.random.uniform(0,1,(n_samples,d)); x = np.zeros((n_samples,d))
        vv = [[None]*d for _ in range(d)]; x[:,0] = W[:,0]
        for i in range(1, d):
            w = W[:,i].copy()
            for lv in range(min(i,T)-1, -1, -1):
                key = (lv, 0, i-lv)
                if key not in self.copulas: continue
                cond = x[:,lv] if lv == 0 else vv[lv][lv-1]
                w = self.copulas[key].h_function_inv(w, cond, deriv=1)
            x[:,i] = w
            key0 = (0,0,i)
            vv[i][0] = (self.copulas[key0].h_function(x[:,i], x[:,0], deriv=1) if key0 in self.copulas else x[:,i])
            for lv in range(1, min(i,T)):
                key = (lv, 0, i-lv); cond = x[:,lv] if lv == 0 else vv[lv][lv-1]
                vv[i][lv] = (self.copulas[key].h_function(vv[i][lv-1], cond, deriv=1) if key in self.copulas else vv[i][lv-1])
            vv[i][i] = x[:,i]
        out = np.zeros_like(x)
        for pos, orig in enumerate(self.order): out[:,orig] = x[:,pos]
        return out


class DVineCopula:
    def __init__(self, n_dim):
        self.n_dim = n_dim; self.copulas = {}; self._cop_r = {}; self.order = None
        self.n_trees_fit = None; self._pair_data = {}; self._keys_by_tree = []

    def _select_order(self, data):
        n = data.shape[1]; tau_mat = np.zeros((n,n))
        for i in range(n):
            for j in range(i+1,n):
                t = abs(PairCopula.kendall_tau(data[:,i], data[:,j]))
                tau_mat[i,j] = tau_mat[j,i] = t
        best_i, best_j, best_score = 0, 1, -1.0
        for i in range(n):
            for j in range(i+1,n):
                if tau_mat[i,j] > best_score: best_score=tau_mat[i,j]; best_i,best_j=i,j
        chain = [best_i, best_j]; remaining = set(range(n))-{best_i, best_j}
        while remaining:
            best_l_c = max(remaining, key=lambda c: tau_mat[chain[0],c])
            best_r_c = max(remaining, key=lambda c: tau_mat[chain[-1],c])
            if tau_mat[chain[0],best_l_c] >= tau_mat[chain[-1],best_r_c]:
                chain.insert(0, best_l_c); remaining.remove(best_l_c)
            else:
                chain.append(best_r_c); remaining.remove(best_r_c)
        self.order = np.array(chain)
        adj_taus = [tau_mat[chain[k],chain[k+1]] for k in range(n-1)]
        logger.info(f"D-vine order={chain}  adj_tau={[round(t,3) for t in adj_taus]}")

    def fit(self, data, families=None, auto_select=True, max_trees=3,
            min_tau=0.0, tail_bias=None, aic_threshold=10.0,
            tail_err_tol=0.12, tail_q=0.10, n_jobs=-1):
        if families is None:
            families = ['gaussian','clayton','gumbel','frank','t','gumbel_180','clayton_180','joe','joe_180','bb1','bb1_180']
        n_trees = min(max_trees, self.n_dim-1) if max_trees else self.n_dim-1
        self.n_trees_fit = n_trees; _clear_tau_cache(); self._select_order(data); d = self.n_dim
        vm = {}
        for j in range(d): vm[(j, frozenset())] = data[:, self.order[j]].copy()
        key_u = {j: (j, frozenset()) for j in range(d-1)}
        key_v = {j: (j+1, frozenset()) for j in range(d-1)}
        self._keys_by_tree = []
        for e in range(n_trees):
            n_pairs = d-e-1; pairs = []; pair_meta = []
            for j in range(n_pairs):
                u_key = key_u[j]; v_key = key_v[j]
                if u_key not in vm or v_key not in vm:
                    raise KeyError(f"D-vine T{e+1} par j={j}: chave ausente")
                pairs.append((vm[u_key].copy(), vm[v_key].copy()))
                pair_meta.append((j, u_key, v_key))
            results = Parallel(n_jobs=n_jobs, backend='loky', prefer='processes')(
                delayed(_fit_one_pair)(u, v, families, tail_bias, aic_threshold, min_tau,
                                      tail_err_tol, tail_q, vine_tree=e+1)
                for (u, v) in pairs)
            new_key_u = {}; new_key_v = {}
            for (j, u_key, v_key), (fam, cop, cop_r, tau) in zip(pair_meta, results):
                u, v = pairs[j]; cop_key = (e, j, j+e+1)
                self.copulas[cop_key] = cop
                self._cop_r[cop_key] = cop_r
                self._pair_data[cop_key] = (u.copy(), v.copy())
                if e < n_trees-1:
                    u_node, u_cond = u_key; v_node, v_cond = v_key
                    new_u_cond = frozenset(u_cond|v_cond|{v_node})
                    new_v_cond = frozenset(v_cond|u_cond|{u_node})
                    new_u_key = (u_node, new_u_cond); new_v_key = (v_node, new_v_cond)
                    if new_u_key not in vm: vm[new_u_key] = cop.h_function(u, v, deriv=1)
                    if new_v_key not in vm: vm[new_v_key] = cop.h_function(u, v, deriv=2)
                    new_key_u[j] = new_u_key; new_key_v[j] = new_v_key
            self._keys_by_tree.append((dict(key_u), dict(key_v)))
            if e < n_trees-1:
                next_key_u = {}; next_key_v = {}
                for j in range(n_pairs-1):
                    next_key_u[j] = new_key_u[j]; next_key_v[j] = new_key_v[j+1]
                key_u = next_key_u; key_v = next_key_v
        return {'n_trees': n_trees, 'order': self.order}

    def simulate(self, n_samples, seed=None):
        if seed is not None:
            np.random.seed(seed)
        d = self.n_dim; T = self.n_trees_fit or d - 1
        W = np.random.uniform(0, 1, (n_samples, d))
        x = np.zeros((n_samples, d))
        m_u = [[None] * (T + 1) for _ in range(d)]
        m_v = [[None] * (T + 1) for _ in range(d)]
        x[:, 0] = W[:, 0]
        m_u[0][0] = W[:, 0].copy()
        m_v[0][0] = W[:, 0].copy()
        for i in range(1, d):
            w = W[:, i].copy()
            for k in range(min(i, T) - 1, -1, -1):
                cop_r = self._cop_r.get((k, i - k - 1, i))
                if cop_r is None: continue
                cond = m_u[i - k - 1][k]
                if cond is None: continue
                w = cop_r.h_function_inv(w, cond, deriv=1)
            x[:, i] = w
            m_u[i][0] = w.copy()
            m_v[i][0] = w.copy()
            if i == d - 1: break
            for k in range(1, min(i, T) + 1):
                j = i - k
                cop = self.copulas.get((k - 1, j, i))
                if cop is None: break
                u_in    = m_u[j][k - 1]
                cond_in = m_v[i][k - 1]
                if u_in is None or cond_in is None: break
                m_u[j][k] = cop.h_function(u_in, cond_in, deriv=1)
                m_v[i][k] = cop.h_function(u_in, cond_in, deriv=2)
        out = np.zeros_like(x)
        for pos, orig in enumerate(self.order):
            out[:, orig] = x[:, pos]
        return out

    def loglikelihood(self): return sum(self.copulas[k].loglikelihood(u,v) for k,(u,v) in self._pair_data.items())
    def n_params(self): return sum(c.n_params() for c in self.copulas.values())
    def aic(self): return 2*self.n_params()-2*self.loglikelihood()
    def bic(self, n): return self.n_params()*np.log(n)-2*self.loglikelihood()


class RVineCopula:
    def __init__(self, n_dim):
        self.n_dim = n_dim; self.n_trees_fit = None
        self._edges_by_tree = []; self._pseudo_fit = {}
        self._sim_order = []; self._pair_ll = {}

    def _find_conditioned_pseudo(self, ov_node, D):
        conditioned = ov_node-D; target = frozenset(ov_node); candidates = []
        for key in self._pseudo_fit:
            if not (isinstance(key, tuple) and len(key)==2): continue
            A, B = key
            if not (isinstance(A, frozenset) and isinstance(B, frozenset)): continue
            if A|B==target and A.issuperset(conditioned) and len(A)==len(target)-1:
                candidates.append(key)
        if not candidates: return None
        return max(candidates, key=lambda k: len(k[0]))

    def fit(self, data, families=None, auto_select=True, max_trees=3, min_tau=0.0,
            tail_bias=None, aic_threshold=10.0,
            tail_err_tol=0.12, tail_q=0.10,
            forced_edges=None,
            n_jobs=-1):
        self.n_dim = data.shape[1]; d = self.n_dim
        if families is None:
            families = ['gaussian','clayton','gumbel','frank','t','gumbel_180','clayton_180','joe','joe_180','bb1','bb1_180']
        n_trees = min(max_trees, d-1) if max_trees else d-1
        self.n_trees_fit = n_trees; _clear_tau_cache()
        self._forced_t1 = (set(frozenset(e) for e in forced_edges)
                           if forced_edges else set())
        logger.info(f"Ajustando R-vine {d}d  arvores={n_trees}  min_tau={min_tau}"
                    + (f"  forced_edges={list(forced_edges)}" if forced_edges else ""))
        pseudo = {frozenset({i}): data[:,i].copy() for i in range(d)}
        self._pseudo_fit = {}; orig_vars = {i: frozenset({i}) for i in range(d)}
        edges_by_tree = []
        for t in range(n_trees):
            nodes = list(orig_vars.keys()); tau_mat = self._build_tau_mat(nodes, orig_vars, pseudo, t)
            if not tau_mat: break
            forced = self._forced_t1 if t == 0 else set()
            mst = self._mst(nodes, tau_mat, forced); tree_edges = []; new_orig = {}; new_pseudo = {}
            pair_inputs = []
            for (ni, nj) in mst:
                ov_ni = orig_vars[ni]; ov_nj = orig_vars[nj]; D = ov_ni&ov_nj
                if t == 0:
                    u_arr = pseudo[ov_ni]; v_arr = pseudo[ov_nj]
                else:
                    u_key = self._find_conditioned_pseudo(ov_ni, D)
                    v_key = self._find_conditioned_pseudo(ov_nj, D)
                    u_arr = (self._pseudo_fit[u_key] if u_key is not None else pseudo.get(ov_ni, data[:,next(iter(ov_ni-D))]))
                    v_arr = (self._pseudo_fit[v_key] if v_key is not None else pseudo.get(ov_nj, data[:,next(iter(ov_nj-D))]))
                pair_inputs.append((ni, nj, ov_ni, ov_nj, D, u_arr.copy(), v_arr.copy()))
            fit_results = Parallel(n_jobs=n_jobs, backend='loky', prefer='processes')(
                delayed(_fit_one_pair)(u, v, families, tail_bias, aic_threshold, min_tau,
                                      tail_err_tol, tail_q, vine_tree=t+1)
                for (_, _, _, _, _, u, v) in pair_inputs)
            for (ni, nj, ov_ni, ov_nj, D, u_arr, v_arr), (fam, cop, cop_r, tau) in zip(pair_inputs, fit_results):
                edge_key = (frozenset(ov_ni), frozenset(ov_nj))
                node_key_ni = (frozenset(ov_ni), frozenset(ov_nj))
                node_key_nj = (frozenset(ov_nj), frozenset(ov_ni))
                tree_edges.append({'ov_ni':ov_ni,'ov_nj':ov_nj,'cop':cop,'cop_r':cop_r,
                                   'edge_key':edge_key,'node_key_ni':node_key_ni,'node_key_nj':node_key_nj})
                self._pair_ll[edge_key] = (u_arr.copy(), v_arr.copy(), cop)
                self._pseudo_fit[node_key_ni] = cop.h_function(u_arr, v_arr, deriv=1)
                self._pseudo_fit[node_key_nj] = cop_r.h_function(v_arr, u_arr, deriv=1)
                merged = ov_ni|ov_nj; new_pseudo[merged] = self._pseudo_fit[node_key_ni]
                new_orig[(ni,nj)] = merged; new_orig[(nj,ni)] = merged
            edges_by_tree.append(tree_edges); pseudo.update(new_pseudo); orig_vars = new_orig
        self._edges_by_tree = edges_by_tree
        self._sim_order = self._extract_sim_order(edges_by_tree)
        logger.info(f"R-vine OK  sim_order={self._sim_order}")
        return {'n_trees': n_trees, 'sim_order': self._sim_order}

    def _extract_sim_order(self, edges_by_tree):
        if not edges_by_tree:
            return list(range(self.n_dim))
        from collections import defaultdict, deque
        adj: dict[int, set] = defaultdict(set)
        for e in edges_by_tree[0]:
            u = next(iter(e['ov_ni']))
            v = next(iter(e['ov_nj']))
            adj[u].add(v)
            adj[v].add(u)
        for v in range(self.n_dim):
            if v not in adj:
                adj[v] = set()
        root = max(range(self.n_dim), key=lambda v: len(adj[v]))
        order: list[int] = []
        visited: set[int] = set()
        queue: deque[int] = deque([root])
        while queue:
            node = queue.popleft()
            if node in visited: continue
            visited.add(node); order.append(node)
            for nb in sorted(adj[node], key=lambda v: -len(adj[v])):
                if nb not in visited:
                    queue.append(nb)
        for v in range(self.n_dim):
            if v not in visited:
                order.append(v)
        return order

    def _build_tau_mat(self, nodes, orig_vars, pseudo, t):
        tau_mat = {}; seen = set()
        for idx, ni in enumerate(nodes):
            for nj in nodes[idx+1:]:
                fni=orig_vars[ni]; fnj=orig_vars[nj]
                if t > 0 and len(fni.symmetric_difference(fnj)) != 2: continue
                cp = (min(str(fni),str(fnj)), max(str(fni),str(fnj)))
                if cp in seen: continue
                seen.add(cp)
                if fni in pseudo and fnj in pseudo:
                    tau_mat[(ni,nj)] = abs(PairCopula.kendall_tau(pseudo[fni], pseudo[fnj]))
        return tau_mat

    def _mst(self, nodes, tau_mat, forced=None):
        par = {n: n for n in nodes}
        def find(x):
            while par[x] != x: par[x]=par[par[x]]; x=par[x]
            return x
        def union(x, y):
            px,py=find(x),find(y)
            if px==py: return False
            par[px]=py; return True
        mst = []
        n_target = len(set(nodes)) - 1
        if forced:
            node_set = set(nodes)
            for fs in forced:
                ab = list(fs)
                if len(ab) != 2: continue
                i, j = ab
                if i not in node_set or j not in node_set: continue
                key = (i, j) if (i, j) in tau_mat else ((j, i) if (j, i) in tau_mat else None)
                if key is None: continue
                if union(i, j):
                    mst.append(key)
                if len(mst) == n_target: return mst
        sorted_e = sorted(tau_mat.items(), key=lambda x: -x[1])
        for (i, j), _ in sorted_e:
            if union(i, j):
                mst.append((i, j))
            if len(mst) == n_target: break
        return mst

    def loglikelihood(self): return sum(cop.loglikelihood(u,v) for (u,v,cop) in self._pair_ll.values())
    def n_params(self): return sum(cop.n_params() for (_,_,cop) in self._pair_ll.values())
    def aic(self): return 2*self.n_params()-2*self.loglikelihood()
    def bic(self, n): return self.n_params()*np.log(n)-2*self.loglikelihood()

    def simulate(self, n_samples, seed=None):
        if seed is not None: np.random.seed(seed)
        d = self.n_dim; pi = self._sim_order
        if len(pi) != d: raise ValueError(f"sim_order tem {len(pi)} elementos, esperado {d}")
        W = np.random.uniform(0,1,(n_samples,d)); X = np.zeros((n_samples,d)); pseudo_sim = {}
        v0 = pi[0]; X[:,v0] = W[:,0]; pseudo_sim[frozenset({v0})] = W[:,0].copy()
        for j in range(1, d):
            vj = pi[j]; prev_vars = frozenset(pi[:j]); w = W[:,j].copy()
            chain = self._build_backward_chain(vj, prev_vars, pseudo_sim)
            for (cop, cond_key) in reversed(chain):
                cond_arr = pseudo_sim.get(cond_key)
                if cond_arr is None: raise ValueError(f"[backward] pseudo_sim[{cond_key}] nao encontrado")
                w = cop.h_function_inv(w, cond_arr, deriv=1)
            X[:,vj] = w; pseudo_sim[frozenset({vj})] = w.copy()
            self._forward_propagate(vj, prev_vars, pseudo_sim)
        return X

    def _build_backward_chain(self, vj, prev_vars, pseudo_sim):
        chain = []; current_node = frozenset({vj})
        for t, tree_edges in enumerate(self._edges_by_tree):
            if t >= self.n_trees_fit: break
            found = False
            for e in tree_edges:
                ov_ni=e['ov_ni']; ov_nj=e['ov_nj']; D=ov_ni&ov_nj
                if ov_ni==current_node and ov_nj.issubset(prev_vars|frozenset({vj})):
                    cond_key = self._find_conditioned_pseudo_in_sim(ov_nj, D, pseudo_sim)
                    if cond_key is None: break
                    chain.append((e['cop'], cond_key)); current_node=ov_ni|ov_nj; found=True; break
                elif ov_nj==current_node and ov_ni.issubset(prev_vars|frozenset({vj})):
                    cond_key = self._find_conditioned_pseudo_in_sim(ov_ni, D, pseudo_sim)
                    if cond_key is None: break
                    chain.append((e['cop_r'], cond_key)); current_node=ov_ni|ov_nj; found=True; break
            if not found: break
        return chain

    def _find_pseudo_key(self, cond_node, level, pseudo_sim):
        fs = frozenset(cond_node)
        if fs in pseudo_sim: return fs
        candidates = []
        for key in pseudo_sim:
            if not (isinstance(key,tuple) and len(key)==2): continue
            A, B = key
            if not (isinstance(A,frozenset) and isinstance(B,frozenset)): continue
            if A|B==fs and A.issubset(fs): candidates.append((key, len(A)))
        if candidates: candidates.sort(key=lambda x: -x[1]); return candidates[0][0]
        for key in pseudo_sim:
            if isinstance(key,tuple) and len(key)==2:
                A, B = key
                if isinstance(A,frozenset) and A==fs: return key
        return None

    def _find_conditioned_pseudo_in_sim(self, cond_node, D, pseudo_sim):
        fs = frozenset(cond_node); conditioned = fs-D
        if len(fs) == 1: return fs if fs in pseudo_sim else None
        if len(conditioned) == 1:
            direct_key = (conditioned, D)
            if direct_key in pseudo_sim: return direct_key
        candidates = []
        for key in pseudo_sim:
            if not (isinstance(key,tuple) and len(key)==2): continue
            A, B = key
            if not (isinstance(A,frozenset) and isinstance(B,frozenset)): continue
            if A|B==fs and A.issuperset(conditioned) and len(A)==len(fs)-1:
                candidates.append((key, len(A)))
        if candidates: candidates.sort(key=lambda x: -x[1]); return candidates[0][0]
        return self._find_pseudo_key(cond_node, -1, pseudo_sim)

    def _forward_propagate(self, vj, prev_vars, pseudo_sim):
        current_node = frozenset({vj}); vj_arr = pseudo_sim[frozenset({vj})]
        for t, tree_edges in enumerate(self._edges_by_tree):
            if t >= self.n_trees_fit: break
            found = False
            for e in tree_edges:
                ov_ni=e['ov_ni']; ov_nj=e['ov_nj']
                node_key_ni=e['node_key_ni']; node_key_nj=e['node_key_nj']; D=ov_ni&ov_nj
                if ov_ni==current_node and ov_nj.issubset(prev_vars|frozenset({vj})):
                    cond_key=self._find_conditioned_pseudo_in_sim(ov_nj, D, pseudo_sim)
                    if cond_key is None: found=False; break
                    cond_arr=pseudo_sim[cond_key]
                    pseudo_sim[node_key_ni]=e['cop'].h_function(vj_arr, cond_arr, deriv=1)
                    pseudo_sim[node_key_nj]=e['cop_r'].h_function(cond_arr, vj_arr, deriv=1)
                    vj_arr=pseudo_sim[node_key_ni]; current_node=ov_ni|ov_nj; found=True; break
                elif ov_nj==current_node and ov_ni.issubset(prev_vars|frozenset({vj})):
                    cond_key=self._find_conditioned_pseudo_in_sim(ov_ni, D, pseudo_sim)
                    if cond_key is None: found=False; break
                    cond_arr=pseudo_sim[cond_key]
                    pseudo_sim[node_key_nj]=e['cop_r'].h_function(vj_arr, cond_arr, deriv=1)
                    pseudo_sim[node_key_ni]=e['cop'].h_function(cond_arr, vj_arr, deriv=1)
                    vj_arr=pseudo_sim[node_key_nj]; current_node=ov_ni|ov_nj; found=True; break
            if not found: break


def _tail_dependence_empirical(data, q_lower=0.05, q_upper=0.95):
    n, d = data.shape; lowers, uppers = [], []
    for i in range(d):
        for j in range(i+1, d):
            lo_i=data[:,i]<q_lower; lo_j=data[:,j]<q_lower
            hi_i=data[:,i]>q_upper; hi_j=data[:,j]>q_upper
            n_lo=lo_i.sum(); n_hi=hi_i.sum()
            lowers.append((lo_i&lo_j).sum()/n_lo if n_lo>0 else 0.0)
            uppers.append((hi_i&hi_j).sum()/n_hi if n_hi>0 else 0.0)
    return float(np.mean(lowers)), float(np.mean(uppers))


def compare_dependence(original, simulated):
    from scipy.stats import kendalltau as _kt
    n_vars = original.shape[1]
    print("comparacao kendall's tau")
    print(f"{'Par':<10}{'Original':>12}{'Simulado':>12}{'Diff':>10}")
    for i in range(n_vars):
        for j in range(i+1, n_vars):
            tau_orig,_ = _kt(original[:,i], original[:,j])
            tau_sim,_  = _kt(simulated[:,i], simulated[:,j])
            print(f"({i},{j}){tau_orig:>16.3f}{tau_sim:>12.3f}{abs(tau_orig-tau_sim):>10.3f}")


def compare_vines(data, families=None, max_trees=3, min_tau=0.0, n_sim=1000, seed=42,
                  q_tail=0.10, tail_bias='None', aic_threshold=4.0,
                  tail_err_tol=0.12):
    from scipy.stats import kendalltau as _kt
    n, d = data.shape
    tlo_orig, thi_orig = _tail_dependence_empirical(data, q_tail, 1-q_tail); results = []
    for name, Cls in [("C-vine", CVineCopula), ("D-vine", DVineCopula), ("R-vine", RVineCopula)]:
        logger.info(f"--- {name} ---")
        try:
            vine = Cls(n_dim=d)
            kw = dict(families=families, max_trees=max_trees, min_tau=min_tau,
                      tail_bias=tail_bias, aic_threshold=aic_threshold,
                      tail_err_tol=tail_err_tol)
            if name == "C-vine": kw["auto_select"] = True
            vine.fit(data, **kw)
            ll=vine.loglikelihood(); aic=vine.aic(); bic=vine.bic(n)
            sim=vine.simulate(n_sim, seed=seed)
            tau_o=np.mean([abs(float(_kt(data[:,i],data[:,j])[0])) for i in range(d) for j in range(i+1,d)])
            tau_s=np.mean([abs(PairCopula.kendall_tau(sim[:,i],sim[:,j])) for i in range(d) for j in range(i+1,d)])
            tlo_s,thi_s=_tail_dependence_empirical(sim, q_tail, 1-q_tail)
            results.append(dict(modelo=name,ll=round(ll,2),n_params=vine.n_params(),aic=round(aic,2),bic=round(bic,2),
                                tau_orig=round(tau_o,4),tau_sim=round(tau_s,4),tau_diff=round(abs(tau_o-tau_s),4),
                                tail_lo_orig=round(tlo_orig,4),tail_lo_sim=round(tlo_s,4),tail_lo_diff=round(abs(tlo_orig-tlo_s),4),
                                tail_hi_orig=round(thi_orig,4),tail_hi_sim=round(thi_s,4),tail_hi_diff=round(abs(thi_orig-thi_s),4)))
        except Exception as ex:
            logger.error(f"{name} falhou: {ex}", exc_info=True)
            results.append(dict(modelo=name,ll=np.nan,n_params=np.nan,aic=np.nan,bic=np.nan,
                                tau_orig=np.nan,tau_sim=np.nan,tau_diff=np.nan,
                                tail_lo_orig=round(tlo_orig,4),tail_lo_sim=np.nan,tail_lo_diff=np.nan,
                                tail_hi_orig=round(thi_orig,4),tail_hi_sim=np.nan,tail_hi_diff=np.nan))
    return pd.DataFrame(results).set_index('modelo')


# ─────────────────────────────────────────────────────────────────────────────
# Public API wrappers  (CVine / DVine / RVine / select_vine)
# Adapts the internal CVineCopula / DVineCopula / RVineCopula to the
# standardised interface expected by tests and the rest of the pipeline.
# ─────────────────────────────────────────────────────────────────────────────

# Mixin that provides the standardised public API on top of the internal vines.
class _VineBase:

    _inner_cls = None

    def __init__(self, n_vars, **kwargs):
        self.n_vars = n_vars
        self._kwargs = kwargs
        self._vine = None
        self._n_obs = None
        self.pair_copulas = None
        self._trees = None

    # ── fit ──────────────────────────────────────────────────────────────────

    def fit(self, u: np.ndarray) -> "_VineBase":
        u = np.asarray(u, dtype=np.float64)
        if u.ndim != 2:
            raise ValueError(f"u must be 2-D, got shape {u.shape}")
        n, d = u.shape
        if d != self.n_vars:
            raise ValueError(
                f"n_vars={self.n_vars} but data has {d} columns"
            )
        if not (np.all(u > 0) and np.all(u < 1)):
            raise ValueError("All values in u must be strictly in (0, 1)")

        self._n_obs = n
        self._vine = self._build_inner(d)
        self._vine.fit(u)

        if hasattr(self._vine, 'copulas'):
            self.pair_copulas = self._vine.copulas
            self._trees = self._vine.copulas
        return self

    def _build_inner(self, d):
        raise NotImplementedError

    # ── simulate ─────────────────────────────────────────────────────────────

    def simulate(self, n: int, seed=None) -> np.ndarray:
        out = self._vine.simulate(n, seed=seed)
        return np.clip(out, 1e-7, 1 - 1e-7)

    # ── loglik ───────────────────────────────────────────────────────────────

    def loglik(self, u: np.ndarray) -> float:
        return float(self._vine.loglikelihood())

    # ── aic / bic ────────────────────────────────────────────────────────────

    @property
    def aic(self) -> float:
        return float(self._vine.aic())

    @property
    def bic(self) -> float:
        n = self._n_obs or 1
        return float(self._vine.bic(n))

    # ── pit  (Rosenblatt / probability integral transform) ───────────────────

    # Rosenblatt transform: applies sequential h-functions to map u → [0,1]^d.
    def pit(self, u: np.ndarray) -> np.ndarray:
        u = np.asarray(u, dtype=np.float64)
        u = np.clip(u, 1e-7, 1 - 1e-7)
        n, d = u.shape
        out = np.empty_like(u)
        out[:, 0] = u[:, 0]
        v = u.copy()
        for j in range(1, d):
            pit_j = self._pit_column(v, j)
            out[:, j] = pit_j
            v[:, j] = pit_j
        return np.clip(out, 1e-7, 1 - 1e-7)

    # Compute the PIT for column j given columns 0..j-1 already transformed.
    def _pit_column(self, v: np.ndarray, j: int) -> np.ndarray:
        vine = self._vine
        if hasattr(vine, 'copulas') and vine.copulas:
            cop = None
            for key in [(0, j - 1, j), (0, 0, j), (0, j, j - 1)]:
                if key in vine.copulas:
                    cop = vine.copulas[key]
                    break
            if cop is not None:
                return cop.h_function(v[:, j], v[:, j - 1], deriv=1)
        if hasattr(vine, '_pair_ll') and vine._pair_ll:
            for (fni, fnj), (u_fit, v_fit, cop) in vine._pair_ll.items():
                ni = next(iter(fni)); nj = next(iter(fnj))
                if len(fni) == 1 and len(fnj) == 1:
                    if nj == j or ni == j:
                        cond = j - 1 if j > 0 else j + 1
                        return cop.h_function(v[:, j], v[:, cond], deriv=1)
        return v[:, j].copy()


# C-Vine copula (star-graph structure at each tree).
class CVine(_VineBase):

    def __init__(self, n_vars: int, pair_copulas=None):
        super().__init__(n_vars)
        self._pair_copulas_init = pair_copulas

    def _build_inner(self, d):
        return CVineCopula(n_dim=d)


# D-Vine copula (path-graph structure at each tree).
class DVine(_VineBase):

    def __init__(self, n_vars: int, pair_copulas=None):
        super().__init__(n_vars)
        self._pair_copulas_init = pair_copulas

    def _build_inner(self, d):
        return DVineCopula(n_dim=d)


# R-Vine copula (general vine with MST-selected structure).
class RVine(_VineBase):

    def __init__(self, n_vars: int, matrix=None, pair_copulas=None):
        super().__init__(n_vars)
        self.matrix = matrix
        self._pair_copulas_init = pair_copulas

    def _build_inner(self, d):
        return RVineCopula(n_dim=d)

    def fit(self, u: np.ndarray) -> "RVine":
        super().fit(u)
        d = self.n_vars
        sim_order = getattr(self._vine, '_sim_order', list(range(d)))
        if self.matrix is None:
            mat = np.zeros((d, d), dtype=int)
            order = list(sim_order)
            for i in range(d):
                root = order[d - 1 - i]
                for j in range(i, d):
                    mat[i, j] = root
            self.matrix = mat
        return self


# Ajusta CVine, DVine e/ou RVine em *u* e retorna a com menor
def select_vine(
    u: np.ndarray,
    vine_types: tuple = ('C', 'D', 'R'),
    criterion: str = 'mbicv',
) -> _VineBase:
    if criterion not in ('aic', 'bic', 'mbicv'):
        raise ValueError(f"criterion deve ser 'aic', 'bic' ou 'mbicv', recebido '{criterion}'")

    _map = {'C': CVine, 'D': DVine, 'R': RVine}
    d = np.asarray(u).shape[1]

    best_vine = None
    best_score = np.inf

    for vt in vine_types:
        if vt not in _map:
            raise ValueError(f"Tipo de vine desconhecido '{vt}'. Escolha entre {list(_map)}")
        vine = _map[vt](n_vars=d).fit(u)
        if criterion == 'mbicv':
            score = vine.bic
        elif criterion == 'aic':
            score = vine.aic
        else:
            score = vine.bic
        if score < best_score:
            best_score = score
            best_vine = vine

    return best_vine


# ── __main__ ──────────────────────────────────────────────────────────────────
# Carrega resíduos padronizados via GARCHFitter.fit_from_loader e
def _load_pseudo_obs_from_loader(start_date="2020-01-01", model_type="gjr", dist="t"):
    import sys, importlib
    from scipy.stats import rankdata

    _fit_from_loader = None
    for _mod in ('src.marginals.garch', 'marginals.garch', 'garch'):
        try:
            _fit_from_loader = importlib.import_module(_mod).fit_from_loader
            break
        except ImportError:
            pass
    if _fit_from_loader is None:
        _here = __import__('pathlib').Path(__file__).resolve().parent
        for _p in (_here.parent, _here.parent.parent):
            if str(_p) not in sys.path:
                sys.path.insert(0, str(_p))
        for _mod in ('src.marginals.garch', 'marginals.garch', 'garch'):
            try:
                _fit_from_loader = importlib.import_module(_mod).fit_from_loader
                break
            except ImportError:
                pass
    if _fit_from_loader is None:
        raise ImportError(
            "Não foi possível importar fit_from_loader do garch.py. "
            "Execute a partir da raiz do projeto ou com "
            "`python -m src.copulas.vine_copulas`."
        )

    std_resid, _, garch_results = _fit_from_loader(
        start_date=start_date, model_type=model_type, dist=dist,
    )
    tickers = list(std_resid.columns)
    d = len(tickers)
    Z = std_resid.dropna().values
    T = len(Z)
    U = np.column_stack([(rankdata(Z[:, j]) - 0.5) / T for j in range(d)])
    U = np.clip(U, 1e-4, 1 - 1e-4)
    return U, tickers, garch_results


PRODUCTION_FAMILIES = ['gaussian', 't', 'gumbel', 'frank', 'clayton',
                       'gumbel_180', 'clayton_180', 'joe', 'joe_180', 'bb1', 'bb1_180']
PRODUCTION_FORCED_EDGES = [(1, 4), (3, 4)]
PRODUCTION_TAIL_Q = 0.15
PRODUCTION_TAIL_ERR_TOL = 0.15


# Ajusta a única vine usada em produção (config v19a, vencedora da
def fit_production_vine(U=None, tickers=None, verbose=True):
    if U is None:
        U, tickers, _ = _load_pseudo_obs_from_loader()
    d = U.shape[1]
    if verbose:
        print(f"Ajustando vine de produção (v19a)  d={d}  T={len(U)}")
    rv = RVineCopula(n_dim=d)
    rv.fit(U, families=PRODUCTION_FAMILIES, forced_edges=PRODUCTION_FORCED_EDGES,
           tail_err_tol=PRODUCTION_TAIL_ERR_TOL, tail_q=PRODUCTION_TAIL_Q)
    if verbose:
        print(f"LL={rv.loglikelihood():.2f}  AIC={rv.aic():.2f}  "
              f"sim_order={rv._sim_order}")
    return rv, U, tickers


# Ferramenta de DIAGNÓSTICO: reproduz a comparação entre as 4 variantes
def compare_configs(U=None, tickers=None):
    from scipy.stats import kendalltau as _kt

    if U is None:
        print("=" * 60)
        print("Carregando dados reais via fit_from_loader …")
        print("=" * 60)
        U, tickers, _ = _load_pseudo_obs_from_loader()

    d = U.shape[1]
    print(f"\nAtivos : {tickers}")
    print(f"Obs    : {len(U)}  |  Dimensão: {d}")
    print(f"Pseudo-obs shape: {U.shape}\n")

    logging.getLogger(__name__).setLevel(logging.DEBUG)

    families_v18 = ['gaussian', 't', 'gumbel', 'frank', 'clayton', 'gumbel_180', 'clayton_180']
    families_v19 = ['gaussian', 't', 'gumbel', 'frank', 'clayton', 'gumbel_180', 'clayton_180',
                    'joe', 'joe_180', 'bb1', 'bb1_180']
    n_sim = len(U)

    print("=" * 60)
    print("BASELINE — R-vine v18, sem forced_edges, mBICV puro")
    print("=" * 60)
    rv_base = RVineCopula(n_dim=d)
    rv_base.fit(U, families=families_v18, tail_err_tol=0.15, tail_q=0.10)
    sim_base = rv_base.simulate(n_sim, seed=42)

    print("\n" + "=" * 60)
    print("v18 — forced=[(1,4),(3,4)], tail_q=0.10")
    print("=" * 60)
    rv_v18 = RVineCopula(n_dim=d)
    rv_v18.fit(U, families=families_v18, forced_edges=[(1,4),(3,4)],
               tail_err_tol=0.15, tail_q=0.10)
    sim_v18 = rv_v18.simulate(n_sim, seed=42)

    print("\n" + "=" * 60)
    print("v19a — Joe+BB1, forced=[(1,4),(3,4)], tail_q=0.15")
    print("=" * 60)
    rv_v19a = RVineCopula(n_dim=d)
    rv_v19a.fit(U, families=families_v19, forced_edges=[(1,4),(3,4)],
                tail_err_tol=0.15, tail_q=0.15)
    sim_v19a = rv_v19a.simulate(n_sim, seed=42)

    print("\n" + "=" * 60)
    print("v19b — Joe+BB1, forced=[(1,4),(3,4),(0,3),(1,2)], tail_q=0.15")
    print("=" * 60)
    rv_v19b = RVineCopula(n_dim=d)
    rv_v19b.fit(U, families=families_v19,
                forced_edges=[(1,4),(3,4),(0,3),(1,2)],
                tail_err_tol=0.15, tail_q=0.15)
    sim_v19b = rv_v19b.simulate(n_sim, seed=42)

    logging.getLogger(__name__).setLevel(logging.INFO)

    print("\n" + "=" * 60)
    print("Kendall τ — dados reais vs simulados")
    print("=" * 60)
    pairs = [(i, j) for i in range(d) for j in range(i+1, d)]
    hdr = f"{'Par':<18}{'Orig':>7}{'base':>8}{'v18':>8}{'v19a':>8}{'v19b':>8}"
    hdr += f"  | {'Δbase':>7}{'Δv18':>7}{'Δv19a':>7}{'Δv19b':>7}"
    print(hdr)
    print("-" * len(hdr))
    for i, j in pairs:
        ti = tickers[i]; tj = tickers[j]
        t_o   = abs(float(_kt(U[:,i],           U[:,j])[0]))
        t_b   = abs(float(_kt(sim_base[:,i],    sim_base[:,j])[0]))
        t_18  = abs(float(_kt(sim_v18[:,i],     sim_v18[:,j])[0]))
        t_19a = abs(float(_kt(sim_v19a[:,i],    sim_v19a[:,j])[0]))
        t_19b = abs(float(_kt(sim_v19b[:,i],    sim_v19b[:,j])[0]))
        label = f"({ti},{tj})"
        print(f"{label:<18}{t_o:>7.3f}{t_b:>8.3f}{t_18:>8.3f}{t_19a:>8.3f}{t_19b:>8.3f}"
              f"  | {abs(t_o-t_b):>7.3f}{abs(t_o-t_18):>7.3f}"
              f"{abs(t_o-t_19a):>7.3f}{abs(t_o-t_19b):>7.3f}")

    best_sim = sim_v19b
    print("\n" + "=" * 60)
    print("Dependência de cauda — empírica vs simulada (v19b)")
    print("=" * 60)
    print(f"{'Par':<18}{'λL_emp':>8}{'λL_sim':>8}{'λU_emp':>8}{'λU_sim':>8}  {'ΔλL':>6}{'ΔλU':>6}")
    print("-" * 62)
    for i, j in pairs:
        ti = tickers[i]; tj = tickers[j]
        lL_e, lU_e = _empirical_tail(U[:,i],       U[:,j],       q=0.05)
        lL_s, lU_s = _empirical_tail(best_sim[:,i], best_sim[:,j], q=0.05)
        label = f"({ti},{tj})"
        print(f"{label:<18}{lL_e:>8.3f}{lL_s:>8.3f}{lU_e:>8.3f}{lU_s:>8.3f}"
              f"  {abs(lL_e-lL_s):>6.3f}{abs(lU_e-lU_s):>6.3f}")

    print("\n" + "=" * 60)
    print("MÉTRICAS AGREGADAS")
    print("=" * 60)
    tlo_o, thi_o = _tail_dependence_empirical(U, 0.05, 0.95)
    rows = []
    for label, rv, sim in [
        ("baseline",  rv_base,   sim_base),
        ("v18",       rv_v18,    sim_v18),
        ("v19a",      rv_v19a,   sim_v19a),
        ("v19b",      rv_v19b,   sim_v19b),
    ]:
        tlo_s, thi_s = _tail_dependence_empirical(sim, 0.05, 0.95)
        tau_o = np.mean([abs(float(_kt(U[:,i],   U[:,j])[0]))   for i,j in pairs])
        tau_s = np.mean([abs(float(_kt(sim[:,i], sim[:,j])[0])) for i,j in pairs])
        rows.append(dict(
            config       = label,
            ll           = round(rv.loglikelihood(), 2),
            aic          = round(rv.aic(), 2),
            tau_diff     = round(abs(tau_o - tau_s), 4),
            tail_lo_diff = round(abs(tlo_o - tlo_s), 4),
            tail_hi_diff = round(abs(thi_o - thi_s), 4),
        ))
    df = pd.DataFrame(rows).set_index('config')
    print(df.to_string())
    print("\nTeste concluído.")
    return dict(baseline=rv_base, v18=rv_v18, v19a=rv_v19a, v19b=rv_v19b)


if __name__ == '__main__':
    import sys
    if '--compare' in sys.argv:
        compare_configs()
    else:
        print("=" * 60)
        print("Carregando dados reais via fit_from_loader …")
        print("=" * 60)
        U, tickers, _ = _load_pseudo_obs_from_loader()
        print(f"\nAtivos : {tickers}")
        print(f"Obs    : {len(U)}  |  Dimensão: {U.shape[1]}")
        print(f"Pseudo-obs shape: {U.shape}\n")

        print("=" * 60)
        print("VINE DE PRODUÇÃO (v19a)")
        print("=" * 60)
        rv, U, tickers = fit_production_vine(U, tickers)

        sim = rv.simulate(len(U), seed=42)
        from scipy.stats import kendalltau as _kt
        d = U.shape[1]
        pairs = [(i, j) for i in range(d) for j in range(i+1, d)]
        print("\nKendall τ — dados reais vs simulados")
        print("-" * 40)
        for i, j in pairs:
            t_o = float(_kt(U[:,i], U[:,j])[0])
            t_s = float(_kt(sim[:,i], sim[:,j])[0])
            print(f"({tickers[i]},{tickers[j]}): orig={t_o:.3f}  sim={t_s:.3f}  Δ={abs(t_o-t_s):.3f}")

        print(f"\nLL={rv.loglikelihood():.2f}  AIC={rv.aic():.2f}")
        print("\nUse --compare para reproduzir a comparação completa entre as 4 variantes testadas.")
