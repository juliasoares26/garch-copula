
import logging
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


# Métricas de retorno e risco

# Sharpe Ratio anualizado. rf em escala diária.
def sharpe_ratio(
    returns: Union[pd.Series, np.ndarray],
    rf: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    r = np.asarray(returns).flatten()
    r = r[~np.isnan(r)]
    excess = r - rf
    if excess.std(ddof=1) < 1e-12:
        return np.nan
    return float(excess.mean() / excess.std(ddof=1) * np.sqrt(periods_per_year))


# Sortino Ratio anualizado.
def sortino_ratio(
    returns: Union[pd.Series, np.ndarray],
    rf: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    r = np.asarray(returns).flatten()
    r = r[~np.isnan(r)]
    excess = r - rf
    downside = excess[excess < 0]
    if len(downside) == 0:
        return np.inf
    downside_dev = np.sqrt((downside ** 2).mean() * periods_per_year)
    if downside_dev < 1e-12:
        return np.nan
    return float(excess.mean() * periods_per_year / downside_dev)


# Máximo drawdown e duração em dias úteis.
def max_drawdown(
    returns: Union[pd.Series, np.ndarray],
) -> Dict[str, float]:
    r = np.asarray(returns).flatten()
    r = r[~np.isnan(r)]
    cum = np.cumprod(1 + r)
    peak = np.maximum.accumulate(cum)
    dd = (cum - peak) / peak

    trough_idx = int(dd.argmin())

    if trough_idx > 0:
        peak_idx = int(np.argmax(cum[:trough_idx]))
    else:
        peak_idx = 0

    duration = trough_idx - peak_idx

    recovery_val = peak[trough_idx]
    post = np.where(cum[trough_idx:] >= recovery_val)[0]
    recovery = int(post[0]) if len(post) > 0 else np.nan

    return {
        'max_dd':     float(dd.min()),
        'peak_idx':   peak_idx,
        'trough_idx': trough_idx,
        'duration':   duration,
        'recovery':   recovery,
    }


# Calmar Ratio = retorno anualizado / |max drawdown|.
def calmar_ratio(
    returns: Union[pd.Series, np.ndarray],
    periods_per_year: int = 252,
) -> float:
    r = np.asarray(returns).flatten()
    r = r[~np.isnan(r)]
    cum = float(np.prod(1 + r))
    ann_ret = cum ** (periods_per_year / len(r)) - 1
    mdd = abs(max_drawdown(r)['max_dd'])
    if mdd < 1e-10:
        return np.inf
    return float(ann_ret / mdd)


# Information Ratio anualizado = alpha / tracking error.
def information_ratio(
    returns: Union[pd.Series, np.ndarray],
    benchmark: Union[pd.Series, np.ndarray],
    periods_per_year: int = 252,
) -> float:
    r = np.asarray(returns).flatten()
    b = np.asarray(benchmark).flatten()
    n = min(len(r), len(b))
    active = r[:n] - b[:n]
    active = active[~np.isnan(active)]
    te = active.std(ddof=1)
    if te < 1e-12:
        return np.nan
    return float(active.mean() / te * np.sqrt(periods_per_year))


# VaR histórico (positivo = perda). Ex: retorna 0.025 para -2.5% loss.
def historical_var(
    returns: Union[pd.Series, np.ndarray],
    confidence_level: float = 0.99,
) -> float:
    r = np.asarray(returns).flatten()
    r = r[~np.isnan(r)]
    return float(-np.quantile(r, 1 - confidence_level))


# Expected Shortfall histórico (positivo = perda).
def historical_es(
    returns: Union[pd.Series, np.ndarray],
    confidence_level: float = 0.99,
) -> float:
    r = np.asarray(returns).flatten()
    r = r[~np.isnan(r)]
    var = historical_var(r, confidence_level)
    tail = r[r <= -var]
    if len(tail) == 0:
        return var
    return float(-tail.mean())


# Omega Ratio = E[max(r-L,0)] / E[max(L-r,0)].
def omega_ratio(
    returns: Union[pd.Series, np.ndarray],
    threshold: float = 0.0,
) -> float:
    r = np.asarray(returns).flatten()
    r = r[~np.isnan(r)]
    gains  = (r - threshold)[r > threshold].sum()
    losses = (threshold - r)[r <= threshold].sum()
    if losses < 1e-12:
        return np.inf
    return float(gains / losses)


# Retorno anualizado geométrico.
def annualized_return(
    returns: Union[pd.Series, np.ndarray],
    periods_per_year: int = 252,
) -> float:
    r = np.asarray(returns).flatten()
    r = r[~np.isnan(r)]
    cum = float(np.prod(1 + r))
    return float(cum ** (periods_per_year / len(r)) - 1)


# Volatilidade anualizada.
def annualized_volatility(
    returns: Union[pd.Series, np.ndarray],
    periods_per_year: int = 252,
) -> float:
    r = np.asarray(returns).flatten()
    r = r[~np.isnan(r)]
    return float(r.std(ddof=1) * np.sqrt(periods_per_year))


# Diagnóstico de outliers e fat tails nos retornos.
def outlier_diagnostics(
    returns: Union[pd.Series, np.ndarray],
    z_threshold: float = 4.0,
) -> Dict:
    r = np.asarray(returns).flatten()
    r = r[~np.isnan(r)]

    mu, sigma = r.mean(), r.std(ddof=1)
    z_scores = np.abs((r - mu) / (sigma + 1e-12))
    outlier_mask = z_scores > z_threshold

    n_out = int(outlier_mask.sum())
    r_clean = r[~outlier_mask]

    skew_full   = float(stats.skew(r))
    kurt_full   = float(stats.kurtosis(r))
    skew_trim   = float(stats.skew(r_clean)) if len(r_clean) > 10 else np.nan
    kurt_trim   = float(stats.kurtosis(r_clean)) if len(r_clean) > 10 else np.nan

    outlier_dates = None
    if isinstance(returns, pd.Series) and isinstance(returns.index, pd.DatetimeIndex):
        idx = returns.dropna().index
        if len(idx) == len(r):
            outlier_dates = idx[outlier_mask].tolist()

    if kurt_full > 10 and kurt_trim < 3:
        interpretation = (
            f"Kurtosis elevada ({kurt_full:.1f}) é inteiramente devida a {n_out} outlier(s) "
            f"(|z| > {z_threshold}). Sem outliers, kurtosis cai para {kurt_trim:.1f}. "
            f"Verifique se esses eventos são dados corretos ou erros de feed."
        )
    elif kurt_full > 10 and kurt_trim >= 3:
        interpretation = (
            f"Kurtosis elevada ({kurt_full:.1f}) é parcialmente estrutural: "
            f"mesmo sem os {n_out} outlier(s) mais extremos, kurtosis = {kurt_trim:.1f} > 3. "
            f"Fat tails persistem — EVT/GPD é o modelo correto para essas caudas."
        )
    else:
        interpretation = (
            f"Kurtosis ({kurt_full:.1f}) compatível com distribuição leptocúrtica moderada. "
            f"{n_out} outlier(s) identificados acima de {z_threshold}σ."
        )

    return {
        "n_outliers":    n_out,
        "outlier_pct":   round(100 * n_out / len(r), 2),
        "skew_full":     round(skew_full, 4),
        "skew_trimmed":  round(skew_trim, 4) if not np.isnan(skew_trim) else np.nan,
        "kurt_full":     round(kurt_full, 4),
        "kurt_trimmed":  round(kurt_trim, 4) if not np.isnan(kurt_trim) else np.nan,
        "max_return":    round(float(r.max()), 6),
        "min_return":    round(float(r.min()), 6),
        "max_zscore":    round(float(z_scores.max()), 2),
        "outlier_dates": outlier_dates,
        "interpretation": interpretation,
    }


# Retorna DataFrame com data, retorno e z-score de cada outlier |z| > z_threshold.
def find_outlier_dates(
    returns: pd.Series,
    z_threshold: float = 4.0,
) -> pd.DataFrame:
    if not isinstance(returns, pd.Series):
        raise TypeError("returns deve ser pd.Series com DatetimeIndex.")
    r = returns.dropna()
    mu, sigma = r.mean(), r.std(ddof=1)
    z = (r - mu) / (sigma + 1e-12)
    mask = z.abs() > z_threshold
    df = pd.DataFrame({
        "date":   r.index[mask],
        "return": r.values[mask],
        "z_score": z.values[mask],
    }).sort_values("z_score").reset_index(drop=True)
    return df


# VaR Rolling — separação correta in-sample / out-of-sample

# Gera previsões de VaR histórico rolling com separação correta
def rolling_var(
    returns: Union[pd.Series, np.ndarray],
    window: int = 252,
    confidence_level: float = 0.99,
) -> pd.Series:
    s = pd.Series(np.asarray(returns).flatten())
    var_rolling = (
        s.rolling(window)
         .apply(lambda x: historical_var(x, confidence_level), raw=True)
         .shift(1)
    )
    return var_rolling


# ES via Cornish-Fisher: combina ES histórico com ajuste de skewness/kurtosis.
def _es_cornish_fisher(x: np.ndarray, confidence_level: float) -> float:
    p = 1 - confidence_level
    mu    = float(np.mean(x))
    sigma = float(np.std(x, ddof=1))
    if sigma < 1e-12:
        return float(-np.quantile(x, p))

    S = float(np.clip(stats.skew(x),     -3.0, 3.0))
    K = float(np.clip(stats.kurtosis(x), -2.0, 8.0))

    z0   = float(stats.norm.ppf(p))
    z_cf = (z0
            + (z0**2 - 1) * S / 6
            + (z0**3 - 3*z0) * K / 24
            - (2*z0**3 - 5*z0) * S**2 / 36)

    phi_z = float(stats.norm.pdf(z_cf))
    es_z  = -phi_z / p

    es_cf = -(mu + sigma * es_z)

    tail = x[x <= np.quantile(x, p)]
    es_hist = float(-np.mean(tail)) if len(tail) > 0 else es_cf

    if not np.isfinite(es_cf) or es_cf <= 0:
        return es_hist

    n      = len(x)
    w_hist = float(np.clip(30 / n, 0.20, 0.60))
    return float((1 - w_hist) * es_cf + w_hist * es_hist)


# ES rolling estabilizado via Cornish-Fisher + blend histórico.
def rolling_es(
    returns: Union[pd.Series, np.ndarray],
    window: int = 252,
    confidence_level: float = 0.99,
) -> pd.Series:
    s = pd.Series(np.asarray(returns).flatten())
    es_rolling = (
        s.rolling(window)
         .apply(lambda x: _es_cornish_fisher(x, confidence_level), raw=True)
         .shift(1)
    )
    return es_rolling


# Backtesting de VaR — Kupiec e Christoffersen

# Teste de Kupiec (1995) — Proportion of Failures (POF).
def kupiec_pof(
    returns: Union[pd.Series, np.ndarray],
    var_forecasts: Union[pd.Series, np.ndarray],
    confidence_level: float = 0.99,
    alpha: float = 0.05,
) -> Dict:
    r   = np.asarray(returns).flatten()
    var = np.asarray(var_forecasts).flatten()
    n   = min(len(r), len(var))
    r, var = r[:n], var[:n]

    mask = ~(np.isnan(r) | np.isnan(var))
    r, var = r[mask], var[mask]
    n = len(r)

    p = 1 - confidence_level
    v = int((r < -var).sum())

    if v == 0 or v == n:
        return {
            'n': n, 'violations': v,
            'obs_rate': v / n, 'exp_rate': p,
            'lr_stat': 0.0, 'p_value': 1.0,
            'reject_h0': False,
            'confidence_level': confidence_level,
        }

    p_hat = v / n
    lr = -2 * (
        v * np.log(p / p_hat)
        + (n - v) * np.log((1 - p) / (1 - p_hat))
    )
    p_val = float(1 - stats.chi2.cdf(lr, df=1))

    return {
        'n': n,
        'violations': v,
        'obs_rate': round(p_hat, 6),
        'exp_rate': p,
        'lr_stat': round(float(lr), 4),
        'p_value': round(p_val, 6),
        'reject_h0': p_val < alpha,
        'confidence_level': confidence_level,
    }


# Teste de Christoffersen (1998) — independência das violações.
def christoffersen_test(
    returns: Union[pd.Series, np.ndarray],
    var_forecasts: Union[pd.Series, np.ndarray],
    confidence_level: float = 0.99,
    alpha: float = 0.05,
) -> Dict:
    r   = np.asarray(returns).flatten()
    var = np.asarray(var_forecasts).flatten()
    n   = min(len(r), len(var))
    r, var = r[:n], var[:n]

    mask = ~(np.isnan(r) | np.isnan(var))
    r, var = r[mask], var[mask]

    ind = (r < -var).astype(int)
    n_total = len(ind)

    n00 = int(((ind[:-1] == 0) & (ind[1:] == 0)).sum())
    n01 = int(((ind[:-1] == 0) & (ind[1:] == 1)).sum())
    n10 = int(((ind[:-1] == 1) & (ind[1:] == 0)).sum())
    n11 = int(((ind[:-1] == 1) & (ind[1:] == 1)).sum())

    pi01 = n01 / (n00 + n01) if (n00 + n01) > 0 else 0.0
    pi11 = n11 / (n10 + n11) if (n10 + n11) > 0 else 0.0
    pi   = (n01 + n11) / n_total if n_total > 0 else 0.0

    eps = 1e-10
    pi_c   = float(np.clip(pi,   eps, 1 - eps))
    pi01_c = float(np.clip(pi01, eps, 1 - eps))
    pi11_c = float(np.clip(pi11, eps, 1 - eps))

    if abs(pi01_c - pi11_c) < 1e-8:
        lr_ind = 0.0
    else:
        lr_ind = -2 * (
            (n00 + n10) * np.log(1 - pi_c)
            + (n01 + n11) * np.log(pi_c)
            - n00 * np.log(1 - pi01_c)
            - n01 * np.log(pi01_c)
            - n10 * np.log(1 - pi11_c)
            - n11 * np.log(pi11_c)
        )

    lr_ind = float(max(lr_ind, 0.0))

    kupiec = kupiec_pof(r, var, confidence_level, alpha)
    lr_uc  = kupiec['lr_stat']
    lr_cc  = lr_uc + lr_ind

    p_ind = float(1 - stats.chi2.cdf(lr_ind, df=1))
    p_cc  = float(1 - stats.chi2.cdf(lr_cc,  df=2))

    return {
        'n': n_total,
        'n00': n00, 'n01': n01, 'n10': n10, 'n11': n11,
        'pi':   round(pi,   6),
        'pi01': round(pi01, 6),
        'pi11': round(pi11, 6),
        'lr_ind':     round(lr_ind, 4),
        'lr_cc':      round(lr_cc,  4),
        'p_ind':      round(p_ind,  6),
        'p_cc':       round(p_cc,   6),
        'reject_ind': p_ind < alpha,
        'reject_cc':  p_cc  < alpha,
        'confidence_level': confidence_level,
    }


# Relatório de backtesting consolidado (VaR)

# Gera tabela de backtesting com Kupiec + Christoffersen para múltiplos níveis.
def backtesting_report(
    returns: Union[pd.Series, np.ndarray],
    var_forecasts: Union[pd.Series, np.ndarray],
    confidence_levels: List[float] = None,
    alpha: float = 0.05,
) -> pd.DataFrame:
    if confidence_levels is None:
        confidence_levels = [0.95, 0.99]

    rows = []
    for cl in confidence_levels:
        k = kupiec_pof(returns, var_forecasts, cl, alpha)
        c = christoffersen_test(returns, var_forecasts, cl, alpha)
        rows.append({
            'Nível (%)':          round(cl * 100, 1),
            'Taxa Observada (%)': round(k['obs_rate'] * 100, 3),
            'Taxa Esperada (%)':  round(k['exp_rate'] * 100, 3),
            'Violações':          k['violations'],
            'N':                  k['n'],
            'Kupiec LR':          k['lr_stat'],
            'Kupiec p':           k['p_value'],
            'Kupiec OK':          not k['reject_h0'],
            'Chris p (ind)':      c['p_ind'],
            'Chris p (cc)':       c['p_cc'],
            'Chris OK':           not c['reject_cc'],
        })
    return pd.DataFrame(rows)


# Integração com EVTWalkForward — backtesting multi-modelo

# Gera tabela de backtesting consumindo diretamente a saída de
def backtesting_report_from_walkforward(
    wf_results: pd.DataFrame,
    confidence_levels: List[float] = None,
    alpha: float = 0.05,
) -> pd.DataFrame:
    if confidence_levels is None:
        confidence_levels = [0.95, 0.99]

    realized = wf_results['return'].values
    rows = []

    for cl in confidence_levels:
        cl_str  = f"{int(cl * 100)}"
        var_col = f"var_{cl_str}"
        es_col  = f"es_{cl_str}"

        if var_col not in wf_results.columns:
            logger.warning(f"Coluna '{var_col}' não encontrada em wf_results. Pulando.")
            continue

        var_f = wf_results[var_col].values

        k = kupiec_pof(realized, var_f, cl, alpha)
        c = christoffersen_test(realized, var_f, cl, alpha)

        es_adequate = None
        es_mf_p     = None
        es_az1_p    = None
        if es_col in wf_results.columns:
            try:
                try:
                    from backtesting.backtesting_walking_forward import ESBacktest
                except ImportError:
                    from backtesting_walking_forward import ESBacktest
                es_bt   = ESBacktest(alpha=alpha)
                es_f    = wf_results[es_col].values
                mask    = ~(np.isnan(realized) | np.isnan(var_f) | np.isnan(es_f))
                es_res  = es_bt.run(
                    realized[mask], var_f[mask], es_f[mask],
                    confidence_level=cl,
                )
                es_adequate = es_res.get('es_model_adequate')
                es_mf_p     = es_res.get('mf_pvalue')
                es_az1_p    = es_res.get('az1_pvalue')
            except ImportError:
                logger.warning(
                    "ESBacktest não importado. Verifique src/backtesting/backtesting_walking_forward.py. "
                    "Métricas de ES omitidas."
                )

        row = {
            'Nível (%)':          round(cl * 100, 1),
            'Taxa Observada (%)': round(k['obs_rate'] * 100, 3),
            'Taxa Esperada (%)':  round(k['exp_rate'] * 100, 3),
            'Violações':          k['violations'],
            'N':                  k['n'],
            'Kupiec LR':          k['lr_stat'],
            'Kupiec p':           k['p_value'],
            'Kupiec OK':          not k['reject_h0'],
            'Chris p (ind)':      c['p_ind'],
            'Chris p (cc)':       c['p_cc'],
            'Chris OK':           not c['reject_cc'],
        }
        if es_adequate is not None:
            row['ES MF p']    = es_mf_p
            row['ES AZ1 p']   = es_az1_p
            row['ES OK']      = es_adequate

        rows.append(row)

    return pd.DataFrame(rows)


# Compara múltiplos modelos de VaR side-by-side quando wf_results contém
def compare_models_backtest(
    wf_results: pd.DataFrame,
    confidence_levels: List[float] = None,
    alpha: float = 0.05,
) -> pd.DataFrame:
    if confidence_levels is None:
        confidence_levels = [0.95, 0.99]

    realized = wf_results['return'].values
    rows = []

    for col in wf_results.columns:
        if '_var_' not in col:
            continue
        parts = col.split('_var_')
        if len(parts) != 2:
            continue
        model = parts[0]
        try:
            cl = float(parts[1]) / 100 if float(parts[1]) > 1 else float(parts[1])
        except ValueError:
            continue
        if cl not in confidence_levels:
            continue

        var_f = wf_results[col].values
        k = kupiec_pof(realized, var_f, cl, alpha)
        c = christoffersen_test(realized, var_f, cl, alpha)

        rows.append({
            'Modelo':             model,
            'Nível (%)':          round(cl * 100, 1),
            'Taxa Observada (%)': round(k['obs_rate'] * 100, 3),
            'Taxa Esperada (%)':  round(k['exp_rate'] * 100, 3),
            'Violações':          k['violations'],
            'N':                  k['n'],
            'Kupiec p':           k['p_value'],
            'Kupiec OK':          not k['reject_h0'],
            'Chris OK':           not c['reject_cc'],
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(['Nível (%)', 'Kupiec p'], ascending=[True, False])
    return df


# Tabela de performance consolidada (múltiplas estratégias)

# Gera tabela de performance para múltiplas estratégias.
def performance_summary(
    returns_dict: Dict[str, Union[pd.Series, np.ndarray]],
    benchmark: Optional[Union[pd.Series, np.ndarray]] = None,
    rf: float = 0.0,
    periods_per_year: int = 252,
) -> pd.DataFrame:
    rows = []
    for label, r in returns_dict.items():
        arr = np.asarray(r).flatten()
        arr = arr[~np.isnan(arr)]

        mdd_info = max_drawdown(arr)

        if isinstance(r, pd.Series) and isinstance(r.index, pd.DatetimeIndex):
            periodo = f"{r.index[0].date()} → {r.index[-1].date()}"
        else:
            periodo = f"{len(arr)} dias"

        row = {
            'Estratégia':           label,
            'N (dias)':             len(arr),
            'Período':              periodo,
            'Ann. Return (%)':      round(annualized_return(arr, periods_per_year) * 100, 2),
            'Ann. Vol (%)':         round(annualized_volatility(arr, periods_per_year) * 100, 2),
            'Sharpe':               round(sharpe_ratio(arr, rf, periods_per_year), 4),
            'Sortino':              round(sortino_ratio(arr, rf, periods_per_year), 4),
            'Calmar':               round(calmar_ratio(arr, periods_per_year), 4),
            'Max DD (%)':           round(mdd_info['max_dd'] * 100, 2),
            'Max DD Duration':      int(mdd_info['duration']),
            'VaR 99% (%)':          round(historical_var(arr, 0.99) * 100, 3),
            'ES 99% (%)':           round(historical_es(arr, 0.99) * 100, 3),
            'VaR 95% (%)':          round(historical_var(arr, 0.95) * 100, 3),
            'ES 95% (%)':           round(historical_es(arr, 0.95) * 100, 3),
            'Skewness':             round(float(stats.skew(arr)), 4),
            'Excess Kurtosis':      round(float(stats.kurtosis(arr)), 4),
            'Omega Ratio':          round(omega_ratio(arr, rf), 4),
        }
        if benchmark is not None:
            row['IR'] = round(information_ratio(arr, benchmark, periods_per_year), 4)

        kurt_val = float(stats.kurtosis(arr))
        skew_val = float(stats.skew(arr))
        if kurt_val > 10 or abs(skew_val) > 1.5:
            diag = outlier_diagnostics(r if isinstance(r, pd.Series) else arr)
            row['Skew (sem outliers)']  = diag['skew_trimmed']
            row['Kurt (sem outliers)']  = diag['kurt_trimmed']
            row['N Outliers (|z|>4σ)']  = diag['n_outliers']
            logger.warning(
                f"[{label}] Alta kurtosis ({kurt_val:.1f}) / skewness ({skew_val:.2f}). "
                f"{diag['interpretation']}"
            )
            if diag['outlier_dates']:
                logger.warning(
                    f"[{label}] Datas dos outliers: "
                    + ", ".join(str(d.date() if hasattr(d, 'date') else d)
                                for d in diag['outlier_dates'])
                )

        rows.append(row)

    df = pd.DataFrame(rows).set_index('Estratégia')
    return df


# Orquestrador completo

# Orquestra o pipeline completo de backtesting:
def full_backtest_report(
    returns_df: pd.DataFrame,
    benchmark: Optional[Union[pd.Series, np.ndarray]] = None,
    rf: float = 0.0,
    estimation_window: int = 500,
    rebalancing_frequency: int = 5,
    confidence_levels: List[float] = None,
    copula_type: str = 'cvine',
    alpha: float = 0.05,
) -> Dict[str, pd.DataFrame]:
    if confidence_levels is None:
        confidence_levels = [0.95, 0.99]

    try:
        try:
            from backtesting.backtesting_walking_forward import EVTWalkForward
        except ImportError:
            from backtesting_walking_forward import EVTWalkForward
    except ImportError:
        raise ImportError(
            "backtesting_walking_forward.py não encontrado. "
            "Verifique se o arquivo está em src/backtesting/."
        )

    logger.info("=== Iniciando pipeline completo de backtesting ===")

    wf = EVTWalkForward(
        estimation_window=estimation_window,
        rebalancing_frequency=rebalancing_frequency,
        copula_type=copula_type,
        confidence_levels=confidence_levels,
    )
    wf_results  = wf.run(returns_df)
    wf_summary  = wf.summary(wf_results)

    logger.info(f"Walk-forward concluído: {len(wf_results)} dias OOS")

    bt_table = backtesting_report_from_walkforward(
        wf_results, confidence_levels, alpha
    )

    port_returns = wf_results['return']
    returns_dict = {'EVT-CVine': port_returns}
    if benchmark is not None:
        b = np.asarray(benchmark).flatten()
        b_aligned = b[-len(port_returns):]
        returns_dict['Benchmark'] = pd.Series(b_aligned, index=port_returns.index)

    perf_table = performance_summary(
        returns_dict,
        benchmark=benchmark,
        rf=rf,
    )

    logger.info("=== Pipeline concluído ===")

    return {
        'performance': perf_table,
        'backtesting': bt_table,
        'wf_results':  wf_results,
        'wf_summary':  wf_summary,
    }


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s | %(message)s')

    import sys
    from pathlib import Path

    _data_arg = None
    for i, arg in enumerate(sys.argv[1:], 1):
        if arg == '--data-dir' and i < len(sys.argv):
            _data_arg = Path(sys.argv[i + 1])
            break

    if _data_arg is not None:
        _panel_path = _data_arg / 'returns_panel.csv'
    else:
        _here = Path(__file__).resolve().parent
        _panel_path = None
        for _ in range(6):
            candidate = _here / 'results' / 'comparison' / 'returns_panel.csv'
            if candidate.exists():
                _panel_path = candidate
                break
            _here = _here.parent
        if _panel_path is None:
            raise FileNotFoundError(
                "returns_panel.csv não encontrado. "
                "Passe --data-dir <pasta> ou execute a partir da raiz do projeto."
            )

    _panel = pd.read_csv(_panel_path, index_col=0, parse_dates=True)
    _STRATEGY  = 'Minimum Variance'
    _BENCHMARK = 'Equal Weight'
    _common = _panel[_STRATEGY].dropna().index.intersection(_panel[_BENCHMARK].dropna().index)
    ret_a = _panel.loc[_common, _STRATEGY].rename(_STRATEGY)
    ret_b = _panel.loc[_common, _BENCHMARK].rename(_BENCHMARK)
    logging.getLogger(__name__).info(
        f"Retornos reais: {_STRATEGY} vs {_BENCHMARK}  "
        f"({_common[0].date()} → {_common[-1].date()}  {len(_common)} obs)"
    )

    print("=== Performance Summary ===")
    summary = performance_summary(
        {_STRATEGY: ret_a, _BENCHMARK: ret_b},
        benchmark=ret_b,
    )
    print(summary.T.to_string())

    window = 252
    var_f_95 = rolling_var(ret_a, window=window, confidence_level=0.95)
    var_f_99 = rolling_var(ret_a, window=window, confidence_level=0.99)

    var_f_95.index = ret_a.index
    var_f_99.index = ret_a.index
    mask       = ~(var_f_99.isna())
    ret_oos    = ret_a[mask]
    var_oos_95 = var_f_95[mask]
    var_oos_99 = var_f_99[mask]

    print(f"\n[Rolling VaR] In-sample: {window} dias | OOS: {mask.sum()} dias")

    print("\n=== Backtesting Report (VaR histórico rolling) ===")

    es_f_95 = rolling_es(ret_a, window, 0.95)
    es_f_99 = rolling_es(ret_a, window, 0.99)
    es_f_95.index = ret_a.index
    es_f_99.index = ret_a.index
    wf_mock = pd.DataFrame({
        'return': ret_oos.values,
        'var_95': var_oos_95.values,
        'var_99': var_oos_99.values,
        'es_95':  es_f_95[mask].values,
        'es_99':  es_f_99[mask].values,
    }, index=ret_oos.index)

    bt = backtesting_report_from_walkforward(wf_mock)
    print(bt.to_string(index=False))

    print("\n[Nota] Para backtesting com EVT real, use full_backtest_report(returns_df).")
    print("       Requer returns_df como DataFrame (T × d) com retornos dos ativos.")
