
import logging
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import (
    ks_2samp, mannwhitneyu, wilcoxon,
    jarque_bera, shapiro, normaltest,
    ttest_1samp, ttest_ind, f_oneway,
)

logger = logging.getLogger(__name__)


# Resultado padronizado de testes

# Resultado de um teste estatístico.
class TestResult:

    def __init__(
        self,
        test_name: str,
        statistic: float,
        p_value: float,
        reject_h0: bool,
        alpha: float = 0.05,
        details: Optional[Dict] = None,
    ):
        self.test_name  = test_name
        self.statistic  = statistic
        self.p_value    = p_value
        self.reject_h0  = reject_h0
        self.alpha      = alpha
        self.details    = details or {}

    def __repr__(self) -> str:
        sig = '**' if self.reject_h0 else ''
        return (
            f"{self.test_name}: stat={self.statistic:.4f}, "
            f"p={self.p_value:.4f}{sig}"
        )

    def to_dict(self) -> Dict:
        return {
            'test':      self.test_name,
            'statistic': round(self.statistic, 6),
            'p_value':   round(self.p_value, 6),
            'reject_h0': self.reject_h0,
            'alpha':     self.alpha,
            **self.details,
        }


# Testes de normalidade

# Bateria de testes de normalidade.
def test_normality(
    returns: Union[pd.Series, np.ndarray],
    alpha: float = 0.05,
) -> Dict[str, TestResult]:
    x = np.asarray(returns).flatten()
    x = x[~np.isnan(x)]

    results: Dict[str, TestResult] = {}

    jb_stat, jb_p = jarque_bera(x)
    results['jarque_bera'] = TestResult(
        'Jarque-Bera', jb_stat, jb_p, jb_p < alpha, alpha,
        details={'skewness': float(stats.skew(x)), 'kurtosis': float(stats.kurtosis(x))}
    )

    if len(x) <= 5000:
        sw_stat, sw_p = shapiro(x)
        results['shapiro_wilk'] = TestResult(
            'Shapiro-Wilk', sw_stat, sw_p, sw_p < alpha, alpha
        )

    if len(x) >= 8:
        dp_stat, dp_p = normaltest(x)
        results['dagostino_pearson'] = TestResult(
            "D'Agostino-Pearson", dp_stat, dp_p, dp_p < alpha, alpha
        )

    ad_result = stats.anderson(x, dist='norm')
    crit_5pct = ad_result.critical_values[2]
    ad_reject = ad_result.statistic > crit_5pct
    results['anderson_darling'] = TestResult(
        'Anderson-Darling', ad_result.statistic, float('nan'), ad_reject, alpha,
        details={'critical_value_5pct': float(crit_5pct)}
    )

    return results


# Testes de autocorrelação

# Ljung-Box: testa ausência de autocorrelação nos retornos.
def test_autocorrelation(
    returns: Union[pd.Series, np.ndarray],
    lags: int = 10,
    alpha: float = 0.05,
) -> TestResult:
    from statsmodels.stats.diagnostic import acorr_ljungbox

    x = np.asarray(returns).flatten()
    x = x[~np.isnan(x)]

    lb = acorr_ljungbox(x, lags=[lags], return_df=True)
    stat  = float(lb['lb_stat'].iloc[-1])
    p_val = float(lb['lb_pvalue'].iloc[-1])

    return TestResult(
        f'Ljung-Box (lags={lags})',
        stat, p_val, p_val < alpha, alpha,
        details={'lags': lags}
    )


# Teste LM para efeitos ARCH (heterocedasticidade condicional).
def test_arch_effects(
    returns: Union[pd.Series, np.ndarray],
    lags: int = 5,
    alpha: float = 0.05,
) -> TestResult:
    from statsmodels.stats.diagnostic import het_arch

    x = np.asarray(returns).flatten()
    x = x[~np.isnan(x)]

    lm_stat, lm_p, f_stat, f_p = het_arch(x, nlags=lags)

    return TestResult(
        f'ARCH-LM (lags={lags})',
        lm_stat, lm_p, lm_p < alpha, alpha,
        details={'f_stat': round(float(f_stat), 4), 'f_pvalue': round(float(f_p), 4)}
    )


# Comparação de Sharpe (Jobson-Korkie / Memmel 2003)

# Teste de Jobson-Korkie (1981) para igualdade de Sharpe Ratios,
def test_sharpe_difference(
    returns_a: Union[pd.Series, np.ndarray],
    returns_b: Union[pd.Series, np.ndarray],
    rf: float = 0.0,
    alpha: float = 0.05,
    label_a: str = 'A',
    label_b: str = 'B',
) -> TestResult:
    a = np.asarray(returns_a).flatten()
    b = np.asarray(returns_b).flatten()

    n = min(len(a), len(b))
    a, b = a[:n], b[:n]

    excess_a = a - rf
    excess_b = b - rf

    mu_a, mu_b       = excess_a.mean(), excess_b.mean()
    var_a, var_b     = excess_a.var(ddof=1), excess_b.var(ddof=1)
    sigma_a, sigma_b = np.sqrt(var_a), np.sqrt(var_b)
    cov_ab           = np.cov(excess_a, excess_b, ddof=1)[0, 1]

    sr_a = mu_a / sigma_a
    sr_b = mu_b / sigma_b

    rho_ab = cov_ab / (sigma_a * sigma_b)

    theta = (
        2 * var_a * var_b
        - 2 * sigma_a * sigma_b * cov_ab
        + 0.5 * mu_a ** 2 * var_b
        + 0.5 * mu_b ** 2 * var_a
        - mu_a * mu_b * rho_ab * cov_ab
    )

    var_sr_diff = theta / (n * (sigma_a * sigma_b) ** 2)
    se = np.sqrt(np.maximum(var_sr_diff, 1e-12))

    z_stat = (sr_a - sr_b) / se
    p_val  = 2 * (1 - stats.norm.cdf(abs(z_stat)))

    return TestResult(
        f'Jobson-Korkie ({label_a} vs {label_b})',
        float(z_stat), float(p_val), p_val < alpha, alpha,
        details={
            f'SR_{label_a}': round(float(sr_a * np.sqrt(252)), 4),
            f'SR_{label_b}': round(float(sr_b * np.sqrt(252)), 4),
        }
    )


# Testes de comparação de distribuições

# Compara duas distribuições de retornos com KS e Wilcoxon/Mann-Whitney.
def test_distribution_comparison(
    returns_a: Union[pd.Series, np.ndarray],
    returns_b: Union[pd.Series, np.ndarray],
    alpha: float = 0.05,
    label_a: str = 'A',
    label_b: str = 'B',
) -> Dict[str, TestResult]:
    a = np.asarray(returns_a).flatten()
    b = np.asarray(returns_b).flatten()

    results: Dict[str, TestResult] = {}

    ks_stat, ks_p = ks_2samp(a, b)
    results['ks_2sample'] = TestResult(
        f'KS 2-amostras ({label_a} vs {label_b})', ks_stat, ks_p, ks_p < alpha, alpha
    )

    mw_stat, mw_p = mannwhitneyu(a, b, alternative='two-sided')
    results['mann_whitney'] = TestResult(
        f'Mann-Whitney ({label_a} vs {label_b})', mw_stat, mw_p, mw_p < alpha, alpha
    )

    tt_stat, tt_p = ttest_ind(a, b, equal_var=False)
    results['welch_t'] = TestResult(
        f'Welch t-test ({label_a} vs {label_b})', tt_stat, tt_p, tt_p < alpha, alpha
    )

    return results


# Teste de dominância estocástica

# Verifica dominância estocástica de primeira ou segunda ordem de A sobre B.
def test_stochastic_dominance(
    returns_a: Union[pd.Series, np.ndarray],
    returns_b: Union[pd.Series, np.ndarray],
    order: int = 1,
    label_a: str = 'A',
    label_b: str = 'B',
) -> Dict:
    a = np.sort(np.asarray(returns_a).flatten())
    b = np.sort(np.asarray(returns_b).flatten())
    grid = np.linspace(min(a.min(), b.min()), max(a.max(), b.max()), 500)

    def ecdf(x, grid):
        return np.searchsorted(x, grid, side='right') / len(x)

    cdf_a = ecdf(a, grid)
    cdf_b = ecdf(b, grid)

    if order == 1:
        dominates  = bool(np.all(cdf_a <= cdf_b))
        dominated  = bool(np.all(cdf_b <= cdf_a))
        max_diff   = float((cdf_a - cdf_b).max())
    elif order == 2:
        integral_a = np.cumsum(cdf_a) * (grid[1] - grid[0])
        integral_b = np.cumsum(cdf_b) * (grid[1] - grid[0])
        dominates  = bool(np.all(integral_a <= integral_b))
        dominated  = bool(np.all(integral_b <= integral_a))
        max_diff   = float((integral_a - integral_b).max())
    else:
        raise ValueError("order deve ser 1 ou 2")

    return {
        'order':               order,
        f'{label_a}_dominates_{label_b}': dominates,
        f'{label_b}_dominates_{label_a}': dominated,
        'neither':             not (dominates or dominated),
        'max_cdf_diff':        round(max_diff, 6),
    }


# Teste de Information Ratio

# Testa se o Information Ratio é significativamente > 0.
def test_information_ratio(
    portfolio_returns: Union[pd.Series, np.ndarray],
    benchmark_returns: Union[pd.Series, np.ndarray],
    alpha: float = 0.05,
) -> TestResult:
    p = np.asarray(portfolio_returns).flatten()
    b = np.asarray(benchmark_returns).flatten()
    n = min(len(p), len(b))
    p, b = p[:n], b[:n]

    active = p - b
    ir     = active.mean() / active.std(ddof=1) * np.sqrt(252)
    t_stat = active.mean() / (active.std(ddof=1) / np.sqrt(n))
    p_val  = 1 - stats.t.cdf(t_stat, df=n - 1)

    return TestResult(
        'Information Ratio (t-test)',
        float(t_stat), float(p_val), p_val < alpha, alpha,
        details={'IR_annualized': round(float(ir), 4), 'n': n}
    )


# Comparação de múltiplas estratégias (ANOVA / Kruskal-Wallis)

# Compara múltiplas estratégias simultaneamente.
def test_multiple_strategies(
    returns_dict: Dict[str, Union[pd.Series, np.ndarray]],
    alpha: float = 0.05,
) -> Dict[str, TestResult]:
    arrays = [np.asarray(v).flatten() for v in returns_dict.values()]
    labels = list(returns_dict.keys())

    n_min = min(len(a) for a in arrays)
    arrays = [a[:n_min] for a in arrays]

    results: Dict[str, TestResult] = {}

    f_stat, f_p = f_oneway(*arrays)
    results['anova'] = TestResult(
        f'ANOVA ({len(labels)} estratégias)', f_stat, f_p, f_p < alpha, alpha,
        details={'n_strategies': len(labels), 'labels': labels}
    )

    kw_stat, kw_p = stats.kruskal(*arrays)
    results['kruskal_wallis'] = TestResult(
        f'Kruskal-Wallis ({len(labels)} estratégias)', kw_stat, kw_p, kw_p < alpha, alpha
    )

    return results


# Relatório consolidado

# Roda bateria completa de testes e retorna DataFrame consolidado.
def summary_report(
    returns: Union[pd.Series, np.ndarray],
    benchmark: Optional[Union[pd.Series, np.ndarray]] = None,
    label: str = 'Portfólio',
    alpha: float = 0.05,
) -> pd.DataFrame:
    all_results: List[Dict] = []

    norm_tests = test_normality(returns, alpha)
    all_results.extend([r.to_dict() for r in norm_tests.values()])

    try:
        ac_test = test_autocorrelation(returns, alpha=alpha)
        all_results.append(ac_test.to_dict())
    except Exception as e:
        logger.warning(f"Ljung-Box falhou: {e}")

    try:
        arch_test = test_arch_effects(returns, alpha=alpha)
        all_results.append(arch_test.to_dict())
    except Exception as e:
        logger.warning(f"ARCH-LM falhou: {e}")

    if benchmark is not None:
        try:
            ir_test = test_information_ratio(returns, benchmark, alpha)
            all_results.append(ir_test.to_dict())

            sr_test = test_sharpe_difference(
                returns, benchmark, label_a=label, label_b='Benchmark', alpha=alpha
            )
            all_results.append(sr_test.to_dict())
        except Exception as e:
            logger.warning(f"Testes vs benchmark falharam: {e}")

    df = pd.DataFrame(all_results)
    logger.info(f"\n=== Relatório de Testes Estatísticos ({label}) ===")
    logger.info(df[['test', 'statistic', 'p_value', 'reject_h0']].to_string(index=False))
    return df


if __name__ == '__main__':
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
                "Passe --data-dir <pasta com returns_panel.csv> ou execute "
                "a partir da raiz do projeto."
            )

    _panel = pd.read_csv(_panel_path, index_col=0, parse_dates=True)

    _STRATEGY  = 'Minimum Variance'
    _BENCHMARK = 'Equal Weight'

    ret = _panel[_STRATEGY].dropna()
    bm  = _panel[_BENCHMARK].dropna()
    _common = ret.index.intersection(bm.index)
    ret, bm = ret.loc[_common], bm.loc[_common]

    report = summary_report(ret, benchmark=bm, label=_STRATEGY)
    print(report[['test', 'statistic', 'p_value', 'reject_h0']].to_string())
