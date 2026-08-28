
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


try:
    from backtesting.performance_metrics import (
        performance_summary,
        backtesting_report as _bt_report,
        kupiec_pof,
        christoffersen_test,
        sharpe_ratio,
        historical_var,
        historical_es,
        max_drawdown,
    )
    _PM_AVAILABLE = True
except ImportError:
    try:
        from performance_metrics import (
            performance_summary,
            backtesting_report as _bt_report,
            kupiec_pof,
            christoffersen_test,
            sharpe_ratio,
            historical_var,
            historical_es,
            max_drawdown,
        )
        _PM_AVAILABLE = True
    except ImportError:
        _PM_AVAILABLE = False
        logger.warning("performance_metrics não disponível — funcionalidade limitada")

try:
    from evaluation.statistical_tests import (
        test_normality,
        test_autocorrelation,
        test_arch_effects,
        test_sharpe_difference,
        test_information_ratio,
        summary_report as stat_summary,
    )
    _ST_AVAILABLE = True
except ImportError:
    try:
        from statistical_tests import (
            test_normality,
            test_autocorrelation,
            test_arch_effects,
            test_sharpe_difference,
            test_information_ratio,
            summary_report as stat_summary,
        )
        _ST_AVAILABLE = True
    except ImportError:
        _ST_AVAILABLE = False
        logger.warning("statistical_tests não disponível — testes estatísticos desabilitados")


# PerformanceReport

# Relatório de performance para múltiplas estratégias.
class PerformanceReport:

    def __init__(
        self,
        returns_dict: Dict[str, Union[pd.Series, np.ndarray]],
        benchmark: Optional[Union[pd.Series, np.ndarray]] = None,
        rf: float = 0.0,
        periods_per_year: int = 252,
        alpha: float = 0.05,
    ):
        self.returns_dict    = returns_dict
        self.benchmark       = benchmark
        self.rf              = rf
        self.periods_per_year = periods_per_year
        self.alpha           = alpha
        self._df: Optional[pd.DataFrame] = None

    # Calcula todas as métricas de performance.
    def compute(self) -> pd.DataFrame:
        if not _PM_AVAILABLE:
            return self._compute_fallback()

        self._df = performance_summary(
            self.returns_dict,
            benchmark=self.benchmark,
            rf=self.rf,
            periods_per_year=self.periods_per_year,
        )
        logger.info(f"PerformanceReport computado: {len(self._df)} estratégias")
        return self._df

    # Versão mínima sem performance_metrics.py.
    def _compute_fallback(self) -> pd.DataFrame:
        rows = []
        for label, r in self.returns_dict.items():
            arr = np.asarray(r).flatten()
            arr = arr[~np.isnan(arr)]
            cum = np.cumprod(1 + arr)
            pk  = np.maximum.accumulate(cum)
            dd  = (cum - pk) / pk

            mu    = arr.mean()
            sigma = arr.std(ddof=1)
            sr    = mu / sigma * np.sqrt(self.periods_per_year) if sigma > 0 else np.nan
            mdd   = float(dd.min())

            rows.append({
                'Estratégia':      label,
                'Sharpe':          round(sr, 4),
                'Sortino':         np.nan,
                'Calmar':          np.nan,
                'Max DD (%)':      round(mdd * 100, 2),
                'VaR 99% (%)':     round(float(-np.quantile(arr, 0.01)) * 100, 3),
                'ES 99% (%)':      round(float(-arr[arr <= np.quantile(arr, 0.01)].mean()) * 100, 3),
                'Ann. Return (%)': round((np.prod(1 + arr) ** (self.periods_per_year / len(arr)) - 1) * 100, 2),
                'Ann. Vol (%)':    round(sigma * np.sqrt(self.periods_per_year) * 100, 2),
                'Skewness':        round(float(stats.skew(arr)), 4),
                'Excess Kurtosis': round(float(stats.kurtosis(arr)), 4),
            })

        return pd.DataFrame(rows).set_index('Estratégia')

    # Roda bateria de testes estatísticos para cada estratégia (ou uma específica).
    def statistical_tests(
        self,
        strategy: Optional[str] = None,
    ) -> Dict[str, pd.DataFrame]:
        if not _ST_AVAILABLE:
            logger.warning("statistical_tests não disponível")
            return {}

        targets = {strategy: self.returns_dict[strategy]} \
            if strategy else self.returns_dict

        out = {}
        for label, r in targets.items():
            bm = self.benchmark
            out[label] = stat_summary(r, benchmark=bm, label=label, alpha=self.alpha)
        return out

    # Exporta tabela para LaTeX.
    def to_latex(
        self,
        path: Union[str, Path],
        caption: str = "Métricas de performance dos portfólios",
        label: str = "tab:performance",
        float_format: str = "%.4f",
    ) -> str:
        df = self._df if self._df is not None else self.compute()

        cols = [c for c in [
            'Ann. Return (%)', 'Ann. Vol (%)', 'Sharpe', 'Sortino',
            'Calmar', 'Max DD (%)', 'VaR 99% (%)', 'ES 99% (%)',
            'Skewness', 'Excess Kurtosis',
        ] if c in df.columns]

        latex = df[cols].to_latex(
            float_format=float_format,
            caption=caption,
            label=label,
            bold_rows=True,
            escape=False,
        )

        if path:
            p = Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(latex, encoding='utf-8')
            logger.info(f"LaTeX salvo em {p}")

        return latex

    def to_csv(self, path: Union[str, Path]) -> None:
        df = self._df if self._df is not None else self.compute()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path)
        logger.info(f"CSV salvo em {path}")


# RiskReport

# Relatório de backtesting de VaR/ES.
class RiskReport:

    def __init__(
        self,
        returns: Union[pd.Series, np.ndarray],
        var_forecasts: Optional[Union[pd.Series, np.ndarray]] = None,
        es_forecasts: Optional[Union[pd.Series, np.ndarray]] = None,
        label: str = 'Portfólio',
        confidence_levels: List[float] = [0.95, 0.99],
        alpha: float = 0.05,
    ):
        self.returns           = returns
        self.var_forecasts     = var_forecasts
        self.es_forecasts      = es_forecasts
        self.label             = label
        self.confidence_levels = confidence_levels
        self.alpha             = alpha
        self._bt_df: Optional[pd.DataFrame] = None

    # Kupiec POF + Christoffersen para cada nível de confiança.
    def backtesting_report(self) -> pd.DataFrame:
        var = self._get_var_forecasts()

        if _PM_AVAILABLE:
            self._bt_df = _bt_report(
                self.returns, var,
                confidence_levels=self.confidence_levels,
                alpha=self.alpha,
            )
        else:
            self._bt_df = self._backtesting_fallback(var)

        logger.info(f"RiskReport ({self.label}): backtesting concluído")
        return self._bt_df

    # Retorna var_forecasts ou constrói rolling histórico como fallback.
    def _get_var_forecasts(self) -> Union[pd.Series, np.ndarray]:
        if self.var_forecasts is not None:
            return self.var_forecasts

        logger.warning(
            "var_forecasts não fornecido — usando VaR histórico rolling (janela=250)"
        )
        r = pd.Series(np.asarray(self.returns).flatten())
        return (-r.rolling(250).quantile(0.01)).shift(1).dropna()

    # Implementação mínima sem performance_metrics.py.
    def _backtesting_fallback(
        self,
        var: Union[pd.Series, np.ndarray],
    ) -> pd.DataFrame:
        rows = []
        r   = np.asarray(self.returns).flatten()
        v   = np.asarray(var).flatten()
        n   = min(len(r), len(v))
        r, v = r[:n], v[:n]
        mask = ~(np.isnan(r) | np.isnan(v))
        r, v = r[mask], v[mask]
        n = len(r)

        for cl in self.confidence_levels:
            p      = 1 - cl
            viols  = int((r < -v).sum())
            obs    = viols / n
            exp    = p

            if viols > 0 and viols < n:
                p_hat = viols / n
                lr = -2 * (
                    viols * np.log(p / p_hat)
                    + (n - viols) * np.log((1 - p) / (1 - p_hat))
                )
                p_val = float(1 - stats.chi2.cdf(lr, df=1))
            else:
                lr, p_val = 0.0, 1.0

            rows.append({
                'Nível (%)':          round(cl * 100, 1),
                'Taxa Observada (%)': round(obs * 100, 3),
                'Taxa Esperada (%)':  round(exp * 100, 3),
                'Violações':          viols,
                'N':                  n,
                'Kupiec LR':          round(lr, 4),
                'Kupiec p':           round(p_val, 6),
                'Kupiec OK':          p_val >= self.alpha,
                'Chris p (ind)':      np.nan,
                'Chris p (cc)':       np.nan,
                'Chris OK':           np.nan,
            })
        return pd.DataFrame(rows)

    # Estatísticas descritivas dos retornos: média, vol, VaR, ES, skew, kurt.
    def summary_statistics(self) -> pd.DataFrame:
        r = np.asarray(self.returns).flatten()
        r = r[~np.isnan(r)]

        rows = []
        for cl in self.confidence_levels:
            rows.append({
                'Nível':    f'{cl:.0%}',
                'VaR (%)':  round(float(-np.quantile(r, 1 - cl)) * 100, 3),
                'ES (%)':   round(float(-r[r <= np.quantile(r, 1 - cl)].mean()) * 100, 3),
            })

        desc = {
            'Média Diária (%)':    round(r.mean() * 100, 4),
            'Vol Diária (%)':      round(r.std(ddof=1) * 100, 4),
            'Skewness':            round(float(stats.skew(r)), 4),
            'Excess Kurtosis':     round(float(stats.kurtosis(r)), 4),
            'Min (%)':             round(r.min() * 100, 3),
            'Max (%)':             round(r.max() * 100, 3),
            'N obs':               len(r),
        }

        desc_df  = pd.DataFrame([desc]).T.rename(columns={0: self.label})
        level_df = pd.DataFrame(rows).set_index('Nível').T.rename(
            index={'VaR (%)': f'VaR ({self.label})', 'ES (%)': f'ES ({self.label})'}
        )
        return pd.concat([desc_df, level_df.T.reset_index(drop=True)])

    def to_latex(
        self,
        path: Union[str, Path],
        caption: str = "Backtesting de VaR",
        label_tex: str = "tab:backtesting",
    ) -> str:
        df = self._bt_df if self._bt_df is not None else self.backtesting_report()

        cols_print = [c for c in [
            'Nível (%)', 'Taxa Observada (%)', 'Taxa Esperada (%)',
            'Violações', 'N', 'Kupiec LR', 'Kupiec p', 'Kupiec OK',
            'Chris p (cc)', 'Chris OK',
        ] if c in df.columns]

        latex = df[cols_print].to_latex(
            index=False,
            float_format="%.4f",
            caption=caption,
            label=label_tex,
            escape=False,
        )

        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(latex, encoding='utf-8')
        logger.info(f"LaTeX salvo em {p}")
        return latex

    def to_csv(self, path: Union[str, Path]) -> None:
        df = self._bt_df if self._bt_df is not None else self.backtesting_report()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        logger.info(f"CSV salvo em {path}")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

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

    logger.info(f"Carregando retornos reais de: {_panel_path}")
    _panel = pd.read_csv(_panel_path, index_col=0, parse_dates=True)

    _STRATEGY  = 'Minimum Variance'
    _BENCHMARK = 'Equal Weight'

    if _STRATEGY not in _panel.columns or _BENCHMARK not in _panel.columns:
        raise ValueError(
            f"Colunas esperadas '{_STRATEGY}' e '{_BENCHMARK}' "
            f"não encontradas. Disponíveis: {list(_panel.columns)}"
        )

    ret_a = _panel[_STRATEGY].dropna().rename(_STRATEGY)
    ret_b = _panel[_BENCHMARK].dropna().rename(_BENCHMARK)

    _common = ret_a.index.intersection(ret_b.index)
    ret_a, ret_b = ret_a.loc[_common], ret_b.loc[_common]
    logger.info(f"Período: {_common[0].date()} → {_common[-1].date()}  ({len(_common)} obs)")

    pr = PerformanceReport({'EVT-CVine': ret_a, 'Benchmark': ret_b}, benchmark=ret_b)
    df_perf = pr.compute()
    print("=== PerformanceReport ===")
    print(df_perf.T.to_string())

    window = 250
    s = ret_a.reset_index(drop=True)
    var_95 = (-s.rolling(window).quantile(0.05)).shift(1)
    var_99 = (-s.rolling(window).quantile(0.01)).shift(1)
    mask       = ~var_99.isna()
    ret_oos    = ret_a.values[mask]
    var_oos_95 = var_95.values[mask]
    var_oos_99 = var_99.values[mask]

    rows = []
    for cl, var_oos in zip([0.95, 0.99], [var_oos_95, var_oos_99]):
        if _PM_AVAILABLE:
            k = kupiec_pof(ret_oos, var_oos, cl)
            c = christoffersen_test(ret_oos, var_oos, cl)
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
        else:
            r2, v2 = ret_oos, var_oos
            mask2 = ~(np.isnan(r2) | np.isnan(v2))
            r2, v2 = r2[mask2], v2[mask2]
            nn = len(r2)
            viols = int((r2 < -v2).sum())
            obs = viols / nn if nn > 0 else 0.0
            exp = 1 - cl
            if viols > 0 and viols < nn:
                lr = -2 * (viols * np.log(exp / obs) + (nn - viols) * np.log((1 - exp) / (1 - obs)))
                pv = float(1 - stats.chi2.cdf(lr, df=1))
            else:
                lr, pv = 0.0, 1.0
            rows.append({
                'Nível (%)': round(cl * 100, 1),
                'Taxa Observada (%)': round(obs * 100, 3),
                'Taxa Esperada (%)': round(exp * 100, 3),
                'Violações': viols, 'N': nn,
                'Kupiec LR': round(lr, 4), 'Kupiec p': round(pv, 6),
                'Kupiec OK': pv >= 0.05,
                'Chris p (ind)': np.nan, 'Chris p (cc)': np.nan, 'Chris OK': np.nan,
            })

    df_bt = pd.DataFrame(rows)
    logger.info(f"RiskReport ({_STRATEGY}): backtesting concluído")
    print("\n=== RiskReport — Backtesting ===")
    print(df_bt.to_string(index=False))
