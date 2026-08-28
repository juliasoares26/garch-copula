
import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


PALETTE = ['#2563EB', '#DC2626', '#16A34A', '#D97706', '#7C3AED',
           '#0891B2', '#BE185D', '#065F46']
ALPHA_FILL = 0.15
FIG_DPI    = 150
FONT_TITLE = 13
FONT_LABEL = 11
FONT_TICK  = 9


def _style():
    plt.rcParams.update({
        'figure.dpi':        FIG_DPI,
        'axes.spines.top':   False,
        'axes.spines.right': False,
        'axes.grid':         True,
        'grid.alpha':        0.3,
        'grid.linewidth':    0.5,
        'font.size':         FONT_TICK,
        'axes.titlesize':    FONT_TITLE,
        'axes.labelsize':    FONT_LABEL,
        'legend.fontsize':   FONT_TICK,
        'legend.framealpha': 0.85,
    })


_style()


def _save(fig: plt.Figure, path: Optional[Union[str, Path]], tight: bool = True) -> None:
    if tight:
        fig.tight_layout()
    if path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=FIG_DPI, bbox_inches='tight')
        logger.info(f"Figura salva: {p}")


# 1. Retornos acumulados

# Retornos acumulados (base 1) para múltiplas estratégias.
def plot_cumulative_returns(
    returns_dict: Dict[str, Union[pd.Series, np.ndarray]],
    title: str = "Retornos Acumulados",
    figsize: Tuple = (10, 5),
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=figsize)
    for (label, r), color in zip(returns_dict.items(), PALETTE):
        arr = np.asarray(r).flatten()
        idx = r.index if isinstance(r, pd.Series) else np.arange(len(arr))
        cum = np.cumprod(1 + arr)
        ax.plot(idx, cum, label=label, color=color, linewidth=1.6)

    ax.axhline(1.0, color='black', linewidth=0.8, linestyle='--', alpha=0.5)
    ax.set_title(title)
    ax.set_ylabel("Riqueza Acumulada")
    ax.set_xlabel("Data")
    ax.legend()
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.2f'))
    _save(fig, save_path)
    return fig


# 2. Drawdowns

# Séries temporais de drawdown para múltiplas estratégias.
def plot_drawdowns(
    returns_dict: Dict[str, Union[pd.Series, np.ndarray]],
    title: str = "Drawdowns",
    figsize: Tuple = (10, 4),
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=figsize)
    for (label, r), color in zip(returns_dict.items(), PALETTE):
        arr = np.asarray(r).flatten()
        idx = r.index if isinstance(r, pd.Series) else np.arange(len(arr))
        cum = np.cumprod(1 + arr)
        pk  = np.maximum.accumulate(cum)
        dd  = (cum - pk) / pk
        ax.fill_between(idx, dd, 0, alpha=ALPHA_FILL, color=color)
        ax.plot(idx, dd, color=color, linewidth=1.2, label=label)

    ax.set_title(title)
    ax.set_ylabel("Drawdown")
    ax.set_xlabel("Data")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=1))
    ax.legend()
    _save(fig, save_path)
    return fig


# 3. Rolling Sharpe

# Sharpe ratio rolling (janela `window` dias úteis).
def plot_rolling_sharpe(
    returns_dict: Dict[str, Union[pd.Series, np.ndarray]],
    window: int = 63,
    rf: float = 0.0,
    periods_per_year: int = 252,
    title: str = "Sharpe Ratio Rolling",
    figsize: Tuple = (10, 4),
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=figsize)
    for (label, r), color in zip(returns_dict.items(), PALETTE):
        s = pd.Series(np.asarray(r).flatten())
        excess = s - rf
        roll_sr = (
            excess.rolling(window).mean()
            / excess.rolling(window).std(ddof=1)
            * np.sqrt(periods_per_year)
        )
        idx = r.index if isinstance(r, pd.Series) else roll_sr.index
        ax.plot(idx, roll_sr.values, label=label, color=color, linewidth=1.4)

    ax.axhline(0.0, color='black', linewidth=0.8, linestyle='--', alpha=0.6)
    ax.set_title(f"{title} (janela={window}d)")
    ax.set_ylabel("Sharpe Ratio")
    ax.set_xlabel("Data")
    ax.legend()
    _save(fig, save_path)
    return fig


# 4. VaR Backtesting — retornos vs VaR com violações

# Retornos realizados vs previsão de VaR com violações destacadas.
def plot_var_backtesting(
    returns: Union[pd.Series, np.ndarray],
    var_forecasts: Union[pd.Series, np.ndarray],
    confidence_level: float = 0.99,
    label: str = 'Portfólio',
    title: Optional[str] = None,
    figsize: Tuple = (12, 5),
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    r   = np.asarray(returns).flatten()
    var = np.asarray(var_forecasts).flatten()
    n   = min(len(r), len(var))
    r, var = r[:n], var[:n]

    idx = returns.index[:n] if isinstance(returns, pd.Series) else np.arange(n)
    violations = r < -var

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(idx, r * 100, color=PALETTE[0], linewidth=0.8,
            alpha=0.7, label='Retorno realizado')
    ax.plot(idx, -var * 100, color=PALETTE[1], linewidth=1.4,
            linestyle='--', label=f'VaR {confidence_level:.0%}')
    ax.scatter(
        np.array(idx)[violations], r[violations] * 100,
        color='red', s=18, zorder=5, label=f'Violações ({violations.sum()})'
    )
    ax.set_title(title or f"Backtesting VaR {confidence_level:.0%} — {label}")
    ax.set_ylabel("Retorno (%)")
    ax.set_xlabel("Data")
    ax.legend()
    _save(fig, save_path)
    return fig


# 5. Clustering de violações

# ACF das violações — detecta clustering (rejeição de Christoffersen).
def plot_violation_clustering(
    returns: Union[pd.Series, np.ndarray],
    var_forecasts: Union[pd.Series, np.ndarray],
    confidence_level: float = 0.99,
    n_lags: int = 20,
    title: str = "Autocorrelação das Violações (Christoffersen)",
    figsize: Tuple = (8, 4),
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    from statsmodels.graphics.tsaplots import plot_acf

    r   = np.asarray(returns).flatten()
    var = np.asarray(var_forecasts).flatten()
    n   = min(len(r), len(var))
    violations = (r[:n] < -var[:n]).astype(float)

    fig, ax = plt.subplots(figsize=figsize)
    plot_acf(violations, lags=n_lags, ax=ax, zero=False, alpha=0.05)
    ax.set_title(title)
    ax.set_xlabel("Lag")
    ax.set_ylabel("Autocorrelação")
    _save(fig, save_path)
    return fig


# 6. Q-Q plot EVT (GPD)

# Q-Q plot: excedências empíricas vs quantis GPD teóricos.
def plot_evt_qq(
    residuals: Union[pd.Series, np.ndarray],
    xi: float,
    sigma: float,
    threshold: float,
    side: str = 'left',
    ticker: str = '',
    figsize: Tuple = (5, 5),
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    x = np.asarray(residuals).flatten()
    x = x[~np.isnan(x)]

    if side == 'left':
        exceedances = np.sort(threshold - x[x < threshold])
    else:
        exceedances = np.sort(x[x > threshold] - threshold)

    n = len(exceedances)
    if n < 5:
        logger.warning(f"plot_evt_qq: apenas {n} excedências para {ticker} ({side})")
        return plt.figure()

    probs  = (np.arange(1, n + 1) - 0.5) / n
    if abs(xi) < 1e-6:
        theoretical = -sigma * np.log(1 - probs)
    else:
        theoretical = (sigma / xi) * ((1 - probs) ** (-xi) - 1)

    fig, ax = plt.subplots(figsize=figsize)
    ax.scatter(theoretical, exceedances, s=12, color=PALETTE[0], alpha=0.7)
    lim = max(theoretical.max(), exceedances.max()) * 1.05
    ax.plot([0, lim], [0, lim], 'r--', linewidth=1.2, label='Linha 45°')
    ax.set_title(f"Q-Q GPD — {ticker} ({side})\nξ={xi:.3f}, σ={sigma:.4f}")
    ax.set_xlabel("Quantis teóricos (GPD)")
    ax.set_ylabel("Quantis empíricos")
    ax.legend()
    _save(fig, save_path)
    return fig


# 7. Ajuste GPD na cauda (escala log)

# Densidade empírica vs GPD ajustada na cauda, escala log no eixo y.
def plot_tail_fit(
    residuals: Union[pd.Series, np.ndarray],
    xi: float,
    sigma: float,
    threshold: float,
    side: str = 'left',
    ticker: str = '',
    figsize: Tuple = (6, 4),
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    x = np.asarray(residuals).flatten()
    x = x[~np.isnan(x)]

    if side == 'left':
        exceedances = np.sort(threshold - x[x < threshold])
    else:
        exceedances = np.sort(x[x > threshold] - threshold)

    if len(exceedances) < 5:
        return plt.figure()

    grid = np.linspace(0, exceedances.max() * 1.1, 300)
    if abs(xi) < 1e-6:
        pdf_gpd = (1 / sigma) * np.exp(-grid / sigma)
    else:
        t = 1 + xi * grid / sigma
        t = np.maximum(t, 1e-10)
        pdf_gpd = (1 / sigma) * t ** (-(1 / xi + 1))

    fig, ax = plt.subplots(figsize=figsize)
    ax.hist(exceedances, bins=30, density=True, alpha=0.4,
            color=PALETTE[0], label='Empírico')
    ax.plot(grid, pdf_gpd, color=PALETTE[1], linewidth=2.0,
            label=f'GPD (ξ={xi:.3f}, σ={sigma:.4f})')
    ax.set_yscale('log')
    ax.set_title(f"Ajuste GPD na cauda — {ticker} ({side})")
    ax.set_xlabel("Excedência")
    ax.set_ylabel("Densidade (log)")
    ax.legend()
    _save(fig, save_path)
    return fig


# 8. GEV Return Levels

# Plota return levels GEV.
def plot_gev_return_levels(
    return_levels_df: pd.DataFrame,
    title: str = "Return Levels GEV por Ativo",
    figsize: Tuple = (9, 5),
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    periods = list(return_levels_df.columns)
    x = np.arange(len(periods))
    n_assets = len(return_levels_df)
    width = 0.8 / n_assets

    fig, ax = plt.subplots(figsize=figsize)
    for i, (ticker, row) in enumerate(return_levels_df.iterrows()):
        vals = [abs(row[p]) * 100 for p in periods]
        ax.bar(x + i * width, vals, width=width * 0.9,
               color=PALETTE[i % len(PALETTE)], label=ticker, alpha=0.85)

    ax.set_title(title)
    ax.set_ylabel("Perda Máxima Esperada (%)")
    ax.set_xlabel("Horizonte")
    ax.set_xticks(x + width * (n_assets - 1) / 2)
    ax.set_xticklabels(periods)
    ax.legend()
    _save(fig, save_path)
    return fig


# 9. Scatter de pseudo-observações (cópula)

# Scatter matrix das pseudo-observações [0,1] com density na diagonal.
def plot_copula_scatter(
    U: np.ndarray,
    asset_names: Optional[List[str]] = None,
    title: str = "Pseudo-Observações Uniformes",
    max_pairs: int = 6,
    figsize_per: float = 3.5,
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    if isinstance(U, pd.DataFrame):
        asset_names = list(U.columns)
        U = U.values

    d = U.shape[1]
    names = asset_names or [f'A{i+1}' for i in range(d)]

    d_plot = min(d, int(np.ceil(np.sqrt(max_pairs * 2))))
    U_plot = U[:, :d_plot]
    names  = names[:d_plot]

    fig, axes = plt.subplots(d_plot, d_plot,
                             figsize=(figsize_per * d_plot, figsize_per * d_plot))
    if d_plot == 1:
        axes = np.array([[axes]])

    for i in range(d_plot):
        for j in range(d_plot):
            ax = axes[i][j]
            if i == j:
                ax.hist(U_plot[:, i], bins=25, density=True,
                        color=PALETTE[i % len(PALETTE)], alpha=0.7)
                ax.set_xlim(0, 1)
            elif i > j:
                ax.scatter(U_plot[:, j], U_plot[:, i],
                           s=4, alpha=0.3, color=PALETTE[0])
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
            else:
                ax.axis('off')

            if j == 0:
                ax.set_ylabel(names[i], fontsize=8)
            if i == d_plot - 1:
                ax.set_xlabel(names[j], fontsize=8)
            ax.tick_params(labelsize=7)

    fig.suptitle(title, fontsize=FONT_TITLE)
    _save(fig, save_path)
    return fig


# 10. Heatmap de correlação

# Heatmap da correlação linear de Pearson entre ativos.
def plot_correlation_heatmap(
    returns_df: pd.DataFrame,
    title: str = "Matriz de Correlação",
    figsize: Tuple = (8, 6),
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    import matplotlib.colors as mcolors

    corr = returns_df.corr()
    n = len(corr)

    fig, ax = plt.subplots(figsize=figsize)
    cmap = plt.get_cmap('RdYlGn')
    im = ax.imshow(corr.values, cmap=cmap, vmin=-1, vmax=1, aspect='auto')

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(corr.columns, rotation=45, ha='right', fontsize=9)
    ax.set_yticklabels(corr.index, fontsize=9)

    for i in range(n):
        for j in range(n):
            val = corr.iloc[i, j]
            color = 'white' if abs(val) > 0.65 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                    fontsize=8, color=color)

    plt.colorbar(im, ax=ax, shrink=0.8, label='Correlação')
    ax.set_title(title)
    _save(fig, save_path)
    return fig


# 11. Tail dependence heatmap

# Heatmaps de tail dependence inferior e superior entre pares de ativos.
def plot_tail_dependence(
    tail_lower: np.ndarray,
    tail_upper: np.ndarray,
    asset_names: List[str],
    figsize: Tuple = (12, 5),
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    n = len(asset_names)

    for ax, mat, title in zip(
        axes,
        [tail_lower, tail_upper],
        ['Tail Dependence Inferior (λ_L)', 'Tail Dependence Superior (λ_U)']
    ):
        im = ax.imshow(mat, cmap='YlOrRd', vmin=0, vmax=1, aspect='auto')
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(asset_names, rotation=45, ha='right', fontsize=8)
        ax.set_yticklabels(asset_names, fontsize=8)
        for i in range(n):
            for j in range(n):
                ax.text(j, i, f'{mat[i, j]:.2f}',
                        ha='center', va='center', fontsize=7,
                        color='black' if mat[i, j] < 0.6 else 'white')
        plt.colorbar(im, ax=ax, shrink=0.8)
        ax.set_title(title)

    _save(fig, save_path)
    return fig


# 12. Regime timeline

# Retornos com fundo colorido por regime detectado.
def plot_regime_timeline(
    returns: Union[pd.Series, np.ndarray],
    regimes: Union[pd.Series, np.ndarray],
    n_regimes: Optional[int] = None,
    regime_labels: Optional[Dict[int, str]] = None,
    title: str = "Regimes de Mercado",
    figsize: Tuple = (12, 4),
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    r   = np.asarray(returns).flatten()
    reg = np.asarray(regimes).flatten()
    n   = min(len(r), len(reg))
    r, reg = r[:n], reg[:n]
    idx = returns.index[:n] if isinstance(returns, pd.Series) else np.arange(n)

    unique_regimes = sorted(set(reg.astype(int)))
    if n_regimes is None:
        n_regimes = len(unique_regimes)
    colors_reg = ['#DBEAFE', '#FEE2E2', '#DCFCE7', '#FEF9C3', '#F3E8FF']
    if regime_labels is None:
        regime_labels = {r: f'Regime {r}' for r in unique_regimes}

    fig, ax = plt.subplots(figsize=figsize)

    prev_r, prev_i = reg[0], 0
    for i in range(1, n):
        if reg[i] != prev_r or i == n - 1:
            color = colors_reg[int(prev_r) % len(colors_reg)]
            ax.axvspan(idx[prev_i], idx[min(i, n - 1)],
                       alpha=0.35, color=color)
            prev_r, prev_i = reg[i], i

    ax.plot(idx, r * 100, color=PALETTE[0], linewidth=0.8, alpha=0.8)
    ax.axhline(0, color='black', linewidth=0.5, linestyle='--')

    patches = [mpatches.Patch(color=colors_reg[r % len(colors_reg)],
                               label=regime_labels[r])
               for r in unique_regimes]
    ax.legend(handles=patches, loc='upper left')
    ax.set_title(title)
    ax.set_ylabel("Retorno (%)")
    ax.set_xlabel("Data")
    _save(fig, save_path)
    return fig


# 13. VaR/ES comparison (métodos)

# Bar chart comparando VaR e ES entre métodos.
def plot_var_es_comparison(
    comparison_df: pd.DataFrame,
    title: str = "Comparação VaR/ES por Estrutura de Dependência",
    figsize: Tuple = (8, 5),
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    methods = comparison_df['Method'].tolist()
    var_vals = (comparison_df['VaR'].abs() * 100).tolist()
    es_vals  = (comparison_df['ES'].abs() * 100).tolist()

    x = np.arange(len(methods))
    w = 0.35

    fig, ax = plt.subplots(figsize=figsize)
    bars1 = ax.bar(x - w/2, var_vals, w, label='VaR', color=PALETTE[0], alpha=0.85)
    bars2 = ax.bar(x + w/2, es_vals,  w, label='ES',  color=PALETTE[1], alpha=0.85)

    for bar in list(bars1) + list(bars2):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.02,
                f'{h:.2f}%', ha='center', va='bottom', fontsize=8)

    ax.set_title(title)
    ax.set_ylabel("Perda (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.legend()
    _save(fig, save_path)
    return fig


# 14. Radar chart de performance

# Radar chart comparando estratégias em múltiplas métricas normalizadas.
def plot_performance_radar(
    performance_df: pd.DataFrame,
    metrics: Optional[List[str]] = None,
    title: str = "Radar de Performance",
    figsize: Tuple = (7, 7),
    save_path: Optional[Union[str, Path]] = None,
) -> plt.Figure:
    if metrics is None:
        metrics = [c for c in [
            'Sharpe', 'Sortino', 'Calmar', 'Ann. Return (%)', 'Omega Ratio'
        ] if c in performance_df.columns]

    if len(metrics) < 3:
        logger.warning("Radar chart requer >=3 métricas disponíveis")
        return plt.figure()

    df = performance_df[metrics].copy()

    df_norm = (df - df.min()) / (df.max() - df.min() + 1e-12)

    N = len(metrics)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))

    for (label, row), color in zip(df_norm.iterrows(), PALETTE):
        values = row.tolist() + row.tolist()[:1]
        ax.plot(angles, values, color=color, linewidth=2, label=label)
        ax.fill(angles, values, color=color, alpha=0.08)

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, size=9)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(['25%', '50%', '75%', '100%'], size=7)
    ax.set_title(title, pad=15)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    _save(fig, save_path)
    return fig


# 15. Exportador batch

# Gera e salva todas as figuras padrão do pipeline.
def save_all_figures(
    returns_dict: Dict[str, Union[pd.Series, np.ndarray]],
    benchmark: Optional[Union[pd.Series, np.ndarray]] = None,
    var_forecasts: Optional[Union[pd.Series, np.ndarray]] = None,
    output_dir: Union[str, Path] = 'results/figures',
    fmt: str = 'pdf',
) -> Dict[str, Path]:
    out = Path(output_dir)
    saved = {}

    def _path(name):
        return out / f"{name}.{fmt}"

    plot_cumulative_returns(returns_dict,
                            save_path=_path('cumulative_returns'))
    saved['cumulative_returns'] = _path('cumulative_returns')

    plot_drawdowns(returns_dict,
                   save_path=_path('drawdowns'))
    saved['drawdowns'] = _path('drawdowns')

    plot_rolling_sharpe(returns_dict,
                        save_path=_path('rolling_sharpe'))
    saved['rolling_sharpe'] = _path('rolling_sharpe')

    if var_forecasts is not None:
        first_label = list(returns_dict.keys())[0]
        plot_var_backtesting(
            list(returns_dict.values())[0],
            var_forecasts,
            label=first_label,
            save_path=_path('var_backtesting'),
        )
        saved['var_backtesting'] = _path('var_backtesting')

        plot_violation_clustering(
            list(returns_dict.values())[0],
            var_forecasts,
            save_path=_path('violation_clustering'),
        )
        saved['violation_clustering'] = _path('violation_clustering')

    logger.info(f"Figuras salvas em {out}: {list(saved.keys())}")
    return saved


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
                "Passe --data-dir <pasta> ou execute a partir da raiz do projeto."
            )

    _panel = pd.read_csv(_panel_path, index_col=0, parse_dates=True)
    _STRATEGY  = 'Minimum Variance'
    _BENCHMARK = 'Equal Weight'
    _common = _panel[_STRATEGY].dropna().index.intersection(_panel[_BENCHMARK].dropna().index)
    ret_a = _panel.loc[_common, _STRATEGY].rename(_STRATEGY)
    ret_b = _panel.loc[_common, _BENCHMARK].rename(_BENCHMARK)

    _window = 252
    var_f = (-ret_a.rolling(_window).quantile(0.05)).shift(1).reindex(ret_a.index)

    logging.getLogger(__name__).info(
        f"Retornos reais: {_STRATEGY} vs {_BENCHMARK}  "
        f"({_common[0].date()} → {_common[-1].date()}  {len(_common)} obs)"
    )

    figs = save_all_figures(
        {_STRATEGY: ret_a, _BENCHMARK: ret_b},
        benchmark=ret_b,
        var_forecasts=var_f,
        output_dir='results/figures',
        fmt='png',
    )
    print(f"Figuras geradas: {list(figs.keys())}")
