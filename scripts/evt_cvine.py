
from __future__ import annotations

import importlib.util
import logging
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import stats

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)


# Resolução de caminhos dos módulos do projeto

def _resolve_search_dirs() -> List[Path]:
    here = Path(__file__).resolve().parent
    root = here
    for _ in range(6):
        if (root / "src").exists():
            break
        root = root.parent
    return [
        here,
        root / "src",
        root / "src" / "comparison",
        root / "src" / "backtesting",
        root / "src" / "marginals",
        root / "src" / "copulas",
        root / "src" / "risk",
        root / "src" / "optimization",
        root / "src" / "utils",
        root / "scripts",
    ]


_SEARCH = _resolve_search_dirs()


def _import_module(name: str, search_dirs: List[Path] = _SEARCH):
    if name in sys.modules:
        return sys.modules[name]
    for d in search_dirs:
        s = str(d)
        if s not in sys.path:
            sys.path.insert(0, s)
    for d in search_dirs:
        p = d / f"{name}.py"
        if p.exists():
            spec = importlib.util.spec_from_file_location(name, p)
            mod  = importlib.util.module_from_spec(spec)
            sys.modules[name] = mod
            spec.loader.exec_module(mod)
            return mod
    raise ImportError(
        f"Módulo '{name}' não encontrado em {[str(d) for d in search_dirs]}."
    )


# Helpers de fitting (top-level para serem picklable pelo joblib)

# PIT empírico: (T, d) → (T, d) em (0, 1).
def _rank_pit(data: np.ndarray) -> np.ndarray:
    T = data.shape[0]
    U = np.zeros_like(data, dtype=float)
    for j in range(data.shape[1]):
        U[:, j] = stats.rankdata(data[:, j]) / (T + 1)
    return np.clip(U, 1e-10, 1 - 1e-10)


# Função executada em paralelo por período de rebalanceamento.
def _fit_one_period(
    train_values: np.ndarray,
    asset_names: List[str],
    search_dirs: List[str],
    garch_dist: str,
    max_trees: int,
    min_tau: float,
    confidence_level: float,
    confidence_levels_bt: Tuple[float, ...],
    n_simulations: int,
    max_weight: float,
    seed: int,
    prev_weights: np.ndarray,
) -> Dict:
    _dirs = [Path(d) for d in search_dirs]
    for _d in search_dirs:
        if _d not in sys.path:
            sys.path.insert(0, _d)

    def _imp(name):
        return _import_module(name, _dirs)

    d = len(asset_names)
    train_df = pd.DataFrame(train_values, columns=asset_names)

    result = {
        "weights":      prev_weights.copy(),
        "risk_metrics": {f"var_{int(cl*100)}": np.nan for cl in confidence_levels_bt}
                      | {f"es_{int(cl*100)}":  np.nan for cl in confidence_levels_bt},
        "ok": False,
    }

    try:
        SemiParam = _imp("semi_parametric").SemiParametricGARCH_EVT
        marginal_models = {}
        Z = np.full_like(train_values, np.nan, dtype=float)

        for j, col in enumerate(asset_names):
            series = train_df[col].dropna()
            valid_pos = series.index.to_numpy()

            m = SemiParam()
            try:
                m.fit(
                    series,
                    dist=garch_dist,
                    threshold_method="quantile",
                    left_quantile=0.05,
                    right_quantile=0.95,
                    run_diagnostics=False,
                    plot_diagnostics=False,
                )
            except Exception:
                if not hasattr(m, "returns") or m.returns is None:
                    m.returns = series

            marginal_models[col] = m

            z = None
            for attr in ("std_residuals", "standardized_residuals", "residuals"):
                if hasattr(m, attr):
                    arr = np.asarray(getattr(m, attr)).ravel()
                    if len(arr) >= len(series) * 0.8:
                        z = arr[-len(series):]
                        break
            if z is None:
                z = series.values

            if len(z) != len(valid_pos):
                n = min(len(z), len(valid_pos))
                z = z[-n:]
                valid_pos = valid_pos[-n:]
            Z[valid_pos, j] = z

        Z_df = pd.DataFrame(Z, columns=asset_names).ffill().bfill()
        if Z_df.isna().any().any():
            Z_df = Z_df.fillna(
                pd.DataFrame(train_values, columns=asset_names)
            )
        Z = Z_df.to_numpy()

    except Exception as e:
        import traceback
        logger.warning(f"GARCH fit falhou: {e}\n{traceback.format_exc()}")
        marginal_models = {col: None for col in asset_names}
        Z = train_values.copy()

    U = _rank_pit(Z)

    try:
        CVine = _imp("vine_copulas").CVineCopula
        vine  = CVine(n_dim=U.shape[1])
        vine.fit(U, max_trees=max_trees, min_tau=min_tau)
    except Exception as e:
        import traceback; tb = traceback.format_exc()
        logger.warning(f"Vine fit falhou: {e} -- {tb}")
        return result

    try:
        EVTOpt = _imp("evt_cvar_optimizer").EVTCVaROptimizer
        opt = EVTOpt(
            confidence_level=confidence_level,
            n_simulations=n_simulations,
            seed=seed,
            max_weight=max_weight,
        )
        opt.fit(train_df, vine, marginal_models, asset_names)
        res_opt   = opt.optimize_min_cvar()
        w_dict    = res_opt.get("weights")
        w_new = (
            np.array([w_dict[name] for name in asset_names], dtype=float)
            if isinstance(w_dict, dict) and all(name in w_dict for name in asset_names)
            else None
        )

        if w_new is not None and len(w_new) == d and np.isfinite(w_new).all():
            w_new = np.clip(w_new, 0, max_weight)
            w_new /= w_new.sum()
            result["weights"] = w_new
        else:
            logger.warning(f"optimize_min_cvar retornou pesos inválidos: {w_new}")
    except Exception as e:
        import traceback; tb = traceback.format_exc()
        logger.warning(f"Otimização falhou: {e} -- {tb}")

    try:
        sim_returns_fn = _imp("evt_cvar_optimizer").simulate_portfolio_returns
        compute_cvar_fn = _imp("evt_cvar_optimizer").compute_cvar

        risk = {}
        _port_r_diag = None
        for cl in confidence_levels_bt:
            label = int(cl * 100)
            try:
                port_r = sim_returns_fn(
                    result["weights"], opt._U_sim,
                    train_df, marginal_models, asset_names,
                )
                if _port_r_diag is None:
                    _port_r_diag = port_r
                    _finite = port_r[np.isfinite(port_r)]
                    if len(_finite) > 0:
                        logger.info(
                            f"[DIAG] port_r: mean={_finite.mean():.6f}  "
                            f"std={_finite.std():.6f}  "
                            f"p01={np.percentile(_finite,1):.6f}  "
                            f"p99={np.percentile(_finite,99):.6f}"
                        )
                var, es = compute_cvar_fn(port_r, cl)
                risk[f"var_{label}"] = float(var)
                risk[f"es_{label}"]  = float(es)
            except Exception as e:
                logger.debug(f"VaR/ES cl={cl} falhou: {e}")
                risk[f"var_{label}"] = np.nan
                risk[f"es_{label}"]  = np.nan

        result["risk_metrics"] = risk
        result["ok"] = True

    except Exception as e:
        logger.debug(f"VaR/ES step falhou: {e}")

    return result


# EVTCVineBacktest

# Walk-forward backtest paralelizado para a estratégia EVT-CVine.
class EVTCVineBacktest:

    def __init__(
        self,
        estimation_window: int     = 252,
        rebalancing_frequency: int = 21,
        window_type: str           = "rolling",
        confidence_level: float    = 0.99,
        confidence_levels_bt: Tuple[float, ...] = (0.95, 0.99),
        n_simulations: int         = 2000,
        max_weight: float          = 0.40,
        max_trees: int             = 3,
        min_tau: float             = 0.05,
        garch_dist: str            = "skewt",
        risk_free_rate: float      = 0.1075,
        seed: int                  = 42,
        n_jobs: int                = -1,
        joblib_backend: str        = "loky",
        verbose: bool              = True,
    ):
        self.estimation_window     = estimation_window
        self.rebalancing_frequency = rebalancing_frequency
        self.window_type           = window_type
        self.confidence_level      = confidence_level
        self.confidence_levels_bt  = confidence_levels_bt
        self.n_simulations         = n_simulations
        self.max_weight            = max_weight
        self.max_trees             = max_trees
        self.min_tau               = min_tau
        self.garch_dist            = garch_dist
        self.rf                    = risk_free_rate
        self.seed                  = seed
        self.n_jobs                = n_jobs
        self.joblib_backend        = joblib_backend
        self.verbose               = verbose

    # ── Ponto de entrada ────────────────────────────────────────────────────

    # Executa o walk-forward paralelizado e retorna BacktestResult.
    def run(
        self,
        returns_df: pd.DataFrame,
        strategy_name: str = "EVT-CVine",
        benchmark_returns: Optional[pd.Series] = None,
        initial_weights: Optional[np.ndarray] = None,
        save_path: Optional[Path] = None,
    ):
        BacktestResult, PeriodResult, VaRBacktestEngine, PerformanceEngine = \
            self._get_backtest_classes()

        T, d          = returns_df.shape
        asset_names   = list(returns_df.columns)
        search_dirs   = [str(p) for p in _SEARCH]

        if initial_weights is None:
            initial_weights = np.ones(d) / d

        rebal_indices = list(range(
            self.estimation_window,
            T - 1,
            self.rebalancing_frequency,
        ))

        logger.info(
            f"\nEVTCVineBacktest [{strategy_name}]  "
            f"T={T}  d={d}  window={self.window_type}  "
            f"est_window={self.estimation_window}  "
            f"rebal_freq={self.rebalancing_frequency}  "
            f"n_rebal={len(rebal_indices)}  "
            f"n_jobs={self.n_jobs}  backend={self.joblib_backend}"
        )

        ew = initial_weights.copy()

        _du = _import_module("dynamic_universe")
        eligible_assets, expand_weights = _du.eligible_assets, _du.expand_weights
        MIN_COVERAGE = 0.95
        MIN_ASSETS   = 5

        eligible_by_period: List[List[str]] = []
        job_args = []
        skipped_periods: List[int] = []
        for i, t_idx in enumerate(rebal_indices):
            t_start  = (t_idx - self.estimation_window
                        if self.window_type == "rolling" else 0)
            window_slice = returns_df.iloc[t_start:t_idx]
            asset_names_t = eligible_assets(window_slice, min_coverage=MIN_COVERAGE)

            if len(asset_names_t) < MIN_ASSETS:
                skipped_periods.append(i)

            eligible_by_period.append(asset_names_t)
            train_vals = (window_slice[asset_names_t].values if asset_names_t
                          else np.empty((len(window_slice), 0)))
            job_args.append(dict(
                train_values        = train_vals,
                asset_names         = asset_names_t,
                search_dirs         = search_dirs,
                garch_dist          = self.garch_dist,
                max_trees           = self.max_trees,
                min_tau             = self.min_tau,
                confidence_level    = self.confidence_level,
                confidence_levels_bt= self.confidence_levels_bt,
                n_simulations       = self.n_simulations,
                max_weight          = self.max_weight,
                seed                = self.seed + i,
                prev_weights        = (np.ones(len(asset_names_t)) / len(asset_names_t)
                                        if asset_names_t else np.empty(0)),
            ))

        logger.info(f"  Iniciando fitting paralelo ({len(job_args)} períodos)...")
        fitted = Parallel(
            n_jobs=self.n_jobs,
            backend=self.joblib_backend,
            verbose=5 if self.verbose else 0,
        )(
            delayed(_fit_one_period)(**args) for args in job_args
        )

        n_ok = sum(1 for f in fitted if f["ok"])
        logger.info(f"  Fitting concluído: {n_ok}/{len(fitted)} períodos OK")

        period_results: List = []

        nan_safe_portfolio_return = _du.nan_safe_portfolio_return

        for i, (t_idx, fit) in enumerate(zip(rebal_indices, fitted)):
            asset_names_t = eligible_by_period[i]
            w_sub         = fit["weights"]
            w_current    = expand_weights(w_sub, asset_names_t, asset_names)
            risk_metrics = fit["risk_metrics"]

            t_oos_end = min(t_idx + self.rebalancing_frequency, T)
            oos_df    = returns_df.iloc[t_idx:t_oos_end]

            var_95 = risk_metrics.get("var_95", np.nan)
            var_99 = risk_metrics.get("var_99", np.nan)
            es_95  = risk_metrics.get("es_95",  np.nan)
            es_99  = risk_metrics.get("es_99",  np.nan)

            for t_oos in range(len(oos_df)):
                date  = oos_df.index[t_oos]
                ret_t = nan_safe_portfolio_return(oos_df.iloc[t_oos].values, w_current)

                period_results.append(PeriodResult(
                    date            = date,
                    weights         = w_current.copy(),
                    realized_return = ret_t,
                    var_95          = float(var_95) if np.isfinite(var_95) else np.nan,
                    var_99          = float(var_99) if np.isfinite(var_99) else np.nan,
                    es_95           = float(es_95)  if np.isfinite(es_95)  else np.nan,
                    es_99           = float(es_99)  if np.isfinite(es_99)  else np.nan,
                    violation_95    = (ret_t < -var_95 if np.isfinite(var_95) else False),
                    violation_99    = (ret_t < -var_99 if np.isfinite(var_99) else False),
                    n_train_obs     = len(job_args[i]["train_values"]),
                ))

        logger.info(f"  Períodos OOS: {len(period_results)}")

        result = self._compile(
            strategy_name, period_results, benchmark_returns,
            BacktestResult, VaRBacktestEngine, PerformanceEngine,
        )

        if save_path is not None:
            self._save(result, Path(save_path))

        return result

    # ── Compilação ───────────────────────────────────────────────────────────

    def _compile(
        self, strategy_name, period_results, benchmark_returns,
        BacktestResult, VaRBacktestEngine, PerformanceEngine,
    ):
        if not period_results:
            raise RuntimeError("Nenhum período OOS — verifique os parâmetros.")

        dates   = pd.DatetimeIndex([r.date for r in period_results])
        returns = pd.Series([r.realized_return for r in period_results],
                            index=dates, name=strategy_name)
        var_95  = pd.Series([r.var_95 for r in period_results], index=dates)
        var_99  = pd.Series([r.var_99 for r in period_results], index=dates)
        es_95   = pd.Series([r.es_95  for r in period_results], index=dates)
        es_99   = pd.Series([r.es_99  for r in period_results], index=dates)
        wdf     = pd.DataFrame([r.weights for r in period_results], index=dates)

        var_engine = VaRBacktestEngine(alpha=0.05)

        def _bt(var_s, cl):
            v = var_s.dropna()
            common = returns.index.intersection(v.index)
            if len(common) < 10:
                return {}
            return var_engine.run(
                returns.loc[common].values,
                v.loc[common].values,
                confidence_level=cl,
                name=f"{strategy_name} {cl:.0%}",
            )

        perf = PerformanceEngine(
            risk_free_rate=self.rf, periods_per_year=252
        ).compute(returns, benchmark_returns, wdf)
        perf["risk_free_rate"] = self.rf

        return BacktestResult(
            strategy_name       = strategy_name,
            period_results      = period_results,
            returns_series      = returns,
            weights_df          = wdf,
            var_95_series       = var_95,
            var_99_series       = var_99,
            es_95_series        = es_95,
            es_99_series        = es_99,
            performance_metrics = perf,
            var_backtest_95     = _bt(var_95, 0.95),
            var_backtest_99     = _bt(var_99, 0.99),
        )

    # ── Save ─────────────────────────────────────────────────────────────────

    def _save(self, result, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        result.returns_series.to_frame("return").to_csv(
            path / "evt_cvine_returns.csv")
        result.weights_df.to_csv(path / "evt_cvine_weights.csv")
        pd.DataFrame({
            "var_95": result.var_95_series,
            "var_99": result.var_99_series,
            "es_95":  result.es_95_series,
            "es_99":  result.es_99_series,
        }).to_csv(path / "evt_cvine_risk_series.csv")
        logger.info(f"EVT-CVine backtest salvo em {path}")

    # ── Imports lazy ─────────────────────────────────────────────────────────

    @staticmethod
    def _get_backtest_classes():
        for _d in _SEARCH:
            s = str(_d)
            if s not in sys.path:
                sys.path.insert(0, s)
        mod = _import_module("backtest_rolling")
        return (
            mod.BacktestResult,
            mod.PeriodResult,
            mod.VaRBacktestEngine,
            mod.PerformanceEngine,
        )


# Integração direta com RollingBacktestRunner

# Pipeline completo: estratégias risk-based + EVT-CVine.
def run_full_comparison(
    returns_df: pd.DataFrame,
    benchmark_returns: Optional[pd.Series] = None,
    output_dir: Optional[Path] = None,
    runner_kwargs: Optional[Dict] = None,
    cvine_kwargs: Optional[Dict] = None,
) -> Dict:
    mod = _import_module("backtest_rolling")
    runner = mod.RollingBacktestRunner(**(runner_kwargs or {}))
    runner.run_all(returns_df, benchmark_returns=benchmark_returns)

    cvine_bt = EVTCVineBacktest(**(cvine_kwargs or {}))
    cvine_result = cvine_bt.run(
        returns_df,
        strategy_name="EVT-CVine",
        benchmark_returns=benchmark_returns,
        save_path=output_dir,
    )

    runner._backtest_results["cvar_evt"] = cvine_result
    panel = runner.returns_panel()
    logger.info(f"Panel final: {panel.shape}  colunas={list(panel.columns)}")

    if output_dir is not None:
        runner.save(output_dir)

    return {"runner": runner, "cvine_result": cvine_result, "panel": panel}


# Carregamento de dados reais (data/processed/returns.parquet)

# Carrega retornos reais de data/processed/returns.parquet.
def load_real_returns(
    tickers: Optional[List[str]] = None,
    parquet_path: Optional[Path] = None,
) -> pd.DataFrame:
    tickers = tickers or ["BRL=X", "^BVSP", "EWZ"]

    if parquet_path is None:
        here = Path(__file__).resolve().parent
        parquet_path = None
        for _ in range(6):
            candidate = here / "data" / "processed" / "returns.parquet"
            if candidate.exists():
                parquet_path = candidate
                break
            here = here.parent
        if parquet_path is None:
            raise FileNotFoundError(
                "data/processed/returns.parquet não encontrado subindo a "
                "partir deste arquivo. Passe parquet_path explicitamente."
            )

    logger.info(f"Carregando retornos reais de: {parquet_path}")
    raw = pd.read_parquet(parquet_path)

    missing = [t for t in tickers if t not in raw.columns]
    if missing:
        raise ValueError(
            f"Tickers {missing} não encontrados em {parquet_path}. "
            f"Disponíveis: {list(raw.columns)}"
        )

    panel = raw[tickers]

    daily = panel.groupby(panel.index.date).sum()
    daily.index = pd.DatetimeIndex(daily.index)
    daily.index.name = "Date"

    daily = daily.replace(0.0, np.nan)

    n_before = len(daily)
    daily = daily.dropna(how="all")
    n_after = len(daily)
    if n_after < n_before:
        logger.info(f"  Removidos {n_before - n_after} dias sem nenhuma observação real.")

    for t in tickers:
        n_nan = daily[t].isna().sum()
        logger.info(f"  {t}: {n_nan}/{len(daily)} dias sem observação (NaN)")

    return daily


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(name)-20s  %(levelname)s  %(message)s",
    )

    use_real = "--real" in sys.argv

    if use_real:
        logger.info("=== Backtest EVTCVineBacktest — dados reais ===")
        returns = load_real_returns(["BRL=X", "^BVSP", "EWZ"])
        logger.info(
            f"  Período disponível: {returns.index[0].date()} → "
            f"{returns.index[-1].date()}  ({len(returns)} obs)"
        )
        bt_kwargs = dict(
            estimation_window     = 750,
            rebalancing_frequency = 63,
            n_simulations         = 10000,
            confidence_level      = 0.99,
            max_weight            = 0.60,
            max_trees             = 2,
            n_jobs                = -1,
            joblib_backend        = "loky",
            verbose               = True,
        )
    else:
        logger.info("=== Smoke test EVTCVineBacktest — dados sintéticos ===")
        np.random.seed(42)
        T, d    = 600, 3
        tickers = ["IBRX50", "GLD", "SPY"]
        vols    = np.array([0.015, 0.010, 0.012])
        corr    = np.array([[1.0, 0.10, 0.25],
                            [0.10, 1.0, 0.08],
                            [0.25, 0.08, 1.0]])
        cov     = np.outer(vols, vols) * corr
        raw     = np.random.multivariate_normal(np.zeros(d) + 3e-4, cov, T)
        returns = pd.DataFrame(
            raw, columns=tickers,
            index=pd.bdate_range("2020-01-01", periods=T),
        )
        bt_kwargs = dict(
            estimation_window     = 252,
            rebalancing_frequency = 63,
            n_simulations         = 500,
            confidence_level      = 0.99,
            max_weight            = 0.60,
            max_trees             = 2,
            n_jobs                = -1,
            joblib_backend        = "loky",
            verbose               = True,
        )

    bt = EVTCVineBacktest(**bt_kwargs)

    try:
        result = bt.run(returns, strategy_name="EVT-CVine")
        r = result.returns_series
        logger.info("\n=== Resultado ===")
        logger.info(f"  N OOS     : {len(r)}")
        logger.info(f"  Período   : {r.index[0].date()} → {r.index[-1].date()}")
        logger.info(f"  Ret. anual: {result.annualized_return:.2%}")
        logger.info(f"  Vol. anual: {result.annualized_vol:.2%}")
        logger.info(f"  Sharpe    : {result.sharpe:.3f}")
        logger.info(f"  Max DD    : {result.performance_metrics.get('max_drawdown', 0):.2%}")
        logger.info(f"  VaR 99%   : {result.var_backtest_99}")
        logger.info("\nBacktest concluído.")
    except Exception as e:
        logger.error(f"Backtest falhou: {e}", exc_info=True)
        sys.exit(1)
