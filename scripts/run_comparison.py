
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR / "src"))
sys.path.insert(0, str(BASE_DIR / "scripts"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)s  %(levelname)s  %(message)s",
)
logger = logging.getLogger(__name__)

RESULTS_DIR     = BASE_DIR / "results"
COMPARISON_DIR  = RESULTS_DIR / "comparison"
DATA_RAW        = BASE_DIR / "data" / "raw"


# 1. Carregamento e pré-processamento de dados

# Lê um arquivo xlsx exportado do Economatica (cabeçalho na linha 3,
def _read_economatica_xlsx(path: str, col_name: str) -> pd.Series:
    df = pd.read_excel(path, header=2, usecols=[0, 4])
    df.columns = ["Data", "Fechamento"]
    df = df[df["Data"] != "Data"]
    df = df[df["Fechamento"].notna() & (df["Fechamento"] != "-")]
    df["Data"]       = pd.to_datetime(df["Data"], errors="coerce")
    df["Fechamento"] = pd.to_numeric(df["Fechamento"],    errors="coerce")
    df = df.dropna().sort_values("Data").set_index("Data")
    return df["Fechamento"].rename(col_name)


ASSET_CURRENCY = {
    "IBRX50": "BRL",
    "IBOV":   "BRL",
    "EWZ":    "USD",
    "GLD":    "USD",
    "SPY":    "USD",
}


# Carrega a série de câmbio USD/BRL (fonte cambial: DOLOF.xlsx, dólar PTAX
def load_usd_brl_fx(args: argparse.Namespace) -> pd.Series:
    data_dir = Path(getattr(args, "data_dir", None) or DATA_RAW)
    candidates = []
    if getattr(args, "fx_file", None):
        candidates.append(Path(args.fx_file))
    candidates.append(data_dir / "DOLOF.xlsx")

    for path in candidates:
        if path.exists():
            s = _read_economatica_xlsx(str(path), "USD_BRL")
            fx_ret = np.log(s / s.shift(1)).iloc[1:]
            logger.info(
                f"  Câmbio USD/BRL carregado de {path} (DOLOF, Dolar Ptax "
                f"Venda, Em R$ Real): {len(fx_ret)} obs "
                f"({fx_ret.index.min().date()} → {fx_ret.index.max().date()})"
            )
            return fx_ret

    raise FileNotFoundError(
        "EWZ, GLD e SPY estão cotados em USD no Economatica, mas o arquivo "
        "de câmbio DOLOF.xlsx não foi encontrado para convertê-los à mesma "
        "unidade de conta de IBRX50/IBOV (BRL) antes de combiná-los no "
        "portfólio.\n"
        f"Procurado em: {[str(p) for p in candidates]}\n"
        "Coloque o arquivo DOLOF.xlsx (mesmo formato dos demais arquivos: "
        "cabeçalho na linha 3, coluna 'Fechamento') em data/raw/, ou aponte "
        "o caminho via --fx-file.\n"
        "Para prosseguir DELIBERADAMENTE sem converter (não recomendado — "
        "mistura BRL e USD na mesma otimização e no Sharpe), use "
        "--allow-mixed-currency."
    )


# Carrega o universo de 81 ações do IBrX-50 (data/raw_b3_ibrx50/*.xlsx,
def load_returns(args: argparse.Namespace) -> pd.DataFrame:
    data_dir = Path(getattr(args, "data_dir", None) or DATA_RAW.parent)
    stocks_dir = data_dir / "raw_b3_ibrx50"
    if not stocks_dir.exists():
        stocks_dir = data_dir if data_dir.name == "raw_b3_ibrx50" else stocks_dir
    if not stocks_dir.exists():
        raise FileNotFoundError(
            f"Pasta de ações não encontrada: {stocks_dir}\n"
            f"Coloque os 81 xlsx do IBrX-50 em data/raw_b3_ibrx50/, ou "
            f"aponte --data-dir para o diretório que a contém."
        )

    import sys as _sys_sa, pathlib as _pathlib_sa
    _data_module_dir = str(_pathlib_sa.Path(__file__).resolve().parent.parent / "src" / "data")
    if _data_module_dir not in _sys_sa.path:
        _sys_sa.path.insert(0, _data_module_dir)
    from split_adjustment import build_adjusted_panel

    logger.info(f"Carregando universo de ações (Economatica): {stocks_dir}")
    raw_panel, adj_panel, events_df, unconfirmed_df = build_adjusted_panel(str(stocks_dir))
    logger.info(
        f"  {adj_panel.shape[1]} ativos carregados. "
        f"Correções de split aplicadas: {len(events_df)} eventos em "
        f"{events_df['ticker'].nunique() if len(events_df) else 0} tickers. "
        f"Candidatos não confirmados (não corrigidos): {len(unconfirmed_df)}."
    )

    audit_dir = RESULTS_DIR / "data_audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    events_df.to_csv(audit_dir / "split_corrections_applied.csv", index=False)
    unconfirmed_df.to_csv(audit_dir / "split_unconfirmed_anomalies.csv", index=False)

    prices = adj_panel.sort_index()

    if args.data_start:
        prices = prices.loc[args.data_start:]

    returns = np.log(prices / prices.shift(1)).iloc[1:]

    logger.info(f"  Retornos: {returns.shape}  "
                f"({returns.index[0].date()} → {returns.index[-1].date()})")
    logger.info(
        f"  Cobertura média por ativo: {returns.notna().mean().mean():.1%} "
        f"(esperado <100%: universo não-balanceado por desenho)"
    )
    return returns


# Benchmark = portfólio Equal Weight (1/N, rebalanceado), conforme
def compute_ew_benchmark(returns: pd.DataFrame, args: argparse.Namespace) -> pd.Series:
    import sys as _sys, pathlib as _pathlib
    _comparison_dir = str(_pathlib.Path(__file__).resolve().parent.parent / "src" / "comparison")
    if _comparison_dir not in _sys.path:
        _sys.path.insert(0, _comparison_dir)
    from backtest_rolling import RollingBacktestRunner

    ew_runner = RollingBacktestRunner(
        estimation_window    = args.window,
        rebalancing_frequency= args.rebal,
        window_type          = args.window_type,
        max_weight            = args.max_weight,
        min_weight            = args.min_weight,
        cov_estimator         = args.cov_estimator,
        risk_fn_type          = args.risk_fn,
        risk_free_rate        = args.risk_free,
        strategies            = ["ew"],
        n_restarts            = args.n_restarts,
        verbose               = False,
    )
    ew_runner.run_all(returns, benchmark_returns=None)
    bm = ew_runner._backtest_results["ew"].returns_series
    logger.info(f"  Benchmark (Equal Weight portfolio): {len(bm)} obs")
    return bm


# Lê a taxa Selic do xlsx do Economatica (Selic_252d.xlsx).
def load_selic(
    args: argparse.Namespace,
    returns_index: Optional[pd.DatetimeIndex] = None,
) -> float:
    data_dir = Path(getattr(args, "data_dir", None) or DATA_RAW)
    path = data_dir / "Selic_252d.xlsx"
    try:
        s = _read_economatica_xlsx(str(path), "Selic")
        if s.max() > 1:
            s = s / 100.0

        if returns_index is not None:
            s_period = s.reindex(returns_index).ffill().dropna()
            if s_period.empty:
                raise ValueError(
                    "Selic sem sobreposição com o período de retornos "
                    f"({returns_index.min()} a {returns_index.max()})."
                )
        else:
            s_period = s

        rf = float(s_period.mean())
        logger.info(
            f"  Selic: {rf:.4%} a.a. "
            f"(média diária, {s_period.index.min().date()} a "
            f"{s_period.index.max().date()}, {len(s_period)} obs)"
        )
        return rf
    except Exception as e:
        logger.warning(f"  Selic não carregada ({e}). Usando 10.75% a.a.")
        return 0.1075


# 2. Estratégias risk-based (backtest rolling)

# Executa RollingBacktestRunner para EW, MV, RP, MD, MDE.
def run_risk_based_backtests(
    returns,
    benchmark,
    args: argparse.Namespace,
):
    import sys as _sys, pathlib as _pathlib
    _comparison_dir = str(_pathlib.Path(__file__).resolve().parent.parent / "src" / "comparison")
    if _comparison_dir not in _sys.path:
        _sys.path.insert(0, _comparison_dir)
    from backtest_rolling import RollingBacktestRunner

    logger.info("\n" + "=" * 60)
    logger.info("  BACKTEST ROLLING — ESTRATÉGIAS RISK-BASED")
    logger.info("=" * 60)
    logger.info(f"  Estratégias   : {args.strategies}")
    logger.info(f"  Janela estim. : {args.window} dias")
    logger.info(f"  Rebalanceamento: {args.rebal} dias")
    logger.info(f"  Tipo janela   : {args.window_type}")
    logger.info(f"  Max weight    : {args.max_weight:.0%}")
    logger.info(f"  Cov estimator : {args.cov_estimator}")
    logger.info(f"  Risk fn       : {args.risk_fn}")
    logger.info("=" * 60)

    runner = RollingBacktestRunner(
        estimation_window    = args.window,
        rebalancing_frequency= args.rebal,
        window_type          = args.window_type,
        max_weight           = args.max_weight,
        min_weight           = args.min_weight,
        cov_estimator        = args.cov_estimator,
        risk_fn_type         = args.risk_fn,
        risk_free_rate       = args.risk_free,
        strategies           = args.strategies,
        n_restarts           = args.n_restarts,
        verbose              = args.verbose,
    )

    runner.run_all(returns, benchmark_returns=benchmark)
    return runner


# ComparisonPipeline — pipeline CVaR-EVT autossuficiente
# Contém apenas os steps necessários para o estudo comparativo:
#   step2b  — GARCH (marginals/garch.py)
#   step3   — PIT → pseudo-observações (copulas/estimation.py)
#   step4   — ajuste da vine/DCC/factor cópula (copulas/vine_copulas.py)
#   step6a  — VaR/ES via simulação Monte Carlo (risk/copula_var_es.py)
#   save / load state — joblib pickle do estado mínimo
# Não depende de main_pipeline.py em nenhum momento.

# Wrapper leve compatível com CopulaEVTRisk.
class _MarginalModelWrapper:
    def __init__(self, returns, std_residuals=None):
        self.returns       = np.asarray(returns, dtype=float)
        self.std_residuals = (
            np.asarray(std_residuals, dtype=float)
            if std_residuals is not None
            else self.returns.copy()
        )
        self.gpd_left  = None
        self.gpd_right = None

    def __repr__(self):
        return f"<_MarginalModelWrapper n={len(self.returns)}>"


# Pipeline CVaR-EVT enxuto para o estudo comparativo.
class ComparisonPipeline:

    # Garante que src/ está no sys.path ANTES de qualquer import local
    # e limpa cache de módulos PyPI que colidem com src/copulas/.
    # Problema: o pacote PyPI "copulas" (se instalado) pode ter sido
    # importado antes desta classe ser instanciada, ficando cacheado
    # em sys.modules como "copulas".  Subimports subsequentes como
    # "copulas.vine_copulas" resolvem para o PyPI em vez de src/.
    # Solução: remover entradas "copulas.*" do cache antes de importar.
    @staticmethod
    def _ensure_src_path() -> None:
        src_dir = str(BASE_DIR / "src")
        if src_dir in sys.path:
            sys.path.remove(src_dir)
        sys.path.insert(0, src_dir)

        stale = [k for k in sys.modules if k == "copulas" or k.startswith("copulas.")]
        for k in stale:
            del sys.modules[k]

    def __init__(
        self,
        garch_model: str   = "gjr",
        pit_method:  str   = "empirical",
        max_trees:   int   = 3,
        min_tau:     float = 0.05,
        n_jobs:      int   = 3,
    ):
        self._ensure_src_path()

        self.returns:            pd.DataFrame = None
        self.asset_names:        list         = []
        self.std_residuals_df:   pd.DataFrame = None
        self.conditional_vol_df: pd.DataFrame = None
        self.garch_results:      dict         = {}
        self.copula_model                     = None
        self.marginal_models:    dict         = {}
        self.results_:           dict         = {}

        self.garch_model = garch_model
        self.pit_method  = pit_method
        self.max_trees   = max_trees
        self.min_tau     = min_tau
        self.n_jobs      = n_jobs

    # step2b — GARCH sobre log-retornos já carregados

    def step2b_fit_garch(self) -> dict:
        from marginals.garch import GARCHFitter, fit_garch_all

        logger.info(f"\n[ComparisonPipeline] step2b — GARCH ({self.garch_model.upper()})")
        std_resid, cond_vol, garch_results = fit_garch_all(
            returns_df = self.returns,
            model_type = self.garch_model,
            dist       = "skewt",
            save_path  = str(RESULTS_DIR / "garch"),
        )
        self.std_residuals_df   = std_resid
        self.conditional_vol_df = cond_vol
        self.garch_results      = garch_results

        fitter = GARCHFitter(model_type=self.garch_model)
        fitter.results_    = garch_results
        fitter._is_fitted  = True
        persist = fitter.get_persistence()
        logger.info(f"  Persistência média  : {persist.mean():.4f}")
        logger.info(f"  Persistência máx    : {persist.max():.4f} ({persist.idxmax()})")
        if (persist > 0.999).any():
            logger.warning(
                f"  Ativos com persistência ≈ 1: "
                f"{list(persist[persist > 0.999].index)}"
            )

        self._build_marginal_models()

        return garch_results

    # Popula self.marginal_models com _MarginalModelWrapper por ativo.
    def _build_marginal_models(self) -> None:
        self.marginal_models = {}
        for ticker in self.returns.columns:
            ret = self.returns[ticker].dropna()
            std = (
                self.std_residuals_df[ticker].dropna()
                if self.std_residuals_df is not None
                   and ticker in self.std_residuals_df.columns
                else None
            )
            self.marginal_models[ticker] = _MarginalModelWrapper(ret, std)
        logger.info(
            f"  marginal_models: {list(self.marginal_models.keys())}"
        )

    # step3 — PIT: resíduos padronizados → pseudo-observações U(0,1)

    # Probability Integral Transform (PIT): converte resíduos padronizados
    def step3_transform_to_uniform(self) -> np.ndarray:
        from scipy.stats import rankdata, norm as _norm

        logger.info(f"\n[ComparisonPipeline] step3 — PIT ({self.pit_method})")

        data = (
            self.std_residuals_df
            if self.std_residuals_df is not None and not self.std_residuals_df.empty
            else self.returns
        )
        arr = data.values
        T   = arr.shape[0]

        if self.pit_method == "empirical":
            pseudo_obs = np.apply_along_axis(
                lambda col: rankdata(col) / (T + 1), axis=0, arr=arr
            )
        elif self.pit_method == "normal":
            pseudo_obs = np.apply_along_axis(
                lambda col: _norm.cdf(
                    (col - col.mean()) / (col.std(ddof=1) or 1.0)
                ),
                axis=0, arr=arr,
            )
        else:
            raise ValueError(
                f"pit_method inválido: '{self.pit_method}'. "
                "Use 'empirical' ou 'normal'."
            )

        pseudo_obs = np.clip(pseudo_obs, 1e-6, 1 - 1e-6)
        logger.info(
            f"  shape={pseudo_obs.shape}  "
            f"range=[{pseudo_obs.min():.4f}, {pseudo_obs.max():.4f}]"
        )
        return pseudo_obs

    # step4 — ajuste da cópula (cvine / dvine / rvine / dcc / factor)

    def step4_fit_copula(
        self,
        pseudo_obs:   np.ndarray,
        copula_type:  str = "cvine",
        n_simulations: int = 10_000,
        seed:         int = 42,
    ):
        logger.info(
            f"\n[ComparisonPipeline] step4 — {copula_type.upper()}  "
            f"max_trees={self.max_trees}  min_tau={self.min_tau}"
        )

        fit_kwargs = dict(
            families    = ["gaussian", "t", "clayton", "clayton_180",
                           "gumbel", "gumbel_180", "frank"],
            auto_select = True,
            max_trees   = self.max_trees,
            min_tau     = self.min_tau,
            n_jobs      = self.n_jobs,
        )

        if copula_type == "cvine":
            from copulas.vine_copulas import CVineCopula
            self.copula_model = CVineCopula(n_dim=len(self.asset_names))
            self.copula_model.fit(pseudo_obs, **fit_kwargs)
            logger.info(f"  C-Vine: {len(self.copula_model.copulas)} pares")

        elif copula_type == "dvine":
            from copulas.vine_copulas import DVineCopula
            self.copula_model = DVineCopula(n_dim=len(self.asset_names))
            self.copula_model.fit(pseudo_obs, **fit_kwargs)
            logger.info(f"  D-Vine: {len(self.copula_model.copulas)} pares")

        elif copula_type == "rvine":
            from copulas.vine_copulas import RVineCopula
            self.copula_model = RVineCopula(n_dim=len(self.asset_names))
            self.copula_model.fit(pseudo_obs, **fit_kwargs)
            logger.info(f"  R-Vine: {len(self.copula_model._pair_ll)} pares")

        elif copula_type == "dcc":
            from copulas.dynamic_copula import DCCCopula
            dcc = DCCCopula(marginal="t", random_state=seed)
            dcc.fit(pseudo_obs)
            s = dcc.get_summary()
            logger.info(
                f"  DCC: a={s['dcc_a']:.4f}  b={s['dcc_b']:.4f}  "
                f"LL={s['log_likelihood']:.2f}  AIC={s['aic']:.2f}"
            )
            _m = dcc
            class _DCCWrapper:
                def simulate(self, n_samples, seed=None):
                    return _m.simulate(n_sim=n_samples, seed=seed)
                def loglikelihood(self):  return float(_m.log_likelihood_)
                def aic(self):            return float(_m.get_summary()["aic"])
                def __getattr__(self, n): return getattr(_m, n)
            self.copula_model = _DCCWrapper()

        elif copula_type == "factor":
            from copulas.factor_estimation import FactorCopulaEstimator
            fe = FactorCopulaEstimator(n_factors=1, random_state=seed)
            fe.fit(pseudo_obs)
            logger.info(
                f"  Factor: ν={fe.nu_factor_:.2f}  "
                f"LL={fe.log_likelihood_:.2f}  AIC={fe.aic_:.2f}"
            )
            _m = fe
            class _FactorWrapper:
                def simulate(self, n_samples, seed=None):
                    return _m.simulate(n_sim=n_samples, seed=seed)
                def loglikelihood(self):  return float(_m.log_likelihood_)
                def aic(self):            return float(_m.aic_)
                def __getattr__(self, n): return getattr(_m, n)
            self.copula_model = _FactorWrapper()

        else:
            raise ValueError(
                f"copula_type inválido: '{copula_type}'. "
                "Use 'cvine', 'dvine', 'rvine', 'dcc' ou 'factor'."
            )

        return self.copula_model

    # step6a — VaR/ES via simulação Monte Carlo

    def step6a_portfolio_risk(
        self,
        weights:       np.ndarray = None,
        n_simulations: int        = 10_000,
        seed:          int        = 42,
    ) -> dict:
        self._ensure_src_path()
        from risk.copula_var_es import CopulaEVTRisk, OptimizedCopulaRisk

        logger.info("\n[ComparisonPipeline] step6a — VaR/ES")
        if weights is None:
            weights = np.ones(len(self.asset_names)) / len(self.asset_names)

        copula_risk = CopulaEVTRisk()
        copula_risk.fit(
            returns        = self.returns,
            marginal_models= self.marginal_models,
            copula_model   = self.copula_model,
            asset_names    = self.asset_names,
        )
        opt_risk = OptimizedCopulaRisk(copula_risk)

        var_es_df = opt_risk.portfolio_var_es_batch(
            weights,
            confidence_levels    = [0.95, 0.99],
            n_simulations        = n_simulations,
            seed                 = seed,
        )
        logger.info(f"\n{var_es_df}")
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        var_es_df.to_csv(RESULTS_DIR / "portfolio_var_es.csv", index=False)

        self.results_["var_es"]          = var_es_df
        self.results_["optimal_weights"] = weights
        return self.results_

    # save / load state

    def save_state(self, path: Path) -> None:
        import joblib
        joblib.dump({
            "returns":            self.returns,
            "asset_names":        self.asset_names,
            "garch_results":      self.garch_results,
            "std_residuals_df":   self.std_residuals_df,
            "conditional_vol_df": self.conditional_vol_df,
            "copula_model":       self.copula_model,
            "marginal_models":    self.marginal_models,
            "results_":           self.results_,
            "garch_model":  self.garch_model,
            "pit_method":   self.pit_method,
            "max_trees":    self.max_trees,
            "min_tau":      self.min_tau,
            "n_jobs":       self.n_jobs,
        }, path)
        logger.info(f"  Estado salvo em: {path}")

    @classmethod
    def load_state(cls, path: Path) -> "ComparisonPipeline":
        import joblib
        state = joblib.load(path)
        pipeline = cls(
            garch_model = state.get("garch_model", "gjr"),
            pit_method  = state.get("pit_method",  "empirical"),
            max_trees   = state.get("max_trees",   3),
            min_tau     = state.get("min_tau",     0.05),
            n_jobs      = state.get("n_jobs",      3),
        )
        for k, v in state.items():
            setattr(pipeline, k, v)
        if not pipeline.marginal_models and pipeline.returns is not None:
            logger.info("  Reconstruindo marginal_models a partir do estado carregado...")
            pipeline._build_marginal_models()
        logger.info(f"  Estado carregado de: {path}")
        return pipeline


# Instancia e executa ComparisonPipeline (sem dependência de main_pipeline).
def run_or_load_pipeline(returns, args: argparse.Namespace):
    if args.no_pipeline:
        logger.info("Pulando pipeline CVaR-EVT (--no-pipeline).")
        return None, None

    if args.load_state:
        logger.info(f"Carregando estado: {args.load_state}")
        try:
            pipeline = ComparisonPipeline.load_state(Path(args.load_state))
            logger.info("  Estado carregado com sucesso.")
            return pipeline, pipeline.results_
        except Exception as e:
            logger.error(f"  Falha ao carregar estado: {e}")
            return None, None

    logger.info("\n" + "=" * 60)
    logger.info("  PIPELINE CVaR-EVT (ComparisonPipeline)")
    logger.info("=" * 60)
    logger.info(f"  Cópula : {args.copula}")
    logger.info(f"  N sim  : {args.n_sim:,}")
    logger.info(f"  Seed   : {args.seed}")
    logger.info("=" * 60)

    try:
        pipeline = ComparisonPipeline(
            garch_model = args.garch_model,
            pit_method  = args.pit_method,
            max_trees   = args.max_trees,
            min_tau     = getattr(args, "min_tau", 0.05),
            n_jobs      = args.n_jobs,
        )
        pipeline.returns     = returns
        pipeline.asset_names = list(returns.columns)

        pipeline.step2b_fit_garch()
        pseudo_obs = pipeline.step3_transform_to_uniform()
        pipeline.step4_fit_copula(
            pseudo_obs    = pseudo_obs,
            copula_type   = args.copula,
            n_simulations = args.n_sim,
            seed          = args.seed,
        )
        results = pipeline.step6a_portfolio_risk(
            n_simulations = args.n_sim,
            seed          = args.seed,
        )
        return pipeline, results

    except Exception as e:
        logger.error(f"Pipeline CVaR-EVT falhou: {e}")
        import traceback
        traceback.print_exc()
        return None, None


# Função top-level picklável para ProcessPoolExecutor
# DEVE estar em top-level do módulo (não dentro de outra função)

# Ajusta GJR-GARCH em um único ativo. Função top-level para ser picklável
def _garch_fit_single_asset(args_tuple):
    import sys as _sys, pathlib as _pathlib, numpy as _np, pandas as _pd
    _src = str(_pathlib.Path(__file__).resolve().parent.parent / "src")
    if _src not in _sys.path:
        _sys.path.insert(0, _src)

    ticker, ret_values, model_type, starting_values = args_tuple
    ret = _pd.Series(ret_values, name=ticker)

    try:
        from marginals.garch import GARCHFitter
        fitter = GARCHFitter(model_type=model_type, dist="normal")
        kw = {"starting_values": starting_values} if starting_values is not None else {}
        gres     = fitter.fit_single(ret, ticker=ticker, **kw)
        std_res  = _np.asarray(gres.std_residuals)
        sigma_t1 = float(gres.conditional_vol.iloc[-1])
        params   = _np.asarray(gres.params)
    except Exception as exc:
        std_res  = ret.values / (ret.std(ddof=1) or 1.0)
        sigma_t1 = float(ret.std(ddof=1))
        params   = None
    return ticker, std_res, sigma_t1, params


# Fábrica de optimizer_fn + risk_fn compatíveis com WalkForwardBacktest.
def make_copula_cvar_optimizer(
    pipeline,
    garch_model: str = "gjr",
    copula_type: str = "cvine",
    n_sim: int       = 5_000,
    seed: int        = 42,
    n_restarts: int  = 5,
):
    import os
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from scipy.optimize import minimize as _minimize
    from scipy.stats import rankdata as _rankdata, norm as _norm

    _n_workers = min(5, max(1, (os.cpu_count() or 2) - 1), 4)

    _state = {
        "window_counter":          0,
        "garch_starting_values":   {},
        "last_sigma_t":            {},
        "last_sim_returns":        None,
        "last_tickers":            [],
    }

    def _fallback_minvar(train_df: pd.DataFrame) -> np.ndarray:
        d = train_df.shape[1]
        try:
            from sklearn.covariance import LedoitWolf
            cov = LedoitWolf().fit(train_df.values).covariance_
        except Exception:
            cov = train_df.cov().values + 1e-6 * np.eye(d)
        if np.linalg.cond(cov) > 1e12:
            cov += 1e-6 * np.eye(d)
        best_w, best_var = np.ones(d) / d, np.inf
        cons = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
        for _ in range(3):
            w0 = np.random.dirichlet(np.ones(d))
            res = _minimize(lambda w: float(w @ cov @ w), w0,
                            method="SLSQP", bounds=[(0.0, 1.0)] * d,
                            constraints=cons, options={"ftol": 1e-12, "maxiter": 1000})
            if res.success and res.fun < best_var:
                best_var, best_w = res.fun, res.x.copy()
        return best_w

    def _optimizer_fn(train_df: pd.DataFrame) -> np.ndarray:
        pipeline._ensure_src_path()
        d       = train_df.shape[1]
        tickers = list(train_df.columns)
        _state["window_counter"] += 1
        win_seed = seed + _state["window_counter"]

        try:
            fit_args = [
                (ticker,
                 train_df[ticker].values,
                 garch_model,
                 _state["garch_starting_values"].get(ticker))
                for ticker in tickers
            ]

            std_resid_cols = {}
            sigma_t_map    = {}

            try:
                with ProcessPoolExecutor(max_workers=_n_workers) as exe:
                    futures = {exe.submit(_garch_fit_single_asset, a): a[0]
                               for a in fit_args}
                    for fut in as_completed(futures):
                        ticker, std_res, sigma_t1, params = fut.result()
                        std_resid_cols[ticker] = std_res
                        sigma_t_map[ticker]    = sigma_t1
                        if params is not None:
                            _state["garch_starting_values"][ticker] = params
            except Exception as par_err:
                logger.warning(f"  GARCH paralelo falhou ({par_err}), usando sequencial.")
                for arg in fit_args:
                    ticker, std_res, sigma_t1, params = _garch_fit_single_asset(arg)
                    std_resid_cols[ticker] = std_res
                    sigma_t_map[ticker]    = sigma_t1
                    if params is not None:
                        _state["garch_starting_values"][ticker] = params

            _state["last_sigma_t"]  = sigma_t_map
            _state["last_tickers"]  = tickers

            std_resid_df = pd.DataFrame(std_resid_cols, index=train_df.index)

            arr  = std_resid_df.values
            T_w  = arr.shape[0]
            pseudo_obs = np.apply_along_axis(
                lambda col: _rankdata(col) / (T_w + 1), axis=0, arr=arr
            )
            pseudo_obs = np.clip(pseudo_obs, 1e-6, 1 - 1e-6)

            from copulas.vine_copulas import CVineCopula
            cop = CVineCopula(n_dim=d)
            cop.fit(
                pseudo_obs,
                families    = ["gaussian", "t", "clayton", "clayton_180",
                               "gumbel", "gumbel_180", "frank"],
                auto_select = True,
                max_trees   = pipeline.max_trees,
                min_tau     = pipeline.min_tau,
                n_jobs      = 1,
            )

            u_sim = cop.simulate(n_samples=n_sim, seed=win_seed)
            u_sim = np.clip(u_sim, 1e-10, 1 - 1e-10)

            sim_returns = np.zeros((n_sim, d))
            ret_arr     = train_df.values
            for i, tkr in enumerate(tickers):
                raw_q        = np.quantile(ret_arr[:, i], u_sim[:, i])
                sigma_uncond = float(np.std(ret_arr[:, i], ddof=1)) or 1.0
                sigma_cond   = sigma_t_map.get(tkr, sigma_uncond)
                sim_returns[:, i] = raw_q * (sigma_cond / sigma_uncond)

            _state["last_sim_returns"] = sim_returns

            alpha = 0.05

            def _cvar(w):
                port_sim = sim_returns @ w
                var_a    = np.quantile(port_sim, alpha)
                return float(-port_sim[port_sim <= var_a].mean())

            best_w, best_cvar = np.ones(d) / d, np.inf
            cons = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
            bnds = [(0.0, 1.0)] * d
            for _ in range(n_restarts):
                w0  = np.random.dirichlet(np.ones(d))
                res = _minimize(_cvar, w0, method="SLSQP",
                                bounds=bnds, constraints=cons,
                                options={"ftol": 1e-10, "maxiter": 500})
                if res.success and res.fun < best_cvar:
                    best_cvar = res.fun
                    best_w    = res.x.copy()

            best_w = np.clip(best_w, 0.0, 1.0)
            best_w /= best_w.sum()
            return best_w

        except Exception as e:
            logger.warning(
                f"  copula_cvar_optimizer falhou na janela "
                f"{_state['window_counter']} ({e}). Fallback: MinVariância."
            )
            return _fallback_minvar(train_df)

    # risk_fn para WalkForwardBacktest. Assinatura: (train_df, weights) → dict.
    def _risk_fn(train_df: pd.DataFrame, weights: np.ndarray) -> dict:
        from scipy.stats import norm as _norm_dist, t as _t_dist

        w           = np.asarray(weights, dtype=float)
        tickers     = _state.get("last_tickers", list(train_df.columns))
        sigma_map   = _state.get("last_sigma_t", {})
        sim_returns = _state.get("last_sim_returns")

        sigma_vec  = np.array([
            sigma_map.get(t, float(train_df[t].std(ddof=1)))
            for t in tickers
        ], dtype=float)
        sigma_port = float(np.sqrt((w * sigma_vec) @ (w * sigma_vec)))
        if sigma_port <= 0 or not np.isfinite(sigma_port):
            sigma_port = float((train_df.values @ w).std(ddof=1))

        if sim_returns is not None and len(sim_returns) > 0:
            port_sim = sim_returns @ w
            var_95   = float(-np.quantile(port_sim, 0.05))
            var_99   = float(-np.quantile(port_sim, 0.01))
            tail_95  = port_sim[port_sim <= np.quantile(port_sim, 0.05)]
            tail_99  = port_sim[port_sim <= np.quantile(port_sim, 0.01)]
            es_95    = float(-tail_95.mean()) if len(tail_95) > 0 else var_95 * 1.3
            es_99    = float(-tail_99.mean()) if len(tail_99) > 0 else var_99 * 1.3
        else:
            var_95 = float(_norm_dist.ppf(0.95) * sigma_port)
            var_99 = float(_norm_dist.ppf(0.99) * sigma_port)
            es_95  = float(sigma_port * _norm_dist.pdf(_norm_dist.ppf(0.05)) / 0.05)
            es_99  = float(sigma_port * _norm_dist.pdf(_norm_dist.ppf(0.01)) / 0.01)

        return {
            "var_95": var_95,
            "var_99": var_99,
            "es_95":  es_95,
            "es_99":  es_99,
        }

    _optimizer_fn._state    = _state
    _optimizer_fn._risk_fn  = _risk_fn
    return _optimizer_fn


# Executa walk-forward do pipeline CVaR-EVT com reotimização por janela.
def build_pipeline_backtest(
    pipeline,
    returns,
    args: argparse.Namespace,
):
    if pipeline is None:
        return None

    logger.info("\nExecutando walk-forward CVaR-EVT (otimização por janela)...")

    try:
        _comparison_dir = str(BASE_DIR / "src" / "comparison")
        if _comparison_dir not in sys.path:
            sys.path.insert(0, _comparison_dir)

        from backtest_rolling import WalkForwardBacktest

        optimizer_fn = make_copula_cvar_optimizer(
            pipeline    = pipeline,
            garch_model = args.garch_model,
            copula_type = args.copula,
            n_sim       = min(args.n_sim, 5_000),
            seed        = args.seed,
            n_restarts  = args.n_restarts,
        )

        risk_fn = optimizer_fn._risk_fn

        backtester = WalkForwardBacktest(
            estimation_window     = args.window,
            rebalancing_frequency = args.rebal,
            window_type           = args.window_type,
            optimizer_fn          = optimizer_fn,
            risk_fn               = risk_fn,
            risk_free_rate        = args.risk_free,
            verbose               = args.verbose,
        )
        result = backtester.run(returns, strategy_name="CVaR-EVT Copula")
        logger.info("  Walk-forward CVaR-EVT concluído.")
        return result

    except Exception as e:
        logger.warning(f"  Walk-forward CVaR-EVT falhou: {e}.")
        import traceback; traceback.print_exc()
        return None


# Instancia PerformanceMetrics, adiciona todas as estratégias
def consolidate_metrics(
    runner,
    pipeline_backtest_result,
    args: argparse.Namespace,
) -> "PerformanceMetrics":
    from performance_metrics import PerformanceMetrics

    pm = PerformanceMetrics(
        risk_free    = args.risk_free,
        periods      = 252,
        n_bootstrap  = args.n_bootstrap,
        block_size   = args.block_size,
        bootstrap_ci = 0.95,
        random_state = args.seed,
    )

    for key, res in runner._backtest_results.items():
        try:
            pm.add_from_backtest_result(
                name   = res.strategy_name,
                result = res,
            )
            logger.info(f"  Adicionado: {res.strategy_name}")
        except Exception as e:
            logger.warning(f"  Falha ao adicionar '{key}': {e}")

    if pipeline_backtest_result is not None:
        try:
            pm.add_from_backtest_result("CVaR-EVT Copula", pipeline_backtest_result)
            logger.info("  Adicionado: CVaR-EVT Copula")
        except Exception as e:
            logger.warning(f"  Falha ao adicionar CVaR-EVT: {e}")

    return pm


# 6. Exportação de resultados

# Salva todos os artefatos em results/comparison/.
def save_results(
    runner,
    pm,
    args: argparse.Namespace,
) -> None:
    COMPARISON_DIR.mkdir(parents=True, exist_ok=True)

    logger.info(f"\nSalvando resultados em {COMPARISON_DIR}")
    runner.save(COMPARISON_DIR)

    table = pm.table()
    table.to_csv(COMPARISON_DIR / "performance_metrics.csv")
    logger.info("  performance_metrics.csv")

    ci_table = pm.bootstrap_ci_table()
    ci_table.to_csv(COMPARISON_DIR / "bootstrap_ci.csv")
    logger.info("  bootstrap_ci.csv")

    if len(pm._returns) >= 2:
        lw_mat = pm.ledoit_wolf_matrix()
        lw_mat.to_csv(COMPARISON_DIR / "lw_pvalues.csv")
        logger.info("  lw_pvalues.csv")

    pm.to_latex(
        path    = str(COMPARISON_DIR / "table_performance.tex"),
        caption = (
            "Métricas de performance OOS — "
            f"janela={args.window}d, rebal={args.rebal}d, "
            f"cov={args.cov_estimator}."
        ),
        label   = "tab:comparison_performance",
        include_bootstrap = True,
        include_lw        = True,
    )
    logger.info("  table_performance.tex  (+ table_performance_lw_pvalues.tex)")

    try:
        panel = runner.returns_panel()
        panel.to_csv(COMPARISON_DIR / "returns_panel.csv")
        logger.info("  returns_panel.csv")
    except Exception:
        pass

    logger.info("\n" + "=" * 60)
    logger.info("  RESUMO DE PERFORMANCE  ")
    logger.info("=" * 60)
    print(table.to_string())

    if len(pm._returns) >= 2:
        logger.info("\n── Matriz Ledoit-Wolf (p-valores) ──")
        print(pm.ledoit_wolf_matrix().round(3).to_string())

    logger.info(f"\nResultados salvos em: {COMPARISON_DIR}")


# CLI

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Estudo comparativo: Copula-EVT-CVaR vs. estratégias risk-based "
            "(EW, MV, RP, MD, MDE) — portfólios B3."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    data = parser.add_argument_group("Dados")
    data.add_argument("--data-start", default="2004-11-18",
                      help="Data de início (YYYY-MM-DD). Default = início do GLD, "
                           "que é o ativo com série mais curta entre os 5.")
    data.add_argument("--data-dir", default=None, type=str,
                      help="Diretório com os xlsx do Economatica "
                           "(IBRX50, IBOV, EWZ, GLD, SPY, Selic_252d). "
                           "Default: data/raw/ relativo à raiz do projeto.")
    data.add_argument("--risk-free",  default=0.1075, type=float,
                      help="Taxa Selic anualizada (sobrescrita por Selic_252d.xlsx se disponível)")
    data.add_argument("--fx-file", default=None, type=str,
                      help="Caminho do xlsx (formato Economatica) com o câmbio "
                           "USD/BRL, usado para converter EWZ/GLD/SPY (cotados "
                           "em USD) para BRL antes de combiná-los com IBRX50/"
                           "IBOV. Default: procura DOLOF.xlsx em --data-dir.")
    data.add_argument("--allow-mixed-currency", action="store_true",
                      help="Pula a conversão USD->BRL de EWZ/GLD/SPY e "
                           "combina os 5 ativos em moedas diferentes na "
                           "mesma otimização/Sharpe. NÃO recomendado — "
                           "mantém deliberadamente o bug de moeda mista.")

    bt = parser.add_argument_group("Backtest Rolling")
    bt.add_argument("--strategies", nargs="+",
                    default=["ew", "mv", "rp", "md", "mde"],
                    choices=["ew", "mv", "rp", "md", "mde"],
                    help="Estratégias risk-based a comparar")
    bt.add_argument("--window",      default=252, type=int,
                    help="Janela de estimação (dias úteis)")
    bt.add_argument("--rebal",       default=21,  type=int,
                    help="Frequência de rebalanceamento (dias úteis)")
    bt.add_argument("--window-type", default="rolling",
                    choices=["rolling", "expanding"])
    bt.add_argument("--max-weight",  default=1.0,  type=float,
                    help="Peso máximo por ativo. Default=1.0 (sem restrição) — "
                         "adequado para universos pequenos (d=5).")
    bt.add_argument("--min-weight",  default=0.0,  type=float)
    bt.add_argument("--cov-estimator", default="ledoit_wolf",
                    choices=["sample", "ledoit_wolf"])
    bt.add_argument("--risk-fn",     default="historical",
                    choices=["historical", "ewma"])
    bt.add_argument("--n-restarts",  default=5, type=int)
    bt.add_argument("--verbose",     action="store_true")

    pipe = parser.add_argument_group("Pipeline CVaR-EVT")
    pipe.add_argument("--no-pipeline", action="store_true",
                      help="Pula o OptimizedPipeline (compara apenas risk-based)")
    pipe.add_argument("--load-state", default=None, type=str,
                      help="Carrega estado de pipeline salvo (.pkl)")
    pipe.add_argument("--copula",     default="cvine",
                      choices=["cvine", "dvine", "rvine", "dcc", "factor"])
    pipe.add_argument("--n-sim",      default=10_000, type=int)
    pipe.add_argument("--seed",       default=42, type=int)
    pipe.add_argument("--garch-model", default="gjr",
                      choices=["garch", "gjr", "auto"])
    pipe.add_argument("--pit-method", default="empirical",
                      choices=["empirical", "normal"])
    pipe.add_argument("--max-trees",  default=3, type=int)
    pipe.add_argument("--n-jobs",     default=3, type=int)
    pipe.add_argument("--cache",      action="store_true")
    pipe.add_argument("--no-ml",      action="store_true")
    pipe.add_argument("--n-regimes",  default=3, type=int)

    met = parser.add_argument_group("Métricas e Testes")
    met.add_argument("--n-bootstrap", default=1_000, type=int,
                     help="Número de réplicas bootstrap")
    met.add_argument("--block-size",  default=20.0, type=float,
                     help="Comprimento médio do bloco (stationary bootstrap)")

    out = parser.add_argument_group("Saída")
    out.add_argument("--save-state",  action="store_true",
                     help="Salva estado do pipeline após execução")
    out.add_argument("--output-dir",  default=None, type=str,
                     help="Diretório de saída (default: results/comparison)")

    return parser.parse_args()


# Main

def main() -> None:
    args = parse_args()
    t0   = time.time()

    if args.output_dir:
        global COMPARISON_DIR
        COMPARISON_DIR = Path(args.output_dir)

    logger.info("=" * 60)
    logger.info("  ESTUDO COMPARATIVO DE PORTFÓLIOS B3")
    logger.info("  Universo: 81 ações do IBrX-50 (data/raw_b3_ibrx50/), universo dinâmico point-in-time")
    logger.info("=" * 60)

    if args.data_dir:
        global DATA_RAW
        DATA_RAW = Path(args.data_dir)

    returns  = load_returns(args)

    if args.risk_free == 0.1075:
        args.risk_free = load_selic(args, returns_index=returns.index)

    benchmark = compute_ew_benchmark(returns, args)

    runner = run_risk_based_backtests(returns, benchmark, args)

    pipeline, pipeline_results = run_or_load_pipeline(returns, args)

    pipeline_bt = build_pipeline_backtest(pipeline, returns, args)

    if args.save_state and pipeline is not None:
        state_path = RESULTS_DIR / f"pipeline_state_{args.copula}.pkl"
        try:
            pipeline.save_state(state_path)
            logger.info(f"  Estado do pipeline salvo em {state_path}")
        except Exception as e:
            logger.warning(f"  Falha ao salvar estado: {e}")

    pm = consolidate_metrics(runner, pipeline_bt, args)

    save_results(runner, pm, args)

    elapsed = time.time() - t0
    logger.info(f"\nConcluído em {elapsed:.1f}s")


if __name__ == "__main__":
    main()
