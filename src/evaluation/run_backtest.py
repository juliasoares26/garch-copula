
import argparse
import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

HERE        = Path(__file__).resolve().parent
SRC         = HERE.parent
ROOT        = SRC.parent
DATA_PATH   = ROOT / "data" / "processed" / "returns.parquet"

sys.path.insert(0, str(SRC))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(SRC / "comparison"))
sys.path.insert(0, str(SRC / "evaluation"))
sys.path.insert(0, str(SRC / "risk"))
sys.path.insert(0, str(SRC / "copulas"))

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)-8s %(name)s | %(message)s",
)
logger = logging.getLogger("run_backtest")

try:
    from evaluation.backtesting_walking_forward import EVTWalkForward, ESBacktest
except ImportError:
    from backtesting_walking_forward import EVTWalkForward, ESBacktest

try:
    from backtest_rolling import RollingBacktestRunner
except ImportError:
    from comparison.backtest_rolling import RollingBacktestRunner

try:
    from evaluation.performance_metrics import (
        performance_summary,
        backtesting_report,
        rolling_var,
        rolling_es,
    )
except ImportError:
    from performance_metrics import (
        performance_summary,
        backtesting_report,
        rolling_var,
        rolling_es,
    )


# ── Configuração via argparse ─────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Backtest EVT-CVine vs risk-based strategies")
    p.add_argument("--data",   default=str(DATA_PATH),
                   help="Caminho para o parquet de retornos (default: %(default)s)")
    p.add_argument("--window", type=int, default=750,
                   help="Janela de estimação em dias úteis (default: 750). "
                        "Aumentado de 500 para 750: com threshold em q=0.10, "
                        "window=500 produz apenas 50 exceedances por cauda, "
                        "amostra pequena o suficiente para o MLE da GPD ter "
                        "viés negativo sistemático em xi (confirmado via "
                        "simulação: ~4%% das janelas convergem no bound -0.5 "
                        "mesmo quando xi verdadeiro é ~0). Em window=750 "
                        "(75 exceedances) esse viés cai para ~1%%. Para "
                        "reproduzir o comportamento anterior, use --window 500.")
    p.add_argument("--rebal",  type=int, default=21,
                   help="Frequência de rebalanceamento em dias úteis (default: 21 = mensal)")
    p.add_argument("--njobs",  type=int, default=-1,
                   help="Paralelismo joblib: -1 = todos os cores, 1 = sequencial (default: -1)")
    p.add_argument("--copula", default="cvine",
                   choices=["cvine", "dvine", "rvine"],
                   help="Tipo de vine copula (default: cvine)")
    p.add_argument("--rf",     type=float, default=0.1075,
                   help="Taxa livre de risco anualizada (default: 10.75%% a.a.)")
    p.add_argument("--benchmark", default=None,
                   help="Coluna a usar como benchmark (ex: '^BVSP'). Omitir = sem benchmark.")
    p.add_argument("--strategies", nargs="+",
                   default=["ew", "mv", "rp", "md", "mde"],
                   help="Estratégias risk-based a comparar")
    p.add_argument("--nsim",   type=int, default=5000,
                   help="Simulações Monte Carlo por janela (default: 5000)")
    p.add_argument("--verbose", action="store_true", default=True)
    return p.parse_args()


# ── Carregamento de dados ─────────────────────────────────────────────────────
def load_returns(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        logger.error(f"Arquivo não encontrado: {p}")
        sys.exit(1)

    if p.suffix == ".parquet":
        df = pd.read_parquet(p)
    elif p.suffix in (".csv", ".tsv"):
        df = pd.read_csv(p, index_col=0, parse_dates=True)
    else:
        logger.error(f"Formato não suportado: {p.suffix}")
        sys.exit(1)

    df = df.dropna(how="all")

    if df.std().median() > 0.10:
        logger.warning("Retornos detectados em escala percentual — dividindo por 100.")
        df = df / 100.0

    logger.info(f"Dados: {df.shape}  ({df.index[0].date()} → {df.index[-1].date()})")
    logger.info(f"Ativos: {list(df.columns)}")
    return df


# ── 1. EVT Walk-Forward ───────────────────────────────────────────────────────
def run_evt_walkforward(returns_df, args):
    logger.info("=" * 60)
    logger.info("1. EVT Walk-Forward (EVT-CVine)")
    logger.info("=" * 60)

    wf = EVTWalkForward(
        estimation_window=args.window,
        rebalancing_frequency=args.rebal,
        copula_type=args.copula,
        confidence_levels=[0.95, 0.99],
        n_simulations=args.nsim,
        risk_free_rate=args.rf / 252,
        seed=42,
        n_jobs=args.njobs,
    )
    wf_results = wf.run(returns_df)
    wf_summary = wf.summary(wf_results)

    logger.info(f"Dias OOS: {len(wf_results)}")

    for cl in [95, 99]:
        kp  = wf_summary.get(f"kupiec_pvalue_{cl}", "N/A")
        ok  = wf_summary.get(f"kupiec_adequate_{cl}", "N/A")
        vio = wf_summary.get(f"violations_{cl}", "N/A")
        n   = wf_summary.get("n_oos", len(wf_results))
        logger.info(f"  VaR {cl}%: {vio}/{n} violações | Kupiec p={kp} | {'✓' if ok else '✗'}")

    return wf_results, wf_summary


# ── 2. ES Backtest ────────────────────────────────────────────────────────────
def run_es_backtest(wf_results):
    logger.info("=" * 60)
    logger.info("2. ES Backtest (McNeil-Frey + Acerbi-Szekely)")
    logger.info("=" * 60)

    es_bt = ESBacktest(alpha=0.05)
    es_results = {}

    for cl_str, cl in [("95", 0.95), ("99", 0.99)]:
        var_col = f"var_{cl_str}"
        es_col  = f"es_{cl_str}"
        if var_col not in wf_results.columns or es_col not in wf_results.columns:
            logger.warning(f"  Colunas {var_col}/{es_col} não encontradas — pulando.")
            continue

        mask = wf_results[[var_col, es_col]].notna().all(axis=1)
        sub  = wf_results[mask]
        if len(sub) < 30:
            logger.warning(f"  ES {cl_str}%: amostra OOS insuficiente ({len(sub)} obs).")
            continue

        res = es_bt.run(
            realized_returns=sub["return"].values,
            var_forecasts=sub[var_col].values,
            es_forecasts=sub[es_col].values,
            confidence_level=cl,
            n_bootstrap=500,
            seed=42,
        )
        es_results[cl_str] = res
        logger.info(
            f"  ES {cl_str}%: MF p={res.get('mf_pvalue', 'N/A')} | "
            f"AZ1 p={res.get('az1_pvalue', 'N/A')} | "
            f"adequate={res.get('es_model_adequate', 'N/A')}"
        )

    return es_results


# ── 3. Risk-Based Strategies ──────────────────────────────────────────────────
def run_risk_based(returns_df, benchmark_series, args):
    logger.info("=" * 60)
    logger.info("3. Risk-Based Strategies (EW / MV / RP / MD / MDE)")
    logger.info("=" * 60)

    runner = RollingBacktestRunner(
        estimation_window=args.window,
        rebalancing_frequency=args.rebal,
        window_type="rolling",
        max_weight=0.40,
        min_weight=0.00,
        cov_estimator="ledoit_wolf",
        risk_fn_type="historical",
        risk_free_rate=args.rf / 252,
        strategies=args.strategies,
        verbose=args.verbose,
    )
    results = runner.run_all(returns_df, benchmark=benchmark_series)
    summary = runner.summary()

    logger.info("\n" + summary.to_string())
    return results, summary


# ── 4. Performance consolidada ────────────────────────────────────────────────
def consolidated_performance(wf_results, risk_results, benchmark_series, rf_daily):
    logger.info("=" * 60)
    logger.info("4. Performance Consolidada")
    logger.info("=" * 60)

    series_dict = {}

    if wf_results is not None and "return" in wf_results.columns:
        evt_ret = pd.Series(
            wf_results["return"].values,
            index=wf_results["date"] if "date" in wf_results.columns else wf_results.index,
            name="EVT-CVine",
        )
        series_dict["EVT-CVine"] = evt_ret

    for name, res in risk_results.items():
        if hasattr(res, "returns"):
            series_dict[name] = pd.Series(res.returns, name=name)

    if benchmark_series is not None:
        series_dict["Benchmark"] = benchmark_series.rename("Benchmark")

    if not series_dict:
        logger.warning("Nenhuma série de retornos disponível para performance summary.")
        return None

    summary = performance_summary(series_dict, benchmark=benchmark_series, rf=rf_daily)
    print("\n=== Performance Summary ===")
    print(summary.T.to_string())
    return summary


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    returns_df = load_returns(args.data)

    benchmark_series = None
    if args.benchmark and args.benchmark in returns_df.columns:
        benchmark_series = returns_df[args.benchmark].rename("Benchmark")
        logger.info(f"Benchmark: {args.benchmark}")
    elif args.benchmark:
        logger.warning(f"Coluna benchmark '{args.benchmark}' não encontrada. Rodando sem benchmark.")

    rf_daily = args.rf / 252

    wf_results, wf_summary = run_evt_walkforward(returns_df, args)

    es_results = run_es_backtest(wf_results)

    risk_results, risk_summary = run_risk_based(returns_df, benchmark_series, args)

    perf = consolidated_performance(wf_results, risk_results, benchmark_series, rf_daily)

    out_dir = HERE / "results"
    out_dir.mkdir(exist_ok=True)

    if wf_results is not None:
        wf_results.to_csv(out_dir / "wf_results.csv")
        logger.info(f"Walk-forward salvo: {out_dir / 'wf_results.csv'}")

    if perf is not None:
        perf.to_csv(out_dir / "performance_summary.csv")
        logger.info(f"Performance salva: {out_dir / 'performance_summary.csv'}")

    if not risk_summary.empty:
        risk_summary.to_csv(out_dir / "risk_based_summary.csv")
        logger.info(f"Risk-based salvo: {out_dir / 'risk_based_summary.csv'}")

    logger.info("=" * 60)
    logger.info("Backtest concluído.")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
