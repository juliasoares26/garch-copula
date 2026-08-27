
import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_RAW = BASE_DIR / 'data' / 'raw'
DATA_EXTERNAL = BASE_DIR / 'data' / 'external'
DATA_RAW.mkdir(parents=True, exist_ok=True)
DATA_EXTERNAL.mkdir(parents=True, exist_ok=True)


DEFAULT_TICKERS_B3 = [
    'PETR4.SA', 'VALE3.SA', 'ITUB4.SA', 'BBDC4.SA', 'ABEV3.SA',
    'WEGE3.SA', 'RENT3.SA', 'HAPV3.SA', 'LREN3.SA', 'MGLU3.SA',
    'BBAS3.SA', 'SUZB3.SA', 'RADL3.SA', 'JBSS3.SA', 'BEEF3.SA',
    'EMBR3.SA', 'CCRO3.SA', 'GGBR4.SA', 'USIM5.SA', 'CSNA3.SA',
]

DEFAULT_INDICES = [
    '^BVSP',
    '^GSPC',
    'BRL=X',
]

SELIC_TICKER = 'IRFM11.SA'


# Funções de download

# Baixa preços de fechamento ajustados para lista de tickers.
def download_stocks(
    tickers: list[str],
    start: str,
    end: str,
    interval: str = '1d',
    progress: bool = True,
) -> pd.DataFrame:
    try:
        import yfinance as yf
    except ImportError:
        logger.error("yfinance não instalado. Execute: pip install yfinance")
        sys.exit(1)

    logger.info(f"Baixando {len(tickers)} tickers de {start} a {end}...")

    data = yf.download(
        tickers,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=True,
        progress=progress,
        threads=True,
    )

    if isinstance(data.columns, pd.MultiIndex):
        prices = data['Close']
    else:
        prices = data[['Close']]
        prices.columns = tickers

    n_missing = prices.isna().sum()
    if n_missing.sum() > 0:
        logger.warning(f"Valores ausentes por ticker:\n{n_missing[n_missing > 0]}")

    prices = prices.ffill().dropna(how='all')

    logger.info(f"Download concluído: {prices.shape[0]} dias × {prices.shape[1]} ativos")
    return prices


# Calcula retornos a partir de preços.
def compute_returns(prices: pd.DataFrame, method: str = 'log') -> pd.DataFrame:
    if method == 'log':
        returns = np.log(prices / prices.shift(1)).dropna()
    elif method == 'simple':
        returns = prices.pct_change().dropna()
    else:
        raise ValueError(f"method deve ser 'log' ou 'simple', recebido: {method}")

    logger.info(f"Retornos calculados: {returns.shape} ({method})")
    return returns


# Baixa taxa livre de risco.
def download_risk_free(start: str, end: str) -> pd.Series:
    try:
        import requests
        start_bcb = pd.to_datetime(start).strftime('%d/%m/%Y')
        end_bcb   = pd.to_datetime(end).strftime('%d/%m/%Y')
        url = (
            f"https://api.bcb.gov.br/dados/serie/bcdata.sgs.12/dados"
            f"?formato=json&dataInicial={start_bcb}&dataFinal={end_bcb}"
        )
        resp = requests.get(url, timeout=15)
        if resp.status_code == 200:
            records = resp.json()
            df = pd.DataFrame(records)
            df['data']  = pd.to_datetime(df['data'], dayfirst=True)
            df['valor'] = df['valor'].str.replace(',', '.').astype(float) / 100
            df = df.set_index('data')['valor']
            df.name = 'cdi_daily'
            logger.info(f"CDI baixado via BCB API: {len(df)} observações")
            return df
    except Exception as e:
        logger.warning(f"BCB API falhou ({e}), tentando yfinance...")

    try:
        import yfinance as yf
        proxy = yf.download(SELIC_TICKER, start=start, end=end, progress=False, auto_adjust=True)
        if not proxy.empty:
            prices_rf = proxy['Close']
            rf = np.log(prices_rf / prices_rf.shift(1)).dropna()
            rf.name = 'rf_proxy'
            logger.info(f"Taxa livre de risco (proxy IRFM11): {len(rf)} obs")
            return rf
    except Exception as e:
        logger.warning(f"yfinance proxy falhou ({e})")

    logger.warning("Usando Selic constante como fallback (10.75% a.a.)")
    selic_annual = 0.1075
    selic_daily  = (1 + selic_annual) ** (1 / 252) - 1
    dates = pd.bdate_range(start=start, end=end)
    return pd.Series(selic_daily, index=dates, name='rf_constant')


# Persistência

# Salva arquivos em data/raw/ e data/external/.
def save_raw_data(
    prices: pd.DataFrame,
    returns: pd.DataFrame,
    indices: pd.DataFrame | None,
    rf: pd.Series,
) -> None:
    prices_path  = DATA_RAW / 'b3_stocks_daily.parquet'
    returns_path = DATA_RAW / 'returns.parquet'
    rf_path      = DATA_EXTERNAL / 'risk_free_rate.csv'

    prices.to_parquet(prices_path)
    logger.info(f"Preços salvos: {prices_path}")

    returns.to_parquet(returns_path)
    logger.info(f"Retornos salvos: {returns_path}")

    rf.to_frame().to_csv(rf_path)
    logger.info(f"Taxa livre de risco salva: {rf_path}")

    if indices is not None and not indices.empty:
        idx_path = DATA_RAW / 'market_indices.parquet'
        indices.to_parquet(idx_path)
        logger.info(f"Índices salvos: {idx_path}")

    logger.info("\n--- Estatísticas dos retornos ---")
    logger.info(f"Período  : {returns.index[0].date()} → {returns.index[-1].date()}")
    logger.info(f"Ativos   : {returns.shape[1]}")
    logger.info(f"Obs/ativo: {returns.shape[0]}")
    logger.info(f"Retorno médio anualizado:\n{(returns.mean() * 252).round(4).to_string()}")
    logger.info(f"Vol anualizada:\n{(returns.std() * np.sqrt(252)).round(4).to_string()}")


# CLI

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download dados B3 e salva em data/raw/",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    default_end   = datetime.today().strftime('%Y-%m-%d')
    default_start = (datetime.today() - timedelta(days=365 * 8)).strftime('%Y-%m-%d')

    parser.add_argument('--start',    default=default_start, help='Data início (YYYY-MM-DD)')
    parser.add_argument('--end',      default=default_end,   help='Data fim (YYYY-MM-DD)')
    parser.add_argument('--tickers',  nargs='+', default=None,
                        help='Lista de tickers (ex: PETR4.SA VALE3.SA). '
                             'Se omitido usa lista padrão.')
    parser.add_argument('--interval', default='1d', choices=['1d', '1wk', '1mo'],
                        help='Intervalo de tempo')
    parser.add_argument('--no-indices', action='store_true',
                        help='Não baixar índices de mercado')
    parser.add_argument('--returns',  default='log', choices=['log', 'simple'],
                        help='Método de cálculo de retornos')
    parser.add_argument('--no-rf',    action='store_true',
                        help='Não baixar taxa livre de risco')
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    tickers = args.tickers if args.tickers else DEFAULT_TICKERS_B3
    logger.info(f"Tickers selecionados ({len(tickers)}): {tickers}")

    prices = download_stocks(tickers, args.start, args.end, args.interval)

    returns = compute_returns(prices, method=args.returns)

    indices = None
    if not args.no_indices:
        try:
            indices = download_stocks(DEFAULT_INDICES, args.start, args.end, args.interval, progress=False)
        except Exception as e:
            logger.warning(f"Download de índices falhou: {e}")

    rf = pd.Series(dtype=float)
    if not args.no_rf:
        rf = download_risk_free(args.start, args.end)

    save_raw_data(prices, returns, indices, rf)

    logger.info("\nDownload concluído com sucesso!")


if __name__ == '__main__':
    main()
