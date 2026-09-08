from pathlib import Path
import pandas as pd
from .sample_data import make_sample_market_data

def download_market_data(tickers, start='2012-01-01', end=None, output='data/raw/market_prices.csv', fallback=False):
    if fallback and 'sample' not in Path(output).parts:
        raise ValueError('Synthetic fallback is permitted only in data/sample, never a production cache.')
    try:
        import yfinance as yf
        raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)['Close']
        if isinstance(raw, pd.Series): raw = raw.to_frame(tickers[0])
        if raw.dropna(how='all').empty: raise ValueError('empty yfinance response')
        # Today's partial close is never a completed observation. Require a prior calendar date.
        from datetime import datetime
        from zoneinfo import ZoneInfo
        today=pd.Timestamp(datetime.now(ZoneInfo('America/Toronto')).date())
        raw=raw.loc[raw.index.tz_localize(None)<today]
        raw.index.name = 'date'
        from src.research.provenance import validate_panel, record_download
        validate_panel(raw, list(tickers), minimum_rows=126)
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        raw.to_csv(output)
        record_download(output, raw, 'Yahoo Finance via yfinance', list(tickers), list(tickers), adjustments='auto_adjust=True; vendor split/dividend adjusted Close')
        return raw
    except Exception:
        if not fallback: raise
        return make_sample_market_data(output)

def load_market_prices(path='data/raw/market_prices.csv'):
    if not Path(path).exists(): path='data/sample/market_prices.csv'
    df = pd.read_csv(path)
    date_col = 'date' if 'date' in df.columns else 'Date' if 'Date' in df.columns else df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.rename(columns={date_col: 'date'}).set_index('date').sort_index()
    df = df.apply(pd.to_numeric, errors='coerce')

    # Yahoo can return today's partial FX row before Canadian equities close.
    # Keep rows with enough actual market observations for cross-asset features.
    min_assets = max(1, min(6, len(df.columns)))
    df = df.dropna(thresh=min_assets)
    return df
