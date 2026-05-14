from pathlib import Path
import pandas as pd
from .sample_data import make_sample_market_data

def download_market_data(tickers, start='2012-01-01', end=None, output='data/raw/market_prices.csv', fallback=True):
    try:
        import yfinance as yf
        raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)['Close']
        if isinstance(raw, pd.Series): raw = raw.to_frame(tickers[0])
        if raw.dropna(how='all').empty: raise ValueError('empty yfinance response')
        Path(output).parent.mkdir(parents=True, exist_ok=True); raw.to_csv(output); return raw
    except Exception:
        if not fallback: raise
        return make_sample_market_data(output)

def load_market_prices(path='data/raw/market_prices.csv'):
    if not Path(path).exists(): path='data/sample/market_prices.csv'
    return pd.read_csv(path, parse_dates=['date'], index_col='date').sort_index()
