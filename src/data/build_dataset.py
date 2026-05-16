from pathlib import Path
import pandas as pd
from .market_data import load_market_prices
from .sample_data import ensure_sample_data
from src.features.market_features import make_market_features
from src.features.macro_features import make_macro_features
from src.features.stress_features import make_contagion_risk_score

def _load_csv_date_index(path):
    df = pd.read_csv(path)
    date_col = 'date' if 'date' in df.columns else 'Date' if 'Date' in df.columns else df.columns[0]
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.rename(columns={date_col: 'date'}).set_index('date').sort_index()
    return df.apply(pd.to_numeric, errors='coerce')

def load_macro_data():
    raw = Path('data/raw/boc_yields.csv')
    sample = Path('data/sample/macro.csv')
    path = raw if raw.exists() else sample
    macro = _load_csv_date_index(path)

    if {'ca_10y', 'ca_2y'}.issubset(macro.columns) and 'slope_10y_2y' not in macro.columns:
        macro['slope_10y_2y'] = macro['ca_10y'] - macro['ca_2y']
    if {'ca_2y', 'ca_5y', 'ca_10y'}.issubset(macro.columns) and 'curvature' not in macro.columns:
        macro['curvature'] = 2 * macro['ca_5y'] - macro['ca_2y'] - macro['ca_10y']
    return macro

def build_dataset(output='data/processed/model_dataset.csv'):
    ensure_sample_data()
    price_path = 'data/raw/market_prices.csv' if Path('data/raw/market_prices.csv').exists() else 'data/sample/market_prices.csv'
    prices=load_market_prices(price_path)
    macro=load_macro_data()
    mf=make_market_features(prices)
    mac=make_macro_features(macro).reindex(mf.index).ffill()
    data=mf.join(mac, how='left').ffill().dropna(how='all')
    data=data.assign(contagion_risk_score=make_contagion_risk_score(data))
    Path(output).parent.mkdir(parents=True, exist_ok=True); data.to_csv(output); prices.to_csv('data/processed/prices.csv')
    return data
if __name__=='__main__': build_dataset()
