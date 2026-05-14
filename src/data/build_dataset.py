from pathlib import Path
import pandas as pd
from .market_data import load_market_prices
from .sample_data import ensure_sample_data
from src.features.market_features import make_market_features
from src.features.macro_features import make_macro_features
from src.features.stress_features import make_contagion_risk_score

def build_dataset(output='data/processed/model_dataset.csv'):
    ensure_sample_data()
    prices=load_market_prices('data/sample/market_prices.csv') if not Path('data/raw/market_prices.csv').exists() else load_market_prices('data/raw/market_prices.csv')
    macro=pd.read_csv('data/sample/macro.csv',parse_dates=['date'],index_col='date')
    mf=make_market_features(prices)
    mac=make_macro_features(macro)
    data=mf.join(mac, how='left').ffill().dropna()
    data['contagion_risk_score']=make_contagion_risk_score(data)
    Path(output).parent.mkdir(parents=True, exist_ok=True); data.to_csv(output); prices.to_csv('data/processed/prices.csv')
    return data
if __name__=='__main__': build_dataset()
