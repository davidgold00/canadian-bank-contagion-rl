from src.data.sample_data import ensure_sample_data
from src.data.market_data import load_market_prices

def test_sample_data_loads():
    ensure_sample_data(); df=load_market_prices('data/sample/market_prices.csv'); assert not df.empty
