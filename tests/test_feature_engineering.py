from src.data.sample_data import make_sample_market_data, make_sample_macro_data
from src.features.market_features import make_market_features
from src.features.stress_features import make_contagion_risk_score

def test_contagion_score_range(tmp_path):
    prices=make_sample_market_data(tmp_path/'p.csv', n=120); f=make_market_features(prices); s=make_contagion_risk_score(f); assert s.dropna().between(0,100).all()
