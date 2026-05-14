from pathlib import Path
import numpy as np
import pandas as pd

BANKS = ["RY.TO","TD.TO","BMO.TO","BNS.TO","CM.TO","NA.TO"]
ASSETS = BANKS + ["XFN.TO","XIU.TO","cash"]

def make_sample_market_data(path="data/sample/market_prices.csv", n=900, seed=7):
    rng=np.random.default_rng(seed); dates=pd.bdate_range("2020-01-01", periods=n)
    factors=rng.normal(0, [0.006,0.004,0.003], size=(n,3))
    tickers=BANKS+["XFN.TO","XIU.TO","CADUSD=X","CL=F","GC=F","^GSPTSE","^VIX"]
    prices={}
    base={t:100+rng.normal(0,5) for t in tickers}
    for i,t in enumerate(tickers):
        loading=np.array([1.0, .4 if t in BANKS+["XFN.TO"] else .2, rng.normal(.1,.3)])
        ret=factors@loading + rng.normal(0,0.006 if t in BANKS else 0.004,n)
        if t=="^VIX": ret=-0.4*factors[:,0]+rng.normal(0,0.015,n)
        prices[t]=base[t]*np.exp(np.cumsum(ret))
    df=pd.DataFrame(prices,index=dates); df.index.name='date'
    Path(path).parent.mkdir(parents=True, exist_ok=True); df.to_csv(path); return df

def make_sample_macro_data(path="data/sample/macro.csv", n=900, seed=8):
    rng=np.random.default_rng(seed); dates=pd.bdate_range("2020-01-01", periods=n)
    y2=1.8+np.cumsum(rng.normal(0,0.015,n)); y10=y2+0.7+np.sin(np.arange(n)/90)*0.5+rng.normal(0,.04,n)
    df=pd.DataFrame({"policy_rate":np.maximum(.25,y2-.6),"ca_2y":y2,"ca_5y":(y2+y10)/2+rng.normal(0,.03,n),"ca_10y":y10}, index=dates)
    df["slope_10y_2y"]=df.ca_10y-df.ca_2y; df["curvature"]=2*df.ca_5y-df.ca_2y-df.ca_10y
    df.index.name='date'; Path(path).parent.mkdir(parents=True, exist_ok=True); df.to_csv(path); return df

def make_templates():
    Path('data/templates').mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"date":["2024-01-31"],"mortgage_arrears_rate":[0.18],"delinquency_90d_rate":[0.22],"mortgage_credit_growth":[3.1],"house_price_index":[100],"unemployment_rate":[6.0],"mortgage_debt_level":[2200]}).to_csv('data/templates/housing_stress_template.csv',index=False)
    rows=[]
    weights=[.22,.20,.15,.13,.11,.06]
    for t,w in zip(BANKS,weights): rows.append({"date":"2024-01-31","ETF ticker":"XFN.TO","holding ticker":t,"holding weight":w})
    pd.DataFrame(rows).to_csv('data/templates/etf_holdings_template.csv',index=False)
    pd.DataFrame({"date":["2024-01-31"],"bank":["RY.TO"],"cds_5y_bps":[75]}).to_csv('data/templates/cds_template.csv',index=False)

def ensure_sample_data():
    make_sample_market_data(); make_sample_macro_data(); make_templates()

if __name__ == '__main__': ensure_sample_data()
