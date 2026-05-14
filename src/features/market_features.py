import pandas as pd, numpy as np
BANKS=["RY.TO","TD.TO","BMO.TO","BNS.TO","CM.TO","NA.TO"]

def rolling_beta(x,y,window=63):
    return x.rolling(window,min_periods=20).cov(y)/y.rolling(window,min_periods=20).var()

def make_market_features(prices):
    rets=prices.pct_change(); out=pd.DataFrame(index=prices.index)
    for t in prices.columns:
        out[f'{t}_ret_1d']=rets[t]
        out[f'{t}_ret_5d']=prices[t].pct_change(5)
        out[f'{t}_ret_21d']=prices[t].pct_change(21)
        out[f'{t}_vol_5d']=rets[t].rolling(5).std()*np.sqrt(252)
        out[f'{t}_vol_21d']=rets[t].rolling(21).std()*np.sqrt(252)
        out[f'{t}_vol_63d']=rets[t].rolling(63).std()*np.sqrt(252)
        out[f'{t}_drawdown_63d']=prices[t]/prices[t].rolling(63,min_periods=10).max()-1
        out[f'{t}_dist_52w_high']=prices[t]/prices[t].rolling(252,min_periods=20).max()-1
    if 'XFN.TO' in prices:
        for b in BANKS:
            if b in prices: out[f'{b}_beta_xfn_63d']=rolling_beta(rets[b],rets['XFN.TO'])
    if '^VIX' in prices: out['VIX_level']=prices['^VIX']; out['VIX_chg_5d']=prices['^VIX'].diff(5)
    bank_cols=[b for b in BANKS if b in prices]
    out['avg_bank_return']=rets[bank_cols].mean(axis=1); out['avg_bank_vol_21d']=out[[f'{b}_vol_21d' for b in bank_cols]].mean(axis=1)
    out['avg_pairwise_corr_63d']=rets[bank_cols].rolling(63,min_periods=20).corr().groupby(level=0).apply(lambda m: m.where(~np.eye(len(m),dtype=bool)).stack().mean() if len(m)>1 else np.nan)
    return out
