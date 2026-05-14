import pandas as pd, numpy as np

def synthetic_credit_spread_proxy(features, banks):
    out=pd.DataFrame(index=features.index)
    vix=features.get('VIX_level', pd.Series(20,index=features.index))
    for b in banks:
        vol=features.get(f'{b}_vol_21d', pd.Series(0.2,index=features.index))
        dd=features.get(f'{b}_drawdown_63d', pd.Series(0,index=features.index)).abs()
        out[f'{b}_cds_proxy_bps']=50+350*(vol.rank(pct=True)*0.4+dd.rank(pct=True)*0.4+vix.rank(pct=True)*0.2)
    return out
