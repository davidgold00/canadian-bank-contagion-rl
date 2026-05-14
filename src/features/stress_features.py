import numpy as np, pandas as pd

def _pct_rank(s): return s.expanding().rank(pct=True)

def make_contagion_risk_score(df):
    comps=[]
    for c in ['avg_bank_vol_21d','avg_pairwise_corr_63d','XFN.TO_drawdown_63d','VIX_level']:
        if c in df: comps.append(_pct_rank(df[c].abs() if 'drawdown' in c else df[c]))
    if 'slope_10y_2y' in df: comps.append(_pct_rank((-df['slope_10y_2y']).clip(lower=0)))
    if not comps: return pd.Series(50,index=df.index)
    score=pd.concat(comps,axis=1).mean(axis=1).clip(0,1)*100
    return score.ffill().fillna(50)

def bank_node_stress_scores(df, banks):
    out=pd.DataFrame(index=df.index)
    for b in banks:
        cols=[f'{b}_vol_21d', f'{b}_drawdown_63d', f'{b}_cds_proxy_bps']
        comps=[_pct_rank(df[c].abs() if 'drawdown' in c else df[c]) for c in cols if c in df]
        out[f'{b}_node_stress_score']=pd.concat(comps,axis=1).mean(axis=1).fillna(0.5)*100 if comps else 50
    return out
