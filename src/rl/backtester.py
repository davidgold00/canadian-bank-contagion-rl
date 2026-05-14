def performance_metrics(values):
    import pandas as pd, numpy as np
    v=pd.Series(values); r=v.pct_change().dropna(); return {'cumulative_return':float(v.iloc[-1]/v.iloc[0]-1),'annualized_vol':float(r.std()*np.sqrt(252)),'max_drawdown':float((v/v.cummax()-1).min()),'sharpe':float(r.mean()/r.std()*np.sqrt(252)) if r.std() else 0}
