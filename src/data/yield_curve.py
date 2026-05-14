import pandas as pd

def transform_yields(df):
    out=df.copy().sort_index().ffill()
    if {'ca_10y','ca_2y'}.issubset(out.columns): out['slope_10y_2y']=out['ca_10y']-out['ca_2y']
    if {'ca_5y','ca_2y','ca_10y'}.issubset(out.columns): out['curvature']=2*out['ca_5y']-out['ca_2y']-out['ca_10y']
    for c in [x for x in out.columns if 'ca_' in x or 'policy' in x]:
        out[f'{c}_chg_5d']=out[c].diff(5); out[f'{c}_chg_21d']=out[c].diff(21); out[f'{c}_vol_63d']=out[c].diff().rolling(63,min_periods=10).std()
    return out
