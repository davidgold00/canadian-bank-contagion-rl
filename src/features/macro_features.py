def make_macro_features(macro):
    out=macro.copy().sort_index().ffill()
    for c in out.columns:
        out[f'{c}_chg_5d']=out[c].diff(5); out[f'{c}_chg_21d']=out[c].diff(21)
    return out
