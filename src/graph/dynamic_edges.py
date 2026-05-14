def tail_dependence_edges(returns, market_col='XFN.TO', threshold=-0.02):
    stress=returns[market_col] < threshold
    return returns.loc[stress].corr().abs().fillna(0)
