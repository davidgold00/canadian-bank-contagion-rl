def make_stress_event_labels(df):
    y=(df['contagion_risk_score']>df['contagion_risk_score'].quantile(.8)).astype(int)
    return y.shift(-5).rename('stress_event_next_5d')
