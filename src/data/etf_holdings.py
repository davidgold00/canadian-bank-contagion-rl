import pandas as pd

def load_etf_holdings(path='data/templates/etf_holdings_template.csv'):
    return pd.read_csv(path, parse_dates=['date'])

def ownership_overlap(holdings, banks):
    pivot=holdings.pivot_table(index='ETF ticker', columns='holding ticker', values='holding weight', aggfunc='sum').fillna(0)
    rows=[]
    for i in banks:
        for j in banks:
            rows.append((i,j,sum(min(pivot.get(i,0).get(etf,0), pivot.get(j,0).get(etf,0)) for etf in pivot.index)))
    return pd.DataFrame(rows, columns=['source','target','overlap'])
