from pathlib import Path
import requests, pandas as pd
from .sample_data import make_sample_macro_data

class BankOfCanadaValetClient:
    def __init__(self, base_url='https://www.bankofcanada.ca/valet'):
        self.base_url=base_url.rstrip('/')
    def fetch_series(self, series_ids, start_date=None, end_date=None):
        ids=','.join(series_ids); url=f'{self.base_url}/observations/{ids}/json'
        params={k:v for k,v in {'start_date':start_date,'end_date':end_date}.items() if v}
        r=requests.get(url, params=params, timeout=20); r.raise_for_status(); js=r.json()
        rows=[]
        for obs in js.get('observations',[]):
            row={'date':obs['d']}
            for sid in series_ids:
                row[sid]=float(obs.get(sid,{}).get('v','nan'))
            rows.append(row)
        return pd.DataFrame(rows).assign(date=lambda d: pd.to_datetime(d.date)).set_index('date')

def download_boc_series(series_map, output='data/raw/boc_yields.csv', fallback=True):
    try:
        df=BankOfCanadaValetClient().fetch_series(list(series_map.values()))
        df=df.rename(columns={v:k for k,v in series_map.items()})
        if df.empty: raise ValueError('empty BoC response')
        Path(output).parent.mkdir(parents=True, exist_ok=True); df.to_csv(output); return df
    except Exception:
        if not fallback: raise
        return make_sample_macro_data(output)
