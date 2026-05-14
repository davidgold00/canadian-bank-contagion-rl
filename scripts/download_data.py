import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.sample_data import ensure_sample_data
from src.data.market_data import download_market_data
TICKERS=['RY.TO','TD.TO','BMO.TO','BNS.TO','CM.TO','NA.TO','XFN.TO','XIU.TO','CADUSD=X','CL=F','GC=F','^GSPTSE','^VIX']
if __name__=='__main__':
    ensure_sample_data(); download_market_data(TICKERS); print('Downloaded or generated fallback data.')
