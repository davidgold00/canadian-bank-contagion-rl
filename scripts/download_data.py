import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.market_data import download_market_data
from src.data.boc_valet import download_boc_series
TICKERS=[
    'RY.TO','TD.TO','BMO.TO','BNS.TO','CM.TO','NA.TO',
    'XFN.TO','XIU.TO','ZEB.TO',
    '^GSPC','^IXIC','^DJI','^GSPTSE','^VIX',
    'CADUSD=X','CL=F','GC=F',
]
BOC_SERIES={
    'policy_rate': 'V39079',
    'ca_2y': 'BD.CDN.2YR.DQ.YLD',
    'ca_5y': 'BD.CDN.5YR.DQ.YLD',
    'ca_10y': 'BD.CDN.10YR.DQ.YLD',
}
if __name__=='__main__':
    download_market_data(TICKERS)
    download_boc_series(BOC_SERIES)
    print('Downloaded and recorded source manifests. No synthetic production fallback is allowed.')
