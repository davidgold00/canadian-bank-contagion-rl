"""Owner-only staged rebuild. Does not call training or evaluation."""
import argparse,json,sys,uuid,fcntl
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.research.provenance import write_json,utcnow,digest,prepare_dataset

def check_history(previous,current):
    import pandas as pd
    import numpy as np
    for name in ['prices','features','macro']:
        old=pd.read_csv(Path(previous)/f'{name}.csv',index_col=0,parse_dates=True)
        new=pd.read_csv(Path(current)/f'{name}.csv',index_col=0,parse_dates=True)
        if not old.index.isin(new.index).all() or list(old.columns)!=list(new.columns):
            raise ValueError(f'{name}: historical coverage or schema changed; explicit new evaluation required')
        a=old.to_numpy();b=new.loc[old.index,old.columns].to_numpy()
        if not np.allclose(a,b,rtol=1e-10,atol=1e-12,equal_nan=True):
            raise ValueError(f'{name}: historical vendor/input revisions detected; retain this intake and explicitly evaluate a new input version')

def rebuild(download=False,config='configs/research-case.json'):
    cfg=json.loads(Path(config).read_text()); job=Path('build/jobs')/(utcnow().replace(':','-')+'-'+uuid.uuid4().hex[:8]);job.mkdir(parents=True)
    state={'started':utcnow(),'status':'running','stages':[]};log=job/'status.json'
    def stage(name):
        state['stage']=name;state['stages'].append(name);write_json(log,state);print(name,flush=True)
    with Path('build/.owner-rebuild.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        try:
            dataset=Path(cfg['dataset'])
            # When continuing an owner-produced case, use its last verified intake.
            pointer=Path('build/current')
            previous=dataset
            prior_state=Path('build/last-intake.json')
            if prior_state.exists():previous=Path(json.loads(prior_state.read_text())['dataset'])
            if download:
                from scripts.download_data import TICKERS,BOC_SERIES
                from src.data.market_data import download_market_data
                from src.data.boc_valet import download_boc_series
                stage('Download verified market observations')
                download_market_data(TICKERS,output=job/'market.csv',fallback=False)
                stage('Download Bank of Canada observations')
                download_boc_series(BOC_SERIES,output=job/'macro.csv',fallback=False)
                stage('Validate completeness, staleness and information alignment')
                dataset=job/'dataset';prepare_dataset(job/'market.csv',job/'macro.csv',dataset)
                stage('Detect historical revisions before any new review')
                check_history(previous,dataset)
            else:stage('Verify pinned inputs and frozen evaluations')
            from src.research.case import build_case
            stage('Prepare the current paper path and proposal; reuse frozen evaluation')
            case=build_case(dataset,cfg['evaluation'],job/'prepared')
            from src.dashboard.case_site import render_case
            stage('Render and validate every route in an immutable release')
            render_case(job/'prepared/case.json')
            stage('Verify the expected snapshot is locally published')
            actual=json.loads((pointer/'snapshot-manifest.json').read_text())
            if actual['snapshot_id']!=case['manifest']['snapshot_id']:raise RuntimeError('Published snapshot differs from expected')
            write_json(cfg['current_case'],case)
            write_json('build/last-intake.json',{'dataset':str(dataset)})
            state.update(status='complete',snapshot_id=actual['snapshot_id'],finished=utcnow(),local_only=True);write_json(log,state)
            print(actual['snapshot_id'],flush=True);return case
        except Exception as error:
            state.update(status='failed',error=str(error),finished=utcnow());write_json(log,state);raise
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--download',action='store_true');p.add_argument('--config',default='configs/research-case.json');a=p.parse_args();rebuild(a.download,a.config)
