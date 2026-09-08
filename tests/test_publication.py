import json,sys,subprocess
from pathlib import Path
import numpy as np,pandas as pd,pytest
from src.research.publication import release_transaction
from src.research.provenance import validate_panel,write_json
from src.dashboard.case_site import render_case,validate_site,tree_hashes
from scripts.rebuild_research import check_history

def test_failure_leaves_complete_prior_release(tmp_path):
    pointer=tmp_path/'current'
    with release_transaction(pointer,'first') as stage:(stage/'marker').write_text('first')
    original=pointer.resolve()
    with pytest.raises(RuntimeError):
        with release_transaction(pointer,'second') as stage:
            (stage/'marker').write_text('partial');raise RuntimeError('failure while building')
    assert pointer.resolve()==original and (pointer/'marker').read_text()=='first'
    with release_transaction(pointer,'second') as stage:(stage/'marker').write_text('second')
    assert (original/'marker').read_text()=='first' and (pointer/'marker').read_text()=='second'

def test_pinned_release_cannot_be_rewritten(tmp_path):
    pointer=tmp_path/'current'
    with release_transaction(pointer,'same') as stage:(stage/'marker').write_text('original')
    with pytest.raises(ValueError,match='different contents'):
        with release_transaction(pointer,'same') as stage:(stage/'marker').write_text('changed')
    assert (pointer/'marker').read_text()=='original'

def test_provider_failure_never_overwrites_cache(tmp_path,monkeypatch):
    import yfinance
    from src.data.market_data import download_market_data
    from src.data.boc_valet import download_boc_series,BankOfCanadaValetClient
    cache=tmp_path/'market.csv';cache.write_text('previous verified bytes')
    def fail(*a,**k):raise ConnectionError('provider offline')
    monkeypatch.setattr(yfinance,'download',fail)
    with pytest.raises(ConnectionError):download_market_data(['RY.TO'],output=cache)
    monkeypatch.setattr(BankOfCanadaValetClient,'fetch_series',fail)
    with pytest.raises(ConnectionError):download_boc_series({'ca_2y':'id'},output=cache)
    assert cache.read_text()=='previous verified bytes'
    assert not Path(str(cache)+'.manifest.json').exists()

def test_invalid_dates_duplicates_and_nonnumeric_rejected():
    for f in [pd.DataFrame({'a':[1,2]},index=pd.to_datetime(['2020-01-01','2020-01-01'])),pd.DataFrame({'a':['oops',2]},index=pd.date_range('2020-01-01',periods=2)),pd.DataFrame({'a':[1,2]},index=['bad','dates'])]:
        with pytest.raises(ValueError):validate_panel(f,['a'])

def test_revisions_require_explicit_evaluation(tmp_path):
    a=tmp_path/'old';b=tmp_path/'new';a.mkdir();b.mkdir()
    old=pd.DataFrame({'a':[1,2]},index=pd.date_range('2020-01-01',periods=2));new=pd.DataFrame({'a':[1,2,3]},index=pd.date_range('2020-01-01',periods=3))
    for name in ['prices','features','macro']:old.to_csv(a/f'{name}.csv');new.to_csv(b/f'{name}.csv')
    check_history(a,b)
    new.iloc[0,0]=10;new.to_csv(b/'prices.csv')
    with pytest.raises(ValueError,match='revisions'):check_history(a,b)

def test_render_has_no_training_or_optimizer_dependency(tmp_path):
    # -S disables site packages, proving rendering does not require sklearn/SB3/scipy.
    code="from src.dashboard.case_site import render_case; render_case('artifacts/current/case.json',"+repr(str(tmp_path/'current'))+")"
    subprocess.run([sys.executable,'-S','-c',code],check=True,capture_output=True,text=True)
    assert validate_site(tmp_path/'current')

def test_case_reconciles_every_asset_and_comparison():
    c=json.loads(Path('artifacts/current/case.json').read_text());p=c['proposal'];r=c['paper_run']
    assert len(c['allocation_rows'])==9
    np.testing.assert_array_equal(c['actual_weights'],r['weights'][-1]);np.testing.assert_array_equal(p['prior_weights'],c['actual_weights'])
    for row in c['allocation_rows']:
        i=r['assets'].index(row['asset']);assert row['delta']==pytest.approx(p['weights'][i]-c['actual_weights'][i])
    dates=[(x['start'],x['end'],len(x['ledger'])) for x in c['benchmarks'].values()];assert len(set(dates))==1
    assert c['ppo']['selected_seed']==max(c['ppo']['candidates'],key=lambda x:x['validation_reward'])['seed']
    for run in c['benchmarks'].values():
        factor=np.prod([1+x['daily_return'] for x in run['ledger']]);assert factor==pytest.approx(run['metrics']['ending_value']/run['capital'])

def test_frozen_evaluation_identity_and_reward_terms():
    from src.research.case import validate_ppo
    c=json.loads(Path('artifacts/current/case.json').read_text())
    validate_ppo(c.get('evaluation_directory','artifacts/evaluations/verified-v2'),c['ppo'],c['classifier']['protocol'])
    terms=[r['terms'] for r in c['reward_diagnostic']['rows']]
    for k in ['dispersion','contagion','stress']:assert len(set(r[k] for r in terms))==1

def test_out_of_order_publication_is_rejected():
    from scripts.guard_publication import may_publish
    current={'snapshot_id':'case-new','feature_date':'2026-09-04','generated_at':'2026-09-07T10:00:00+00:00'}
    may_publish(current,current)
    with pytest.raises(ValueError,match='older data'):may_publish({**current,'snapshot_id':'case-old','feature_date':'2026-09-03'},current)
    with pytest.raises(ValueError,match='older prepared'):may_publish({**current,'snapshot_id':'case-old','generated_at':'2026-09-07T09:00:00+00:00'},current)
