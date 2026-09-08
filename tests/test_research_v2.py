import json
import numpy as np
import pandas as pd
import pytest
from src.research.provenance import validate_panel,TRADABLES,ASSETS,write_json,digest,verify_download
from src.research.execution import CloseLedger,ObservationSpec,observation,constrained_ppo,simulate,report_metrics
from src.research.evaluation import labeled_frame,freeze_protocol,partition
from src.research.case import scenario_paths,PRESETS,review_status,editorial,allocation_rows
from src.rl.env import CanadianBankContagionEnv

def fixture_frames(n=300):
    dates=pd.bdate_range('2020-01-01',periods=n)
    prices=pd.DataFrame({a:100*np.cumprod(1+.001*np.sin(np.arange(n)/9+i)) for i,a in enumerate(TRADABLES)},index=dates)
    features=pd.DataFrame({f'f{i}':np.cos(np.arange(n)/11+i) for i in range(40)},index=dates)
    features['contagion_risk_score']=50+30*np.sin(np.arange(n)/15)
    return prices,features

def test_no_fallback_in_production():
    from src.data.market_data import download_market_data
    from src.data.boc_valet import download_boc_series
    with pytest.raises(ValueError,match='Synthetic'): download_market_data(['RY.TO'],fallback=True)
    with pytest.raises(ValueError,match='Synthetic'): download_boc_series({'rate':'x'},fallback=True)

def test_provenance_rejects_mutation(tmp_path):
    p=tmp_path/'data.csv'; p.write_text('date,value\n2020-01-01,1\n')
    write_json(str(p)+'.manifest.json',{'status':'downloaded_verified','sha256':digest(p)})
    verify_download(p);p.write_text('different')
    with pytest.raises(ValueError): verify_download(p)

def test_panel_quality():
    p,_=fixture_frames()
    validate_panel(p,TRADABLES,positive=True)
    with pytest.raises(ValueError): validate_panel(p.drop(columns=TRADABLES[0]),TRADABLES)
    p.iloc[0,0]=np.inf
    with pytest.raises(ValueError): validate_panel(p,TRADABLES)

def test_labels_and_purge_no_future_calibration():
    _,f=fixture_frames(); frame,cols=labeled_frame(f); protocol=freeze_protocol(frame,cols,'test')
    a,b,c,q=partition(frame,protocol)
    assert len(frame)==len(f)-5
    assert a.outcome_date.max()<pd.Timestamp(protocol['validation_start'])
    assert b.outcome_date.max()<pd.Timestamp(protocol['test_start'])
    changed=frame.copy(); changed.loc[changed.index>=pd.Timestamp(protocol['test_start']),'future_score']=10000
    assert partition(changed,protocol)[3]==q
    assert (frame.outcome_date>frame.index).all()

def test_cannot_earn_return_before_execution():
    ledger=CloseLedger(['stock','cash'],capital=100,cost_bps=0,threshold=0)
    first=ledger.advance([1.,0],[1.,0])
    assert first['portfolio_value']==100 # doubled before buying; no gain
    second=ledger.advance([.1,0],None)
    assert second['portfolio_value']==pytest.approx(110)

def test_costs_and_drift_match_training_and_inference():
    p,f=fixture_frames(70); env=CanadianBankContagionEnv(p,f); obs,_=env.reset(); ledger=CloseLedger(ASSETS,capital=1.)
    target=constrained_ppo(np.zeros(9));r=p.pct_change(fill_method=None).fillna(0);r['cash']=0
    expected=observation(r[ASSETS].to_numpy(),f.iloc[:,:40].to_numpy(),ledger.weights,20)
    np.testing.assert_array_equal(obs,expected)
    for i in range(21,28):
        actual=env.step(np.zeros(9)); standalone=ledger.advance(r.loc[p.index[i],ASSETS].to_numpy(),target)
        assert actual[4]['value']==pytest.approx(standalone['portfolio_value'])
        np.testing.assert_allclose(actual[4]['weights'],standalone['weights'])
    assert ledger.values[-1]>=0

def test_schema_reordering_rejected():
    p,f=fixture_frames(); spec=ObservationSpec.from_features(f)
    with pytest.raises(ValueError,match='Feature'): spec.validate(p,f[list(reversed(f.columns))])
    with pytest.raises(ValueError,match='Asset'): spec.validate(p[list(reversed(p.columns))],f)

def test_original_capital_and_undefined_ratios():
    m=report_metrics([100,99.95,109.945],[0,-.0005,.1],capital=100)
    assert m['cumulative_return']==pytest.approx(.09945)
    assert m['cagr']==pytest.approx((109.945/100)**126-1)
    cash=report_metrics([100,100,100],[0,0,0],capital=100)
    assert cash['sharpe'] is None and cash['legacy_sortino_variant'] is None

def test_review_gates_and_cash_action():
    assert review_status(False,True,True,.1,editorial(65))['status'].startswith('Blocked')
    assert review_status(True,True,True,.1,editorial(65))['status'].startswith('Held')
    assert review_status(True,True,True,.25,editorial(65))['status'].startswith('Ready')
    rows=allocation_rows(np.ones(9)/9,np.r_[np.zeros(6),.5,.45,.05])
    assert next(r for r in rows if r['asset']=='cash')['action']=='Decrease cash'

@pytest.mark.parametrize('preset',PRESETS)
@pytest.mark.parametrize('q',[.5,1,1.5])
def test_scenario_controls_and_uncapped_identity(preset,q):
    a=(np.ones((6,6))-np.eye(6))/5
    path=np.asarray(scenario_paths(a,PRESETS[preset],q))
    assert path.shape==(6,6) and path.min()>=0 and path.max()<=100
    for before,after in zip(path[:-1],path[1:]):
        if after.max()<100: assert after.sum()==pytest.approx(before.sum()*1.15)
    if preset=='Liquidity squeeze' and q==1.5: assert np.all(path[-1]==100)

def test_zero_row_not_assumed_to_amplify():
    a=np.zeros((6,6));path=np.asarray(scenario_paths(a,PRESETS['Liquidity squeeze']))
    assert path[-1].mean()==pytest.approx(37*.7**5)
