from __future__ import annotations
import json
import subprocess
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from src.dashboard.reporting import (portfolio_action, portfolio_comparison, return_reconciliation,
    reported_summary, scenario_leaders, constraint_status, legacy_classifier_dataset as classifier_dataset)
from src.portfolio.portfolio_constraints import PortfolioConstraints
from src.features.stress_features import make_contagion_risk_score
from scripts.export_static_site import component_scores, composite_attribution, stress_path_chart, stress_paths, SCENARIO_SHOCKS, BANKS


def test_actual_holdings_rebalance_covers_all_assets_and_no_phantom_sales():
    current=pd.Series({'XFN.TO':.59,'XIU.TO':.31,'cash':.10})
    target=pd.Series({'XFN.TO':.51,'XIU.TO':.44,'cash':.05})
    result=portfolio_comparison(current,target)
    assert set(result.Bank)==set(BANKS+['XFN.TO','XIU.TO','cash'])
    banks=result.set_index('Bank').loc[BANKS]
    assert (banks.Action=='HOLD').all()
    assert np.isclose(result.Delta.sum(),0)
    assert result.Bank.tolist()[:3]==['XIU.TO','XFN.TO','cash']
    assert result['Current Weight'].sum()==pytest.approx(1)
    assert result['Target Weight'].sum()==pytest.approx(1)
    with pytest.raises(ValueError): portfolio_comparison(current,target*.55)


@pytest.mark.parametrize('delta,expected', [(-.002,'SELL'),(-.001,'HOLD'),(0,'HOLD'),(.001,'HOLD'),(.002,'BUY')])
def test_action_tolerance(delta,expected):
    assert portfolio_action(delta)==expected


def test_return_denominator_includes_first_day_cost_and_compounds():
    ledger=pd.DataFrame({'portfolio_value':[99950.,109945.],'daily_pnl':[-50.,9995.],
                         'daily_return':[-.0005,.1],'transaction_costs':[50.,0.],'turnover':[1.,0.]})
    before=ledger.copy(deep=True)
    r=return_reconciliation(ledger)
    assert r['initial_capital']==100000
    assert r['gross_initial_return']==pytest.approx(.09945)
    assert r['post_cost_nav_return']==pytest.approx(.1)
    assert r['compounded_daily_return']==pytest.approx(r['gross_initial_return'])
    assert reported_summary(ledger)['cumulative_return']==pytest.approx(.09945)
    pd.testing.assert_frame_equal(ledger,before)


def test_adverse_reporting_and_yield_transform_reconcile_to_score():
    f=pd.DataFrame({'avg_bank_vol_21d':[1.,2.,3.], 'avg_pairwise_corr_63d':[.2,.4,.9],
                    'XFN.TO_drawdown_63d':[-.2,-.1,-.3], 'VIX_level':[15.,25.,20.],
                    'slope_10y_2y':[-1.,1.,2.], 'CL=F_ret_21d':[-.2,0.,.5], 'CADUSD=X_ret_21d':[-.1,0.,.2]})
    f['contagion_risk_score']=make_contagion_risk_score(f)
    c=component_scores(f); a=composite_attribution(f)
    assert c.iloc[-1]['Oil shock']==pytest.approx(100/3)
    assert c.iloc[-1]['CAD pressure']==pytest.approx(100/3)
    assert 'Oil shock' not in set(a.Component)
    assert a.Contribution.sum()==pytest.approx(f.contagion_risk_score.iloc[-1])
    for _,row in a.iterrows(): assert c.iloc[-1][row.Component]==pytest.approx(row.Percentile)
    # Earlier breadth values do not depend on later observations.
    pd.testing.assert_frame_equal(component_scores(f.iloc[:2]),c.iloc[:2])


def test_constraint_slack_and_active_floor():
    weights=pd.Series({**{b:0. for b in BANKS},'XFN.TO':.51,'XIU.TO':.44,'cash':.05})
    c=constraint_status(weights,PortfolioConstraints(max_bank_exposure=.70,min_cash_weight=.05)).set_index('Constraint')
    assert c.loc['Cash floor','Status']=='Active'
    assert c.loc['Modeled financial-exposure proxy (Big Six + XFN)','Slack']=='19.00 pp'


def test_exact_and_display_ties_are_distinct():
    assert len(scenario_leaders(pd.Series(100.,index=BANKS))['exact'])==6
    result=scenario_leaders(pd.Series({'RY.TO':71.81,'CM.TO':71.82}))
    assert result['exact']==['CM.TO']
    assert result['displayed']==['RY.TO','CM.TO']


@pytest.mark.parametrize('name',SCENARIO_SHOCKS)
@pytest.mark.parametrize('severity',[.5,1.,1.5])
def test_scenario_js_python_path_identity_and_title(name,severity):
    prices=pd.DataFrame({bank:100*np.cumprod(1+np.sin(np.arange(150)/7+i/3)*.005) for i,bank in enumerate(BANKS)})
    paths,_=stress_paths(prices,name,severity)
    corr=prices[BANKS].pct_change().tail(126).corr().fillna(0).clip(lower=0)
    values=corr.to_numpy(copy=True)
    np.fill_diagonal(values,0)
    corr=pd.DataFrame(values,index=BANKS,columns=BANKS)
    adj=corr.div(corr.sum(axis=1).replace(0,1),axis=0)
    data={'banks':BANKS,'shocks':SCENARIO_SHOCKS,'adjacency':adj.values.tolist()}
    script="const a=require('./src/dashboard/scenario_reporting.js'); const d=JSON.parse(process.argv[1]); const p=a.calculate(d,process.argv[2],Number(process.argv[3])); console.log(JSON.stringify({paths:p,summary:a.summarize(d,p,process.argv[2],Number(process.argv[3])*100)}));"
    out=json.loads(subprocess.check_output(['node','-e',script,json.dumps(data),name,str(severity)],text=True))
    np.testing.assert_allclose(out['paths'],paths.values,atol=1e-12)
    assert out['summary']['title']==f'{name} ({severity:.0%}): Contagion Propagation'
    assert stress_path_chart(paths,name,severity).layout.title.text==out['summary']['title']
    assert 'does not rerun' in out['summary']['response']


def test_saturation_never_singles_out_a_bank_for_reduction():
    script="const a=require('./src/dashboard/scenario_reporting.js'); const b=['RY','TD','BMO','BNS','CM','NA']; console.log(JSON.stringify(a.summarize({banks:b}, [Array(6).fill(60),Array(6).fill(100)],'Liquidity Squeeze',150)));"
    out=json.loads(subprocess.check_output(['node','-e',script],text=True))
    assert 'no unique terminal leader' in out['response']
    assert '6 of 6 banks' in out['response']
    assert len(out['leaders']['exact'])==6


def test_classifier_intake_preserves_existing_label_math():
    f=pd.DataFrame({'x':np.arange(30.),'contagion_risk_score':np.arange(30.)})
    ds,cols,split,cut=classifier_dataset(f)
    y=(f.contagion_risk_score.shift(-5)>=f.contagion_risk_score.shift(-5).quantile(.8)).astype(int)
    assert ds.target.tolist()==y.tolist()
    assert ds.target.tail(5).sum()==0  # documented, not silently corrected
    assert split==21


def test_generated_run_identity_reconciliation_and_denominator_contract():
    from html.parser import HTMLParser
    from urllib.parse import unquote
    class Downloads(HTMLParser):
        def handle_starttag(self,tag,attrs):
            attrs=dict(attrs)
            if attrs.get('download')=='northern-signal-run-evidence.json':
                self.data=json.loads(unquote(attrs['href'].split(',',1)[1]))
    manifests=[]
    for name in ['models','decision','performance','research']:
        p=Downloads();p.feed(Path(f'public/{name}.html').read_text());manifests.append(p.data)
    assert all(m==manifests[0] for m in manifests)
    m=manifests[0]
    assert m['runs']['PPO common period']['start']==m['runs']['CVaR common period']['start']
    assert m['runs']['PPO extended period']['start']<m['runs']['PPO common period']['start']
    for run in m['runs'].values():
        assert sum(run['latest_weights'].values())==pytest.approx(1)
        r=run['return_reconciliation']
        assert r['gross_initial_return']==pytest.approx(r['ending_value']/r['initial_capital']-1)
        assert r['gross_initial_return']==pytest.approx(r['compounded_daily_return'])
    assert sum(m['current_cvar_target'].values())==pytest.approx(1)
    decision=Path('public/decision.html').read_text()
    assert 'policy reconciliation pending' in decision
    assert 'Signal-derived target weights' not in decision
    assert 'Validation confidence' not in decision
    overview=Path('public/index.html').read_text()
    assert 'with 20–35% cash' in overview or 'cash as editorial guidance' in overview
    assert 'XFN beta hedge worthwhile' not in overview
