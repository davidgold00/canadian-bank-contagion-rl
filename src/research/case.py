"""Prepare one coherent review artifact. This stage solves; rendering only reads."""
from dataclasses import asdict
import json,subprocess
from pathlib import Path
import numpy as np
import pandas as pd
from .provenance import ASSETS,TRADABLES,digest,object_id,write_json,utcnow
from .execution import simulate,CloseLedger,constrained_ppo
from src.portfolio.cvar_optimizer import optimize_cvar_portfolio
from src.portfolio.portfolio_constraints import PortfolioConstraints,constraint_diagnostics
from src.portfolio.regime_detection import regime_constraints
from src.portfolio.allocation_policy import bank_node_stress
BANKS=TRADABLES[:6]
PRESETS={'Housing crisis':[35,35,30,30,45,32],'Oil crash':[20,18,26,25,22,18],'Liquidity squeeze':[40,38,36,36,38,34],'Yield-curve inversion':[24,24,22,22,28,20],'Global risk-off':[32,32,30,31,33,29]}
COMPONENTS=[('Bank volatility','avg_bank_vol_21d','annualized volatility','21 sessions'),('Bank correlation','avg_pairwise_corr_63d','correlation','63 sessions'),('Financials drawdown','XFN.TO_drawdown_63d','return below recent peak','63 sessions'),('Global volatility','VIX_level','VIX index','daily level'),('Yield-curve inversion','slope_10y_2y','percentage-point slope','latest lagged observation')]

def mandate(score):
    r=regime_constraints(score)
    return PortfolioConstraints(max_single_name_weight=min(.20,r['max_single_name_weight']),max_bank_exposure=min(.70,r['max_bank_exposure']),min_cash_weight=max(.04,r['min_cash_weight']),max_cash_weight=.60)

def solve(prices,features,previous,**changes):
    kwargs=dict(confidence_level=.95,lookback_window=126,risk_aversion=6.,cvar_penalty=8.,volatility_penalty=1.,turnover_penalty=.20,contagion_penalty=.80,graph_penalty_strength=.40,constraints=mandate(float(features.contagion_risk_score.iloc[-1])),previous_weights=pd.Series(previous,index=ASSETS))
    kwargs.update(changes)
    return optimize_cvar_portfolio(prices,features,assets=ASSETS,**kwargs)

def score_evidence(features):
    available=[]
    rank_columns={}
    for label,col,unit,window in COMPONENTS:
        if col not in features: continue
        raw=features[col]; transformed=(-raw).clip(lower=0) if col=='slope_10y_2y' else raw.abs() if 'drawdown' in col else raw
        ranks=transformed.expanding().rank(pct=True); rank_columns[label]=ranks
        if pd.notna(ranks.iloc[-1]): available.append({'component':label,'feature':col,'raw':float(raw.iloc[-1]),'unit':unit,'window':window,'transformed':float(transformed.iloc[-1]),'percentile':float(100*ranks.iloc[-1]),'history_observations':int(raw.notna().sum())})
    for r in available: r['weight']=1/len(available); r['contribution']=r['percentile']/len(available)
    count=pd.DataFrame(rank_columns).notna().sum(axis=1)
    return sorted(available,key=lambda r:-r['contribution']),count

def scenario_paths(matrix,initial,severity=1.,spillover=.45):
    s=np.clip(np.asarray(initial,float)*severity,0,100); out=[s.tolist()]
    for _ in range(5): s=np.clip(.70*s+spillover*np.asarray(matrix).T@s,0,100); out.append(s.tolist())
    return out

def editorial(score):
    if score<30: return {'cash_min':.05,'cash_max':None,'cash_text':'5–10% minimum guidance; no hard upper bound','bank_text':'65–75% editorial range'}
    if score<60: return {'cash_min':.10,'cash_max':.15,'cash_text':'10–15%','bank_text':'50–65% editorial range'}
    if score<80: return {'cash_min':.20,'cash_max':.35,'cash_text':'20–35%','bank_text':'35–50% editorial range'}
    return {'cash_min':.35,'cash_max':None,'cash_text':'35–50%+; no hard upper bound','bank_text':'20–35% editorial range'}

def review_status(verified,compatible,constraints_pass,cash,guidance):
    reasons=[]
    if not verified: reasons.append('Input provenance is incomplete.')
    if not compatible: reasons.append('Proposal method or artifact identity is incomplete.')
    if not constraints_pass: reasons.append('The proposal does not satisfy its model constraints.')
    cash_conflict=cash<guidance['cash_min']-1e-6 or (guidance['cash_max'] is not None and cash>guidance['cash_max']+1e-6)
    if cash_conflict: reasons.append(f"Proposal cash {cash:.1%} lies outside editorial guidance {guidance['cash_text']}.")
    if cash<guidance['cash_min']-1e-6: reasons.append(f"Cash is {(guidance['cash_min']-cash)*100:.1f} percentage points below the editorial lower bound.")
    status='Blocked — evidence incomplete' if not verified or not compatible or not constraints_pass else 'Held for review — guidance conflict' if cash_conflict else 'Ready for research review'
    return {'status':status,'reasons':reasons or ['Model constraints and the specified cash guidance are satisfied.'],
            'economic_exposure':'Not assessable — verified ETF look-through unavailable',
            'meaning':'A research-review disposition, not investment approval.',
            'next_evidence':['Dated ETF holdings are needed to assess economic bank and issuer exposure.','Resolve any cash-guidance conflict before endorsing the proposal.','Consider the matched historical comparison and its period limits.']}

def allocation_rows(current,target):
    rows=[]
    for a,c,t in zip(ASSETS,current,target):
        d=float(t-c); material=abs(d)>.001
        action=('Increase cash' if d>.001 else 'Decrease cash' if d<-.001 else 'No material cash change') if a=='cash' else ('BUY' if d>.001 else 'SELL' if d<-.001 else 'HOLD')
        rows.append({'asset':a,'current':float(c),'target':float(t),'delta':d,'action':action,'trade_threshold_met':a!='cash' and abs(d)>=.01})
    return sorted(rows,key=lambda r:(-round(abs(r['delta'])*100,2),r['asset']))

def summarize_solution(result,prices):
    w=result.weights.reindex(ASSETS).to_numpy(); returns=prices[TRADABLES].pct_change(fill_method=None).tail(126).fillna(0); returns['cash']=0
    port=returns[ASSETS].to_numpy()@w; tail=port[port<=np.quantile(port,.05)]
    return {'weights':w.tolist(),'diagnostics':{k:float(v) if isinstance(v,(float,int,np.number)) else str(v) for k,v in result.diagnostics.items()},
            'expected_return':float(result.expected_returns.reindex(ASSETS)@result.weights.reindex(ASSETS)),
            'expected_returns':result.expected_returns.to_dict(),'daily_cvar_95':float(-tail.mean()),'tail_count':len(tail),'sample_count':len(port),
            'volatility':float(np.sqrt(w@result.adjusted_covariance.reindex(index=ASSETS,columns=ASSETS).to_numpy()@w)),
            'risk_contributions':result.risk_contributions.to_dict('records')}

def validate_ppo(directory,manifest,protocol):
    from .execution import ObservationSpec
    from dataclasses import fields
    if manifest['identity']['protocol']!=protocol or manifest['identity']['input_hash']!=protocol['input_hash']:
        raise ValueError('PPO protocol/input identity mismatch')
    schema=manifest['schema'].copy(); schema['assets']=tuple(schema['assets']); schema['features']=tuple(schema['features'])
    if ObservationSpec(**schema).id!=manifest['schema_id']: raise ValueError('PPO schema identity mismatch')
    for candidate in manifest['candidates']:
        if digest(Path(directory)/'ppo'/candidate['artifact'])!=candidate['sha256']: raise ValueError('PPO artifact hash mismatch')
        run=json.loads((Path(directory)/'ppo'/f"test-seed-{candidate['seed']}.json").read_text())
        if run['artifact_sha256']!=candidate['sha256']: raise ValueError('PPO ledger/model mismatch')

def freeze_portfolio_evaluation(dataset,evaluation_dir):
    """Explicit evaluation stage, never invoked by data refresh or rendering."""
    dataset,evaluation_dir=map(Path,[dataset,evaluation_dir])
    provenance=json.loads((dataset/'provenance.json').read_text())
    for name in ['prices','features','macro']:
        if digest(dataset/f'{name}.csv')!=provenance['hashes'][name]: raise ValueError('Dataset hash mismatch')
    classifier=json.loads((evaluation_dir/'classifier.json').read_text()); protocol=classifier['protocol']
    if protocol['input_hash']!=provenance['hashes']['features']: raise ValueError('Evaluation data differs from classifier protocol')
    path=evaluation_dir/'portfolio-evaluation.json'
    if path.exists():
        saved=json.loads(path.read_text())
        if saved['input_hashes']!=provenance['hashes']: raise ValueError('Use a new evaluation directory for revised inputs')
        return saved
    prices=pd.read_csv(dataset/'prices.csv',index_col=0,parse_dates=True); features=pd.read_csv(dataset/'features.csv',index_col=0,parse_dates=True)
    start=protocol['test_start'];end=protocol['portfolio_evaluation_end']
    equal=constrained_ppo(np.zeros(len(ASSETS)))
    dev_end=prices.index[prices.index<pd.Timestamp(protocol['test_start'])][-1]
    static=solve(prices.loc[:dev_end],features.loc[:dev_end],np.eye(1,len(ASSETS),len(ASSETS)-1).ravel()).weights.reindex(ASSETS).to_numpy()
    benchmarks={'CVaR paper mandate':simulate(prices,features,start,end,lambda p,f,w:solve(p,f,w).weights.reindex(ASSETS).to_numpy(),frequency=10),
                'Equal-weight feasible':simulate(prices,features,start,end,lambda p,f,w:equal),
                'Development-fixed allocation':simulate(prices,features,start,end,lambda p,f,w:static)}
    for name,asset in [('XFN buy-and-hold (costed)','XFN.TO'),('XIU buy-and-hold (costed)','XIU.TO'),('Cash (zero interest)','cash')]:
        vector=np.zeros(len(ASSETS)); vector[ASSETS.index(asset)]=1
        benchmarks[name]=simulate(prices,features,start,end,lambda p,f,w,v=vector:v,frequency=10**9)
    ppo_manifest=None
    if (evaluation_dir/'ppo/manifest.json').exists():
        ppo_manifest=json.loads((evaluation_dir/'ppo/manifest.json').read_text())
        run=json.loads((evaluation_dir/f'ppo/test-seed-{ppo_manifest["selected_seed"]}.json').read_text())
        validate_ppo(evaluation_dir,ppo_manifest,protocol)
        if run['start']!=start or run['end']!=end: raise ValueError('PPO evaluation dates differ from frozen comparison')
        benchmarks['PPO corrected comparator']=run
    result={'id':'comparison-v2-'+object_id({'inputs':provenance['hashes'],'protocol':protocol,'method':'next-close-v2'}),
            'input_hashes':provenance['hashes'],'protocol':protocol,'benchmarks':benchmarks,'ppo':ppo_manifest,
            'static_weights':static.tolist(),'static_fixed_on':str(dev_end.date()),'frozen_at':utcnow(),
            'dataset_reference':str(dataset)}
    write_json(path,result,immutable=True)
    return result

def build_case(dataset,evaluation_dir,output):
    dataset,evaluation_dir,output=map(Path,[dataset,evaluation_dir,output])
    provenance=json.loads((dataset/'provenance.json').read_text())
    for name in ['prices','features','macro']:
        if digest(dataset/f'{name}.csv')!=provenance['hashes'][name]: raise ValueError('Dataset modified after provenance')
    prices=pd.read_csv(dataset/'prices.csv',index_col=0,parse_dates=True); features=pd.read_csv(dataset/'features.csv',index_col=0,parse_dates=True); macro=pd.read_csv(dataset/'macro.csv',index_col=0,parse_dates=True)
    classifier=json.loads((evaluation_dir/'classifier.json').read_text())
    protocol=classifier['protocol']
    start=protocol['test_start']; end=str(prices.index[-1].date())
    print('Computing corrected CVaR paper ledger',flush=True)
    frozen=json.loads((evaluation_dir/'portfolio-evaluation.json').read_text())
    cvar=frozen['benchmarks']['CVaR paper mandate'] if frozen['input_hashes']==provenance['hashes'] and end==protocol['portfolio_evaluation_end'] else simulate(prices,features,start,end,lambda p,f,w:solve(p,f,w).weights.reindex(ASSETS).to_numpy(),frequency=10)
    current=np.asarray(cvar['weights'][-1]); proposal_result=solve(prices,features,current); proposal=summarize_solution(proposal_result,prices)
    proposal['id']='proposal-'+object_id({'inputs':provenance['hashes'],'previous':current.tolist(),'weights':proposal['weights']})
    proposal['as_of']=end; proposal['prior_weights']=current.tolist(); proposal['kind']='Off-cycle indicative research proposal'
    constraints=mandate(float(features.contagion_risk_score.iloc[-1])); proposal['constraints']=asdict(constraints)
    proposal['constraint_checks']=constraint_diagnostics(proposal_result.weights,constraints).to_dict('records')
    proposal['configuration']={'risk_aversion':6,'cvar_penalty':8,'contagion_penalty':.8,'turnover_penalty':.2,'volatility_penalty':1,'graph_strength':.4,'lookback':126}
    rows=allocation_rows(current,proposal['weights'])
    preview=CloseLedger(); preview.values=current*cvar['metrics']['ending_value']; s=preview.advance(np.zeros(len(ASSETS)),proposal['weights'])
    proposal['preview']={'cost':s['transaction_costs'],'post_cost_weights':s['weights'].tolist(),'notionals':dict(zip(ASSETS,s['trades'].tolist())),'assumption':'Current marks held fixed for illustration; actual next-close execution will differ.'}
    diags=[]
    for name,changes in [('Current graph treatment',{}),('Covariance uplift off',{'graph_penalty_strength':0.}),('Node penalty off',{'contagion_penalty':0.}),('Both graph treatments off',{'graph_penalty_strength':0.,'contagion_penalty':0.})]:
        result=proposal_result if not changes else solve(prices,features,current,**changes)
        diags.append({'name':name,**summarize_solution(result,prices)})
    sensitivities=[]
    for risk in [1,2,4,6,8,10,14,18]:
        result=proposal_result if risk==6 else solve(prices,features,current,risk_aversion=risk)
        sensitivities.append({'risk_aversion':risk,**summarize_solution(result,prices)})
    components,coverage=score_evidence(features); score=float(features.contagion_risk_score.iloc[-1]); guidance=editorial(score)
    corr=prices[BANKS].pct_change(fill_method=None).tail(63).corr()
    adj=prices[BANKS].pct_change(fill_method=None).tail(126).corr().fillna(0).clip(lower=0).to_numpy(); np.fill_diagonal(adj,0); denom=adj.sum(axis=1,keepdims=True); adj=np.divide(adj,denom,out=np.zeros_like(adj),where=denom!=0)
    scenarios={'banks':BANKS,'matrix':adj.tolist(),'presets':PRESETS,'default':'Liquidity squeeze','steps':5,'persistence':.70,'spillover':.45}
    scenarios['default_path']=scenario_paths(adj,PRESETS['Liquidity squeeze'])
    evaluation=json.loads((evaluation_dir/'portfolio-evaluation.json').read_text())
    if evaluation['protocol']!=protocol: raise ValueError('Portfolio/classifier protocol mismatch')
    benchmarks=evaluation['benchmarks']; ppo_manifest=evaluation['ppo']
    if ppo_manifest: validate_ppo(evaluation_dir,ppo_manifest,protocol)
    legacy=json.loads(Path('artifacts/legacy/2026-09-04/run-evidence.json').read_text())
    alt=optimize_cvar_portfolio(prices,features,lookback_window=126,risk_aversion=7,cvar_penalty=9,contagion_penalty=.9,graph_penalty_strength=.4,constraints=PortfolioConstraints(max_single_name_weight=.2,max_bank_exposure=.7,min_cash_weight=.05,max_cash_weight=.6))
    node=bank_node_stress(features,BANKS).to_dict()
    mean_corr=float(corr.to_numpy()[~np.eye(6,dtype=bool)].mean())
    review=review_status(provenance['status']=='downloaded_verified',proposal['diagnostics'].get('status')=='optimal',all(r['Status']=='Pass' for r in proposal['constraint_checks']),proposal['weights'][-1],guidance)
    source_hash=object_id({str(p):digest(p) for folder in ['src','scripts','configs','reports','docs','.github'] for p in sorted(Path(folder).rglob('*')) if p.is_file() and p.suffix in ['.py','.js','.css','.yaml','.yml','.md']})
    revision=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()
    snapshot_id='case-'+object_id({'source':source_hash,'revision':revision,'input':provenance['hashes'],'proposal':proposal['id'],'classifier':classifier['id'] if classifier else None,'ppo':ppo_manifest['id'] if ppo_manifest else None,'method':'research-review-v2'})
    manifest={'schema_version':1,'snapshot_id':snapshot_id,'generated_at':utcnow(),'feature_date':end,'source_status':provenance['status'],
              'case_url':f'/snapshots/{snapshot_id}/','source_revision':revision,
              'source_tree_hash':source_hash,
              'method':'research-review-v2','source_observation_dates':{'market':provenance['market']['quality']['last_observation'],'macro':provenance['macro']['quality']['last_observation']},
              'feature_sha256':provenance['hashes']['features'],'model_id':ppo_manifest['id'] if ppo_manifest else None,'classifier_id':classifier['id'] if classifier else None}
    reward_rows=[]
    prior=np.asarray(cvar['weights'][-2]); last_returns=prices[TRADABLES].pct_change(fill_method=None).iloc[-1].tolist()+[0.]
    for label,target in [('Keep prior target',prior),('All cash',np.r_[np.zeros(8),1.]),('80% XFN / 20% cash',np.r_[np.zeros(6),.8,0,.2])]:
        ledger=CloseLedger();ledger.values=prior*cvar['ledger'][-2]['portfolio_value'];ledger.peak=max(x['portfolio_value'] for x in cvar['ledger'][:-1])
        state=ledger.advance(last_returns,target)
        terms={'return':state['daily_return'],'dispersion':-.20*np.std(last_returns)*np.sqrt(252),'drawdown':-.30*abs(state['drawdown']),'turnover':-.02*state['turnover'],'contagion':-.15*score/100,'stress':-.10*score/100,'excess':.25*(state['daily_return']-last_returns[6])}
        reward_rows.append({'action':label,'terms':terms,'total':float(sum(terms.values()))})
    case={'reward_diagnostic':{'date':end,'rows':reward_rows,'scope':'Same prior holdings, peak and next-close market returns; only queued target differs. This is a diagnostic, not a model action.'},'manifest':manifest,'provenance':provenance,'score':score,'score_percentile':float(features.contagion_risk_score.rank(pct=True).iloc[-1]*100),'components':components,'component_coverage':{'current':int(coverage.iloc[-1]),'incomplete_history_rows':int((coverage<5).sum())},
          'history':[{'date':str(d.date()),'score':float(v),'components':int(coverage.loc[d])} for d,v in features.contagion_risk_score.tail(252).items()],
          'correlation':corr.to_dict(),'mean_correlation':mean_corr,'node_stress':node,'proposal':proposal,'allocation_rows':rows,'review':review,'guidance':guidance,
          'scenarios':scenarios,'classifier':classifier,'ppo':ppo_manifest,'benchmarks':benchmarks,'legacy':legacy,'alternative':summarize_solution(alt,prices),
          'diagnostics':diags,'sensitivities':sensitivities,'evaluation_directory':str(evaluation_dir),'evaluation_id':evaluation['id'],'evaluation_input_hashes':evaluation['input_hashes'],'static_weights':evaluation['static_weights'],'static_fixed_on':evaluation['static_fixed_on'],'paper_run':cvar,'actual_weights':current.tolist(),
          'last_scheduled_target':cvar['targets'][-1],'macro':{c:float(macro[c].dropna().iloc[-1]) for c in ['ca_2y','ca_10y','policy_rate']}}
    output.mkdir(parents=True,exist_ok=True)
    archive=output.parent/'cases'/snapshot_id/'case.json'
    if archive.exists():
        saved=json.loads(archive.read_text()); case['manifest']['generated_at']=saved['manifest']['generated_at']
    write_json(archive,case,immutable=True); write_json(output/'case.json',case)
    print('Prepared',snapshot_id,review['status'],flush=True)
    return case
