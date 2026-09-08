"""Explicit owner-invoked PPO training; never an export side effect."""
import json
from dataclasses import asdict
from pathlib import Path
import importlib.metadata
import numpy as np
import pandas as pd
from .provenance import ASSETS,TRADABLES,digest,object_id,write_json,utcnow
from .execution import ObservationSpec,observation,constrained_ppo,simulate
from src.rl.env import CanadianBankContagionEnv
from src.rl.reward import portfolio_reward

def ppo_policy(model,spec,prices,features):
    p=prices[[a for a in spec.assets if a!='cash']]; spec.validate(p,features)
    r=p.pct_change(fill_method=None).fillna(0); r['cash']=0; rr=r[list(spec.assets)].to_numpy(float)
    ff=np.nan_to_num(features.reindex(p.index)[list(spec.features)].to_numpy(float),nan=0,posinf=0,neginf=0)
    def predict(history,feature_history,current):
        t=p.index.get_loc(history.index[-1]); obs=observation(rr,ff,current,t,spec.lookback)
        action,_=model.predict(obs,deterministic=True)
        return constrained_ppo(action,list(spec.assets))
    return predict

def train_and_evaluate(prices,features,protocol,input_hash,output,steps=100000,seeds=(17,42,83)):
    import torch
    from stable_baselines3 import PPO
    torch.set_num_threads(1)
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    manifest_path=output/'manifest.json'
    files=['src/research/training.py','src/research/execution.py','src/rl/env.py','src/rl/reward.py','src/features/market_features.py','src/features/macro_features.py','src/features/stress_features.py']
    source_hashes={name:digest(name) for name in files}
    identity={'source_hashes':source_hashes,'input_hash':input_hash,'protocol':protocol,'seeds':list(seeds),'requested_steps':steps,'method':'next-close-v2','reward':'unchanged v1 coefficients/economics','network':'SB3 PPO MlpPolicy defaults'}
    if manifest_path.exists():
        old=json.loads(manifest_path.read_text())
        if old['identity']!=identity: raise ValueError('Training artifacts immutable; use a new run directory')
        return old
    import shutil
    for name in files:
        saved=output/'sources'/name;saved.parent.mkdir(parents=True,exist_ok=True)
        if saved.exists() and digest(saved)!=source_hashes[name]:raise ValueError('Training source bundle changed')
        shutil.copyfile(name,saved)
    write_json(output/'training-intent.json',identity,immutable=True)
    write_json(output/'started.json',{'started_at':utcnow(),'requested_steps_per_seed':steps,'stopping':'Requested budget; SB3 completes full rollouts','device':'cpu','threads':1},immutable=True)
    start=pd.Timestamp(protocol['train_start']); val=pd.Timestamp(protocol['validation_start']); test=pd.Timestamp(protocol['test_start'])
    train_p=prices.loc[(prices.index>=start)&(prices.index<val),TRADABLES]
    spec=ObservationSpec.from_features(features,ASSETS)
    write_json(output/'observation-schema.json',asdict(spec),immutable=True)
    validation_end=prices.index[prices.index<test][-1]
    candidates=[]
    for seed in seeds:
        env=CanadianBankContagionEnv(train_p,features.loc[train_p.index],spec=spec)
        model=PPO('MlpPolicy',env,seed=seed,device='cpu',verbose=0)
        model.learn(total_timesteps=steps)
        path=output/f'ppo-seed-{seed}.zip'; model.save(path)
        policy=ppo_policy(model,spec,prices,features)
        run=simulate(prices,features,val,validation_end,policy)
        # Selection by the predeclared unchanged reward, measured on validation only.
        rp=prices[TRADABLES].pct_change(fill_method=None).fillna(0); rp['cash']=0
        rewards=[]
        for row in run['ledger'][1:]:
            dt=pd.Timestamp(row['date']); risk=float(features.loc[dt,'contagion_risk_score'])
            rewards.append(portfolio_reward(row['daily_return'],vol=float(rp.loc[dt].std(ddof=0)*np.sqrt(252)),drawdown=row['drawdown'],turnover=row['turnover'],contagion=risk,stress=risk,excess=row['daily_return']-float(rp.loc[dt,'XFN.TO'])))
        candidate={'seed':seed,'artifact':path.name,'sha256':digest(path),'completed_steps':model.num_timesteps,'validation_reward':float(np.mean(rewards)),'validation_metrics':run['metrics']}
        write_json(output/f'seed-{seed}.json',candidate,immutable=True)
        candidates.append(candidate)
        print(f'Seed {seed}: {model.num_timesteps} steps; validation reward {candidate["validation_reward"]:.6f}',flush=True)
    winner=max(candidates,key=lambda c:c['validation_reward'])
    # Final-test evaluation begins only after checkpoint selection is frozen.
    write_json(output/'selection.json',winner,immutable=True)
    for candidate in candidates:
        model=PPO.load(output/candidate['artifact'],device='cpu')
        run=simulate(prices,features,test,protocol['portfolio_evaluation_end'],ppo_policy(model,spec,prices,features))
        run['policy_source']='trained PPO'; run['artifact_sha256']=candidate['sha256']; run['seed']=candidate['seed']
        write_json(output/f'test-seed-{candidate["seed"]}.json',run,immutable=True)
        candidate['test_metrics']=run['metrics']
    result={'id':'ppo-v2-'+object_id(identity),'identity':identity,'status':'completed','trained_at':utcnow(),
            'selected_seed':winner['seed'],'selected_artifact':winner['artifact'],'schema_id':spec.id,
            'schema':asdict(spec),'observation_units':{name:'annualized standard deviation of daily simple returns' if '_vol_' in name else 'dimensionless simple return or relative price distance' for name in spec.features},'return_units':'daily simple return fractions; cash zero','holdings_units':'post-fill NAV fractions','candidates':candidates,'selection_rule':'Highest mean unchanged validation reward; no final-test selection',
            'packages':{p:importlib.metadata.version(p) for p in ['torch','stable-baselines3','gymnasium','numpy','pandas']},
            'risk_inputs':{'composite_score': 'contagion_risk_score' in spec.features,'named_features':list(spec.features)},
            'limitations':['Reward volatility is asset-return dispersion, not weighted portfolio volatility.','Market contagion/stress terms are allocation-independent.','Historical macro release timestamps are unavailable.']}
    write_json(manifest_path,result,immutable=True)
    return result
