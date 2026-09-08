"""Next-close environment. Original environment preserved in legacy_env.py."""
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
from src.research.execution import CloseLedger,ObservationSpec,observation,constrained_ppo
from src.research.provenance import ASSETS
from .reward import portfolio_reward

class CanadianBankContagionEnv(gym.Env):
    metadata={'render_modes':['human']}
    def __init__(self,prices,features,assets=None,lookback=21,transaction_cost_bps=5,discrete=False,spec=None):
        if discrete: raise ValueError('DQN is a legacy experiment; use src.rl.legacy_env explicitly')
        super().__init__(); self.assets=list(assets or [a for a in ASSETS if a=='cash' or a in prices]); self.prices=prices[[a for a in self.assets if a!='cash']].sort_index()
        if self.prices.isna().any().any(): raise ValueError('Missing tradable prices')
        self.features=features.reindex(self.prices.index); self.spec=spec or ObservationSpec.from_features(self.features,self.assets); self.spec.validate(self.prices,self.features)
        if lookback!=self.spec.lookback:raise ValueError('Lookback schema mismatch')
        self.lookback=lookback; self.tc_bps=transaction_cost_bps; self.discrete=discrete
        r=self.prices.pct_change(fill_method=None).fillna(0); r['cash']=0; self._returns=r[self.assets].to_numpy(float)
        self._features=np.nan_to_num(self.features[list(self.spec.features)].to_numpy(float),nan=0,posinf=0,neginf=0)
        self._risk=self.features.get('contagion_risk_score',pd.Series(50.,index=self.features.index)).fillna(50).to_numpy()
        n=len(self.assets); self.observation_space=spaces.Box(-np.inf,np.inf,shape=(lookback*n+len(self.spec.features)+n,),dtype=np.float32)
        self.action_space=spaces.Discrete(12) if discrete else spaces.Box(-5,5,shape=(n,),dtype=np.float32)
    def _obs(self): return observation(self._returns,self._features,self.ledger.weights,self.t,self.lookback)
    def reset(self,seed=None,options=None):
        super().reset(seed=seed); self.t=self.lookback-1; self.ledger=CloseLedger(self.assets,capital=1.,cost_bps=self.tc_bps); return self._obs(),{}
    def step(self,action):
        target=constrained_ppo(action,self.assets); self.t+=1; s=self.ledger.advance(self._returns[self.t],target); risk=float(self._risk[self.t])
        xfn=self._returns[self.t,self.assets.index('XFN.TO')] if 'XFN.TO' in self.assets else 0
        reward=portfolio_reward(s['daily_return'],vol=float(np.std(self._returns[self.t])*np.sqrt(252)),drawdown=s['drawdown'],turnover=s['turnover'],contagion=risk,stress=risk,excess=s['daily_return']-float(xfn))
        return self._obs(),reward,self.t>=len(self.prices)-1,False,{'value':s['portfolio_value'],'weights':s['weights'],'contagion':risk}
