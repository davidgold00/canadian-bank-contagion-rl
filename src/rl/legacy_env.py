import numpy as np, pandas as pd
try:
    import gymnasium as gym
    from gymnasium import spaces
except Exception:  # lightweight fallback for environments without gymnasium installed
    class _Env:
        metadata={}
        def reset(self, seed=None):
            self.np_random=np.random.default_rng(seed)
    class _Box:
        def __init__(self, low, high, shape, dtype): self.low=low; self.high=high; self.shape=shape; self.dtype=dtype
        def sample(self): return np.random.uniform(-1,1,self.shape).astype(self.dtype)
    class _Discrete:
        def __init__(self, n): self.n=n
        def sample(self): return int(np.random.randint(self.n))
    class gym: Env=_Env
    class spaces:
        Box=_Box; Discrete=_Discrete
from .reward import portfolio_reward

class CanadianBankContagionEnv(gym.Env):
    metadata={'render_modes':['human']}
    def __init__(self, prices, features, assets=None, lookback=21, transaction_cost_bps=5, discrete=False):
        super().__init__(); self.prices=prices.sort_index(); self.returns=self.prices.pct_change().fillna(0); self.features=features.reindex(self.prices.index).ffill().fillna(0)
        self.assets=assets or [c for c in ['RY.TO','TD.TO','BMO.TO','BNS.TO','CM.TO','NA.TO','XFN.TO','XIU.TO','cash'] if c in list(prices.columns)+['cash']]
        if 'cash' not in self.returns: self.returns['cash']=0.0
        self.lookback=lookback; self.tc=transaction_cost_bps/10000; self.discrete=discrete
        n=len(self.assets); f=min(40, self.features.shape[1]); self._fcols=list(self.features.columns[:f])
        self.observation_space=spaces.Box(-np.inf,np.inf,shape=(lookback*n+f+n,),dtype=np.float32)
        self.action_space=spaces.Discrete(12) if discrete else spaces.Box(-5,5,shape=(n,),dtype=np.float32)
    def _obs(self):
        r=self.returns[self.assets].iloc[self.t-self.lookback:self.t].values.flatten(); f=self.features[self._fcols].iloc[self.t].values; return np.concatenate([r,f,self.weights]).astype(np.float32)
    def reset(self, seed=None, options=None):
        try:
            super().reset(seed=seed)
        except TypeError:
            super().reset(seed)
        self.t=self.lookback; self.weights=np.array([1/len(self.assets)]*len(self.assets)); self.value=1.0; self.peak=1.0; return self._obs(), {}
    def _action_to_weights(self, action):
        if self.discrete:
            n=len(self.assets); w=np.zeros(n); banks=[i for i,a in enumerate(self.assets) if a.endswith('.TO') and a not in ['XFN.TO','XIU.TO']]
            cash=self.assets.index('cash') if 'cash' in self.assets else n-1
            if action==0: w[cash]=1
            elif action==1: w[banks]=1/len(banks)
            elif action==5 and 'XFN.TO' in self.assets: w[self.assets.index('XFN.TO')]=1
            elif action==6: w[cash]=.5; w[banks]=.5/len(banks)
            else: w[:]=1/n
            return w
        z=np.exp(action-np.max(action)); return z/z.sum()
    def step(self, action):
        new_w=self._action_to_weights(action); turnover=float(np.abs(new_w-self.weights).sum()); asset_ret=self.returns[self.assets].iloc[self.t].values
        pret=float(new_w@asset_ret - self.tc*turnover); self.value*=1+pret; self.peak=max(self.peak,self.value); dd=self.value/self.peak-1
        contagion=float(self.features.get('contagion_risk_score',pd.Series(50,index=self.features.index)).iloc[self.t]); rew=portfolio_reward(pret, vol=float(np.std(asset_ret)*np.sqrt(252)), drawdown=dd, turnover=turnover, contagion=contagion, stress=contagion, excess=pret-float(self.returns.get('XFN.TO',pd.Series(0,index=self.returns.index)).iloc[self.t]))
        self.weights=new_w; self.t+=1; done=self.t>=len(self.prices)-1; return self._obs(), rew, done, False, {'value':self.value,'weights':new_w,'contagion':contagion}
