"""Shared next-close ledger for corrected training and evaluation. Legacy runs are untouched."""
from dataclasses import dataclass, asdict
import numpy as np
import pandas as pd
from .provenance import ASSETS, TRADABLES, object_id
METHOD = 'next-close-v2'

def constrained_ppo(action, assets=ASSETS):
    z=np.asarray(action,float)
    if not np.isfinite(z).all() or len(z)!=len(assets): raise ValueError('Invalid policy action')
    w=np.exp(z-z.max()); w/=w.sum(); cash=assets.index('cash')
    for i,a in enumerate(assets):
        if a in TRADABLES[:6] and w[i]>.22: w[cash]+=w[i]-.22; w[i]=.22
    fi=[i for i,a in enumerate(assets) if a in TRADABLES[:6]+['XFN.TO']]; total=w[fi].sum()
    if total>.80: w[fi]*=.80/total; w[cash]+=total-.80
    return w

@dataclass(frozen=True)
class ObservationSpec:
    assets: tuple
    features: tuple
    lookback: int=21
    timing: str='returns/features through close t; execution at close t+1'
    missing: str='feature NaN/inf to zero; no missing tradable prices'
    @classmethod
    def from_features(cls,features,assets=ASSETS): return cls(tuple(assets),tuple(features.columns[:40]))
    @property
    def id(self): return object_id(asdict(self))
    def validate(self,prices,features):
        if [a for a in self.assets if a!='cash']!=list(prices.columns): raise ValueError('Asset schema/order mismatch')
        if list(features.columns[:len(self.features)])!=list(self.features): raise ValueError('Feature schema/order mismatch')

def observation(returns,features,weights,t,lookback=21):
    if t<lookback-1: raise ValueError('Insufficient lookback')
    return np.concatenate((returns[t-lookback+1:t+1].ravel(),features[t],weights)).astype(np.float32)

class CloseLedger:
    """An action at close t fills after old holdings are marked to t+1.
    The new allocation cannot earn the just-observed return. Next state is post-fill.
    """
    def __init__(self,assets=ASSETS,capital=100000.,cost_bps=5.,threshold=.01):
        self.assets=list(assets); self.cash=self.assets.index('cash'); self.values=np.zeros(len(assets)); self.values[self.cash]=capital
        self.initial=float(capital); self.cost_rate=cost_bps/10000; self.threshold=threshold; self.peak=capital
    @property
    def weights(self): return self.values/self.values.sum()
    def advance(self,returns,target=None):
        returns=np.asarray(returns,float)
        if not np.isfinite(returns).all() or np.any(returns < -1): raise ValueError('Invalid execution returns')
        before=self.values.sum(); self.values*=1+returns; pre_nav=self.values.sum(); pre_weights=self.weights.copy(); trades=np.zeros(len(self.assets))
        if target is not None:
            target=np.asarray(target,float)
            if not np.isfinite(target).all() or target.min() < -1e-8 or abs(target.sum()-1)>1e-6: raise ValueError('Invalid target')
            desired=target*pre_nav-self.values; desired[self.cash]=0; desired[np.abs(desired)<self.threshold*pre_nav]=0
            sales=np.minimum(desired,0); self.values+=sales; self.values[self.cash]+=-sales.sum()*(1-self.cost_rate)
            buys=np.maximum(desired,0); spend=buys.sum()*(1+self.cost_rate); available=max(0,self.values[self.cash])
            if spend>available and spend>0: buys*=available/spend
            self.values+=buys; self.values[self.cash]-=buys.sum()*(1+self.cost_rate); trades=sales+buys
        cost=np.abs(trades).sum()*self.cost_rate; nav=self.values.sum(); self.peak=max(self.peak,nav)
        if self.values[self.cash]<-1e-7: raise RuntimeError('Negative cash')
        return {'portfolio_value':float(nav),'daily_return':float(nav/before-1),'daily_pnl':float(nav-before),
                'transaction_costs':float(cost),'turnover':float(np.abs(trades).sum()/pre_nav),'weights':self.weights.copy(),
                'pre_weights':pre_weights,'trades':trades,'drawdown':float(nav/self.peak-1)}

def report_metrics(values,returns,costs=None,turnover=None,capital=100000.):
    v,r=np.asarray(values,float),np.asarray(returns,float); sd=r.std(); negative=r[r<0]; dsd=negative.std() if len(negative) else 0
    dd=v/np.maximum.accumulate(np.r_[capital,v])[1:]-1; tail=r[r<=np.quantile(r,.05)]
    return {'ending_value':float(v[-1]),'cumulative_return':float(v[-1]/capital-1),'cagr':float((v[-1]/capital)**(252/max(1,len(r)-1))-1),
            'volatility':float(sd*np.sqrt(252)),'sharpe':float(r.mean()/sd*np.sqrt(252)) if sd>1e-12 else None,
            'legacy_sortino_variant':float(r.mean()/dsd*np.sqrt(252)) if dsd>1e-12 else None,
            'max_drawdown':float(dd.min()),'daily_cvar_95':float(-tail.mean()),'tail_observations':len(tail),
            'costs':float(np.sum(costs)) if costs is not None else 0,'average_turnover':float(np.mean(turnover)) if turnover is not None else 0,
            'days_below_original_capital':int((v<capital).sum()),'observations':len(v)}

def simulate(prices,features,start,end,policy,assets=ASSETS,frequency=1,capital=100000.):
    p=prices.loc[:end,[a for a in assets if a!='cash']]
    if p.isna().any().any(): raise ValueError('Missing tradable execution prices')
    r=p.pct_change(fill_method=None).fillna(0); r['cash']=0; r=r[list(assets)]
    dates=p.index[(p.index>=pd.Timestamp(start))&(p.index<=pd.Timestamp(end))]
    if len(dates)<2: raise ValueError('Evaluation requires at least two sessions')
    ledger=CloseLedger(assets,capital=capital); rows=[]; weights=[ledger.weights.tolist()]; trades=[]; targets=[]
    rows.append({'date':str(dates[0].date()),'portfolio_value':capital,'daily_return':0.,'daily_pnl':0.,'transaction_costs':0.,'turnover':0.,'drawdown':0.})
    for i in range(len(dates)-1):
        decision,execution=dates[i],dates[i+1]; target=None
        if i%frequency==0:
            target=np.asarray(policy(p.loc[:decision],features.loc[:decision],ledger.weights.copy()),float)
            targets.append({'decision_date':str(decision.date()),'execution_date':str(execution.date()),'weights':target.tolist()})
        state=ledger.advance(r.loc[execution].to_numpy(),target); weights.append(state.pop('weights').tolist()); pre=state.pop('pre_weights'); tr=state.pop('trades')
        for j in np.flatnonzero(np.abs(tr)>1e-8): trades.append({'date':str(execution.date()),'decision_date':str(decision.date()),'asset':assets[j],'notional':float(tr[j]),'cost':float(abs(tr[j])*ledger.cost_rate),'pre_weight':float(pre[j]),'target_weight':float(target[j])})
        rows.append({'date':str(execution.date()),**state})
    metrics=report_metrics([x['portfolio_value'] for x in rows],[x['daily_return'] for x in rows],[x['transaction_costs'] for x in rows],[x['turnover'] for x in rows],capital)
    w=np.asarray(weights); fi=[assets.index(a) for a in TRADABLES[:6]+['XFN.TO'] if a in assets]; variability={}
    for name,a in [('financial_proxy',w[:,fi].sum(axis=1)),('cash',w[:,assets.index('cash')])]: variability[name]={'mean':float(a.mean()),'min':float(a.min()),'max':float(a.max()),'std_pp':float(a.std()*100)}
    variability['individual_std_pp']=dict(zip(assets,(w.std(axis=0)*100).tolist()))
    return {'method':METHOD,'assets':list(assets),'start':str(dates[0].date()),'end':str(dates[-1].date()),'capital':capital,'frequency':frequency,'metrics':metrics,'ledger':rows,'weights':weights,'trades':trades,'targets':targets,'variability':variability}
