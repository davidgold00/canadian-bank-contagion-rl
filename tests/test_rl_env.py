from src.data.build_dataset import build_dataset
import pandas as pd
from src.rl.env import CanadianBankContagionEnv

def test_env_reset_step():
    df=build_dataset(); p=pd.read_csv('data/processed/prices.csv',parse_dates=['date'],index_col='date'); env=CanadianBankContagionEnv(p,df); obs,_=env.reset(); obs,r,d,t,info=env.step(env.action_space.sample()); assert abs(info['weights'].sum()-1)<1e-6
