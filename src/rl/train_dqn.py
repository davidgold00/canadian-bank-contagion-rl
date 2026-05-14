from pathlib import Path
import pandas as pd
from src.data.build_dataset import build_dataset
from src.rl.env import CanadianBankContagionEnv
from src.rl.agents import make_dqn

def train_dqn(total_timesteps=2000, out='artifacts/rl/dqn_model'):
    df=build_dataset(); prices=pd.read_csv('data/processed/prices.csv',parse_dates=['date'],index_col='date')
    env=CanadianBankContagionEnv(prices, df, discrete=True); model=make_dqn(env); model.learn(total_timesteps=total_timesteps); Path(out).parent.mkdir(parents=True,exist_ok=True); model.save(out); return out
if __name__=='__main__': train_dqn()
