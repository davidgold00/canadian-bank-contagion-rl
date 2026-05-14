import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import argparse
from src.rl.train_ppo import train_ppo
from src.rl.train_dqn import train_dqn
if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--agent',choices=['ppo','dqn'],default='ppo'); p.add_argument('--timesteps',type=int,default=2000); a=p.parse_args()
    print(train_ppo(a.timesteps) if a.agent=='ppo' else train_dqn(a.timesteps))
