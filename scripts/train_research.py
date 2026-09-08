"""Explicit, versioned classifier and PPO evaluation. Rendering never calls this."""
import argparse,json,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import pandas as pd
from src.research.provenance import digest
from src.research.evaluation import evaluate_classifier
from src.research.training import train_and_evaluate

def main():
    p=argparse.ArgumentParser(); p.add_argument('--dataset',required=True); p.add_argument('--output',required=True); p.add_argument('--ppo',action='store_true'); p.add_argument('--steps',type=int,default=100000); args=p.parse_args()
    base=Path(args.dataset); out=Path(args.output); out.mkdir(parents=True,exist_ok=True)
    provenance=json.loads((base/'provenance.json').read_text())
    for name in ['prices','features','macro']:
        if digest(base/f'{name}.csv')!=provenance['hashes'][name]: raise ValueError('Input hash mismatch')
    if provenance['status']!='downloaded_verified': raise ValueError('Verified data required for current research training')
    prices=pd.read_csv(base/'prices.csv',index_col=0,parse_dates=True); features=pd.read_csv(base/'features.csv',index_col=0,parse_dates=True)
    result=evaluate_classifier(features,digest(base/'features.csv'),out/'classifier.json')
    print('Classifier:',result['selected_model'],result['test'],flush=True)
    if args.ppo: train_and_evaluate(prices,features,result['protocol'],digest(base/'features.csv'),out/'ppo',steps=args.steps)
if __name__=='__main__': main()
