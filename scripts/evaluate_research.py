"""Explicitly freeze the corrected strategy comparison; never called by refresh."""
import argparse,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.research.case import freeze_portfolio_evaluation
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);p.add_argument('--evaluation',required=True);a=p.parse_args()
    r=freeze_portfolio_evaluation(a.dataset,a.evaluation);print(r['id'])
