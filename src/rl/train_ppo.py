"""Corrected training requires a verified dataset and a new versioned output."""
from scripts.train_research import main

def train_ppo(*args, **kwargs):
    raise RuntimeError("Use python scripts/train_research.py --dataset VERIFIED_DATASET --output NEW_RUN --ppo. Legacy training is archived in legacy_train_ppo.py.")
if __name__ == '__main__': main()
