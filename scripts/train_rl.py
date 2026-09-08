"""Corrected owner training entry point. DQN remains an explicitly legacy experiment."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.train_research import main
if __name__=='__main__': main()
