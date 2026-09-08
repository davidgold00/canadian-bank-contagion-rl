"""Dependency-light static deployment: renders saved evidence, never solves or fits."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from src.dashboard.case_site import render_case
if __name__=='__main__': render_case('artifacts/current/case.json')
