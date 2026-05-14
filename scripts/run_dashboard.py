import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import subprocess, sys
subprocess.run([sys.executable,'-m','streamlit','run','src/dashboard/app.py'])
