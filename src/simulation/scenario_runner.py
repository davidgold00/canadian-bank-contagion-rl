from .stress_scenarios import SCENARIOS
from .contagion_simulator import propagate_shock

def run_scenario(name, A, source_index=0, shock_size=60, steps=5):
    import numpy as np
    s0=np.zeros(A.shape[0]); s0[source_index]=shock_size
    return {'scenario':SCENARIOS.get(name,{}),'stress_path':propagate_shock(A,s0,steps=steps)}
