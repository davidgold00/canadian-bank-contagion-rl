import numpy as np, pandas as pd

def propagate_shock(A, s0, alpha=.65, beta=.25, decay=.35, steps=5, exogenous=None):
    A=np.asarray(A,dtype=float); s=np.asarray(s0,dtype=float); ex=np.zeros_like(s) if exogenous is None else np.asarray(exogenous,dtype=float)
    hist=[s.copy()]
    row=A.sum(axis=1,keepdims=True); A_norm=np.divide(A,row,out=np.zeros_like(A),where=row!=0)
    for _ in range(steps):
        s=np.clip(alpha*(A_norm@s)+beta*ex+decay*s,0,100); hist.append(s.copy())
    return np.vstack(hist)
