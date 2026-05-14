import numpy as np

def largest_eigenvalue(A): return float(max(np.linalg.eigvals(A).real)) if len(A) else 0.0

def systemic_importance(A):
    vals=A.sum(axis=1); total=vals.sum() or 1
    return vals/total
