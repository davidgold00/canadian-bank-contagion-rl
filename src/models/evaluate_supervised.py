def precision_at_k(y_true, scores, k=20):
    import numpy as np
    idx=np.argsort(scores)[-k:]
    return float(sum(y_true.iloc[idx])/len(idx))
