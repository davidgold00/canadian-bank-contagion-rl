def assert_no_future_leakage(df):
    assert df.index.is_monotonic_increasing
    return True

def validate_weights(weights, tol=1e-6):
    import numpy as np
    return abs(float(np.sum(weights))-1.0) <= tol
