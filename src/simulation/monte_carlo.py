def monte_carlo_losses(mu, sigma, n=1000, seed=42):
    import numpy as np
    rng=np.random.default_rng(seed); return rng.normal(mu,sigma,n)
