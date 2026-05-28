import numpy as np


def sample_demands(mu, sigma, rng):
    eps = rng.standard_normal(len(mu))
    d = mu + sigma * eps
    return np.maximum(0, np.round(d)).astype(int)
