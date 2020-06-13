"""Auxiliary functions used for this project"""

import numpy as np
from scipy.stats import binom


def igamrnd(a, b, size=None):
    """Inverse gamma random number generator"""
    return 1 / np.random.gamma(shape=a, scale=1 / b, size=size)


def binom_by_normal(n, p, N=1):
    """Sample from binomial distribution, using normal approximation for large population"""
    n = np.atleast_1d(n)  # This is a bit annoying, but I don't see any simpler way to handle the possibility of scalar inputs
    small_n = n < (1 << 31)

    if len(n) == 1:  # Single state in
        if small_n:
            return np.int64(binom.rvs(n, p, size=N))
        else:
            return np.int64(np.random.normal(n * p, np.sqrt(n * p * (1 - p)), size=N))
    else:  # Multiple states in
        samples = np.zeros(N,dtype=np.int64)
        large_n = np.invert(small_n)
        num_small_n = sum(small_n)
        samples[small_n] = np.int64(binom.rvs(n[small_n].astype(np.int32), p, size=num_small_n))
        samples[large_n] = np.int64(
            np.random.normal(n[large_n] * p, np.sqrt(n[large_n] * p * (1 - p)), size=N - num_small_n))
        return samples
