"""Stochastic coordinated turn motion model"""
import numpy as np
from scipy.stats import multivariate_normal as mvn
from post_lin_filt.slr.distributions import Conditional


class CoordTurn(Conditional):
    def __init__(self, sampling_period, process_noise):
        self.sampling_period = sampling_period
        self.process_noise = process_noise

    def sample(self, states):
        means = np.apply_along_axis(self.mean, 1, states)
        num_samples, mean_dim = means.shape
        noise = mvn.rvs(mean=np.zeros((mean_dim, )),
                        cov=self.process_noise,
                        size=num_samples)
        return means + noise

    def mean(self, state):
        v = state[2]
        phi = state[3]
        omega = state[4]
        delta = np.array([
            self.sampling_period * v * np.cos(phi),
            self.sampling_period * v * np.sin(phi), 0,
            self.sampling_period * omega, 0
        ])
        return state + delta
