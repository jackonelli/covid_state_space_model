"""Stochastic coordinated turn motion model"""
import numpy as np
from scipy.stats import multivariate_normal as mvn
from post_lin_filt.slr.distributions import Conditional


class RangeBearing(Conditional):
    """ pos np.array(2,)"""
    def __init__(self, pos, meas_noise):
        self.pos = pos
        self.meas_noise = meas_noise

    def sample(self, states):
        means = np.apply_along_axis(self.mean, 1, states)
        num_samples, mean_dim = means.shape
        noise = mvn.rvs(mean=np.zeros((mean_dim, )),
                        cov=self.meas_noise,
                        size=num_samples)
        return means + noise

    def mean(self, state):
        range_ = np.sqrt(np.sum((state[:2] - self.pos)**2))
        bearing = np.arctan2(state[1] - self.pos[1], state[0] - self.pos[0])
        return np.array([range_, bearing])
