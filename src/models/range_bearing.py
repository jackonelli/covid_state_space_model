"""Stochastic coordinated turn motion model"""
import numpy as np
from scipy.stats import multivariate_normal as mvn
from post_lin_smooth.slr.distributions import Conditional


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


def to_cartesian_coords(meas, pos):
    """Maps a range and bearing measurement to cartesian coords

    Args:
        meas np.array(D_y,)
        pos np.array(2,)

    Returns:
        coords np.array(2,)
    """
    delta_x = meas[0] * np.cos(meas[1])
    delta_y = meas[0] * np.sin(meas[1])
    coords = np.array([delta_x, delta_y]) + pos
    return coords
