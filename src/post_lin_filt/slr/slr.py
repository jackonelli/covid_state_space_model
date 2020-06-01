"""Stochastic linear regression (SLR)"""
import numpy as np
from post_lin_filt.slr.distributions import Gaussian, Conditional
from models.range_bearing import to_cartesian_coords


class Slr:
    """SLR"""
    def __init__(self, p_x: Gaussian, p_z_given_x: Conditional):
        self.p_x = p_x
        self.p_z_given_x = p_z_given_x

    def linear_parameters(self, num_samples):
        """Estimate linear paramters"""
        x_sample, z_sample = self._sample(num_samples)
        z_bar = self._z_bar(z_sample)
        print("z_bar", z_bar)
        psi = self._psi(x_sample, z_sample, z_bar)
        print("psi", psi.shape)
        phi = self._phi(z_sample, z_bar)
        print("phi", phi.shape)
        print("phi", phi)

        A = psi.T @ self.p_x.P
        b = z_bar - A @ _bar(x_sample)
        Sigma = phi - A @ self.p_x.P @ A.T
        return A, b, Sigma

    def _sample(self, num_samples: int):
        x_sample = self.p_x.sample(num_samples)
        z_sample = self.p_z_given_x.sample(x_sample)
        return (x_sample, z_sample)

    @staticmethod
    def _z_bar(z_sample):
        return _bar(z_sample)

    def _psi(self, x_sample, z_sample, z_bar):
        sample_size, x_dim = x_sample.shape
        z_dim = z_sample.shape[1]
        x_diff = x_sample - self.p_x.x_bar
        z_diff = z_sample - z_bar
        cov = np.zeros((x_dim, z_dim))
        for s in np.arange(sample_size):
            temp = np.expand_dims(x_diff[s, :], 0).T @ np.expand_dims(
                z_diff[s, :], 0)
            cov += temp
        return cov / sample_size

    def _phi(self, z_sample, z_bar):
        sample_size = z_sample.shape[0]
        z_diff = z_sample - z_bar
        # return z_diff.T @ z_diff / sample_size
        return np.cov(z_sample.T)


import matplotlib.pyplot as plt
import time


def plot_state(states):
    plt.plot(states[:, 0], states[:, 1], "r*")
    plt.show()
    time.sleep(1)


def plot_meas(meas):
    cartes_meas = np.apply_along_axis(
        lambda x: to_cartesian_coords(x, pos=np.array([280, -140])), 1, meas)
    plot_state(cartes_meas)


def plot_corr(x_sample, z_sample):
    plt.plot(x_sample[:, 1], z_sample[:, 1], "r*")
    plt.show()
    time.sleep(1)


def _bar(sample):
    return np.mean(sample, 0)
