"""Stochastic linear regression (SLR)"""
import numpy as np
from post_lin_filt.slr.distributions import Gaussian, Conditional


class Slr:
    """SLR"""
    def __init__(self, p_x: Gaussian, p_z_given_x: Conditional):
        self.p_x = p_x
        self.p_z_given_x = p_z_given_x

    def linear_parameters(self, num_samples):
        """Estimate linear parameters"""
        x_sample, z_sample = self._sample(num_samples)
        z_bar = self._z_bar(z_sample)
        psi = self._psi(x_sample, z_sample, z_bar)
        phi = self._phi(z_sample, z_bar)

        A = psi.T @ np.linalg.inv(self.p_x.P)
        b = z_bar - A @ _bar(x_sample)
        Sigma = phi - A @ self.p_x.P @ A.T
        return A, b, Sigma

    def _sample(self, num_samples: int):
        x_sample = self.p_x.sample(num_samples)
        z_sample = self.p_z_given_x.sample(x_sample)
        return (x_sample, z_sample)

    @staticmethod
    def _z_bar(z_sample):
        """Calc z_bar

        Args:
            z_sample (N, D_z)

        Returns:
            z_bar (D_z,)
        """
        return _bar(z_sample)

    def _psi(self, x_sample, z_sample, z_bar):
        """Calc Psi = Cov[x, z]
        Vectorization:
        x_diff.T @ z_diff is a matrix mult with dim's:
        (D_x, N) * (N, D_z): The sum of the product of
        each element in x_i and y_i will be computed.

        Args:
            x_sample (N, D_x)
            z_sample (N, D_z)
            z_bar (D_z,)

        Returns:
            Psi (D_x, D_z)
        """
        sample_size, x_dim = x_sample.shape
        x_diff = x_sample - self.p_x.x_bar
        z_diff = z_sample - z_bar
        cov = (x_diff.T @ z_diff)
        return cov / sample_size

    def _phi(self, z_sample, z_bar):
        """Calc Phi = Cov[z, z]
        Vectorization:
        z_diff.T @ z_diff is a matrix mult with dim's:
        (D_z, N) * (N, D_z): The sum of the product of
        each element in x_i and y_i will be computed.

        Args:
            z_sample (N, D_z)
            z_bar (D_z,)

        Returns:
            Psi (D_z, D_z)
        """
        sample_size = z_sample.shape[0]
        z_diff = z_sample - z_bar
        return z_diff.T @ z_diff / sample_size


import time
import matplotlib.pyplot as plt
from models.range_bearing import to_cartesian_coords


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
