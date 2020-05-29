"""Stochastic linear regression (SLR)"""
import numpy as np
from post_lin_filt.slr.distributions import Gaussian, Conditional


class Slr:
    """SLR"""
    def __init__(self, p_x: Gaussian, p_z_given_x: Conditional):
        self.p_x = p_x
        self.p_z_given_x = p_z_given_x

    def sample(self, num_samples: int):
        x_sample = self.p_x.sample(num_samples)
        print("x_sample", x_sample.shape)
        z_sample = self.p_z_given_x.sample(x_sample)
        return (x_sample, z_sample)

    @staticmethod
    def _z_bar(z_sample):
        return _bar(z_sample)

    def _psi(self, x_sample, z_sample, z_bar):
        sample_size = x_sample.shape[0]
        x_diff = x_sample - self.p_x.x_bar
        z_diff = z_sample - z_bar
        return (x_diff.T @ z_diff) / sample_size

    def _phi(self, z_sample, z_bar):
        sample_size = z_sample.shape[0]
        z_diff = z_sample - z_bar
        return z_diff.T @ z_diff / sample_size

    def linear_parameters(self, num_samples):
        """Estimate linear paramters"""
        x_sample, z_sample = self.sample(num_samples)
        z_bar = self._z_bar(z_sample)
        psi = self._psi(x_sample, z_sample, z_bar)
        phi = self._phi(z_sample, z_bar)

        A = psi.T @ self.p_x.P
        b = z_bar - A @ _bar(x_sample)
        Sigma = phi - A @ self.p_x.P @ A.T
        return A, b, Sigma


def _bar(sample):
    return np.mean(sample, 0)
