"""Stochastic linear regression (SLR)"""
import numpy as np
from post_lin_filt.slr.distribution import Gaussian, Conditional


class SLR:
    """SLR"""
    def __init__(self, p_x: Gaussian, p_z_given_x: Conditional):
        self.p_x = p_x
        self.p_z_given_x = p_z_given_x

    def sample(self, num_samples):
        x_sample = self.p_x.sample(num_samples)
        z_sample = self.p_z_given_x.sample(x_sample, num_samples)
        return (x_sample, z_sample)

    @staticmethod
    def _z_bar(z_sample):
        return _bar(z_sample)

    def _psi(self, x_sample, z_sample):
        z_bar = self._z_bar(z_sample)
        return np.mean((x_sample - self.p_x.x_bar) @ (z_sample - z_bar).T)

    def _phi(self, z_sample):
        z_bar = self._z_bar(z_sample)
        return np.mean((z_sample - z_bar) @ (z_sample - z_bar).T)

    def linear_parameters(self, num_samples):
        """Estimate linear paramters"""
        x_sample, z_sample = self.sample(num_samples)
        z_bar = self._z_bar(z_sample)
        psi = self._psi(x_sample, z_sample)
        phi = self._phi(z_sample)

        A = psi.T @ self.p_x.P
        b = z_bar - A @ _bar(x_sample)
        Sigma = phi - A @ self.p_x.P @ A.T
        return A, b, Sigma


def _bar(sample):
    return np.mean(sample)
