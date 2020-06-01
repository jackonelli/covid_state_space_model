"""Stochastic affine model for comparison with analytical KF"""
import numpy as np
from scipy.stats import multivariate_normal as mvn
from post_lin_smooth.slr.distributions import Conditional


class Affine(Conditional):
    def __init__(self, linear_map, translation, process_noise):
        self.linear_map = linear_map
        self.process_noise = process_noise
        self.translation = translation

    def sample(self, x_sample):
        new_mean = (self.linear_map @ x_sample.T).T + self.translation
        num_samples, meas_dim = new_mean.shape
        if self.process_noise.size == 1:
            proc_noise = self.process_noise[0]
        else:
            proc_noise = self.process_noise
        noise = mvn.rvs(mean=np.zeros((meas_dim, )),
                        cov=proc_noise,
                        size=x_sample.shape[0]).reshape(
                            (num_samples, meas_dim))

        return new_mean + noise
