"""Helper model for the FHM models"""
import numpy as np
from scipy.stats import multivariate_normal as mvn
from post_lin_smooth.slr.distributions import Prior
from utils import calc_subspace_proj_matrix


class ProjectedTruncGauss(Prior):
    def __init__(self, mean, cov):
        # Truncate neg. components: mean <- el. wise max(mean, 0)
        trunc_mean = mean * (mean > 0)
        U_orth = calc_subspace_proj_matrix(mean.shape[0])
        proj_cov = U_orth @ cov @ U_orth
        self._distr = TruncGauss(trunc_mean, proj_cov)

    def __str__(self):
        return "TruncGauss: mean={}, cov={}".format(self._distr.mean,
                                                    self._distr.cov)

    def sample(self, num_samples):
        print("Sampling with: {}".format(self))
        return self._distr.sample(num_samples)


class TruncGauss(Prior):
    def __init__(self, mean, cov):
        self.mean = mean
        self.cov = cov

    def __str__(self):
        return "TruncGauss: mean={}, cov={}".format(self.mean, self.cov)

    def sample(self, num_samples):
        successful_samples = 0
        sample = np.empty((num_samples, self.mean.shape[0]))
        while successful_samples < num_samples:
            candidate = mvn.rvs(mean=self.mean, cov=self.cov)
            if (candidate > 0).all():
                sample[successful_samples, :] = candidate
                successful_samples += 1
        return sample / sample.sum(1, keepdims=True)
