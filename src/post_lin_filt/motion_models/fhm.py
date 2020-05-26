"""Stochastic version of the Folkhälsomyndigheten (FHM) model

The state is:

    x = [
        s
        e
        i^u
        i^r
        r
    ]
"""

from post_lin_filt.motion_models.interface import MotionModel
from post_lin_filt.slr.distributions import Gaussian, Conditional
from post_lin_filt.slr.slr import SLR
import numpy as np
from scipy.stats import binom


class Fhm(MotionModel):
    def __init__(self, num_samples, b):
        self.num_samples = num_samples
        self.b = b

    def predict(self, prior_mean, prior_cov, process_noise_cov):
        slr = SLR(Gaussian(prior_mean, prior_cov), _FhmDistr(self.b))
        A, b, Sigma = slr.linear_parameters(self.num_samples)


class _FhmDistr(Conditional):
    def __init__(self, b):
        self.b = b

    def sample(self, state):
        s, e, i_u, i_r, r = destructure_state(state)
        delta_e = self._delta_e(s, i_u, i_r)
        delta_i_u = self._delta_i_u(e)
        delta_i_r = self._delta_i_u(e)
        delta_r_u = self._delta_r_u(i_u)
        delta_r_r = self._delta_r_u(i_r)
        s_new = s - delta_e
        e_new = e + delta_e - delta_i_u - delta_i_r
        i_u_new = i_u + delta_i_u - delta_r_u
        i_r_new = i_r + delta_i_r - delta_r_r
        r_new = r + delta_r_u + delta_r_r
        return structure_state(s_new, e_new, i_u_new, i_r_new, r_new)

    def _delta_e(self, s, i_u, i_r):
        unrep_interactions = int(s) * int(i_u)
        delta_e_u = binom.rvs(unrep_interactions, self.b)
        rep_interactions = int(s) * int(i_r)
        delta_e_r = binom.rvs(rep_interactions, self.q * self.b)
        return delta_e_u + delta_e_r

    def _delta_i_u(self, e):
        return binom.rvs(e, self.p_0 * self.p_ei)

    def _delta_i_r(self, e):
        return binom.rvs(e, (1 - self.p_0) * self.p_ei)

    def _delta_r_u(self, i_u):
        return binom.rvs(i_u, self.p_er)

    def _delta_r_r(self, i_r):
        return binom.rvs(i_r, self.p_er)


def structure_state(s, e, i_u, i_r, r):
    return np.array([s, e, i_u, i_r, r])


def destructure_state(state):
    s = state[0]
    e = state[1]
    i_u = state[2]
    i_r = state[3]
    r = state[4]
    return s, e, i_u, i_r, r
