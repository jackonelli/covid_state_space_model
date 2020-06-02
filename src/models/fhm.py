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
from dataclasses import dataclass
import numpy as np
from scipy.stats import binom
from post_lin_smooth.motion_models.interface import MotionModel
from post_lin_smooth.slr.distributions import Conditional


@dataclass
class Params:
    b: float
    q: float
    p_0: float
    p_ei: float
    p_ei: float
    p_er: float


class Motion(Conditional):
    def __init__(self, params: Params, population_size: int):

        self.params = params
        self.population_size = int(population_size)

    def sample(self, states):
        s, e, i_u, i_r, r = destructure_state(
            denormalize_state(states, self.population_size))
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
        sample = normalize_state(
            structure_state(s_new, e_new, i_u_new, i_r_new, r_new),
            self.population_size)
        return sample

    def _delta_e(self, s, i_u, i_r):
        unrep_interactions = s * i_u
        delta_e_u = binom.rvs(unrep_interactions, self.params.b)
        rep_interactions = s * i_r
        delta_e_r = binom.rvs(rep_interactions, self.params.q * self.params.b)
        return delta_e_u + delta_e_r

    def _delta_i_u(self, e):
        return binom.rvs(e, self.params.p_0 * self.params.p_ei)

    def _delta_i_r(self, e):
        return binom.rvs(e, (1 - self.params.p_0) * self.params.p_ei)

    def _delta_r_u(self, i_u):
        return binom.rvs(i_u, self.params.p_er)

    def _delta_r_r(self, i_r):
        return binom.rvs(i_r, self.params.p_er)


class Meas(Conditional):
    def __init__(self, population_size):
        self.population_size = population_size

    def sample(self, states):
        _, _, _, i_r, _ = destructure_state(states)
        return np.reshape(i_r, (i_r.shape[0], 1))


def normalize_state(states):
    """Normalize multiple states at once
    This works for both states represented in full population
    and states represented with portion of population
    N = number of states
    D_x = state dimension

    Args:
        states (N, D_x)
    Returns
        normalized states (N, D_x):
            (columns sum to 1)
    """
    return states / states.sum(1)


def denormalize_state(state, population_size: int):
    return (state * population_size).astype("int64")


def structure_state(s, e, i_u, i_r, r):
    return np.column_stack((s, e, i_u, i_r, r))


def destructure_state(state):
    s = state[:, 0]
    e = state[:, 1]
    i_u = state[:, 2]
    i_r = state[:, 3]
    r = state[:, 4]
    return s, e, i_u, i_r, r
