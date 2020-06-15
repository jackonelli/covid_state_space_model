"""Stochastic version of the Folkhälsomyndigheten (FHM) model
'Fourth' version, see section 5.6 in document.

The state is:

    x = [
        s
        e
        i
        r
    ]
"""
from dataclasses import dataclass
import numpy as np
from scipy.stats import binom
from scipy.stats import multivariate_normal as mvn
from scipy.stats import norm
from post_lin_smooth.slr.distributions import Prior, Conditional


@dataclass
class Params:
    """Model parameters
    p_s_e: referred to as b_k/N in the FHM paper.
    p_e_i:
    p_i_r:
    """
    p_s_e: float
    p_e_i: float
    p_i_r: float


class Motion(Conditional):
    def __init__(self, params: Params, population_size: int):

        self.params = params
        self.population_size = int(population_size)

    def __str__(self):
        return "Fourth version FHM: {}".format(self.params)

    def sample(self, states):
        states = _normalize_state(states)
        if np.any(states < 0.0):
            ValueError("Cond. on negative values in motion model")
        s, e, i, r = _destructure_state(
            _denormalize_state(states, self.population_size))
        delta_e = self._delta_e(s, i)
        delta_i = self._delta_i(e)
        delta_r = self._delta_r(i)
        s_new = s - delta_e
        e_new = e + delta_e - delta_i
        i_new = i + delta_i - delta_r
        r_new = r + delta_r
        sample = _normalize_state(_structure_state(s_new, e_new, i_new, r_new))
        return sample

    def _delta_e(self, s, i):
        interactions = s * i
        return np.array(binom.rvs(interactions, self.params.p_s_e))

    def _delta_i(self, e):
        return np.array(binom.rvs(e, self.params.p_e_i))

    def _delta_r(self, i):
        return binom.rvs(i, self.params.p_i_r)


class Meas(Conditional):
    """Measurement model
    p_i_c: The probability of transition I --> C but with delay.
    """
    def __init__(self, population_size, p_i_c):
        self.population_size = population_size
        self.p_i_c = p_i_c

    def sample(self, states):
        _, _, i, _ = _destructure_state(_denormalize_state(states, self.population_size))
        new_icu = binom.rvs(i, self.p_i_c, size=states.shape[0])
        return new_icu.reshape((states.shape[0], 1))


def generate_true_state(params: Params, num_time_steps: int, pop_start_state):
    motion_model = Motion(params, pop_start_state.sum())
    state_k = _normalize_state(pop_start_state)
    gen_states = np.empty((num_time_steps, state_k.shape[0]))
    gen_states[0, :] = state_k
    for time_k in np.arange(1, num_time_steps):
        state_kplus1 = motion_model.sample(state_k)
        gen_states[time_k, :] = state_kplus1
        # Transition to next time step
        state_k = state_kplus1
    return gen_states


def _normalize_state(states):
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
    if states.ndim > 1:
        norm_states = states / states.sum(1, keepdims=True)
    else:
        norm_states = states / states.sum()
    return norm_states


def _denormalize_state(state, population_size: int):
    return (state * population_size).astype("int64")


def _structure_state(s, e, i, r):
    return np.column_stack((s, e, i, r))


def _destructure_state(state):
    if state.ndim > 1:
        s = state[:, 0]
        e = state[:, 1]
        i = state[:, 2]
        r = state[:, 3]
    else:
        s = state[0]
        e = state[1]
        i = state[2]
        r = state[3]
    return s, e, i, r
