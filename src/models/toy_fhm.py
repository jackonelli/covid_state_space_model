"""Stochastic version of the Folkhälsomyndigheten (FHM) model

The state is:

    x = [
        s
        i
        r
    ]
"""
from dataclasses import dataclass
import numpy as np
from scipy.stats import binom
from scipy.stats import multivariate_normal as mvn
from post_lin_smooth.slr.distributions import Prior, Conditional


@dataclass
class Params:
    p_s_i: float
    p_i_r: float


class Motion(Conditional):
    def __init__(self, params: Params, population_size: int):

        self.params = params
        self.population_size = int(population_size)

    def __str__(self):
        return "Toy FHM: {}".format(self.params)

    def sample(self, states):
        if np.any(states < 0.0):
            ValueError("Cond. on negative values in motion model")
        s, i, r = _destructure_state(
            _denormalize_state(states, self.population_size))
        delta_i = self._delta_i(s, i)
        delta_r = self._delta_r(i)
        s_new = s - delta_i
        i_new = i + delta_i - delta_r
        r_new = r + delta_r
        sample = _normalize_state(_structure_state(s_new, i_new, r_new))
        return sample

    def _delta_i(self, s, i):
        interactions = s * i
        return np.array(binom.rvs(interactions, self.params.p_s_i))

    def _delta_r(self, i):
        return binom.rvs(i, self.params.p_i_r)


class Meas(Conditional):
    def __init__(self, population_size):
        self.population_size = population_size

    def sample(self, states):
        _, i, _ = _destructure_state(states)
        return np.reshape(i, (i.shape[0], 1))


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


def _structure_state(s, i, r):
    return np.column_stack((s, i, r))


def _destructure_state(state):

    if state.ndim > 1:
        s = state[:, 0]
        i = state[:, 1]
        r = state[:, 2]
    else:
        s = state[0]
        i = state[1]
        r = state[2]
    return s, i, r
