"""Here we intend to use an SEIR model.

The state is:

    x = [
        s
        e
        i
        r
    ]
"""
"""Runs a toy example to illustrate the output"""
import numpy as np
from scipy.stats import binom
import matplotlib.pyplot as plt


def main():
    """We separate the parameters for the dynamic model into three parts:
        1) pei and pir,
        2) parameters needed to define b and
        3) size of population
    """
    pei = 1 / 5.1
    pir = 1 / 5
    dp = np.array([pei, pir])
    """For the FHM model, the parameters needed to define the function b are theta, delta,
    epsilon and the offset between day 0 and March 16.
    For instance, if we want to UCI measurements to start on March 13,
    and we assume a 7 day delay in the measurement model day 0
    would be March 6 and the offset would be -10."""
    b_par = np.array([2, 0.1, -0.12, -10])
    """For Stockholm the population is probably roughly 2.5 million."""
    N = int(2.5e6)
    """All the above parameters are stored in params."""
    params = Param(dp, b_par, N)
    """We start our simulation with an initial state, selected fairly arbitrarily."""
    i0 = 400
    e0 = 400
    r0 = 1000
    s0 = params.pop - i0 - e0 - r0
    state = np.array([s0, e0, i0, r0])
    """We initiate our state at time 0 and observe measurements from time 1 to T. What's the true 
    value of T? We use 60 here but we have more measurements."""
    T = 60
    num_samples = 10
    traj_mean, traj_std = mean_and_std_traj(state, params, T, num_samples)
    # traj_mean = sample_full_traj(state, params, T)
    """Let us generate a random sequence of states and store it in X"""
    """Finally, we visualize the results."""
    print(traj_mean.shape)
    visualize_mean_std(traj_mean, traj_std)


def mean_and_std_traj(initial_state,
                      params: "Param",
                      T: int,
                      num_samples: int):
    all_trajs = np.empty((num_samples, initial_state.shape[0], T + 1))
    for i in np.arange(num_samples):
        all_trajs[i, :, :] = sample_full_traj(initial_state, params, T).T
    return all_trajs.mean(0), all_trajs.std(0)


def sample_full_traj(initial_state, params: "Param", T: int):
    time = np.arange(T) + 1
    traj = np.zeros((T + 1, initial_state.size))
    traj[0, :] = initial_state
    state = initial_state
    for k in time:
        state = dyn_sampling(state, params, k)
        traj[k, :] = state
    return traj


def dyn_sampling(state, params, time):
    N = params.pop
    b = b_val_FHM(params, time)
    de = binom.rvs(state[0] * state[2], b / N)
    di = binom.rvs(state[1], params.ei)
    dr = binom.rvs(state[2], params.ir)
    state = state + [-de, de - di, di - dr, dr]
    return state


def visualize_mean_std(traj_mean, traj_std):
    """Visualize"""
    s_l, e_l, i_l, r_l = traj_mean - traj_std
    s_u, e_u, i_u, r_u = traj_mean + traj_std
    times = np.arange(traj_mean.shape[1])
    plt.figure()
    plt.title("Simulating a stochastic SEIR model")
    plt.xlabel("Days, Day 0 = March 10")
    plt.ylabel("Number of individuals")
    label = r"$\overline{{{}}} \pm \sigma_{}$"
    plt.fill_between(times, s_l, s_u, label=label.format("s", "s"))
    plt.fill_between(times, e_l, e_u, label=label.format("e", "e"))
    plt.fill_between(times, i_l, i_u, label=label.format("i", "i"))
    plt.fill_between(times, r_l, r_u, label=label.format("r", "r"))
    plt.legend()
    plt.show()


def visualize(states):
    """Visualize"""
    plt.figure()
    plt.title("Simulating a stochastic SEIR model")
    plt.xlabel("Days, Day 0 = March 10")
    plt.ylabel("Number of individuals")
    s_line, = plt.plot(states[0, :])
    e_line, = plt.plot(states[1, :])
    i_line, = plt.plot(states[2, :])
    r_line, = plt.plot(states[3, :])
    plt.legend([s_line, e_line, i_line, r_line], ['s', 'e', 'i', 'r'])
    plt.show()


class Param:
    def __init__(self, d_param, b_param, pop):
        self.ei = d_param[0]
        self.ir = d_param[1]
        self.bp = b_param
        self.pop = pop


def b_val_FHM(params, time):
    b_par = params.bp
    theta = b_par[0]
    delta = b_par[1]
    eps = b_par[2]
    t_offset = b_par[3]
    b = theta * (delta + (1 - delta) / (1 + np.exp(-eps * (time + t_offset))))
    return b


if __name__ == "__main__":
    main()
