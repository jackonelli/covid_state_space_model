"""C19 inference with SLR Kalman filter"""
from functools import partial
import numpy as np
import matplotlib.pyplot as plt
from post_lin_smooth.iterative import iterative_post_lin_smooth
from models import toy_fhm, fhm_utils
from data.c19 import C19Data


def main():
    np.random.seed(0)
    num_time_steps = 5
    num_samples = 1000
    num_iterations = 1
    population_size = 1e6
    params = toy_fhm.Params(p_s_i=0.3 / population_size, p_i_r=0.1)
    num_initially_exposed = 3
    prior_mean = np.array([(population_size - num_initially_exposed),
                           num_initially_exposed,
                           0])
    prior_cov = np.eye(prior_mean.shape[0]) / population_size
    true_states = toy_fhm.generate_true_state(params,
                                              num_time_steps,
                                              prior_mean)
    # print(true_states)

    motion_model = toy_fhm.Motion(params, population_size)
    print(motion_model)
    measurements = true_states[:, 1]
    (xs_slr,
     Ps_slr,
     xf_slr,
     Pf_slr,
     linearizations) = iterative_post_lin_smooth(
         measurements,
         toy_fhm._normalize_state(prior_mean),
         prior_cov,
         fhm_utils.ProjectedTruncGauss,
         motion_model,
         toy_fhm.Meas(population_size),
         num_samples,
         num_iterations)
    plot_fhm_res(true_states, xf_slr)


def plot_fhm_res(true_states, estimates):
    _, ax = plt.subplots()
    plot_trajectory(ax, true_states, mark="*")
    plot_trajectory(ax, estimates, mark="-")
    ax.legend()
    ax.set_xlabel("k")
    ax.set_xlabel("%")
    plt.show()


def plot_trajectory(ax, sir_states, mark):
    ax.plot(sir_states[:, 0], mark, label="S")
    ax.plot(sir_states[:, 1], mark, label="I")
    ax.plot(sir_states[:, 2], mark, label="R")


if __name__ == "__main__":
    main()
