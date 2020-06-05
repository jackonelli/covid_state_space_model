"""C19 inference with SLR Kalman filter"""
from functools import partial
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from post_lin_smooth.iterative import iterative_post_lin_smooth
from post_lin_smooth.slr.slr import Slr
from post_lin_smooth.slr.distributions import ProjectedTruncGauss, TruncGauss
from models import toy_fhm
from data.c19 import C19Data
import visualization as vis


def main():
    # np.random.seed(0)
    num_time_steps = 10
    num_samples = 1000
    num_iterations = 1
    population_size = 1e6
    params = toy_fhm.Params(p_s_i=0.3 / population_size, p_i_r=0.1)
    num_initially_exposed = 1e4
    num_initially_recovered = 1e3
    prior_mean = np.array([
        (population_size - num_initially_exposed - num_initially_recovered),
        num_initially_exposed,
        num_initially_recovered
    ])
    prior_cov = 0.01 * np.eye(prior_mean.shape[0])

    meas_noise = 0.1
    true_states = toy_fhm.generate_true_state(params,
                                              num_time_steps,
                                              prior_mean)
    # print(true_states)

    motion_lin = Slr(p_x=ProjectedTruncGauss(prior_mean.shape[0]),
                     p_z_given_x=toy_fhm.Motion(params,
                                                population_size),
                     num_samples=num_samples)
    meas_lin = Slr(p_x=ProjectedTruncGauss(prior_mean.shape[0]),
                   p_z_given_x=toy_fhm.Meas(population_size,
                                            meas_noise),
                   num_samples=num_samples)
    measurements = list()
    for k in np.arange(true_states.shape[0]):
        noise = norm.rvs(true_states[k, 1], meas_noise)
        meas = true_states[k, 1] + noise
        meas *= meas > 0
        measurements.append(meas)
    measurements = np.array(measurements)
    (xs_slr,
     Ps_slr,
     xf_slr,
     Pf_slr,
     linearizations) = iterative_post_lin_smooth(
         measurements,
         toy_fhm._normalize_state(prior_mean),
         prior_cov,
         motion_lin,
         meas_lin,
         num_iterations,
         normalize=True)
    plot_fhm_res(true_states, measurements, xf_slr, Pf_slr)


def plot_fhm_res(true_states, measurements, est_mean, est_cov):
    _, ax = plt.subplots()
    sigma_level = 1
    ax.plot(true_states[:, 0], label="True S")
    ax.plot(true_states[:, 1], label="True I")
    ax.plot(true_states[:, 2], label="True R")
    ax.plot(measurements, "r*", label="Meas")
    vis.plot_mean_and_cov_1d(ax,
                             est_mean[:,
                                      0],
                             est_cov[:,
                                     0,
                                     0],
                             sigma_level,
                             "S",
                             "b",
                             skip_cov=1)
    vis.plot_mean_and_cov_1d(ax,
                             est_mean[:,
                                      1],
                             est_cov[:,
                                     1,
                                     1],
                             sigma_level,
                             "I",
                             "g",
                             skip_cov=1)
    vis.plot_mean_and_cov_1d(ax,
                             est_mean[:,
                                      2],
                             est_cov[:,
                                     2,
                                     2],
                             sigma_level,
                             "R",
                             "k",
                             skip_cov=1)

    ax.legend()
    ax.set_xlabel("k")
    ax.set_ylabel("%")
    plt.show()


def plot_trajectori(ax, sir_states, mark):
    ax.plot(sir_states[:, 0], mark, label="S")
    ax.plot(sir_states[:, 1], mark, label="I")
    ax.plot(sir_states[:, 2], mark, label="R")


if __name__ == "__main__":
    main()
