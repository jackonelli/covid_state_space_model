"""C19 inference with SLR Kalman filter"""
from functools import partial
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
from post_lin_smooth.iterative import iterative_post_lin_smooth
from post_lin_smooth.slr.slr import Slr
from post_lin_smooth.slr.distributions import ProjectedTruncGauss, TruncGauss
from models import v4_fhm
from data.daily_icu import DailyIcu
import visualization as vis


def main():
    np.random.seed(0)
    num_samples = 1000
    num_iterations = 1
    population_size = int(2.5e6)
    params = v4_fhm.Params(p_s_e=0.4 / population_size,
                           p_e_i=1 / 5.1,
                           p_i_r=1 / 5)
    p_i_c = 1e-4
    num_initially_exposed = 1e3
    num_initially_infected = 1e3
    num_initially_recovered = 1e3
    prior_mean = np.array([(population_size - num_initially_exposed -
                            num_initially_infected - num_initially_recovered),
                           num_initially_exposed,
                           num_initially_infected,
                           num_initially_recovered])
    prior_cov = 0.01 * np.eye(prior_mean.shape[0])

    time_delay = 7
    data = DailyIcu.from_csv("data/New_UCI_June10.csv")
    measurements = data.daily_icu.y.reshape((len(data), 1))[time_delay: 80, :] / population_size

    motion_lin = Slr(p_x=ProjectedTruncGauss(prior_mean.shape[0]),
                     p_z_given_x=v4_fhm.Motion(params,
                                               population_size),
                     num_samples=num_samples)
    meas_lin = Slr(p_x=ProjectedTruncGauss(prior_mean.shape[0]),
                   p_z_given_x=v4_fhm.Meas(population_size,
                                           p_i_c),
                   num_samples=num_samples)
    (xs_slr,
     Ps_slr,
     xf_slr,
     Pf_slr,
     linearizations) = iterative_post_lin_smooth(
         measurements,
         v4_fhm._normalize_state(prior_mean),
         prior_cov,
         motion_lin,
         meas_lin,
         num_iterations,
         normalize=True)
    plot_fhm_res(measurements, xf_slr, Pf_slr)


def plot_fhm_res(measurements, est_mean, est_cov):
    _, ax = plt.subplots()
    sigma_level = 1
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
