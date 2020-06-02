"""C19 inference with SLR Kalman filter"""
from functools import partial
import numpy as np
import matplotlib.pyplot as plt
from post_lin_smooth.iterative import iterative_post_lin_smooth
from post_lin_smooth.filter_type.slr import SlrFilter
from models import fhm
from data.c19 import C19Data


def main():
    num_samples = 1000
    num_iterations = 3
    population_size = 1e7
    params = fhm.Params(b=0.3 / population_size,
                        q=0.1,
                        p_0=0.3,
                        p_ei=0.1,
                        p_er=0.1)
    num_initially_exposed = 3
    reported_cases = get_measurements()
    motion_model = fhm.Motion(params, population_size)
    meas_model = fhm.Meas(population_size)
    x_0 = np.array([
        (population_size - num_initially_exposed) / population_size,
        num_initially_exposed / population_size, 0, 0, 0
    ])

    x_0 = 0.2 * np.ones((5, ))
    sigma_sq_0 = 0.0001
    P_0 = sigma_sq_0 * np.eye(5)
    print("x_0", x_0)
    print("P_0", P_0)

    xs, Ps = iterative_post_lin_smooth(measurements=reported_cases,
                                       prior_mean=x_0,
                                       prior_cov=P_0,
                                       motion_model=motion_model,
                                       meas_model=meas_model,
                                       num_samples=num_samples
                                       num_iterations=num_iterations)


def plot_(true_states, meas, filtered_mean, filtered_cov):
    plt.plot(true_states[:, 0], true_states[:, 1], "b-")
    plt.plot(filtered_mean[:, 0], filtered_mean[:, 1])
    plt.plot(meas[:, 0], meas[:, 1], "r*")
    plt.show()


def get_measurements():
    c19_data = C19Data.from_json_file("../data/data.json")
    reported_cases = c19_data.cases["y"]
    return np.reshape(np.array(reported_cases), (len(reported_cases), 1))


if __name__ == "__main__":
    main()
