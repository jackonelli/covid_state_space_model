"""Smoother testing script"""
from functools import partial
import numpy as np
import matplotlib.pyplot as plt
from post_lin_filt.filtering import non_linear_kalman_filter
from post_lin_filt.meas_models.range_bearing import range_bearing_meas
from post_lin_filt.motion_models.coord_turn import coord_turn_motion
from post_lin_filt.utils import gen_dummy_data, gen_non_lin_meas


def main():
    K = 600
    sampling_period = 0.1
    pos = np.array([280, -140])
    sigma_r = 15
    sigma_phi = 4 * np.pi / 180

    R = np.diag([sigma_r**2, sigma_phi**2])
    meas_model = partial(range_bearing_meas, pos=pos)
    true_states, meas = gen_dummy_data(K, sampling_period, meas_model, R)
    cartes_meas = to_cartesian_coords(meas, pos)

    x_0 = np.zeros((5, ))
    P_0 = np.diag(
        [10**2, 10**2, 10**2, (5 * np.pi / 180)**2, (1 * np.pi / 180)**2])
    v_scale = 0.01
    omega_scale = 1
    sigma_v = v_scale * 1
    sigma_omega = omega_scale * np.pi / 180
    Q = np.diag([
        0, 0, sampling_period * sigma_v**2, 0, sampling_period * sigma_omega**2
    ])
    motion_model = partial(coord_turn_motion, sampling_period=sampling_period)

    xf, Pf, xp, Pp = non_linear_kalman_filter(meas, x_0, P_0, motion_model, Q,
                                              meas_model, R)
    plot_(true_states, cartes_meas, xf, Pf)


def plot_(true_states, meas, filtered_mean, filtered_cov):
    plt.plot(true_states[0, :], true_states[1, :])
    plt.plot(filtered_mean[0, :], filtered_mean[1, :])
    plt.plot(meas[0, :], meas[1, :], "r*")
    plt.show()


def to_cartesian_coords(meas, pos):
    delta_x = meas[:, 0] * np.cos(meas[:, 1])
    delta_y = meas[:, 0] * np.sin(meas[:, 1])
    coords = np.array([delta_x, delta_y]) + pos
    return coords


if __name__ == "__main__":
    main()
