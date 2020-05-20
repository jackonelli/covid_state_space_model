"""Smoother testing script"""
from functools import partial
import numpy as np
from scipy.stats import multivariate_normal as mvn
import matplotlib.pyplot as plt
from post_lin_filt.filtering import non_linear_kalman_filter


def main():
    K = 600
    sampling_period = 0.1
    pos = np.array([280, -140])
    sigma_r = 15
    sigma_phi = 4 * np.pi / 180

    R = np.diag([sigma_r**2, sigma_phi**2])
    meas_model = partial(range_bearing_meas, pos=pos)
    true_states, meas = gen_dummy_data(K, sampling_period, meas_model, pos, R)
    cartes_meas = to_cartesian_coords(meas, pos)
    plot_(true_states, cartes_meas)

    x_0 = np.zeros((5, 1))
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

    xs, Ps, xf, Pf, xp, Pp = non_linear_kalman_filter(meas, x_0, P_0,
                                                      motion_model, Q,
                                                      meas_model, R)


def plot_(true_states, meas):
    plt.plot(true_states[0, :], true_states[1, :])
    plt.plot(meas[0, :], meas[1, :], "r*")
    plt.show()


def range_bearing_meas(states, pos):
    print(states.shape)
    range_ = np.sum((states[0:1, :] - pos)**2)
    bearing = np.arctan2(states[1, :] - pos[1], states[0, :] - pos[0])
    meas_mean = np.array([range_, bearing]).T
    jacobian = 0

    return meas_mean, jacobian


def coord_turn_motion(state, sampling_time):
    v = state[2]
    phi = state[3]
    omega = state[4]

    mean = state + np.array([
        sampling_time * v * np.cos(phi), sampling_time * v * np.sin(phi), 0,
        sampling_time * omega, 0
    ])

    jacobian = np.array([[
        1, 0, sampling_time * np.cos(phi), -sampling_time * v * np.sin(phi), 0
    ], [0, 1, sampling_time * np.sin(phi), sampling_time * v * np.cos(phi), 0],
                         [0, 0, 1, 0, 0], [0, 0, 0, 1, sampling_time],
                         [0, 0, 0, 0, 1]])
    return mean, jacobian


def gen_dummy_data(num_samples, sampling_period, meas_model, pos, R):
    omega = np.zeros((num_samples + 1))
    omega[200:401] = -np.pi / 201 / sampling_period
    # Initial state
    initial_state = np.array([0, 0, 20, 0, omega[0]])
    # Allocate memory
    true_states = np.zeros((initial_state.shape[0], num_samples + 1))
    true_states[:, 0] = initial_state
    # Create true track
    for k in range(1, num_samples + 1):
        new_state, _ = coord_turn_motion(true_states[:, k - 1],
                                         sampling_period)
        true_states[:, k] = new_state
        true_states[4, k] = omega[k]

    return true_states, gen_non_lin_meas(true_states, meas_model, R)


def gen_non_lin_meas(states, meas_model, R):
    meas_mean, _ = meas_model(states)
    noise = mvn.rvs(mean=np.zeros(meas_mean.shape), cov=R, size=1)
    return meas_mean + noise


def to_cartesian_coords(meas, pos):
    return np.array([meas[0] * np.cos(meas[1]), meas[0] * np.sin(meas[1])
                     ]).T + pos


if __name__ == "__main__":
    main()
