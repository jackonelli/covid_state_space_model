"""Smoother testing script"""
from functools import partial
import numpy as np
from scipy.stats import multivariate_normal as mvn
import matplotlib.pyplot as plt
from post_lin_filt.filtering import non_linear_kalman_filter
from post_lin_filt.meas_models.range_bearing import range_bearing_meas, to_cartesian_coords
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
    true_states, measurements = gen_dummy_data(K, sampling_period, meas_model,
                                               R)
    cartes_meas = np.array(
        [to_cartesian_coords(meas, pos) for meas in measurements])
    print(cartes_meas.shape)
    print(true_states.shape)

    x_0 = np.zeros((5, ))
    P_0 = np.diag(
        [10**2, 10**2, 10**2, (5 * np.pi / 180)**2, (1 * np.pi / 180)**2])
    #v_scale = 0.01
    #omega_scale = 1
    #sigma_v = v_scale * 1
    #sigma_omega = omega_scale * np.pi / 180
    #Q = np.diag([
    #    0, 0, sampling_period * sigma_v**2, 0, sampling_period * sigma_omega**2
    #])
    #motion_model = partial(coord_turn_motion, sampling_period=sampling_period)

    #xf, Pf, xp, Pp = non_linear_kalman_filter(measurements, x_0, P_0,
    #                                          motion_model, Q, meas_model, R)
    plot_(true_states, cartes_meas, None, None)


def plot_(true_states, meas, filtered_mean, filtered_cov):
    plt.plot(true_states[:, 0], true_states[:, 1], "b-")
    # plt.plot(filtered_mean[0, :], filtered_mean[1, :])
    plt.plot(meas[:, 0], meas[:, 1], "r*")
    plt.show()


def gen_dummy_data(num_samples, sampling_period, meas_model, R):
    omega = np.zeros((num_samples + 1))
    omega[200:401] = -np.pi / 201 / sampling_period
    # Initial state
    initial_state = np.array([0, 0, 20, 0, omega[0]])
    # Allocate memory
    true_states = np.zeros((num_samples + 1, initial_state.shape[0]))
    true_states[0, :] = initial_state
    # Create true track
    for k in range(1, num_samples + 1):
        new_state, _ = coord_turn_motion(true_states[k - 1, :],
                                         sampling_period)
        true_states[k, :] = new_state
        true_states[k, 4] = omega[k]

    return true_states, gen_non_lin_meas(true_states, meas_model, R)


def gen_non_lin_meas(states, meas_model, R):
    """Generate non-linear measurements

    Args:
        states np.array((K, D_x))
        meas_model
        R np.array((D_y, D_y))
    """

    # Transpose juggling to get np to apply a function row-wise
    meas_mean = np.array([meas_model(state)[0] for state in states])
    num_states, meas_dim = meas_mean.shape
    noise = mvn.rvs(mean=np.zeros((meas_dim, )), cov=R, size=num_states)
    return meas_mean + noise


if __name__ == "__main__":
    main()
