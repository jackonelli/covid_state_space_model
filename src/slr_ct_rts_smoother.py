"""Smoother testing script"""
from functools import partial
import numpy as np
from scipy.stats import multivariate_normal as mvn
import matplotlib.pyplot as plt
from post_lin_smooth.iterative import iterative_post_lin_smooth
from post_lin_smooth.slr.distributions import Gaussian
from models.range_bearing import to_cartesian_coords
from models.coord_turn import CoordTurn
from models.range_bearing import RangeBearing
import visualization as vis


def main():
    # np.random.seed(0)
    num_samples = 1000
    num_iterations = 1
    range_ = (0, 30)

    prior = Gaussian

    # Motion model
    sampling_period = 0.1
    v_scale = 0.01
    omega_scale = 1
    sigma_v = v_scale * 1
    sigma_omega = omega_scale * np.pi / 180
    Q = np.diag([
        0, 0, sampling_period * sigma_v**2, 0, sampling_period * sigma_omega**2
    ])
    motion_model = CoordTurn(sampling_period, Q)

    # Meas model
    pos = np.array([100, -100])
    sigma_r = 1
    sigma_phi = .5 * np.pi / 180

    R = np.diag([sigma_r**2, sigma_phi**2])
    meas_model = RangeBearing(pos, R)

    # Generate data
    # K = 600
    # true_states, measurements, obs_dims = gen_dummy_data(
    #     K, sampling_period, meas_model, R)
    true_states, measurements = gen_tricky_data(meas_model, R, range_)
    obs_dims = true_states.shape[1]
    cartes_meas = np.apply_along_axis(partial(to_cartesian_coords, pos=pos), 1,
                                      measurements)

    # Prior distr.
    x_0 = np.array([4.4, 0, 4, 0, 0])
    P_0 = np.diag(
        [10**2, 10**2, 10**2, (5 * np.pi / 180)**2, (1 * np.pi / 180)**2])

    xs, Ps, xf, Pf, _ = iterative_post_lin_smooth(measurements, x_0, P_0,
                                                  prior, motion_model,
                                                  meas_model, num_samples,
                                                  num_iterations)

    vis.plot_nees_and_2d_est(true_states[range_[0]:range_[1], :],
                             cartes_meas,
                             xf[:, :obs_dims],
                             Pf[:, :obs_dims, :obs_dims],
                             xs[:, :obs_dims],
                             Ps[:, :obs_dims, :obs_dims],
                             sigma_level=3,
                             skip_cov=5)


def gen_tricky_data(meas_model, R, range_):
    true_states = np.loadtxt("ct_data.csv")
    start, stop = range_
    return true_states, gen_non_lin_meas(true_states[start:stop, :],
                                         meas_model, R)


def gen_dummy_data(num_samples, sampling_period, meas_model, R):
    omega = np.zeros((num_samples + 1))
    omega[200:401] = -np.pi / 201 / sampling_period
    # Initial state
    initial_state = np.array([0, 0, 20, 0, omega[0]])
    # Allocate memory
    true_states = np.zeros((num_samples + 1, initial_state.shape[0]))
    true_states[0, :] = initial_state
    coord_turn = CoordTurn(sampling_period, None)
    # Create true track
    for k in range(1, num_samples + 1):
        new_state = coord_turn.mean(true_states[k - 1, :])
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

    meas_mean = np.apply_along_axis(meas_model.mean, 1, states)
    num_states, meas_dim = meas_mean.shape
    noise = mvn.rvs(mean=np.zeros((meas_dim, )), cov=R, size=num_states)
    return meas_mean + noise


if __name__ == "__main__":
    main()
