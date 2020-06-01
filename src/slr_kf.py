import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal as mvn
from post_lin_filt.filtering import slr_kalman_filter
from post_lin_filt.smoothing import slr_rts_smoothing
from post_lin_filt.slr.slr import Slr, _bar
from models.affine import Affine
from post_lin_filt.slr.distributions import Gaussian
from slr_debug import plot_sigma_level


def main():
    prior_mean = np.array([1, 3])
    prior_cov = 1 * np.eye(2)
    T = 1
    A = np.array([[1, T], [0, 1]])
    b = 0 * np.ones((2, ))
    Q = np.array([[0, 0], [0, 1.5]])
    H = np.array([[1, 0]])
    c = np.zeros((H @ prior_mean).shape)
    R = np.array([2])
    K = 20

    num_samples = 100
    motion_model = Affine(A, b, Q)
    meas_model = Affine(H, c, R)
    true_x = gen_linear_state_seq(prior_mean, prior_cov, A, Q, K)
    y = gen_linear_meas_seq(true_x, H, R)

    test_slr_kf_filter(true_x, y, prior_mean, prior_cov, motion_model,
                       meas_model, num_samples)


def true_kf_param(A, b, Q, H, c, R, prior_mean, prior_cov, meas):
    pred_mean = A @ prior_mean + b
    pred_cov = A @ prior_cov @ A.T + Q
    print("pred_mean", pred_mean)
    print("pred_cov", pred_cov)


def test_slr_kf_filter(true_x, y, prior_mean, prior_cov, motion_model,
                       meas_model, num_samples):
    print("\nFILTERING\n")
    xf, Pf, xp, Pp = slr_kalman_filter(y, prior_mean, prior_cov, motion_model,
                                       meas_model, num_samples)

    print("\nSMOOTHING\n")
    xs, Ps = slr_rts_smoothing(xf, Pf, xp, Pp, motion_model, num_samples)
    fig, ax = plt.subplots()
    plot_filtered(ax, true_x, y, xf, Pf, xs)
    ax.legend()
    plt.show()


def plot_filtered(ax, true_x, meas, xf, Pf, xs):
    plot_states_meas(ax, true_x, meas)
    ax.plot(xf[:, 0], "r-", label="x_f")
    ax.plot(xs[:, 0], "g-", label="x_s")


def plot_states_meas(ax, true_x, meas):
    ax.plot(true_x[:, 0], "b-", label="true x")
    ax.plot(meas, "r*", label="meas")
    return ax


def plot_states_meas(ax, true_x, meas):
    ax.plot(true_x[:, 0], "b-")
    ax.plot(meas, "r*")


def gen_linear_state_seq(x_0, P_0, A, Q, K):
    """Generates an K-long sequence of states using a
       Gaussian prior and a linear Gaussian process model

       Args:
          x_0         [n x 1] Prior mean
          P_0         [n x n] Prior covariance
          A           [n x n] State transition matrix
          Q           [n x n] Process noise covariance
          K           [1 x 1] Number of states to generate

       Returns:
          X           [n x K+1] State vector sequence
    """
    X = np.zeros((K, x_0.shape[0]))

    X[0, :] = mvn.rvs(mean=x_0, cov=P_0, size=1)

    q = mvn.rvs(mean=np.zeros(x_0.shape), cov=Q, size=K)

    for k in np.arange(1, K):
        X[k, :] = A @ X[k - 1, :] + q[k - 1, :]
    return X


def gen_linear_meas_seq(X, H, R):
    """generates a sequence of observations of the state
    sequence X using a linear measurement model.
    Measurement noise is assumed to be zero mean and Gaussian.

    Args:
        X [K+1 x n] State vector sequence. The k:th state vector is X(:,k+1)
        H [m x n] Measurement matrix
        R [m x m] Measurement noise covariance

    Returns:
        Y [K, m] Measurement sequence
    """

    r = mvn.rvs(mean=np.zeros((R.shape[0], 1)), cov=R, size=X.shape[0])

    return (H @ X.T + r).T


if __name__ == "__main__":
    main()
