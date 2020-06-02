import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal as mvn
from post_lin_smooth.iterative import iterative_post_lin_smooth
from models.affine import Affine
from slr_debug import plot_sigma_level


def main():
    prior_mean = np.array([1, 1, 3, 2])
    prior_cov = 1 * np.eye(4)
    T = 1
    A = np.array([[1, 0, T, 0], [0, 1, 0, T], [0, 0, 1, 0], [0, 0, 0, 1]])
    b = 0 * np.ones((4, ))
    Q = np.array([
        [0, 0, 0, 0],
        [0, 0, 0, 0],
        [0, 0, 1.5, 0],
        [0, 0, 0, 1.5],
    ])
    H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
    c = np.zeros((H @ prior_mean).shape)
    R = 2 * np.eye(2)
    K = 20

    num_samples = 1000
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
    xs, Ps = iterative_post_lin_smooth(y, prior_mean, prior_cov, motion_model,
                                       meas_model, num_samples, 1)
    fig, ax = plt.subplots()
    plot_filtered(ax, true_x, y, xs)
    ax.legend()
    plt.show()


def plot_filtered(ax, true_x, meas, xs):
    plot_states_meas(ax, true_x, meas)
    # ax.plot(xf[:, 0], xf[:, 1], "r-", label="x_f")
    ax.plot(xs[:, 0], xs[:, 1], "g-", label="x_s")


def plot_states_meas(ax, true_x, meas):
    #ax.plot(true_x[:, 0], true_x[:, 1], "b-", label="true x")
    ax.plot(meas[:, 0], meas[:, 1], "r*", label="meas")
    return ax


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
        X [K x n] State vector sequence. The k:th state vector is X(k, :)
        H [m x n] Measurement matrix
        R [m x m] Measurement noise covariance

    Returns:
        Y [K, m] Measurement sequence
    """

    r = mvn.rvs(mean=np.zeros((R.shape[0], )), cov=R, size=X.shape[0])

    return (H @ X.T).T + r


if __name__ == "__main__":
    main()
