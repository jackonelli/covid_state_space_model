import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal as mvn
from post_lin_filt.filtering import slr_kalman_filter
from post_lin_filt.models import
prior_mean = np.array([1, 3])
prior_var = 4 * np.eye(2)


def main():
    T = 1
    A = np.array([[1, T], [0, 1]])
    Q = np.array([[0, 0], [0, 1.5]])
    H = np.array([[1, 0]])
    R = np.array([2])
    N = 20

    # Generate true states and measurements
    true_x = gen_linear_state_seq(prior_mean, prior_var, A, Q, N)
    y = gen_linear_meas_seq(true_x, H, R)
    plot_states_meas(true_x, y)
    plt.show()

    # [est_x, est_P] = slr_kalman_filter(y, prior_mean, prior_var, A, Q, H, R)


def plot_states_meas(true_x, meas):
    plt.plot(true_x[:, 0], "b-")
    plt.plot(meas, "r*")


def gen_linear_state_seq(x_0, P_0, A, Q, N):
    """Generates an N-long sequence of states using a
       Gaussian prior and a linear Gaussian process model

       Args:
          x_0         [n x 1] Prior mean
          P_0         [n x n] Prior covariance
          A           [n x n] State transition matrix
          Q           [n x n] Process noise covariance
          N           [1 x 1] Number of states to generate

       Returns:
          X           [n x N+1] State vector sequence
    """
    X = np.zeros((N, x_0.shape[0]))

    X[0, :] = mvn.rvs(mean=x_0, cov=P_0, size=1)

    q = mvn.rvs(mean=np.zeros(x_0.shape), cov=Q, size=N)

    for k in np.arange(1, N):
        X[k, :] = A @ X[k - 1, :] + q[k - 1, :]
    return X


def gen_linear_meas_seq(X, H, R):
    """generates a sequence of observations of the state
    sequence X using a linear measurement model.
    Measurement noise is assumed to be zero mean and Gaussian.

    Args:
        X [N+1 x n] State vector sequence. The k:th state vector is X(:,k+1)
        H [m x n] Measurement matrix
        R [m x m] Measurement noise covariance

    Returns:
        Y [m x N] Measurement sequence
    """

    r = mvn.rvs(mean=np.zeros((R.shape[0], 1)), cov=R, size=X.shape[0] - 1)

    return (H @ X[1:, :].T + r).T


if __name__ == "__main__":
    main()
