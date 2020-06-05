"""Example: Iterative post. lin. smoothing with affine models"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal as mvn
from post_lin_smooth.iterative import iterative_post_lin_smooth
from post_lin_smooth.smoothing import rts_smoothing
from post_lin_smooth.slr.distributions import Gaussian
from post_lin_smooth.slr.slr import Slr
from post_lin_smooth.linearizer import Identity
from models.affine import Affine
from post_lin_smooth.analytics import nees
import visualization as vis


def main():
    analytical_linearizer = True
    num_samples = 20000
    K = 20
    num_iterations = 3

    prior_mean = np.array([1, 1, 3, 2])
    prior_cov = 1 * np.eye(4)
    T = 1
    A = np.array([[1, 0, T, 0], [0, 1, 0, T], [0, 0, 1, 0], [0, 0, 0, 1]])
    b = 0 * np.ones((4, ))
    Q = np.array([
        [0,
         0,
         0,
         0],
        [0,
         0,
         0,
         0],
        [0,
         0,
         1.5,
         0],
        [0,
         0,
         0,
         1.5],
    ])
    H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
    c = np.zeros((H @ prior_mean).shape)
    R = 2 * np.eye(2)

    if analytical_linearizer:
        print("Using analytical linearizer")
        motion_lin = Identity(A, b, Q)
        meas_lin = Identity(H, c, R)
    else:
        print("Using SLR linearizer")
        motion_lin = Slr(p_x=Gaussian(),
                         p_z_given_x=Affine(A,
                                            b,
                                            Q),
                         num_samples=num_samples)
        meas_lin = Slr(p_x=Gaussian(),
                       p_z_given_x=Affine(H,
                                          c,
                                          R),
                       num_samples=num_samples)
    true_x = gen_linear_state_seq(prior_mean, prior_cov, A, Q, K)
    y = gen_linear_meas_seq(true_x, H, R)

    (xs_slr,
     Ps_slr,
     xf_slr,
     Pf_slr,
     linearizations) = iterative_post_lin_smooth(y,
                                                 prior_mean,
                                                 prior_cov,
                                                 motion_lin,
                                                 meas_lin,
                                                 num_iterations)

    # xs_kf, Ps_kf, xf_kf, Pf_kf = iterative_kf_rts(y, prior_mean, prior_cov,
    #                                               (A, b, Q), (H, c, R),
    #                                               num_iterations)

    # A_avg = np.zeros(A.shape)
    # b_avg = np.zeros(b.shape)
    # Q_avg = np.zeros(Q.shape)
    # for lin in linearizations:
    #     A_k, b_k, Q_k = lin
    #     A_avg += A_k / K
    #     b_avg += b_k / K
    #     Q_avg += Q_k / K
    # print("A:\n", A)
    # print("A_hat:\n", A_avg)
    # print("Norm A", ((A - A_avg)**2).sum())

    #vis.plot_nees_comp(true_x, xs_kf, Ps_kf, xs_slr, Ps_slr)
    vis.plot_nees_and_2d_est(true_x,
                             y,
                             xf_slr,
                             Pf_slr,
                             xs_slr,
                             Ps_slr,
                             sigma_level=3,
                             skip_cov=2)


def true_kf_param(A, b, Q, H, c, R, prior_mean, prior_cov, meas):
    pred_mean = A @ prior_mean + b
    pred_cov = A @ prior_cov @ A.T + Q
    print("pred_mean", pred_mean)
    print("pred_cov", pred_cov)


def test_slr_kf_filter(true_x,
                       y,
                       prior_mean,
                       prior_cov,
                       motion_model,
                       meas_model,
                       num_samples):
    print("\nFILTERING\n")


def plot_filtered(ax, true_x, meas, xf, Pf, xs, Ps):
    plot_states_meas(ax, true_x, meas)
    ax.plot(xf[:, 0], xf[:, 1], "r-", label="x_f")
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

    X[0, :] = mvn.rvs(mean=A @ x_0, cov=P_0, size=1)

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
