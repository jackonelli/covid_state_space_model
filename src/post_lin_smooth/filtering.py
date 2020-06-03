"""Kalman filter (KF)"""
import numpy as np
from post_lin_smooth.slr.distributions import Prior, Conditional
from post_lin_smooth.slr.slr import Slr
from analytics import pos_def_check, pos_def_ratio


def slr_kf(measurements,
           x_0_0,
           P_0_0,
           prior: Prior,
           motion_model: Conditional,
           meas_model: Conditional,
           num_samples: int):
    """Kalman filter with SLR linearization
    Filters a measurement sequence using a linear Kalman filter.
    Linearization is done with SLR

    Args:
        measurements (K, D_y): Measurement sequence for times 1,..., K
        x_0_0 (D_x,): Prior mean for time 0
        P_0_0 (D_x, D_x): Prior covariance for time 0
        motion_model
        meas_model
        num_samples

    Returns:
        filter_means (K, D_x): Filtered estimates for times 1,..., K
        filter_covs (K, D_x, D_x): Filter error covariance
        pred_means (K, D_x): Predicted estimates for times 1,..., K
        pred_covs (K, D_x, D_x): Filter error covariance
        linearizations List(np.array, np.array, np.array):
            List of tuples (A, b, Q), param's for linear approx of the motion model.
    """

    K = measurements.shape[0]

    filter_means, filter_covs = _init_estimates(x_0_0, P_0_0, K)
    pred_means, pred_covs = _init_estimates(x_0_0, P_0_0, K)
    linearizations = [None] * K

    x_kminus1_kminus1 = x_0_0
    P_kminus1_kminus1 = P_0_0
    for k in np.arange(1, K + 1):
        print("Time step: ", k)
        # measurment vec is zero-indexed
        # this really gives y_k
        y_k = measurements[k - 1]
        slr = Slr(prior(x_bar=x_kminus1_kminus1,
                        P=P_kminus1_kminus1),
                  motion_model)
        motion_lin = slr.linear_parameters(num_samples)
        x_k_kminus1, P_k_kminus1 = _predict(x_kminus1_kminus1, P_kminus1_kminus1, motion_lin)

        slr = Slr(prior(x_bar=x_k_kminus1, P=P_k_kminus1), meas_model)
        meas_lin = slr.linear_parameters(num_samples)
        x_k_k, P_k_k = _update(y_k, x_k_kminus1, P_k_kminus1, meas_lin)

        linearizations[k - 1] = motion_lin
        pred_means[k, :] = x_k_kminus1
        pred_covs[k, :, :] = P_k_kminus1
        filter_means[k, :] = x_k_k
        filter_covs[k, :, :] = P_k_k
        # Shift to next time step
        x_kminus1_kminus1 = x_k_k
        P_kminus1_kminus1 = P_k_k

    return filter_means, filter_covs, pred_means, pred_covs, linearizations


def slr_kf_known_priors(measurements,
                        x_0_0,
                        P_0_0,
                        prev_smooth_means,
                        prev_smooth_covs,
                        prior,
                        motion_model: Conditional,
                        meas_model: Conditional,
                        num_samples: int):

    K = measurements.shape[0]

    filter_means, filter_covs = _init_estimates(x_0_0, P_0_0, K)
    pred_means, pred_covs = _init_estimates(x_0_0, P_0_0, K)
    linearizations = [None] * K

    x_kminus1_kminus1 = x_0_0
    P_kminus1_kminus1 = P_0_0
    for k in np.arange(1, K + 1):
        print("Time step: ", k)
        # measurment vec is zero-indexed
        # this really gives y_k
        y_k = measurements[k - 1]
        x_kminus1_K = prev_smooth_means[k - 1, :]
        P_kminus1_K = prev_smooth_covs[k - 1, :, :]
        x_k_K = prev_smooth_means[k, :]
        P_k_K = prev_smooth_covs[k, :, :]
        slr = Slr(prior(x_bar=x_kminus1_K, P=P_kminus1_K), motion_model)
        motion_lin = slr.linear_parameters(num_samples)
        x_k_kminus1, P_k_kminus1 = _predict(x_kminus1_kminus1, P_kminus1_kminus1, motion_lin)

        slr = Slr(prior(x_bar=x_k_K, P=P_k_K), meas_model)
        meas_lin = slr.linear_parameters(num_samples)
        x_k_k, P_k_k = _update(y_k, x_k_kminus1, P_k_kminus1, meas_lin)

        linearizations[k - 1] = motion_lin
        pred_means[k, :] = x_k_kminus1
        pred_covs[k, :, :] = P_k_kminus1
        filter_means[k, :] = x_k_k
        filter_covs[k, :, :] = P_k_k
        # Shift to next time step
        x_kminus1_kminus1 = x_k_k
        P_kminus1_kminus1 = P_k_k
    return filter_means, filter_covs, pred_means, pred_covs, linearizations


def _init_estimates(x_0_0, P_0_0, K):
    D_x = x_0_0.shape[0]
    est_means = np.empty((K + 1, D_x))
    est_covs = np.empty((K + 1, D_x, D_x))
    est_means[0, :] = x_0_0
    est_covs[0, :, :] = P_0_0
    return est_means, est_covs


def analytical_kf(measurements, x_0_0, P_0_0, motion_lin, meas_lin):
    """SLR Kalman filter with SLR linearization
    Filters a measurement sequence using a linear Kalman filter.
    Linearization is done with SLR
    Args:
        measurements np.array(K, D_y): Measurement sequence for times 1,..., K
        x_0_0 np.array(D_x,): Prior mean for time 0
        P_0_0 np.array(D_x, D_x): Prior covariance
        motion_model
        meas_model
        num_samples

    Returns:
        filter_means np.array(K, D_x): Filtered estimates for times 1,..., K
        filter_covs np.array(K, D_x, D_x): Filter error covariance
        pred_means np.array(): Predicted estimates for times 1,..., K
        pred_covs np.array(): Filter error covariance
        linearizations List(np.array, np.array, np.array):
            List of tuples (A, b, Q), param's for linear approx
    """

    K = measurements.shape[0]
    D_x = x_0_0.shape[0]

    filter_means = np.empty((K, D_x))
    filter_covs = np.empty((K, D_x, D_x))
    pred_means = np.empty((K, D_x))
    pred_covs = np.empty((K, D_x, D_x))
    x_kminus1_kminus1 = x_0_0
    P_kminus1_kminus1 = P_0_0
    for k in np.arange(1, K + 1):
        y_k = measurements[k - 1]

        x_k_kminus1, P_k_kminus1 = _predict(x_kminus1_kminus1, P_kminus1_kminus1, motion_lin)

        x_k_k, P_k_k = _update(y_k, x_k_kminus1, P_k_kminus1, meas_lin)

        pred_means[k, :] = x_k_kminus1
        pred_covs[k, :, :] = P_k_kminus1
        filter_means[k, :] = x_k_k
        filter_covs[k, :, :] = P_k_k
        x_0_0 = x_k_k
        P_0_0 = P_k_k
    return filter_means, filter_covs, pred_means, pred_covs


def _predict(x_kminus1_kminus1, P_kminus1_kminus1, linearization):
    """KF prediction step
        linearizations List(np.array, np.array, np.array):
            List of tuples (A, b, Q), param's for linear approx
    """
    A, b, Q = linearization
    x_k_kminus1 = A @ x_kminus1_kminus1 + b
    P_k_kminus1 = A @ P_kminus1_kminus1 @ A.T + Q
    P_k_kminus1 = (P_k_kminus1 + P_k_kminus1.T) / 2
    return x_k_kminus1, P_k_kminus1


def _update(y_k, x_k_kminus1, P_k_kminus1, linearization):
    """KF update step"""
    H, c, R = linearization
    y_mean = H @ x_k_kminus1 + c
    S = H @ P_k_kminus1 @ H.T + R
    K = (P_k_kminus1 @ H.T @ np.linalg.inv(S))

    x_k_k = x_k_kminus1 + (K @ (y_k - y_mean)).reshape(x_k_kminus1.shape)
    P_k_k = P_k_kminus1 - K @ S @ K.T
    if not pos_def_check(P_k_k):
        raise ValueError("updated cov not pos def")

    return x_k_k, P_k_k
