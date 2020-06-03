"""Kalman filter (KF)"""
import numpy as np
from post_lin_smooth.slr.distributions import Prior, Conditional
from post_lin_smooth.slr.slr import Slr
from analytics import pos_def_check, pos_def_ratio


def slr_kf(measurements,
           prior_mean,
           prior_cov,
           prior: Prior,
           motion_model: Conditional,
           meas_model: Conditional,
           num_samples: int):
    """Kalman filter with SLR linearization
    Filters a measurement sequence using a linear Kalman filter.
    Linearization is done with SLR

    Args:
        measurements (K, D_y): Measurement sequence for times 1,..., K
        prior_mean (D_x,): Prior mean for time 0
        prior_cov (D_x, D_x): Prior covariance
        motion_model
        meas_model
        num_samples

    Returns:
        filter_means (K, D_x): Filtered estimates for times 1,..., K
        filter_covs (K, D_x, D_x): Filter error covariance
        pred_means (K, D_x): Predicted estimates for times 1,..., K
        pred_cov (K, D_x, D_x): Filter error covariance
        linearizations List(np.array, np.array, np.array):
            List of tuples (A, b, Q), param's for linear approx
    """

    K = measurements.shape[0]

    filter_means, filter_covs = _init_estimates(prior_mean, prior_cov, K)
    pred_means, pred_covs = _init_estimates(prior_mean, prior_cov, K)
    linearizations = [None] * K
    for k in np.arange(1, K + 1):
        # measurment vec is zero-indexed
        # this really gives y_k
        meas_k = measurements[k - 1]
        print("Time step: ", k)
        slr = Slr(prior(x_bar=prior_mean, P=prior_cov), motion_model)
        motion_lin = slr.linear_parameters(num_samples)
        pred_mean, pred_cov = _predict(prior_mean, prior_cov, motion_lin)

        slr = Slr(prior(x_bar=pred_mean, P=pred_cov), meas_model)
        meas_lin = slr.linear_parameters(num_samples)
        updated_mean, updated_cov = _update(meas_k, pred_mean, pred_cov,
                                            meas_lin)

        # linearization list is zero-indexed
        # this really gives (A_k, b_k, Sigma_k)
        linearizations[k - 1] = motion_lin
        pred_means[k, :] = pred_mean
        pred_covs[k, :, :] = pred_cov
        filter_means[k, :] = updated_mean
        filter_covs[k, :, :] = updated_cov
        prior_mean = updated_mean
        prior_cov = updated_cov

    return filter_means, filter_covs, pred_means, pred_covs, linearizations


def slr_kf_known_priors(measurements,
                        prior_mean,
                        prior_cov,
                        prev_smooth_means,
                        prev_smooth_covs,
                        prior,
                        motion_model: Conditional,
                        meas_model: Conditional,
                        num_samples: int):

    K = measurements.shape[0]

    filter_means, filter_covs = _init_estimates(prior_mean, prior_cov, K)
    pred_means, pred_covs = _init_estimates(prior_mean, prior_cov, K)
    linearizations = [None] * K
    for k in np.arange(1, K + 1):
        # measurment vec is zero-indexed
        # this really gives y_k
        meas_k = measurements[k - 1]
        print("Time step: ", k)
        slr = Slr(
            prior(x_bar=prev_smooth_means[k-1, :], P=prev_smooth_covs[k-1, :, :]),
            motion_model)
        motion_lin = slr.linear_parameters(num_samples)
        pred_mean, pred_cov = _predict(prior_mean, prior_cov, motion_lin)

        slr = Slr(prior(x_bar=prev_smooth_means[k, :], P=prev_smooth_covs[k, :, :]), meas_model)
        meas_lin = slr.linear_parameters(num_samples)
        updated_mean, updated_cov = _update(meas_k, pred_mean, pred_cov,
                                            meas_lin)
        prior_mean = updated_mean
        prior_cov = updated_cov

        linearizations[k - 1] = motion_lin
        pred_means[k, :] = pred_mean
        pred_covs[k, :, :] = pred_cov
        filter_means[k, :] = updated_mean
        filter_covs[k, :, :] = updated_cov
    return filter_means, filter_covs, pred_means, pred_covs, linearizations


def _init_estimates(prior_mean, prior_cov, K):
    D_x = prior_mean.shape[0]
    est_means = np.empty((K + 1, D_x))
    est_covs = np.empty((K + 1, D_x, D_x))
    est_means[0, :] = prior_mean
    est_covs[0, :, :] = prior_cov
    return est_means, est_covs


def analytical_kf(measurements, prior_mean, prior_cov, motion_lin, meas_lin):
    """SLR Kalman filter with SLR linearization
    Filters a measurement sequence using a linear Kalman filter.
    Linearization is done with SLR
    Args:
        measurements np.array(K, D_y): Measurement sequence for times 1,..., K
        prior_mean np.array(D_x,): Prior mean for time 0
        prior_cov np.array(D_x, D_x): Prior covariance
        motion_model
        meas_model
        num_samples

    Returns:
        filter_means np.array(K, D_x): Filtered estimates for times 1,..., K
        filter_covs np.array(K, D_x, D_x): Filter error covariance
        pred_means np.array(): Predicted estimates for times 1,..., K
        pred_cov np.array(): Filter error covariance
        linearizations List(np.array, np.array, np.array):
            List of tuples (A, b, Q), param's for linear approx
    """

    K = measurements.shape[0]
    D_x = prior_mean.shape[0]

    filter_means = np.empty((K, D_x))
    filter_covs = np.empty((K, D_x, D_x))
    pred_means = np.empty((K, D_x))
    pred_covs = np.empty((K, D_x, D_x))
    for k, meas in enumerate(measurements):
        pred_mean, pred_cov = _predict(prior_mean, prior_cov, motion_lin)

        updated_mean, updated_cov = _update(meas, pred_mean, pred_cov,
                                            meas_lin)

        pred_means[k, :] = pred_mean
        pred_covs[k, :, :] = pred_cov
        filter_means[k, :] = updated_mean
        filter_covs[k, :, :] = updated_cov
        prior_mean = updated_mean
        prior_cov = updated_cov
    return filter_means, filter_covs, pred_means, pred_covs


def _predict(prior_mean, prior_cov, linearization):
    """KF prediction step
        linearizations List(np.array, np.array, np.array):
            List of tuples (A, b, Q), param's for linear approx
    """
    A, b, Q = linearization
    pred_mean = A @ prior_mean + b
    pred_cov = A @ prior_cov @ A.T + Q
    pred_cov = (pred_cov + pred_cov.T) / 2
    return pred_mean, pred_cov


def _update(meas, pred_mean, pred_cov, linearization):
    """KF update step"""
    H, c, R = linearization
    meas_mean = H @ pred_mean + c
    S = H @ pred_cov @ H.T + R
    K = (pred_cov @ H.T @ np.linalg.inv(S))

    updated_mean = pred_mean + (K @ (meas - meas_mean)).reshape(
        pred_mean.shape)
    updated_cov = pred_cov - K @ S @ K.T
    if not pos_def_check(updated_cov):
        raise ValueError("updated cov not pos def")
    updated_cov = (updated_cov + updated_cov.T) / 2

    return updated_mean, updated_cov
