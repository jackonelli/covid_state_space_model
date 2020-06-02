"""Kalman filter (KF)"""
import numpy as np
from post_lin_smooth.slr.distributions import Gaussian, Conditional
from post_lin_smooth.slr.slr import Slr


def slr_kf(measurements, prior_mean, prior_cov, motion_model: Conditional,
           meas_model: Conditional, num_samples: int):
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
        filtered_means np.array(K, D_x): Filtered estimates for times 1,..., K
        filtered_cov np.array(K, D_x, D_x): Filter error covariance
        pred_means np.array(): Predicted estimates for times 1,..., K
        pred_cov np.array(): Filter error covariance
        linearizations List(np.array, np.array, np.array):
            List of tuples (A, b, Q), param's for linear approx
    """

    K = measurements.shape[0]
    dim_x = prior_mean.shape[0]

    filtered_means = np.zeros((K, dim_x))
    filtered_cov = np.zeros((K, dim_x, dim_x))
    pred_means = np.zeros((K, dim_x))
    pred_covs = np.zeros((K, dim_x, dim_x))
    linearizations = [None] * K
    for k in range(K):
        print("Time step: ", k)
        meas = measurements[k, :]

        slr = Slr(Gaussian(x_bar=prior_mean, P=prior_cov), motion_model)
        motion_lin = slr.linear_parameters(num_samples)
        pred_mean, pred_cov = _predict(prior_mean, prior_cov, motion_lin)

        slr = Slr(Gaussian(x_bar=pred_mean, P=pred_cov), meas_model)
        meas_lin = slr.linear_parameters(num_samples)
        updated_mean, updated_cov = _update(meas, pred_mean, pred_cov,
                                            meas_lin)

        linearizations[k] = motion_lin
        pred_means[k, :] = pred_mean
        pred_covs[k, :, :] = pred_cov
        filtered_means[k, :] = updated_mean
        filtered_cov[k, :, :] = updated_cov
        prior_mean = updated_mean
        prior_cov = updated_cov
    return filtered_means, filtered_cov, pred_means, pred_covs, linearizations


def slr_kf_known_priors(measurements, prev_smooth_means, prev_smooth_covs,
                        motion_model: Conditional, meas_model: Conditional,
                        num_samples: int):

    K = measurements.shape[0]
    dim_x = prev_smooth_means.shape[1]

    filtered_means = np.zeros((K, dim_x))
    filtered_cov = np.zeros((K, dim_x, dim_x))
    pred_means = np.zeros((K, dim_x))
    pred_covs = np.zeros((K, dim_x, dim_x))
    linearizations = [None] * K
    for k in range(K):
        meas = measurements[k, :]
        prior_mean = prev_smooth_means[k, :]
        prior_cov = prev_smooth_covs[k, :, :]

        slr = Slr(Gaussian(x_bar=prior_mean, P=prior_cov), motion_model)
        motion_lin = slr.linear_parameters(num_samples)
        pred_mean, pred_cov = _predict(prior_mean, prior_cov, motion_lin)

        slr = Slr(Gaussian(x_bar=pred_mean, P=pred_cov), meas_model)
        meas_lin = slr.linear_parameters(num_samples)
        updated_mean, updated_cov = _update(meas, pred_mean, pred_cov,
                                            meas_lin)

        linearizations[k] = motion_lin
        pred_means[k, :] = pred_mean
        pred_covs[k, :, :] = pred_cov
        filtered_means[k, :] = updated_mean
        filtered_cov[k, :, :] = updated_cov
    return filtered_means, filtered_cov, pred_means, pred_covs, linearizations


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
        filtered_means np.array(K, D_x): Filtered estimates for times 1,..., K
        filtered_cov np.array(K, D_x, D_x): Filter error covariance
        pred_means np.array(): Predicted estimates for times 1,..., K
        pred_cov np.array(): Filter error covariance
        linearizations List(np.array, np.array, np.array):
            List of tuples (A, b, Q), param's for linear approx
    """

    K = measurements.shape[0]
    dim_x = prior_mean.shape[0]

    filtered_means = np.zeros((K, dim_x))
    filtered_cov = np.zeros((K, dim_x, dim_x))
    pred_means = np.zeros((K, dim_x))
    pred_covs = np.zeros((K, dim_x, dim_x))
    for k in range(K):
        meas = measurements[k, :]

        pred_mean, pred_cov = _predict(prior_mean, prior_cov, motion_lin)

        updated_mean, updated_cov = _update(meas, pred_mean, pred_cov,
                                            meas_lin)

        pred_means[k, :] = pred_mean
        pred_covs[k, :, :] = pred_cov
        filtered_means[k, :] = updated_mean
        filtered_cov[k, :, :] = updated_cov
        prior_mean = updated_mean
        prior_cov = updated_cov
    return filtered_means, filtered_cov, pred_means, pred_covs


def _predict(prior_mean, prior_cov, linearization):
    """KF prediction step
        linearizations List(np.array, np.array, np.array):
            List of tuples (A, b, Q), param's for linear approx
    """
    A, b, Q = linearization
    pred_mean = A @ prior_mean + b
    pred_cov = A @ prior_cov @ A.T + Q
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

    return updated_mean, updated_cov
