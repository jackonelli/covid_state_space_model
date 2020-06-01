"""Kalman filter (KF)"""
import numpy as np
from post_lin_smooth.slr.distributions import Gaussian, Conditional
from post_lin_smooth.slr.slr import Slr


def slr_kalman_filter(measurements, prior_mean, prior_cov,
                      motion_model: Conditional, meas_model: Conditional,
                      num_samples):
    """SLR Kalman filter
    Filters a measurement sequence using a non-linear Kalman filter.
    Args:
        measurements np.array(K, D_y): Measurement sequence for times 1,..., K
        prior_mean np.array(D_x,): Prior mean for time 0
        prior_cov np.array(D_x, D_x): Prior covariance
        motion_model:
        process_noise_cov np.array(D_x, D_x): Process noise covariance
        meas_model:
        meas_noise_cov np.array(D_y, D_y): Measurement noise covariance

    Returns:
        filtered_means np.array(K, D_x): Filtered estimates for times 1,..., K
        filtered_cov np.array(K, D_x, D_x): Filter error covariance
        pred_means np.array(): Predicted estimates for times 1,..., K
        pred_cov np.array(): Filter error covariance
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

        pred_mean, pred_cov, A, b, Q = _predict(prior_mean, prior_cov,
                                                motion_model, num_samples)

        updated_mean, updated_cov = _update(meas, pred_mean, pred_cov,
                                            meas_model, num_samples)

        linearizations[k] = (A, b, Q)
        pred_means[k, :] = pred_mean
        pred_covs[k, :, :] = pred_cov
        filtered_means[k, :] = updated_mean
        filtered_cov[k, :, :] = updated_cov
        prior_mean = updated_mean
        prior_cov = updated_cov
    return filtered_means, filtered_cov, pred_means, pred_covs, linearizations


def _predict(prior_mean, prior_cov, motion_model: Conditional,
             num_samples: int):
    slr = Slr(Gaussian(x_bar=prior_mean, P=prior_cov), motion_model)
    A, b, Q = slr.linear_parameters(num_samples)
    pred_mean = A @ prior_mean + b
    pred_cov = A @ prior_cov @ A.T + Q
    return pred_mean, pred_cov, A, b, Q


def _update(meas, pred_mean, pred_cov, meas_model: Conditional,
            num_samples: int):
    slr = Slr(Gaussian(x_bar=pred_mean, P=pred_cov), meas_model)
    H, c, R = slr.linear_parameters(num_samples)
    meas_mean = H @ pred_mean + c
    S = H @ pred_cov @ H.T + R
    K = (pred_cov @ H.T @ np.linalg.inv(S))

    updated_mean = pred_mean + (K @ (meas - meas_mean)).reshape(
        pred_mean.shape)
    updated_cov = pred_cov - K @ S @ K.T

    return updated_mean, updated_cov


from post_lin_smooth.deprecated.filter_type.interface import FilterType
from post_lin_smooth.deprecated.motion_models.interface import MotionModel
from post_lin_smooth.deprecated.meas_models.interface import MeasModel


def non_linear_kalman_filter(filter_type: FilterType, measurements, prior_mean,
                             prior_cov, motion_model: MotionModel,
                             process_noise_cov, meas_model, meas_noise_cov):
    """Non-linear Kalman filter
    Filters a measurement sequence using a non-linear Kalman filter.
    Args:
        measurements np.array(K, D_y): Measurement sequence for times 1,..., K
        prior_mean np.array(D_x,): Prior mean for time 0
        prior_cov np.array(D_x, D_x): Prior covariance
        motion_model: Motion model function handle
                      Takes as input x (state)
                      Returns predicted mean and Jacobian evaluated at x
        process_noise_cov np.array(D_x, D_x): Process noise covariance
        meas_model: Measurement model function handle
                    Takes as input x (state),
                    Returns measurement mean and Jacobian evaluated at x
        meas_noise_cov np.array(D_y, D_y): Measurement noise covariance

    Returns:
        filtered_means np.array(): Filtered estimates for times 1,..., K
        filtered_cov np.array(): Filter error covariance
        pred_means np.array(): Predicted estimates for times 1,..., K
        pred_cov np.array(): Filter error covariance
    """

    K = measurements.shape[0]
    dim_x = prior_mean.shape[0]
    # Data allocation
    filtered_means = np.zeros((K, dim_x))
    filtered_cov = np.zeros((K, dim_x, dim_x))
    pred_means = np.zeros((K, dim_x))
    pred_covs = np.zeros((K, dim_x, dim_x))
    for k in range(K):
        meas = measurements[k, :]
        # Run filter iteration
        pred_mean, pred_cov = filter_type._predict(prior_mean, prior_cov,
                                                   motion_model,
                                                   process_noise_cov)
        updated_mean, updated_cov = filter_type._update(
            pred_mean, pred_cov, meas, meas_model, meas_noise_cov)
        # Store the parameters for use in next step
        pred_means[k, :] = pred_mean
        pred_covs[k, :, :] = pred_cov
        filtered_means[k, :] = updated_mean
        filtered_cov[k, :, :] = updated_cov
        prior_mean = updated_mean
        prior_cov = updated_cov
    return filtered_means, filtered_cov, pred_means, pred_covs
