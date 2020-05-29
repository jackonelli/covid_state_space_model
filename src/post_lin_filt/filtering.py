"""Kalman filter (KF)"""
import numpy as np
from post_lin_filt.slr.distributions import Conditional
from post_lin_filt.deprecated.filter_type.slr import SlrFilter


def slr_kalman_filter(measurements, prior_mean, prior_cov,
                      motion_model: Conditional, meas_model: Conditional,
                      num_samples):
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
        filtered_mean np.array(): Filtered estimates for times 1,..., K
        filtered_cov np.array(): Filter error covariance
        pred_states np.array(): Predicted estimates for times 1,..., K
        pred_cov np.array(): Filter error covariance
    """

    K = measurements.shape[0]
    dim_x = prior_mean.shape[0]
    # Data allocation
    filtered_states = np.zeros((K, dim_x))
    filtered_cov = np.zeros((K, dim_x, dim_x))
    pred_states = np.zeros((K, dim_x))
    pred_covs = np.zeros((K, dim_x, dim_x))
    filter_ = SlrFilter(motion_model, meas_model, num_samples)
    print("P_0", prior_cov.shape)
    for k in range(K):
        print("time step:", k)
        meas = measurements[k, :]
        # Run filter iteration
        pred_mean, pred_cov = filter_.predict(prior_mean, prior_cov)

        updated_mean, updated_cov = filter_.update(pred_mean, pred_cov, meas)
        # Store the parameters for use in next step
        pred_states[k, :] = pred_mean
        pred_covs[k, :, :] = pred_cov
        filtered_states[k, :] = updated_mean
        filtered_cov[k, :, :] = updated_cov
        prior_mean = updated_mean
        prior_cov = updated_cov
    return filtered_states, filtered_cov, pred_states, pred_covs


from post_lin_filt.deprecated.filter_type.slr import SlrFilter
from post_lin_filt.deprecated.filter_type.interface import FilterType
from post_lin_filt.deprecated.motion_models.interface import MotionModel
from post_lin_filt.deprecated.meas_models.interface import MeasModel


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
        filtered_mean np.array(): Filtered estimates for times 1,..., K
        filtered_cov np.array(): Filter error covariance
        pred_states np.array(): Predicted estimates for times 1,..., K
        pred_cov np.array(): Filter error covariance
    """

    K = measurements.shape[0]
    dim_x = prior_mean.shape[0]
    # Data allocation
    filtered_states = np.zeros((K, dim_x))
    filtered_cov = np.zeros((K, dim_x, dim_x))
    pred_states = np.zeros((K, dim_x))
    pred_covs = np.zeros((K, dim_x, dim_x))
    for k in range(K):
        meas = measurements[k, :]
        # Run filter iteration
        pred_mean, pred_cov = filter_type.predict(prior_mean, prior_cov,
                                                  motion_model,
                                                  process_noise_cov)
        updated_mean, updated_cov = filter_type.update(pred_mean, pred_cov,
                                                       meas, meas_model,
                                                       meas_noise_cov)
        # Store the parameters for use in next step
        pred_states[k, :] = pred_mean
        pred_covs[k, :, :] = pred_cov
        filtered_states[k, :] = updated_mean
        filtered_cov[k, :, :] = updated_cov
        prior_mean = updated_mean
        prior_cov = updated_cov
    return filtered_states, filtered_cov, pred_states, pred_covs
