"""Filtering"""
import numpy as np


def non_linear_kalman_filter(measurements, prior_mean, prior_cov, motion_model,
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

    K = measurements.shape[1]
    dim_x = prior_mean.shape[0]
    # Data allocation
    filtered_states = np.zeros((K, dim_x))
    filtered_cov = np.zeros((K, dim_x, dim_x))
    pred_states = np.zeros((K, dim_x))
    pred_cov = np.zeros((K, dim_x, dim_x))
    for k in range(K):
        meas = measurements[k, :]
        # Run filter iteration
        pred_mean, pred_cov = _prediction(prior_mean, prior_cov, motion_model,
                                          process_noise_cov)
        updated_mean, updated_cov = _update(pred_mean, pred_cov, meas,
                                            meas_model, meas_noise_cov)
        # Store the parameters for use in next step
        pred_states[:, k] = pred_mean
        pred_cov[:, :, k] = pred_cov
        filtered_states[:, k] = updated_mean
        filtered_cov[:, :, k] = updated_cov
        prior_mean = updated_mean
        prior_cov = updated_cov
    return filtered_states, filtered_cov, pred_states, pred_cov


def _prediction(prior_mean, prior_cov, motion_model, process_noise_cov):
    """Non-linear Kalman filter prediction
    calculates mean and covariance of predicted state density
    using a non-linear Gaussian model.

    Args:
        prior_mean np.array(D_x): Prior mean
        prior_cov np.array(D_x, D_x): Prior covariance
        motion_model: Motion model function handle
                      Takes as input x (state)
                      Returns predicted mean and Jacobian evaluated at x
       process_noise_cov np.array(D_x, D_x): Process noise covariance

    Returns:
       pred_mean np.array(D_x, D_x): predicted state mean
       pred_cov np.array(D_x, D_x): predicted state covariance
    """
    pred_state, jacobian = motion_model(prior_mean)
    pred_cov = jacobian @ prior_cov @ jacobian.T + process_noise_cov
    return pred_state, pred_cov


def _update(prior_mean, prior_cov, meas, meas_model, meas_noise_cov):
    """Non-linear Kalman filter prediction
    calculates mean and covariance of predicted state density
    using a non-linear Gaussian model.

    Args:
        prior_mean np.array(D_x,): Prior mean
        prior_cov np.array(D_x, D_x): Prior covariance
        meas np.array(d,): Measurement
        meas_model: Measurement model function handle
                    Takes as input x (state),
                    Returns measurement mean and Jacobian evaluated at x
        meas_noise_cov np.array(D_y, D_y): Measurement noise covariance

    Returns:
       updated_mean np.array(D_x, D_x): updated state mean
       updated_cov np.array(D_x, D_x): updated state covariance
    """
    [meas_mean, jacobian] = meas_model(prior_mean)
    print("meas_mean", meas_mean)
    S = jacobian @ prior_cov @ jacobian.T + meas_noise_cov
    print("S", S)
    K = prior_cov @ jacobian.T @ np.linalg.inv(S)
    updated_mean = prior_mean + K @ (meas - meas_mean)
    updated_cov = prior_cov - K @ S @ K.T
    return updated_mean, updated_cov
