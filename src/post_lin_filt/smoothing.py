"""Rauch-Tung-Striebel (RTS) smoothing"""
import numpy as np


def rts_smoothing(filtered_means, filtered_covs, pred_means, pred_covs,
                  motion_model):
    """Rauch-Tung-Striebel smoothing
    Smooths a measurement sequence and outputs from a Kalman filter.

    Args:
        filtered_means np.array(): Filtered estimates for times 1,..., K
        filtered_covs np.array(): Filter error covariance
        pred_means np.array(): Predicted estimates for times 1,..., K
        pred_covs np.array(): Filter error covariance
        motion_model: Motion model function handle
                      Takes as input x (state)
                      Returns predicted mean and Jacobian evaluated at x

    Returns:
        smooth_means np.array(): Smooth estimates for times 1,..., K
        smooth_covs np.array(): Smooth error covariance
    """

    K = filtered_means.shape[0]

    # Allocation
    smooth_means = np.zeros(filtered_means.shape)
    smooth_covs = np.zeros(filtered_covs.shape)
    smooth_mean = filtered_means[K - 1, :]
    smooth_cov = filtered_covs[K - 1, :, :]
    for k in np.flip(np.arange(K - 2)):
        # Run filter iteration
        smooth_mean, smooth_cov = _rts_update(smooth_mean, smooth_cov,
                                              filtered_means[k, :],
                                              filtered_covs[k, :, :],
                                              pred_means[k + 1, :],
                                              pred_covs[k + 1, :, :],
                                              motion_model)
        smooth_means[k, :] = smooth_mean
        smooth_covs[k, :, :] = smooth_cov
    return smooth_means, smooth_covs


def _rts_update(xs_kplus1, Ps_kplus1, xf_k, Pf_k, xp_kplus1, Pp_kplus1,
                motion_model):
    """Non-linear Kalman filter prediction
    calculates mean and covariance of predicted state density
    using a non-linear Gaussian model.

    Args:
        prior_mean np.array(D_x): Prior mean
        prior_cov np.array(D_x, D_x): Prior covariance
        motion_model: Motion model function handle
                      Takes as input x (state)
                      Returns predicted mean and Jacobian evaluated at x

    Returns:
       pred_mean np.array(D_x, D_x): predicted state mean
       pred_cov np.array(D_x, D_x): predicted state covariance
    """
    _, jacobian = motion_model(xf_k)
    P_kkplus1 = Pf_k * jacobian.T

    G_k = P_kkplus1 * np.linalg.inv(Pp_kplus1)
    xs_k = xf_k + G_k @ (xs_kplus1 - xp_kplus1)
    # TODO Check order in Pp Ps
    Ps_k = Pf_k - G_k @ (Ps_kplus1 - Pp_kplus1) * G_k.T
    return xs_k, Ps_k
