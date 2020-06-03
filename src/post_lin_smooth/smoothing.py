"""Rauch-Tung-Striebel (RTS) smoothing"""
import numpy as np
from analytics import pos_def_check


def rts_smoothing(filter_means, filter_covs, pred_means, pred_covs,
                  linearizations):
    """Rauch-Tung-Striebel smoothing
    Smooths a measurement sequence and outputs from a Kalman filter.

    Args:
        filter_means (K, D_x): Filtered estimates for times 1,..., K
        filter_covs (K, D_x, D_x): Filter error covariance
        pred_means (K, D_x): Predicted estimates for times 1,..., K
        pred_covs (K, D_x, D_x): Filter error covariance
        linearizations List(np.array, np.array, np.array):
            List of tuples (A, b, Q), param's for linear approx

    Returns:
        smooth_means (K, D_x): Smooth estimates for times 1,..., K
        smooth_covs (K, D_x, D_x): Smooth error covariance
    """

    K, D_x = filter_means.shape

    smooth_means = np.zeros((K, D_x))
    smooth_covs = np.zeros((K, D_x, D_x))
    smooth_mean = filter_means[-1, :]
    smooth_cov = filter_covs[-1, :, :]
    smooth_means[-1, :] = smooth_mean
    smooth_covs[-1, :, :] = smooth_cov
    for k in np.flip(np.arange(K - 1)):
        linear_params = linearizations[k]
        smooth_mean, smooth_cov = _rts_update(
            smooth_mean, smooth_cov, filter_means[k, :], filter_covs[k, :, :],
            pred_means[k + 1, :], pred_covs[k + 1, :, :], linear_params)
        if not pos_def_check(smooth_cov):
            raise ValueError("Smooth cov not pos def")
        smooth_means[k, :] = smooth_mean
        smooth_covs[k, :, :] = smooth_cov
    return smooth_means, smooth_covs


def _rts_update(xs_kplus1, Ps_kplus1, xf_k, Pf_k, xp_kplus1, Pp_kplus1,
                linear_params):
    """RTS update step"""
    A, b, Q = linear_params

    G_k = Pf_k @ A.T @ np.linalg.inv(Pp_kplus1)
    xs_k = xf_k + G_k @ (xs_kplus1 - xp_kplus1)
    Ps_k = Pf_k + G_k @ (Ps_kplus1 - Pp_kplus1) @ G_k.T
    Ps_k = (Ps_k + Ps_k.T) / 2
    return xs_k, Ps_k
