"""Rauch-Tung-Striebel (RTS) smoothing"""
import numpy as np
from analytics import pos_def_check
from post_lin_smooth.filtering import _init_estimates


def rts_smoothing(filter_means,
                  filter_covs,
                  pred_means,
                  pred_covs,
                  linearizations):
    """Rauch-Tung-Striebel smoothing
    Smooths a measurement sequence and outputs from a Kalman filter.

    Args:
        filter_means (K+1, D_x): Filtered estimates for times 0,..., K
        filter_covs (K+1, D_x, D_x): Filter error covariance
        pred_means (K+1, D_x): Predicted estimates for times 0,..., K
        pred_covs (K+1, D_x, D_x): Filter error covariance
        linearizations List(np.array, np.array, np.array):
            List of tuples (A, b, Q), param's for linear approx

    Returns:
        smooth_means (K+1, D_x): Smooth estimates for times 1,..., K
        smooth_covs (K+1, D_x, D_x): Smooth error covariance
    """

    K = filter_means.shape[0] - 1
    smooth_means, smooth_covs = _init_smooth_estimates(filter_means,
                                                       filter_covs)
    for k in np.flip(np.arange(1, K + 1)):
        print("Time step: ", k)
        linear_params = linearizations[k - 1]
        smooth_mean, smooth_cov = _rts_update(smooth_means[k, :],
                                              smooth_covs[k, :, :],
                                              filter_means[k-1, :],
                                              filter_covs[k-1, :, :],
                                              pred_means[k, :],
                                              pred_covs[k, :, :],
                                              linear_params)
        if not pos_def_check(smooth_cov):
            raise ValueError("Smooth cov not pos def")
        smooth_means[k, :] = smooth_mean
        smooth_covs[k, :, :] = smooth_cov
    print(smooth_means)
    return smooth_means, smooth_covs


def _rts_update(xs_kplus1,
                Ps_kplus1,
                xf_k,
                Pf_k,
                xp_kplus1,
                Pp_kplus1,
                linear_params):
    """RTS update step"""
    A, b, Q = linear_params

    G_k = Pf_k @ A.T @ np.linalg.inv(Pp_kplus1)
    xs_k = xf_k + G_k @ (xs_kplus1 - xp_kplus1)
    Ps_k = Pf_k + G_k @ (Ps_kplus1 - Pp_kplus1) @ G_k.T
    Ps_k = (Ps_k + Ps_k.T) / 2
    return xs_k, Ps_k


def _init_smooth_estimates(filter_means, filter_covs):
    K = filter_means.shape[0]
    smooth_means, smooth_covs = _init_estimates(filter_means[0, :], filter_covs[0, :, :], K)
    smooth_means[-1, :] = filter_means[-1, :]
    smooth_covs[-1, :, :] = filter_covs[-1, :, :]
    return smooth_means, smooth_covs
