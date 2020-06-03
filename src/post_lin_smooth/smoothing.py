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
        x_k_K, P_k_K = smooth_means[k, :], smooth_covs[k, :, :]
        x_k_kminus1, P_k_kminus1 = pred_means[k, :], pred_covs[k, :, :]
        x_kminus1_kminus1 = filter_means[k - 1, :]
        P_kminus_kminus1 = filter_covs[k - 1, :, :]
        x_kminus1_K, P_kminus1_K = _rts_update(x_k_K,
                                               P_k_K,
                                               x_kminus1_kminus1,
                                               P_kminus_kminus1,
                                               x_k_kminus1,
                                               P_k_kminus1,
                                               linear_params)
        if not pos_def_check(P_kminus1_K):
            raise ValueError("Smooth cov not pos def")
        smooth_means[k - 1, :] = x_kminus1_K
        smooth_covs[k - 1, :, :] = P_kminus1_K
    return smooth_means, smooth_covs


def _rts_update(x_k_K,
                P_k_K,
                x_kminus1_kminus1,
                P_kminus_kminus1,
                x_k_kminus1,
                P_k_kminus1,
                linear_params):
    """RTS update step"""
    A, b, Q = linear_params

    G_k = P_kminus_kminus1 @ A.T @ np.linalg.inv(P_k_kminus1)
    x_kminus1_K = x_kminus1_kminus1 + G_k @ (x_k_K - x_k_kminus1)
    P_kminus1_K = P_kminus_kminus1 + G_k @ (P_k_K - P_k_kminus1) @ G_k.T
    return x_kminus1_K, P_kminus1_K


def _init_smooth_estimates(filter_means, filter_covs):
    K, D_x = filter_means.shape
    smooth_means = np.empty((K, D_x))
    smooth_covs = np.empty((K, D_x, D_x))
    smooth_means[-1, :] = filter_means[-1, :]
    smooth_covs[-1, :, :] = filter_covs[-1, :, :]
    return smooth_means, smooth_covs
