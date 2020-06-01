"""Rauch-Tung-Striebel (RTS) smoothing"""
import numpy as np
from post_lin_filt.slr.distributions import Conditional, Gaussian
from post_lin_filt.slr.slr import Slr


def slr_rts_smoothing(filtered_means, filtered_covs, pred_means, pred_covs,
                      motion_model: Conditional, num_samples):
    """Rauch-Tung-Striebel smoothing
    Smooths a measurement sequence and outputs from a Kalman filter.

    Args:
        filtered_means np.array(): Filtered estimates for times 1,..., K
        filtered_covs np.array(): Filter error covariance
        pred_means np.array(): Predicted estimates for times 1,..., K
        pred_covs np.array(): Filter error covariance
        motion_model (Conditional):

    Returns:
        smooth_means np.array(): Smooth estimates for times 1,..., K
        smooth_covs np.array(): Smooth error covariance
    """

    K, dim_x = filtered_means.shape

    # Allocation
    smooth_means = np.zeros((K, dim_x))
    smooth_covs = np.zeros((K, dim_x, dim_x))
    smooth_mean = filtered_means[-1, :]
    smooth_cov = filtered_covs[-1, :, :]
    smooth_means[-1, :] = smooth_mean
    smooth_covs[-1, :, :] = smooth_cov
    for k in np.flip(np.arange(K - 1)):
        # Run filter iteration
        print("Time step: ", k)
        smooth_mean, smooth_cov = _rts_update(smooth_mean, smooth_cov,
                                              filtered_means[k, :],
                                              filtered_covs[k, :, :],
                                              pred_means[k + 1, :],
                                              pred_covs[k + 1, :, :],
                                              motion_model, num_samples)
        smooth_means[k, :] = smooth_mean
        smooth_covs[k, :, :] = smooth_cov
    return smooth_means, smooth_covs


def _rts_update(xs_kplus1, Ps_kplus1, xf_k, Pf_k, xp_kplus1, Pp_kplus1,
                motion_model: Conditional, num_samples):
    """Non-linear Kalman filter prediction
    calculates mean and covariance of predicted state density
    using a non-linear Gaussian model.

    Args:
        prior_mean np.array(D_x): Prior mean
        prior_cov np.array(D_x, D_x): Prior covariance
        motion_model (Conditional):

    Returns:
       pred_mean np.array(D_x, D_x): predicted state mean
       pred_cov np.array(D_x, D_x): predicted state covariance
    """
    slr = Slr(Gaussian(x_bar=xf_k, P=Pf_k), motion_model)
    A, b, Q = slr.linear_parameters(num_samples)

    G_k = Pf_k @ A.T @ np.linalg.inv(Pp_kplus1)
    xs_k = xf_k + G_k @ (xs_kplus1 - xp_kplus1)
    Ps_k = Pf_k - G_k @ (Ps_kplus1 - Pp_kplus1) @ G_k.T
    return xs_k, Ps_k
