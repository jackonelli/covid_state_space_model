"""Iterative wrapper for the posterior linearization smoother"""
import numpy as np
import matplotlib.pyplot as plt
from post_lin_smooth.slr.distributions import Prior, Conditional
from post_lin_smooth.filtering import slr_kf, slr_kf_known_priors
from post_lin_smooth.smoothing import rts_smoothing
# Debug imports
from functools import partial
from models.range_bearing import to_cartesian_coords
import visualization as vis
from analytics import pos_def_ratio


def iterative_post_lin_smooth(measurements,
                              x_0_0,
                              P_0_0,
                              prior: Prior,
                              motion_model: Conditional,
                              meas_model: Conditional,
                              num_samples: int,
                              num_iterations: int):
    """Iterative posterior linearization smoothing
    First iteration performs Kalman filtering with SLR and RTS smoothing
    Subsequent iterations use smooth estimates from previous iteration
    in the linearization.

    TODO: Remove linearization as an output. Only used for debug.

    Args:
        measurements (K, D_y)
        x_0_0 (D_x,): Prior mean for time 0
        P_0_0 (D_x, D_x): Prior covariance for time 0
        prior: p(x) used in SLR.
            Note that this is given as a class prototype,
            it is instantiated multiple times in the function
        motion_model: p(x_k+1 | x_k) used in SLR
        meas_model: p(y_k | x_k) used in SLR
        num_samples
        num_iterations
    """
    print("Iter: ", 1)
    (smooth_means,
     smooth_covs,
     filter_means,
     filter_covs,
     linearizations) = _first_iter(measurements,
                                   x_0_0,
                                   P_0_0,
                                   prior,
                                   motion_model,
                                   meas_model,
                                   num_samples)
    for iter_ in np.arange(2, num_iterations + 1):
        print("Iter: ", iter_)
        (smooth_means,
         smooth_covs,
         filter_means,
         filter_covs,
         linearizations) = _iteration(measurements,
                                      x_0_0,
                                      P_0_0,
                                      smooth_means,
                                      smooth_covs,
                                      prior,
                                      motion_model,
                                      meas_model,
                                      num_samples)
    return smooth_means, smooth_covs, filter_means, filter_covs, linearizations


def _first_iter(measurements,
                x_0_0,
                P_0_0,
                prior,
                motion_model: Conditional,
                meas_model: Conditional,
                num_samples: int):
    """First iteration
    Special case since no smooth estimates exist from prev iteration
    Performs KF with SLR, then RTS smoothing.
    """
    filter_means, filter_covs, pred_means, pred_covs, linearizations = slr_kf(
        measurements, x_0_0, P_0_0, prior, motion_model, meas_model,
        num_samples)
    smooth_means, smooth_covs = rts_smoothing(filter_means, filter_covs,
                                              pred_means, pred_covs,
                                              linearizations)
    return smooth_means, smooth_covs, filter_means, filter_covs, linearizations


def _iteration(measurements,
               x_0_0,
               P_0_0,
               prev_smooth_means,
               prev_smooth_covs,
               prior: Prior,
               motion_model: Conditional,
               meas_model: Conditional,
               num_samples: int):
    """General non-first iteration
    Performs KF but uses smooth estimates from prev iteration as priors in
    the filtering.
    Standard RTS
    """
    (filter_means,
     filter_covs,
     pred_means,
     pred_covs,
     linearizations) = slr_kf_known_priors(measurements,
                                           x_0_0,
                                           P_0_0,
                                           prev_smooth_means,
                                           prev_smooth_covs,
                                           prior,
                                           motion_model,
                                           meas_model,
                                           num_samples)

    smooth_means, smooth_covs = rts_smoothing(filter_means, filter_covs,
                                              pred_means, pred_covs,
                                              linearizations)
    return smooth_means, smooth_covs, filter_means, filter_covs, linearizations
