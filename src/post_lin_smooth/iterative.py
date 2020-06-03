"""Iterative wrapper for the posterior linearization smoother"""
import numpy as np
import matplotlib.pyplot as plt
from post_lin_smooth.slr.distributions import Prior, Conditional
from post_lin_smooth.filtering import slr_kf, slr_kf_known_priors
from post_lin_smooth.smoothing import rts_smoothing
import visualization as vis
from analytics import pos_def_ratio


def iterative_post_lin_smooth(measurements,
                              prior_mean,
                              prior_cov,
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
        prior_mean (D_x,)
        prior_cov (D_x, D_x)
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
                                   prior_mean,
                                   prior_cov,
                                   prior,
                                   motion_model,
                                   meas_model,
                                   num_samples)
    _, ax = plt.subplots()
    print("Pct of smooth covs that are pos def: {}".format(
        pos_def_ratio(smooth_covs)))
    vis.plot_mean_and_cov(ax, filter_means[:, :2], filter_covs[:,:2, :2], 3, "filter", "b", 1)
    vis.plot_mean_and_cov(ax, smooth_means[:, :2], smooth_covs[:,:2, :2], 3, "smooth", "g", 1)
    ax.set_title("Iter: {}".format(1))
    plt.show()
    for iter_ in np.arange(2, num_iterations + 1):
        print("Iter: ", iter_)
        (smooth_means,
         smooth_covs,
         filter_means,
         filter_covs,
         linearizations) = _iteration(measurements,
                                      prior_mean,
                                      prior_cov,
                                      smooth_means,
                                      smooth_covs,
                                      prior,
                                      motion_model,
                                      meas_model,
                                      num_samples)
        _, ax = plt.subplots()
        vis.plot_mean_and_cov(ax,
                              filter_means[:, :2],
                              filter_covs[:,:2, :2],
                              3,
                              "filter",
                              "b",
                              1)
        vis.plot_mean_and_cov(ax,
                              smooth_means[:, :2],
                              smooth_covs[:,:2, :2],
                              3,
                              "smooth",
                              "g",
                              1)
        ax.set_title("Iter: {}".format(iter_))
        plt.show()
    return smooth_means, smooth_covs, filter_means, filter_covs, linearizations


def _first_iter(measurements,
                prior_mean,
                prior_cov,
                prior,
                motion_model: Conditional,
                meas_model: Conditional,
                num_samples: int):
    """First iteration
    Special case since no smooth estimates exist from prev iteration
    Performs KF with SLR, then RTS smoothing.
    """
    filter_means, filter_covs, pred_means, pred_covs, linearizations = slr_kf(
        measurements, prior_mean, prior_cov, prior, motion_model, meas_model,
        num_samples)

    smooth_means, smooth_covs = rts_smoothing(filter_means, filter_covs,
                                              pred_means, pred_covs,
                                              linearizations)
    return smooth_means, smooth_covs, filter_means, filter_covs, linearizations


def _iteration(measurements,
               prior_mean,
               prior_cov,
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
                                           prior_mean,
                                           prior_cov,
                                           prev_smooth_means,
                                           prev_smooth_covs,
                                           prior,
                                           motion_model,
                                           meas_model,
                                           num_samples)

    print("Pct of pred covs that are pos def: {}".format(
        pos_def_ratio(pred_covs)))
    print("Pct of filter covs that are pos def: {}".format(
        pos_def_ratio(filter_covs)))
    smooth_means, smooth_covs = rts_smoothing(filter_means, filter_covs,
                                              pred_means, pred_covs,
                                              linearizations)
    print("Pct of smooth covs that are pos def: {}".format(
        pos_def_ratio(smooth_covs)))
    return smooth_means, smooth_covs, filter_means, filter_covs, linearizations
