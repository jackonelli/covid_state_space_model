"""Iterative wrapper for the posterior linearization smoother"""
import numpy as np
from post_lin_smooth.slr.distributions import Conditional
from post_lin_smooth.filtering import slr_kf, slr_kf_known_priors
from post_lin_smooth.smoothing import rts_smoothing


def iterative_post_lin_smooth(measurements, prior_mean, prior_cov,
                              motion_model: Conditional,
                              meas_model: Conditional, num_samples: int,
                              num_iterations: int):
    print("Iter: ", 1)
    smooth_means, smooth_covs = _first_iter(measurements, prior_mean,
                                            prior_cov, motion_model,
                                            meas_model, num_samples)
    for iter_ in np.arange(2, num_iterations):
        print("Iter: ", iter_)
        smooth_means, smooth_covs = _iteration(measurements, smooth_means,
                                               smooth_covs, motion_model,
                                               meas_model, num_samples)
    return smooth_means, smooth_covs


def _first_iter(measurements, prior_mean, prior_cov, motion_model: Conditional,
                meas_model: Conditional, num_samples: int):
    filter_means, filter_covs, pred_means, pred_covs, linearizations = slr_kf(
        measurements, prior_mean, prior_cov, motion_model, meas_model,
        num_samples)

    smooth_means, smooth_covs = rts_smoothing(filter_means, filter_covs,
                                              pred_means, pred_covs,
                                              linearizations)
    return smooth_means, smooth_covs


def _iteration(measurements, prev_smooth_means, prev_smooth_covs,
               motion_model: Conditional, meas_model: Conditional,
               num_samples: int):
    (filter_means, filter_covs, pred_means, pred_covs,
     linearizations) = slr_kf_known_priors(measurements, prev_smooth_means,
                                           prev_smooth_covs, motion_model,
                                           meas_model, num_samples)

    smooth_means, smooth_covs = rts_smoothing(filter_means, filter_covs,
                                              pred_means, pred_covs,
                                              linearizations)
    return smooth_means, smooth_covs
