"""SLR filter type"""
from abc import ABC, abstractmethod
import numpy as np
from scipy.stats import multivariate_normal as mvn
from post_lin_filt.filter_type.interface import FilterType
from post_lin_filt.motion_models.interface import MotionModel
from post_lin_filt.meas_models.interface import MeasModel
from post_lin_filt.slr.distributions import Conditional, Gaussian
from post_lin_filt.slr.slr import Slr


class SlrFilter(FilterType):
    def __init__(self, motion_model: Conditional, meas_model: Conditional,
                 num_samples: int):
        self.motion_model = motion_model
        self.meas_model = meas_model
        self.num_samples = num_samples

    def predict(self, prior_mean, prior_cov):
        slr = Slr(Gaussian(x_bar=prior_mean, P=prior_cov), self.motion_model)
        A, b, Q = slr.linear_parameters(self.num_samples)
        pred_mean = A @ prior_mean + b
        pred_cov = A @ prior_cov @ A.T + Q
        return pred_mean, pred_cov

    def update(self, prior_mean, prior_cov, meas):
        print("Update")
        slr = Slr(Gaussian(x_bar=prior_mean, P=prior_cov), self.meas_model)
        H, c, R = slr.linear_parameters(self.num_samples)
        meas_mean = H @ prior_mean + c
        print("prior", prior_mean)
        print("H", H)
        S = H @ prior_cov @ H.T + R
        print("S", S)
        K = prior_cov @ H.T @ np.linalg.inv(S)
        updated_mean = prior_mean + K @ (meas - meas_mean)
        updated_cov = prior_cov - K @ S @ K.T
        return updated_mean, updated_cov
