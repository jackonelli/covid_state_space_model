"""EKF filter type"""
import numpy as np
from post_lin_filt.deprecated.filter_type.interface import FilterType
from post_lin_filt.deprecated.motion_models.interface import MotionModel
from post_lin_filt.deprecated.meas_models.interface import MeasModel


class Ekf(FilterType):
    def predict(self, prior_mean, prior_cov, motion_model: MotionModel,
                process_noise_cov):
        pred_mean, jacobian = motion_model.mean_and_jacobian(prior_mean)
        pred_cov = jacobian @ prior_cov @ jacobian.T + process_noise_cov
        return pred_mean, pred_cov

    def update(self, prior_mean, prior_cov, meas, meas_model: MeasModel,
               meas_noise_cov):
        [meas_mean, jacobian] = meas_model.mean_and_jacobian(prior_mean)
        S = jacobian @ prior_cov @ jacobian.T + meas_noise_cov
        K = prior_cov @ jacobian.T @ np.linalg.inv(S)
        updated_mean = prior_mean + K @ (meas - meas_mean)
        updated_cov = prior_cov - K @ S @ K.T
        return updated_mean, updated_cov
