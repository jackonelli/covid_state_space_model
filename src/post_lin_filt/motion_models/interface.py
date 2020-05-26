"""Motion model interface"""
from abc import ABC, abstractmethod
import numpy as np


class MotionModel(ABC):
    """Non-linear Kalman filter prediction
    calculates mean and covariance of predicted state density
    using a non-linear Gaussian model.

    Args:
        prior_mean np.array(D_x): Prior mean
        prior_cov np.array(D_x, D_x): Prior covariance
        motion_model: Motion model function handle
                      Takes as input x (state)
                      Returns predicted mean and Jacobian evaluated at x
       process_noise_cov np.array(D_x, D_x): Process noise covariance

    Returns:
       pred_mean np.array(D_x, D_x): predicted state mean
       pred_cov np.array(D_x, D_x): predicted state covariance
    """
    @abstractmethod
    def mean_and_jacobian(self, prior_mean, prior_cov,
                          process_noise_covcurrent_state):
        """Predict next mean and covariance given the current mean and cov"""
        pass
