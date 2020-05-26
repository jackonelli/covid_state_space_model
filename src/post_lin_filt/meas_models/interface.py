"""Measurement model interface"""
from abc import ABC, abstractmethod
import numpy as np


class MeasModel(ABC):
    """Non-linear Kalman filter update.
    Calculates an updated mean and covariance of the state density
    using a non-linear Gaussian model.

    Args:
        prior_mean np.array(D_x,): Prior mean
        prior_cov np.array(D_x, D_x): Prior covariance
        meas_noise_cov np.array(D_y, D_y): Measurement noise covariance
        meas np.array(d,): Measurement

    Returns:
       updated_mean np.array(D_x, D_x): updated state mean
       updated_cov np.array(D_x, D_x): updated state covariance
    """
    @abstractmethod
    def mean_and_jacobian(self, current_state):
        """Return mean and jacobian of the measurement
        evaluated at the current state
        """
        pass
