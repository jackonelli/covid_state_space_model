"""Range-bearing measurement model"""
import numpy as np
from post_lin_filt.meas_models.interface import MeasModel


class RangeBearing:
    """ pos np.array(2,)"""
    def __init__(self, pos):
        self.pos = pos

    def update(self, prior_mean, prior_cov, meas, meas_noise_cov):
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

        [meas_mean, jacobian] = self.mean_and_jacobian(prior_mean)
        S = jacobian @ prior_cov @ jacobian.T + meas_noise_cov
        K = prior_cov @ jacobian.T @ np.linalg.inv(S)
        updated_mean = prior_mean + K @ (meas - meas_mean)
        updated_cov = prior_cov - K @ S @ K.T
        return updated_mean, updated_cov

    def mean_and_jacobian(self, state):
        """Range bearing measurement

        Args:
            state np.array(D_x,)

        Returns:
            meas_mean np.array(D_y,)
            jacobian np.array(D_y, D_x)
        """

        range_ = np.sqrt(np.sum((state[:2] - self.pos)**2))
        bearing = np.arctan2(state[1] - self.pos[1], state[0] - self.pos[0])
        meas_mean = np.array([range_, bearing])
        jacobian = np.array([[(state[0] - self.pos[0]) / range_,
                              (state[1] - self.pos[1]) / range_, 0, 0, 0],
                             [
                                 -(state[1] - self.pos[1]) / (range_**2),
                                 (state[0] - self.pos[0]) / (range_**2), 0, 0,
                                 0
                             ]])

        return meas_mean, jacobian


def to_cartesian_coords(meas, pos):
    """Maps a range and bearing measurement to cartesian coords

    Args:
        meas np.array(D_y,)
        pos np.array(2,)

    Returns:
        coords np.array(2,)
    """
    delta_x = meas[0] * np.cos(meas[1])
    delta_y = meas[0] * np.sin(meas[1])
    coords = np.array([delta_x, delta_y]) + pos
    return coords
