"""Range-bearing measurement model"""
import numpy as np
from post_lin_filt.deprecated.meas_models.interface import MeasModel


class RangeBearing:
    """ pos np.array(2,)"""
    def __init__(self, pos):
        self.pos = pos

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
