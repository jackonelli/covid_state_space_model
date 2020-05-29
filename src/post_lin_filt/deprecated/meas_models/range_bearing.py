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
