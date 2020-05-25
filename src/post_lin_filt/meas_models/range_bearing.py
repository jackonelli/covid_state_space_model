"""Range-bearing measurement model"""
from functools import partial
import numpy as np
from post_lin_filt.utils import gen_dummy_data


def range_bearing_meas(state, pos):
    """Range bearing measurement

    Args:
        state np.array(D_x,)
        pos np.array(D_y,)

    Returns:
        meas_mean np.array(D_y,)
        jacobian np.array(D_y, D_x)
    """

    range_ = np.sqrt(np.sum((state[:2] - pos)**2))
    bearing = np.arctan2(state[1] - pos[1], state[0] - pos[0])
    meas_mean = np.array([range_, bearing])
    jacobian = np.array([[(state[0] - pos[0]) / range_,
                          (state[1] - pos[1]) / range_, 0, 0, 0],
                         [
                             -(state[1] - pos[1]) / (range_**2),
                             (state[0] - pos[0]) / (range_**2), 0, 0, 0
                         ]])

    return meas_mean, jacobian
