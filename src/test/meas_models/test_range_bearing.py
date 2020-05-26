import unittest
from functools import partial
import numpy as np
from post_lin_filt.meas_models.range_bearing import range_bearing_meas


class TestRangeBearing(unittest.TestCase):
    def test_dim_single_state(self):
        pos = np.array([0, 0])
        meas_model = partial(range_bearing_meas, pos=pos)
        state = np.ones((5, ))
        mean, jacobian = meas_model(state)
        self.assertEqual(mean.shape, (2, ))
        self.assertEqual(jacobian.shape, (2, 5))
        # Most important to check that there are no sub-arrays
        self.assertEqual(mean[0], np.sqrt(2))
