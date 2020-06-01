import unittest
from functools import partial
import numpy as np
from models.range_bearing import RangeBearing


class TestRangeBearing(unittest.TestCase):
    def test_dim_single_state(self):
        pos = np.array([0, 0])
        R = np.eye(2)
        meas_model = RangeBearing(pos, R)
        state = np.ones((5, ))
        mean = meas_model.mean(state)
        self.assertEqual(mean.shape, (2, ))
        # Most important to check that there are no sub-arrays
        self.assertEqual(mean[0], np.sqrt(2))
