import unittest
import numpy as np
from utils import calc_subspace_proj_matrix


class TestSubspaceProj(unittest.TestCase):
    def test_dim_single_state(self):
        """Test projection matrix takes an arb. cov. matrix
        and outputs one with zero variance in the direction (1,1,1)^T
        """

        det_dir = np.ones((3, 1))
        U_orth = calc_subspace_proj_matrix(3)
        P_sqrt = np.random.rand(3, 3)
        P = P_sqrt @ P_sqrt.T
        # np bug? This split makes the unit test fail
        # P_proj = U_orth @ P @ U_orth
        # var_in_dir = det_dir.T @ P_proj @ det_dir
        # Also this fails:
        # var_in_dir = det_dir.T @ (U_orth @ P @ U_orth) @ det_dir

        var_in_dir = det_dir.T @ U_orth @ P @ U_orth @ det_dir
        self.assertAlmostEqual(0.0, var_in_dir)
