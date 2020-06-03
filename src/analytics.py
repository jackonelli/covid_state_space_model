"""Analytics"""
import numpy as np


def nees(true, est, cov):
    """Calculate NEES
    Normalized estimation error squared (NEES) for all timesteps

    eps_k = e_k^T P_k^-1 e_k

    Args:
        true (K, D_x)
        est (K, D_x)
        cov (K, D_x, D_x)

    Returns:
        nees (K, 1)
    """
    K, D_x = true.shape
    err = (true - est)
    nees = np.empty((K, 1))
    for k, (err_k, cov_k) in enumerate(zip(err, cov)):
        err_k = err_k.reshape((D_x, 1))
        nees[k] = _single_nees(err_k, cov_k)
    return nees


def _single_nees(err, cov):
    return err.T @ np.linalg.inv(cov) @ err


def _is_pos_def(x):
    return np.all(np.linalg.eigvals(x) > 0)


def pos_def_check(x, disabled=True):
    return _is_pos_def(x) or disabled


def pos_def_ratio(covs):
    pos_defs = [pos_def_check(cov, False) for cov in covs]
    return sum(pos_defs) / len(pos_defs)
