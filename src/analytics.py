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
    nees = np.zeros((K, 1))
    for k in np.arange(K):
        err_k = err[k, :].reshape((D_x, 1))
        P_inv_k = np.linalg.inv(cov[k, :, :])
        nees[k] = err_k.T @ P_inv_k @ err_k
    return nees
