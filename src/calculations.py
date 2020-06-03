import numpy as np


def make_pos_def(x, eps=0.1):
    u, s, vh = np.linalg.svd(x, hermitian=True)
    neg_sing_vals = s < 0
    s_hat = s * np.logical_not(neg_sing_vals) + eps * neg_sing_vals
    return np.dot(u, np.dot(np.diag(s_hat), vh)), s
