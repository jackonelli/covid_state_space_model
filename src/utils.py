import numpy as np
from scipy.stats import multivariate_normal as mvn
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def calc_subspace_proj_matrix(num_dim: int):
    det_dir = np.ones((num_dim, 1))
    Q, _ = np.linalg.qr(det_dir, mode="complete")
    U = Q[:, 1:]
    return U @ U.T


if __name__ == "__main__":
    num_dims = 3
    P_sqrt = np.random.rand(num_dims, num_dims)
    P = P_sqrt @ P_sqrt.T
    U_orth = calc_subspace_proj_matrix(num_dims)
    P_proj = U_orth @ P @ U_orth
    x = mvn.rvs(np.array([1, 0, 0]), P_proj, 1000)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x[:, 0], x[:, 1], x[:, 2])
    print(x.sum(1))
    plt.show()
