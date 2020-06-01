import numpy as np
from scipy.stats import multivariate_normal as mvn
from scipy.linalg import sqrtm
import matplotlib.pyplot as plt
from post_lin_filt.slr.slr import Slr
from models.affine import Affine
from post_lin_filt.slr.distributions import Conditional, Gaussian


def main():
    num_samples = 1000
    prior_mean = np.ones((2, ))
    prior_cov = np.eye(2)
    A = np.array([[2, -0.5], [-0.5, 1]])
    b = 3 * np.ones((2, ))
    Q = np.array([[0, 0], [0, 1.5]])
    motion_model = Affine(A, b, Q)
    slr = Slr(Gaussian(x_bar=prior_mean, P=prior_cov), motion_model)
    print("SLR: \nA: {}\nb: {}\nQ: {}".format(
        *slr.linear_parameters(num_samples)))
    x_sample, z_sample = slr._sample(num_samples)

    post_mean = A @ prior_mean + b
    post_cov = A @ prior_cov @ A.T + Q
    _, ax = plt.subplots()
    res = 100
    sigma_level = 1
    plot_state(ax, x_sample, z_sample)
    plot_sigma_level(ax,
                     post_mean,
                     post_cov,
                     sigma_level,
                     res,
                     label="Analytical")
    A_hat, b_hat, Q_hat = slr.linear_parameters(num_samples)
    post_mean_hat = A_hat @ prior_mean + b_hat
    post_cov_hat = A_hat @ prior_cov @ A_hat.T + Q_hat
    plot_sigma_level(ax,
                     post_mean_hat,
                     post_cov_hat,
                     sigma_level,
                     res,
                     label="SLR")
    ax.legend()
    plt.show()


def plot_state(ax, x_sample, z_sample):
    ax.plot(x_sample[:, 0], x_sample[:, 1], "b*")
    ax.plot(z_sample[:, 0], z_sample[:, 1], "r*")


def plot_sigma_level(ax, mean, cov, level, resolution, label):
    ellips = ellips_points(mean, cov, level, resolution)
    ax.plot(ellips[:, 0],
            ellips[:, 1],
            "--",
            label=r"{} (${} \sigma$)".format(label, level))


def ellips_points(center, transf, scale, resolution):
    """Transform the circle to the sought ellipse"""
    angles = np.linspace(0, 2 * np.pi, resolution)
    curve_parameter = np.row_stack((np.cos(angles), np.sin(angles)))

    level_sigma_offsets = scale * sqrtm(transf) @ curve_parameter

    return center + level_sigma_offsets.T


if __name__ == "__main__":
    main()
