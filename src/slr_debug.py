import numpy as np
from scipy.stats import multivariate_normal as mvn
from scipy.linalg import sqrtm
import matplotlib.pyplot as plt
from post_lin_smooth.slr.slr import Slr
from post_lin_smooth.slr.distributions import Conditional, Gaussian
from models.coord_turn import CoordTurn
from models.range_bearing import RangeBearing


def main():
    num_samples = 1000
    sampling_period = 0.1

    v_scale = 0.01
    omega_scale = 1
    sigma_v = v_scale * 1
    sigma_omega = omega_scale * np.pi / 180
    # Prior distr.
    x_0 = np.array([0, 0, 20, 0, 0])
    P_0 = np.diag(
        [10**2, 10**2, 10**2, (5 * np.pi / 180)**2, (1 * np.pi / 180)**2])
    Q = np.diag([
        0, 0, sampling_period * sigma_v**2, 0, sampling_period * sigma_omega**2
    ])
    motion_model = CoordTurn(sampling_period, Q)

    pos = np.array([280, -140])
    sigma_r = 15
    sigma_phi = 4 * np.pi / 180

    R = np.diag([sigma_r**2, sigma_phi**2])
    meas_model = RangeBearing(pos, R)

    slr = Slr(Gaussian(x_bar=x_0, P=P_0), meas_model)
    print("SLR: \nA: {}\nb: {}\nQ: {}".format(
        *slr.linear_parameters(num_samples)))
    x_sample, z_sample = slr._sample(num_samples)

    _, ax = plt.subplots()
    plot_state(ax, x_sample, z_sample)
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
