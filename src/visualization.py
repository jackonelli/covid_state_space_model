"""Vizualisation"""
import numpy as np
from scipy.linalg import sqrtm
import matplotlib.pyplot as plt
from analytics import nees


def plot_nees_and_2d_est(true_x,
                         meas,
                         xf,
                         Pf,
                         xs,
                         Ps,
                         sigma_level=3,
                         skip_cov=1):
    filter_nees = nees(true_x, xf, Pf)
    smooth_nees = nees(true_x, xs, Ps)

    _, (ax_1, ax_2) = plt.subplots(1, 2)
    ax_1.plot(filter_nees, "-b", label="filter")
    ax_1.plot(smooth_nees, "--g", label="smooth")

    ax_2.plot(true_x[:, 0], true_x[:, 1], ".k", label="true")
    ax_2.plot(meas[:, 0], meas[:, 1], ".r", label="meas")
    plot_mean_and_cov(ax_2, xf[:, :2], Pf[:, :2, :2], sigma_level, "$x_f$",
                      "b", skip_cov)
    plot_mean_and_cov(ax_2, xs[:, :2], Ps[:, :2, :2], sigma_level, "$x_s$",
                      "g", skip_cov)
    ax_1.set_xlabel("k")
    ax_1.set_ylabel(r"$\epsilon_{x, k}$")
    ax_2.set_xlabel("$pos_x$")
    ax_2.set_ylabel("$pos_y$")
    ax_1.legend()
    ax_2.legend()
    plt.show()


def plot_mean_and_cov(ax, means, covs, sigma_level, label, color, skip_cov):
    fmt = "{}-*".format(color)
    ax.plot(means[:, 0], means[:, 1], fmt, label=label)
    for k in np.arange(0, len(means), skip_cov):
        last_handle, = plot_sigma_level(ax, means[k, :], covs[k, :, :],
                                        sigma_level, "", color)
    last_handle.set_label(r"${} \sigma$".format(sigma_level))


def plot_sigma_level(ax, means, covs, level, label, color, resolution=50):
    fmt = "{}--".format(color)
    ellips = ellips_points(means, covs, level, resolution)
    return ax.plot(ellips[:, 0], ellips[:, 1], fmt)


def ellips_points(center, transf, scale, resolution):
    """Transform the circle to the sought ellipse"""
    angles = np.linspace(0, 2 * np.pi, resolution)
    curve_parameter = np.row_stack((np.cos(angles), np.sin(angles)))

    level_sigma_offsets = scale * sqrtm(transf) @ curve_parameter

    return center + level_sigma_offsets.T
