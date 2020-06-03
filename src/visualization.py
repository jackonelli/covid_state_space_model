"""Vizualisation"""
import numpy as np
from scipy.linalg import sqrtm
import matplotlib.pyplot as plt
from post_lin_smooth.analytics import nees


def plot_nees_comp(true_x, x_1, P_1, x_2, P_2):
    nees_1 = nees(true_x, x_1, P_1)
    nees_2 = nees(true_x, x_2, P_2)
    _, ax = plt.subplots()
    ax.plot(nees_1, "-b", label="kf")
    ax.plot(nees_2, "--g", label="slr")
    plt.show()


def plot_nees_and_2d_est(true_x,
                         meas,
                         xf,
                         Pf,
                         xs,
                         Ps,
                         sigma_level=3,
                         skip_cov=1):
    filter_nees = nees(true_x, xf[1:, :], Pf[1:, :, :])
    smooth_nees = nees(true_x, xs[1:, :], Ps[1:, :, :])
    print("Filter NEES avg: {}".format(filter_nees.mean()))
    print("Smooth NEES avg: {}".format(smooth_nees.mean()))

    _, (ax_1, ax_2) = plt.subplots(1, 2)
    ax_1.plot(filter_nees, "-b", label="filter")
    ax_1.plot(smooth_nees, "--g", label="smooth")
    K, D_x = true_x.shape
    ax_1.plot([0, K], [D_x, D_x], "--k", label="ref")

    ax_2.plot(true_x[:, 0], true_x[:, 1], ".k", label="true")
    ax_2.plot(meas[:, 0], meas[:, 1], ".r", label="meas")
    plot_mean_and_cov(ax_2, xf[:, :2], Pf[:, :2, :2], sigma_level, "$x_f$",
                      "b", skip_cov)
    plot_mean_and_cov(ax_2, xs[:, :2], Ps[:, :2, :2], sigma_level, "$x_s$",
                      "g", skip_cov)

    ax_1.set_title("NEES")
    ax_1.set_xlabel("k")
    ax_1.set_ylabel(r"$\epsilon_{x, k}$")
    ax_1.legend()

    ax_2.set_title("Estimates")
    ax_2.set_xlabel("$pos_x$")
    ax_2.set_ylabel("$pos_y$")
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
