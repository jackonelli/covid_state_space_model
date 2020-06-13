"""Gibbs sampler for LGSS1d with unknown noise variances"""
import numpy as np
from kalman.kfs import KFS
from smc.bPF import bPF
from models.lgss1d import LGSS1d
from helpers import *


def sample_parameter_posterior(model: LGSS1d, x, y):
    """Helper function to sample from the parameter complete data posterior"""
    T = y.shape[1]  # Second dimension is always time

    # Process noise variance
    err_Q = x[:, 1:] - model.A * x[:, :-1]
    Q = igamrnd(model.aQ + (T - 1) / 2, model.bQ + np.sum(err_Q ** 2) / 2)

    # Measurement noise variance
    err_R = y - model.C * x
    R = igamrnd(model.aR + T / 2, model.bR + np.sum(err_R ** 2) / 2)

    return Q, R


def gibbs_sampler(theta_init, y, numIter, model : LGSS1d, statesampler="Kalman", numParticles=None):
    """Main Gibbs sampling loop"""

    # Handle scalar data
    if y.ndim == 1:
        y = y[np.newaxis, :]

    # Initialize/store traces
    theta_log = np.zeros((2,numIter))
    theta_log[:,0] = theta_init
    model.Q = theta_init[0]
    model.R = theta_init[1]

    # Sample state trajectory
    if statesampler=="Kalman":
        kf = KFS(model,y)
        kf.filter()
        X = kf.backward_simulator(1)
    elif statesampler=="PGAS":
        pf = bPF(model, y, N=numParticles)
        pf.filter()
        X = pf.sample_trajectory()
    else:
        raise Exception("Unknown state sampler.")

    for r in range(1,numIter):
        if r % 10 == 0:
            print(f"Iteration {r}/{numIter}")

        # Sample new parameters
        Q, R = sample_parameter_posterior(model, X, y)
        # This will update the model in the state inference object as well!
        model.Q = Q
        model.R = R

        # Store trace
        theta_log[:, r] = [Q,R]

        # Sample state trajectory
        if statesampler == "Kalman":
            kf.filter()
            X = kf.backward_simulator(1)
        elif statesampler == "PGAS":
            pf.filter(X,ancestor_sampling=True)
            X = pf.sample_trajectory()
        else:
            raise Exception("Unkown state sampler.")

    return theta_log

