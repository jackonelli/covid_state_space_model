"""PMMH sampler for SEIR model"""
import numpy as np
from smc.bPF import bPF
#from models.seir import SEIR, Param
import models.seir as md
from helpers import *


class Proposal:
    def __init__(self):
        pass

    def sample(self, theta_now: md.Param):
        return theta_now

    def log_pdf(self, theta_now, theta_new):
        return .0


def pmmh_sampler(theta_now, y, numMCMC, model: md.SEIR, numParticles=500):
    """Main PMMH sampling loop"""

    # Initialize/store traces
    theta_log = np.zeros((model.param.dth, numMCMC))
    logZ = np.zeros(numMCMC)
    theta_log[:, 0] = theta_now
    model.param.set(theta_now)

    # Set up proposal
    prop = Proposal()

    # Run PF to get initial likelihood estimate
    pf = bPF(model, y, N=numParticles)
    pf.filter()
    logZ_now = pf.logZ
    logZ[0] = logZ_now

    # Main MCMC loop
    for r in range(1, numMCMC):
        if r % 10 == 0:
            print(f"Iteration {r}/{numMCMC}")

        # Propose
        theta_new = prop.sample(theta_now)
        model.param.set(theta_new)
        pf.filter()
        logZ_new = pf.logZ

        # Accept/reject
        log_acceptance_prob = logZ_new - logZ_now + md.prior_log_pdf(theta_new) - md.prior_log_pdf(
            theta_now) + prop.log_pdf(theta_new, theta_now) - prop.log_pdf(theta_now, theta_new)
        accept = np.random.random() < np.exp(log_acceptance_prob)
        if accept:
            logZ_now = logZ_new
            theta_now = theta_new

        # Store trace
        logZ[r] = logZ_now
        theta_log[:, r] = theta_now
    return theta_log
