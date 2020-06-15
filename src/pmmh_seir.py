"""PMMH sampler for SEIR model"""
import numpy as np
from smc.bPF import bPF
#from models.seir import SEIR, Param
import models.seir as md
from helpers import *


class Proposal:
    def __init__(self):
        # Prior means
        mu_prior = [logit(1 / 5.1), logit(1 / 5), logit(1 / 1000), 2, logit(0.1), -.12]
        self.Sigma = np.diag(np.sqrt(np.abs(mu_prior)) * 0.001)  # Independent RW

    def sample(self, theta_now: md.Param):
        sample_these_inds = [0,1,2]
        theta_new = np.copy(theta_now)
        theta_new[sample_these_inds] = np.random.multivariate_normal(theta_now[sample_these_inds], cov=self.Sigma[sample_these_inds,:][:,sample_these_inds])
        return theta_new

    def log_pdf_ratio(self, theta_now, theta_new):
        """Implement ratio directly. Zero for symmetric random walk."""
        return .0


def pmmh_sampler(theta_now, y, numMCMC, model: md.SEIR, numParticles=500):
    """Main PMMH sampling loop"""

    # Initialize/store traces
    theta_log = np.zeros((model.param.dth, numMCMC))
    logZ_log = np.zeros(numMCMC)
    theta_log[:, 0] = theta_now
    model.param.set(theta_now)

    # Diagnostics
    accept_log = np.zeros(numMCMC)

    # Set up proposal
    prop = Proposal()

    # Run PF to get initial likelihood estimate
    pf = bPF(model, y, N=numParticles)
    pf.filter()
    logZ_now = pf.logZ
    logZ_log[0] = logZ_now

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
            theta_now) + prop.log_pdf_ratio(theta_new, theta_now)
        accept = np.random.random() < np.exp(log_acceptance_prob)
        accept_log[r] = np.min([1, np.exp(log_acceptance_prob)])
        if accept:
            logZ_now = logZ_new
            theta_now = theta_new

        # Store trace
        logZ_log[r] = logZ_now
        theta_log[:, r] = theta_now
    return theta_log, logZ_log, accept_log
