"""Simple SEIR model - Section 5.6 in Overleaf document"""

import numpy as np
from scipy.stats import norm
from helpers import *


def prior_log_pdf(theta):
    """Computes parameter prior log pdf"""

    # Prior means
    #mu_prior = [logit(1 / 5.1), logit(1 / 5), logit(1 / 1000), 2, logit(0.1), -.12]
    mu_prior = [logit(1 / 5.1), logit(1 / 5), logit(1 / 1000), np.log(0.2), np.log((0.9 + 1) / 2),  np.log(0.1)]  # For RW

    sigma_prior = np.sqrt(np.abs(mu_prior)) * 3  # <-- No idea how wide the priors should be. Do we want them to be informative?

    logZ = np.sum(norm.logpdf(theta, loc=mu_prior, scale=sigma_prior))
    return logZ



class Param:
    def __init__(self, d_param, b_param, pop):
        self.ei = d_param[0]
        self.ir = d_param[1]
        self.ic = d_param[2]
        self.b_scale = b_param[0]
        self.b_corr = b_param[1]
        self.b_std = b_param[2]
        self.dth = 6  # Number of learnable parameters
        self.pop = pop
        self.lag = 7  # Hard coded time lag for observations

    def set(self, theta):
        """This sets the *learnable* parameters"""
        self.ei = logistic(theta[0])
        self.ir = logistic(theta[1])
        self.ic = logistic(theta[2])
        self.b_scale = np.exp(theta[3])
        self.b_corr = logistic(theta[4]) * 2 - 1
        self.b_std = np.exp(theta[5])

    def get(self):
        theta = np.zeros(self.dth)
        theta[0] = logit(self.ei)
        theta[1] = logit(self.ir)
        theta[2] = logit(self.ic)
        theta[3] = np.log(self.b_scale)
        theta[4] = logit((self.b_corr+1)/2)
        theta[5] = np.log(self.b_std)
        return theta


class SEIR:
    def __init__(self, param: Param):
        self.dx = 4  # SEIR, but R is deterministc so we only represent SEI in state vector. Extra state for b
        #self.x_type = np.int64 # Hacky...
        self.x_type = np.float64  # Hacky...
        self.dy = 1  # ICU measurements
        self.init_state = None
        self.param = param

    def log_transition(self, x1, x0):
        """Computes the model log transition p(x_new|x_now) for all values in array x_now"""
        raise NotImplementedError

    def log_lik(self, y, x, time):
        """Computes the model log likelihood p(y|x) for all values in array x"""
        if y.item() is not None:
            return binom.logpmf(y.item(), x[2, :], self.param.ic)
        else:
            return np.zeros((1, x.shape[1]))

    def sample_state(self, x0=None, time=0, N=1):
        """Samples N states from transition dynamics. Returns (dx,N)"""

        if x0 is None:  # Initialize
            if self.init_state is not None:
                x1 = np.array(self.init_state)
                x1 = np.repeat(x1[:, None], N, axis=1)

            else:
                x1 = np.zeros((self.dx, N), dtype=np.int64)
                i0 = 400
                e0 = 400
                r0 = 1000
                s0 = self.param.pop - i0 - e0 - r0
                # x1[0, :] = binom_by_normal(self.param.pop, s0 / self.param.pop, N=N)
                # x1[1, :] = binom_by_normal(self.param.pop, e0 / self.param.pop, N=N)
                # x1[2, :] = binom_by_normal(self.param.pop, i0 / self.param.pop, N=N)

                x1[0, :] = np.maximum(0, np.random.normal(loc=s0, scale=np.sqrt(s0)/3, size=N)) #.astype(np.int64)
                x1[1, :] = np.maximum(0, np.random.normal(loc=e0, scale=np.sqrt(e0)/3, size=N)) #.astype(np.int64)
                x1[2, :] = np.maximum(0, np.random.normal(loc=i0, scale=np.sqrt(i0)/3, size=N)) #.astype(np.int64)
                # Initial state for b[0] - chosen quite arbitrary here (as all initial states)
                x1[3, :] = np.random.normal(loc=np.log(1.6/self.param.b_scale), scale=0.1, size=N)


        else:  # Propagate

            #b = b_val_FHM(self.param, time)
            b = self.param.b_scale * np.exp(x0[3,:]) # Stochastic b(t), part of state
            de = binom_by_normal(x0[0, :] * x0[2, :], b / self.param.pop, N)
            di = binom_by_normal(x0[1, :], self.param.ei, N)
            dr = binom_by_normal(x0[2, :], self.param.ir, N)
            x1 = x0 + [-de, de - di, di - dr, np.zeros(N)]  # Last zero is just to get the correct size, it will be overwritten below. Seems like it has to be an array for broadcasting to work?!
            # Random walk for b
            x1[3,:] = np.random.normal(loc = self.param.b_corr*x0[3,:], scale=self.param.b_std, size=N)

        return x1

    def sample_obs(self, x, time, N=1):
        y = binom_by_normal(x[2, :], self.param.ic, N)
        return y

    def simulate(self, T, N=1):
        """We work with 3d arrays for the states/obs (d,T,N)"""
        x = np.zeros((self.dx, T, N), dtype=self.x_type)
        y = np.full((self.dy, T, N), None)

        for t in range(T):
            if t == 0:
                x[:, 0, :] = self.sample_state(N=N)
            else:
                x[:, t, :] = self.sample_state(x[:, t - 1, :], t, N=N)

            if t >= self.param.lag:
                y[:, t, :] = self.sample_obs(x[:, t - self.param.lag, :], t, N=N)

        return x, y

