"""Simple SEIR model - Section 5.6 in Overleaf document"""

import numpy as np
from scipy.stats import norm
from helpers import *


def b_val_FHM(params, time):
    b_par = params.bp
    theta = b_par[0]
    delta = b_par[1]
    eps = b_par[2]
    t_offset = b_par[3]
    b = theta * (delta + (1 - delta) / (1 + np.exp(-eps * (time + t_offset))))
    return b


def prior_log_pdf(theta):
    """Computes parameter prior log pdf"""
    return 0.


class Param:
    def __init__(self, d_param, b_param, pop):
        self.ei = d_param[0]
        self.ir = d_param[1]
        self.ic = d_param[2]
        self.bp = b_param # 4d
        self.dth = 7  # Number of learnable parameters
        self.pop = pop
        self.lag = 7  # Hard coded time lag for observations

    def set(self, theta):
        self.ei = theta[0]
        self.ir = theta[1]
        self.ic = theta[2]
        self.bp = theta[3:]

    def get(self):
        theta = np.zeros(self.dth)
        theta[0] = self.ei
        theta[1] = self.ir
        theta[2] = self.ic
        theta[3:] = self.bp
        return theta


class SEIR:
    def __init__(self, param: Param):
        self.dx = 3  # SEIR, but R is deterministc so we only represent SEI in state vector
        self.x_type = np.int64 # Hacky...
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

                x1[0, :] = np.maximum(0, np.random.normal(loc=s0, scale=np.sqrt(s0)/3, size=N)).astype(np.int64)
                x1[1, :] = np.maximum(0, np.random.normal(loc=e0, scale=np.sqrt(s0)/3, size=N)).astype(np.int64)
                x1[2, :] = np.maximum(0, np.random.normal(loc=i0, scale=np.sqrt(s0)/3, size=N)).astype(np.int64)


        else:  # Propagate
            b = b_val_FHM(self.param, time)
            de = binom_by_normal(x0[0, :] * x0[2, :], b / self.param.pop, N)
            di = binom_by_normal(x0[1, :], self.param.ei, N)
            dr = binom_by_normal(x0[2, :], self.param.ir, N)
            x1 = x0 + [-de, de - di, di - dr]
        return x1

    def sample_obs(self, x, time, N=1):
        y = binom_by_normal(x[2, :], self.param.ic, N)
        return y

    def simulate(self, T, N=1):
        """We work with 3d arrays for the states/obs (d,T,N)"""
        x = np.zeros((self.dx, T, N), dtype=np.int64)
        y = np.full((self.dy, T, N), None)

        for t in range(T):
            if t == 0:
                x[:, 0, :] = self.sample_state(N=N)
            else:
                x[:, t, :] = self.sample_state(x[:, t - 1, :], t, N=N)

            if t >= self.param.lag:
                y[:, t, :] = self.sample_obs(x[:, t - self.param.lag, :], t, N=N)

        return x, y

