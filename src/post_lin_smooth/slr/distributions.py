"""Distribution interfaces for SLR"""
from abc import ABC, abstractmethod
import numpy as np
from scipy.stats import multivariate_normal as mvn
from analytics import pos_def_check
from calculations import make_pos_def


class Prior(ABC):
    """Prior distribution p(x)
    This prior should in principle be a Gaussian
    but some modifications might be necessary to fulfill
    constraints in the approximated process.
    """
    @abstractmethod
    def __init__(self, x_bar, P):
        pass

    @abstractmethod
    def sample(self, num_samples):
        pass


class Conditional(ABC):
    """Conditional distribution p(z | x)"""
    @abstractmethod
    def sample(self, x_sample, num_samples: int):
        pass


class Gaussian(Prior):
    """Gaussian distribution"""
    def __init__(self, x_bar, P):
        self.x_bar = x_bar
        self.P = P

    def sample(self, num_samples):
        self.P, sing_vals = make_pos_def(self.P)
        if not pos_def_check(self.P, False):
            print("\n")
            print("Gaussian p(x) cov is pos def: False")
            print("P:", np.round(self.P, decimals=2))
            print("P eig:", np.linalg.eigvals(self.P))
            print("sing vals:", sing_vals)
            print("\n")
        return mvn.rvs(mean=self.x_bar, cov=self.P, size=num_samples)
