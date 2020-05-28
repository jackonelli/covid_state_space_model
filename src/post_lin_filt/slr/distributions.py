"""Distribution interfaces for SLR"""
from abc import ABC, abstractmethod
from scipy.stats import multivariate_normal as mvn


class Gaussian:
    """Gaussian distribution"""
    def __init__(self, x_bar, P):
        self.x_bar = x_bar
        self.P = P

    def sample(self, num_samples):
        print(self.x_bar.shape)
        return mvn.rvs(mean=self.x_bar, cov=self.P, size=num_samples)


class Conditional(ABC):
    """Conditional distribution
    p(z | x)
    """
    @abstractmethod
    def sample(self, x_sample, num_samples: int):
        pass
