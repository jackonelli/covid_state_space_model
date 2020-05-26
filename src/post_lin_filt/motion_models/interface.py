"""Motion model interface"""
from abc import ABC, abstractmethod
import numpy as np


class MotionModel(ABC):
    @abstractmethod
    def predict(self, prior_mean, prior_cov, process_noise_covcurrent_state):
        """Predict next mean and covariance given the current mean and cov"""
        pass
