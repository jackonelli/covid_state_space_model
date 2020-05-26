"""Filter type interface"""
from abc import ABC, abstractmethod


class FilterType(ABC):
    @abstractmethod
    def predict(self, prior_mean, prior_cov, motion_model, process_noise_cov):
        pass

    @abstractmethod
    def update(self, prior_mean, prior_cov, meas, meas_model, meas_noise_cov):
        pass
