"""Measurement model interface"""
from abc import ABC, abstractmethod
import numpy as np


class MeasModel(ABC):
    @abstractmethod
    def update(self, current_state):
        """Return mean and jacobian of the measurement
        evaluated at the current state
        """
        pass
