"""Motion model interface"""
from abc import ABC, abstractmethod
import numpy as np


class MotionModel(ABC):
    @abstractmethod
    def predict(self, current_state):
        """Return mean and jacobian evaluated at the current state"""
        pass
