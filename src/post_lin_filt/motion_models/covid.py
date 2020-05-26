from post_lin_filt.motion_models.interface import MotionModel
import numpy as np


class Covid(MotionModel):
    def __init__(self):

    def predict(self, state):
        v = state[2]
        phi = state[3]
        omega = state[4]

        mean = state + np.array([
            self.sampling_period * v * np.cos(phi), self.sampling_period * v *
            np.sin(phi), 0, self.sampling_period * omega, 0
        ])

        jacobian = np.array([[
            1, 0, self.sampling_period * np.cos(phi),
            -self.sampling_period * v * np.sin(phi), 0
        ],
                             [
                                 0, 1, self.sampling_period * np.sin(phi),
                                 self.sampling_period * v * np.cos(phi), 0
                             ], [0, 0, 1, 0, 0],
                             [0, 0, 0, 1, self.sampling_period],
                             [0, 0, 0, 0, 1]])
        return mean, jacobian
