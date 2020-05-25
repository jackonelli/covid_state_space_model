import numpy as np


def coord_turn_motion(state, sampling_period):
    v = state[2]
    phi = state[3]
    omega = state[4]

    mean = state + np.array([
        sampling_period * v * np.cos(phi), sampling_period * v * np.sin(phi),
        0, sampling_period * omega, 0
    ])

    jacobian = np.array([[
        1, 0, sampling_period * np.cos(phi),
        -sampling_period * v * np.sin(phi), 0
    ],
                         [
                             0, 1, sampling_period * np.sin(phi),
                             sampling_period * v * np.cos(phi), 0
                         ], [0, 0, 1, 0, 0], [0, 0, 0, 1, sampling_period],
                         [0, 0, 0, 0, 1]])
    return mean, jacobian
