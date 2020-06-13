"""Kalman filter and smoother"""
import numpy as np
from models.lgss1d import LGSS1d


def rdiv(B, A):
    """Computes B * A^{-1}"""
    return np.linalg.solve(A.T, B.T).T


def ldiv(A, B):
    """Computes A^{-1} * B"""
    return np.linalg.solve(A, B)


class KFS:
    def __init__(self, model: LGSS1d, y):
        self.model = model
        dx = model.dx
        dy = model.dy
        # Handle scalar data
        if y.ndim == 1:
            if dy != 1:
                raise Exception("Observation dimension mismatch")
            y = y[np.newaxis, :]

        self.dx = dx  # state dimension
        self.dy = dy  # observation dimension
        self.y = y  # Data

        #self.t = 0  # Time pointer
        self.T = y.shape[1]  # Second dimension is always time

        # Filter and 1-step predictor
        self.x_filt = np.zeros((dx, self.T))
        self.P_filt = np.zeros((dx, dx, self.T))
        self.x_pred = np.zeros((dx, self.T))
        self.P_pred = np.zeros((dx, dx, self.T))
        # Innovations
        self.y_pred = np.zeros((dy, self.T))
        self.S = np.zeros((dy, dy, self.T))

    def filter(self):
        # Get the model parameters and convert to numpy arrays of dimension 2
        A, Q, C, R = map(np.atleast_2d, self.model.get_params())

        for t in range(self.T):

            # Time update (predict)
            if t == 0:
                # Initialize predictions
                self.x_pred[:, 0] = self.model.mu1
                self.P_pred[:, :, 0] = self.model.P1
            else:
                self.x_pred[:, t] = A @ self.x_filt[:, t - 1]
                self.P_pred[:, :, t] = A @ self.P_filt[:, :, t - 1] @ A.T + Q

            # Compute prediction of current output (used for likelihood computation)
            self.y_pred[:, t] = C @ self.x_pred[:, t]
            self.S[:, :, t] = C @ self.P_pred[:, :, t] @ C.T + R

            # Measurement update (correct)
            K = rdiv(self.P_pred[:, :, t] @ C.T, self.S[:, :, t])
            self.x_filt[:, t] = self.x_pred[:, t] + K @ (self.y[:, t] - self.y_pred[:, t])
            self.P_filt[:, :, t] = (np.identity(self.dx) - K @ C) @ self.P_pred[:, :, t]

    def backward_simulator(self,M=1):
        """Samples from joint smoothing distribution"""
        A, Q, C, R = map(np.atleast_2d, self.model.get_params())

        X = np.zeros((self.dx, self.T, M))

        # Initialize
        X[:,-1,:] = np.random.multivariate_normal(mean=self.x_filt[:,-1], cov=self.P_filt[:,:,-1],size=M).T

        # Loop backward
        for t in range(self.T-1,0,-1): # T-1, ..., 1
            L = rdiv(self.P_filt[:,:,t-1] @ A.T,   Q + A @ self.P_filt[:,:,t-1] @ A.T)
            mu = self.x_filt[:,t-1] + L @ (X[:,t,:] - A @ self.x_filt[:,t-1])
            Sigma = self.P_filt[:,:,t-1] - L @ A @ self.P_filt[:,:,t-1]
            X[:,t-1,:] = mu + np.random.multivariate_normal(mean=np.zeros(self.dx), cov=Sigma, size=M).T

        # Better way to remove trailing singular dimension?
        if M==1:
            X = X.squeeze(axis=2)

        return X

    # def initialize_filter(self):
    #     self.x_pred[:,0] = self.model.mu1
    #     self.P_pred[:,:,0] = self.model.P1
    #     self.t = 0 # Time pointer
    #
    # def update_filter(self):
    #     self.compute_innovation()
    #
    # def compute_innovation(self):
    #     if self.dx==1:
    #         self.yerr[self.t] = self.y[self.t] - self.model.C * self.x_pred[:,self.t]
    #     else:
    #         self.yerr[self.t] = self.y[self.t] - self.model.C @ self.x_pred[:, self.t]
