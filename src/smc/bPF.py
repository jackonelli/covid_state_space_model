"""Bootstrap particle filter"""
import numpy as np
from models.lgss1d import LGSS1d


def exp_norm(logW):
    const = np.max(logW)
    W = np.exp(logW - const)
    sumofweights = np.sum(W)
    logZ = np.log(sumofweights) + const
    W = W / sumofweights
    return W, logZ


class bPF:
    def __init__(self, model, y, N: int):
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

        self.N = N  # Number of particles
        #self.t = 0  # Time pointer
        self.T = y.shape[1]  # Second dimension is always time
        self.X = np.zeros((dx, self.T, self.N), dtype=model.x_type)  # Particles
        self.ancestor_indices = np.zeros((self.T, self.N), dtype=int)  # Store all ancestor indices for tracing genealogy
        self.logW = np.zeros((self.T, self.N))  # Unnormalized log-weight
        self.W = np.zeros((self.T, self.N))  # Normalized weight
        self.x_filt = np.zeros((dx, self.T))  # Store filter mean
        self.N_eff = np.zeros(self.T) # Efficient number of particles
        self.logZ = 0. # Log-likelihood

    def filter(self, X_ref=None, ancestor_sampling=None):
        for t in range(self.T):
            # Sample from bootstrap proposal
            if t == 0:  # Initialize from prior
                self.X[:, 0, :] = self.model.sample_state(N=self.N)
            else:  # Resample and propagate according to dynamics
                ind = np.random.choice(self.N, self.N, replace=True, p=self.W[t-1, :])

                # Conditioning
                if X_ref is not None:
                    if ancestor_sampling:
                        logW_as = self.logW[t-1, :] + self.model.log_transition(X_ref[:,t], self.X[:, t-1, :]).squeeze()
                        W_as, dummy = exp_norm(logW_as)
                        ind[0] = np.random.choice(self.N, 1, p=W_as)
                    else:
                        ind[0] = 0

                self.ancestor_indices[t, :] = ind
                resampledX = self.X[:, t-1, ind]
                self.X[:, t, :] = self.model.sample_state(resampledX, time=t, N=self.N)

            # Conditioning
            if X_ref is not None:
                self.X[:, t, 0] = X_ref[:, t]

            # Compute weights
            self.logW[t, :] = self.model.log_lik(self.y[:, t], self.X[:, t, :], time=t)
            self.W[t, :], logZ = exp_norm(self.logW[t, :])
            self.logZ += logZ # Update log-likelihood estimate
            self.N_eff[t] = 1/np.sum(self.W[t,:]**2)

            # Compute filter estimates
            self.x_filt[:, t] = np.sum(self.W[t, :] * self.X[:, t, :],axis=1)  # numpy broadcasts along 1st dimension...

    def sample_trajectory(self):
        """Sample a single trajectory (genealogy tracing)"""
        ind = np.random.choice(self.N, 1, p=self.W[-1, :])
        return self.genealogy(ind)

    def genealogy(self, ind):
        """Trace genealogy, starting at ind"""
        M = len(ind)
        Xs = np.zeros((self.dx, self.T, M))

        # Trace genealogy backwards
        for t in range(self.T, 0, -1):
            Xs[:, t - 1, :] = self.X[:, t - 1, ind]
            ind = self.ancestor_indices[t - 1, ind]

        return Xs
