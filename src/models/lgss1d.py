"""One-dimensional LGSS model"""
import numpy as np
from scipy.stats import norm


class LGSS1d:
    def __init__(self, A: float, Q: float, C:float, R:float):
        self.dx = 1
        self.dy = 1
        self.set_params(A,Q,C,R)

    def set_params(self,A,Q,C,R):
        self.A = A # Process model
        self.Q = Q # Process noise variance
        self.C = C # Measurement model
        self.R = R # Measurement noise variance
        self.mu1 = 0.0 # Initial mean
        self.P1 = Q/(1-A**2) # Initial variance. We assume stationarity for simplicity

    def get_params(self):
        return self.A, self.Q, self.C, self.R

    def log_transition(self, x1, x0):
        """Computes the model log transition p(x_new|x_now) for all values in array x_now"""
        return norm.logpdf(x1-self.A*x0, 0, np.sqrt(self.Q))

    def log_lik(self, y, x):
        """Computes the model log likelihood p(y|x) for all values in array x"""
        return norm.logpdf(y-self.C*x, 0, np.sqrt(self.R))

    def sample_state(self, x0=None,N=1):
        if x0 is None:
            x1 = np.random.normal(loc=self.mu1, scale=np.sqrt(self.P1),size=N)
        else:
            x1 = np.random.normal(loc=self.A*x0, scale=np.sqrt(self.Q), size=N)
        return x1

    def sample_obs(self, x):
        y = np.random.normal(loc=self.C*x, scale=np.sqrt(self.R))
        return y

    def simulate(self, T):
        x = np.zeros(T)
        y = np.zeros(T)
        x[0] = self.sample_state()
        y[0] = self.sample_obs(x[0])
        for t in range(1,T):
            x[t]=self.sample_state(x[t-1])
            y[t]=self.sample_obs(x[t])

        return x,y

    # Hard coded IG priors on variances !!!
    def set_parameter_prior(self, aQ, bQ, aR, bR):
        self.aQ = aQ
        self.bQ = bQ
        self.aR = aR
        self.bR = bR


