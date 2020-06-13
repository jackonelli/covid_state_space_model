"""SMC/Gibbs/PGAS testing script"""

import numpy as np
import matplotlib.pyplot as plt
from models.lgss1d import LGSS1d
from kalman.kfs import KFS
from gibbs_lgss1d import gibbs_sampler
import time
#from helpers import *
# from models.sm_lgss1d import smLGSS1d

# def main():
# np.random.seed(1)
T = 50

# Motion model
sys0 = LGSS1d(0.9, 0.5, 1., 1.)
x, y = sys0.simulate(T)
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.plot(x, label="State")
ax1.plot(y, '.', label="Observation")

# Run Kalman filter and backward simulator
kf = KFS(sys0, y)
kf.filter()
X = kf.backward_simulator(500)
ax1.plot(np.squeeze(kf.x_filt), label="KF mean")
ax1.plot(np.squeeze(np.mean(X, axis=2)), label="KS mean")
ax1.legend()

# Gibbs sampler
t0 = time.time()
numParticles = 50
numIter = 500
theta_init = np.array([1., .1]) # Q,R
model = LGSS1d(0.9, theta_init[0], 1., theta_init[1])
model.set_parameter_prior(.01, .01, .01, .01) # Set hyperparameters for IG prior

theta_gibbs = gibbs_sampler(theta_init, y, numIter=numIter, model=model, statesampler="PGAS", numParticles=numParticles)
t1 = time.time()
print(f"Elapsed time for Gibbs sampler. Total {t1-t0:.2f} s. Per 100 iteration {100*(t1-t0)/numIter:.2f} s")


fig,(ax3,ax4) = plt.subplots(1,2)
ax3.plot([1, numIter], sys0.Q*np.array([1,1]),"k--")
ax3.plot(theta_gibbs[0,:],label="Trace of Q")
ax3.set_title("Trace of Q")
ax4.plot([1, numIter], sys0.R*np.array([1,1]),"k--")
ax4.plot(theta_gibbs[1,:],label="Trace of R")
ax3.set_title("Trace of R")
plt.show()

# fig,ax2 = plt.subplots(1,1)
# pf = bPF(model,y,5000)
# pf.filter()
# ax2.plot(np.squeeze(pf.x_filt), label="PF mean")
# ax2.plot(np.squeeze(kf.x_filt), label="KF mean")
# ax2.legend()

# plt.show()
plt.show()

# if __name__ == "__main__":
#     main()
