"""Here we intend to use an SEIR model.

The state is:

    x = [
        s
        e
        i
        r
    ]
"""
import numpy as np
import matplotlib.pyplot as plt
from models.seir import SEIR, Param
from pmmh_seir import pmmh_sampler
from smc.bPF import bPF
from helpers import *


"""
Test PMMH on SEIR model
"""

"""" We separate the parameters for the dynamic model into three parts:
1) pei and pir, 2) parameters needed to define b and 3) size of population"""
pei = 1 / 5.1
pir = 1 / 5
pic = 1 / 1000 # ?????
dp = np.array([pei, pir, pic])

"""For Stockholm the population is probably roughly 2.5 million."""
population_size = 2500000
""" For setting the initial state of the simulation"""
i0 = 400
e0 = 400
r0 = 1000
s0 = population_size - i0 - e0 - r0

FHM = False

if FHM:
    """For the FHM model, the parameters needed to define the function b are theta, delta,
    epsilon and the offset between day 0 and March 16. For instance, if we want the UCI 
    measurements to start on March 13, and we assume a 7 day delay in the measurement model day 0 
    would be March 6 and the offset would be -10. """
    from models.seir import SEIR, Param
    b_par = np.array([2, 0.1, -0.12, -10])
    init_state = np.array([s0, e0, i0], dtype=np.int64)  # For FHM model
else:
    """ If we instead use a a random walk model for the log of b,
      b[t] = b_scale * exp(x3[t])
      x3[t] = b_corr*xt[t-1] + b_std*v[t], v[t]~N(0,1)
    """
    from models.seir_rwb import SEIR, Param
    b_scale = 0.2
    b_corr = 0.96
    b_std = 0.05
    b_par = np.array([b_scale, b_corr, b_std])
    init_state = np.array([s0, e0, i0, np.log(1.6/b_scale)], dtype=np.float64)  # For RW model

"""All the above parameters are stored in params."""
params = Param(dp, b_par, population_size)


""" Create a model instance"""
sys0 = SEIR(params)

"""We start our simulation with an initial state, selected fairly arbitrarily."""
sys0.init_state = init_state

"""We initiate our state at time 0 and observe measurements from time 1 to T. What's the true 
value of T? We use 60 here but we have more measurements."""
T = 120

"""Let us generate a random sequence of states and store it in x"""
x, y = sys0.simulate(T)


"""Filter using bootstrap PF"""
model_params = Param(dp, b_par, population_size)
model = SEIR(model_params)  # Can we make a copy of params here instead?
numParticles = 500
"""Since there is a 7 day delay in the observations we shift the data sequence to account for this.
That means that we in fact estimate p(x_t | y_{1:t+7}). However, to get the filter estimate we just
need to predict 7 steps ahead."""
y_shift = np.concatenate((y[:,params.lag:,:], y[:,0:params.lag,:]), axis=1)

# Alternatively, use Stockholm data ####
y_sthlm = np.genfromtxt('./data/New_UCI_June10.csv', delimiter=',')
y_sthlm = y_sthlm[np.newaxis, 1:, np.newaxis]
########################################

pf = bPF(model, y_shift, N=numParticles)  # y_shift is simulated
pf.filter()

"""Finally, we visualize the results."""
plt.figure()
plt.title("Simulating a stochastic SEIR model")
plt.xlabel("Days, Day 0 = March 10")
plt.ylabel("Number of individuals")
#s_line, = plt.plot(x[0, :])
e_line = plt.plot(x[1,:,:], 'b-')[0]
i_line = plt.plot(x[2,:,:], 'r-')[0]
r_line = plt.plot(population_size - np.sum(x[0:3,:], axis=0), 'g-')[0]
# Plot filter estimate
plt.plot(pf.x_filt[1,:], 'b--')
plt.plot(pf.x_filt[2,:], 'r--')
pf_line = plt.plot([None],[None],'k--')[0]
plt.plot(population_size - np.sum(pf.x_filt[0:3,:], axis=0), 'g--')
plt.legend([e_line, i_line, r_line, pf_line], ['e', 'i', 'r', 'PF mean'])

""" Plot b(t) """
plt.figure()
if FHM:
    pass
else:
    plt.plot(params.b_scale * np.exp(x[3,:]),'r-')
    plt.plot(model_params.b_scale * np.exp(pf.x_filt[3, :]), 'r--')

plt.title("b(t)")

plt.figure()
plt.plot(y_shift[0,:,:])
plt.plot(pf.x_filt[2,:]*logistic(pf.model.param.get()[2]),'--')
plt.title("Observations (ICU/day)")

plt.figure()
plt.plot(pf.N_eff)
plt.title("Effective number of particles")


"""""""""""""""""""""""""""""""Run PMMH sampler"""""""""""""""""""""""""""""""
numMCMC = 200
theta_init = params.get()
theta_init *= 0.5
#theta_init[0:3] = theta_init[0:2]*0.5  # Only first three parameters sampled now (hard coded in PMMH script)

# Get initial filter estimate as reference
model_params.set(theta_init)
pf.filter()
x_filt_init = pf.x_filt.copy()

# PMMH
th_pmmh, logZ, accept_prob = pmmh_sampler(theta_init, y_shift, numMCMC, model, numParticles=500)

# Plot
plot_these = [0,1,2]
plt.figure()
plt.plot(th_pmmh[plot_these,:].T)
plt.gca().set_prop_cycle(None)
plt.plot([0, numMCMC-1],np.ones((2,1))*params.get()[plot_these],'--')
plt.xlabel("MCMC iteration")

plt.figure()
plt.plot(accept_prob)

pf.filter()
plt.figure()
plt.title("Simulating a stochastic SEIR model")
plt.xlabel("Days, Day 0 = March 10")
plt.ylabel("Number of individuals")
#s_line, = plt.plot(x[0, :])
e_line = plt.plot(x[1,:,:], 'b-')[0]
i_line = plt.plot(x[2,:,:], 'r-')[0]
r_line = plt.plot(population_size - np.sum(x[0:2,:], axis=0), 'g-')[0]
# Plot filter estimate
plt.plot(pf.x_filt[1,:], 'b--')
plt.plot(pf.x_filt[2,:], 'r--')
pf_line = plt.plot([None],[None],'k--')[0]
plt.plot(population_size - np.sum(pf.x_filt[0:2,:], axis=0), 'g--')
plt.legend([e_line, i_line, r_line, pf_line], ['e', 'i', 'r', 'PF mean'])

""" Plot ICU from estimated state """
plt.figure()
plt.plot(y_shift[0,:,:],label='y')
plt.plot(x_filt_init[2,:]*logistic(theta_init[2]),'--',label='7-day smooth (init)')
plt.plot(pf.x_filt[2,:]*logistic(pf.model.param.get()[2]),'--',label='7-day smooth (final MCMC sample)')
plt.legend()
plt.title("Observations (ICU/day)")


""" Some diagnostics """
plt.figure()
plt.plot(pf.N_eff)
plt.title("Effective number of particles")

plt.figure()
plt.plot(logZ)
plt.title("Likelihood estimates")

#if __name__ == "__main__":
#    main()

plt.show()