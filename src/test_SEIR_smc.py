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
from smc.bPF import bPF
from helpers import *


"""
Run SMC on the SEIR model (no PMCMC)
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
    from models.seir import SEIR, Param, b_val_FHM
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
    b_std = 0.02
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

y = y_sthlm  # <-- this is what we filter and plot below (y_shift is simulated)

pf = bPF(model, y, N=numParticles)
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
    plt.plot(b_val_FHM(params, np.arange(T)),'r-')
else:
    plt.plot(params.b_scale * np.exp(x[3,:]),'r-')
    plt.plot(model_params.b_scale * np.exp(pf.x_filt[3, :]), 'r--')

plt.title("b(t)")

plt.figure()
plt.plot(y[0,:,:])
plt.plot(pf.x_filt[2,:]*logistic(pf.model.param.get()[2]),'--')
plt.title("Observations (ICU/day)")

plt.figure()
plt.plot(pf.N_eff)
plt.title("Effective number of particles")

plt.show()