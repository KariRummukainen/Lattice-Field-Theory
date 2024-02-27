import numpy as np
import copy
import random as random
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
from scipy.optimize import curve_fit

import numpy as np

def computeMagnetization(lattice_size, l, k, phi):
    return np.abs(np.mean(phi))

def updateMetropolisHastings(lattice_size, l, k, phi):


    x = int( lattice_size*np.random.random() )
    y = int( lattice_size*np.random.random() )

    phi_new = phi[x][y] + np.random.randn()

    neighbour_sum = phi[(x+1)%lattice_size][y] + phi[x][(y+1)%lattice_size] + phi[x-1][y] + phi[x][y-1]

    V_old = (1. - 2. * l) * phi[x][y]**2 + l * phi[x][y]**4
    V_new = (1. - 2. * l) * phi_new**2 + l * phi_new**4

    #dS = (S_new - S_old)
    dS = 2.0 * k * neighbour_sum * (phi[x][y]-phi_new) 
    dS += (V_new - V_old)
    if dS < 0 or np.random.random() < np.exp(-dS):
        phi[x][y] = phi_new
    return phi

def getGxy(phi: np.ndarray):
    mag_sq = np.mean(phi)**2
    corr_func = []
    for shift in range(1, phi.shape[1], 1):
        corrs = []
        # We will find phi_x phi_x+\mu over only one direction
        # ie. over only rows or columns, although this results
        # in low statistics, it has the advantage of being fast and easy
        phi_shift = np.roll(phi, shift, 1)
        # shift in y-axis
        corr_mean = np.mean(phi * phi_shift)
        corr_func.append([shift, corr_mean - mag_sq])
    return np.array(corr_func)

warm_up = 1000
steps = 1000
lattice_size = 32
l = 0.02
kappa = np.arange(0.22, 0.32, 0.01)
Skip = 10
MT = np.zeros(len(kappa))
ET = np.zeros(len(kappa))
tauT = np.zeros(len(kappa))
colours = pl.cm.coolwarm(np.linspace(0.0,1.0,len(kappa)))
plt.figure()      

for k in kappa: 
    E = 0.0
    M = 0.0
    num_measure = 0
    #reset phis before every new kappa!
    # Warm ICs
    phi = np.random.randn(lattice_size,lattice_size) 
    for _ in range(0, warm_up*lattice_size**2):
        phi = updateMetropolisHastings(lattice_size, l, k, phi)

    M_list = []
    for _ in range(0, steps*lattice_size**2):
        phi = updateMetropolisHastings(lattice_size, l, k, phi)
        if steps % Skip == 0:
            M+=computeMagnetization(lattice_size, l, k, phi)
            M_list.append(np.abs(computeMagnetization(lattice_size, l, k, phi)))
            num_measure+=1
    MT[ (np.where(kappa == k)) ] = M/num_measure


plt.figure()
plt.plot(kappa,MT)
plt.xlabel('$\kappa$')
plt.ylabel('M')
plt.savefig('magnetization.pdf')

Gxy = getGxy(phi)

plt.figure()
plt.plot(Gxy[:,0], Gxy[:,1])
plt.savefig('Gxy.pdf')
