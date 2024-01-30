import numpy as np
import random as random
import matplotlib.pyplot as plt

def computeEnergy(lattice_size, T, J, h, spin):
    energy = 0.0
    #for x in range(lattice_size):
    #    for y in range(lattice_size):
    #        energy += - spin[x][y] * spin[x][(y+1)%lattice_size]
    #        energy += - spin[x][y] * spin[(x+1)%lattice_size][y]
    #energy = energy / (lattice_size**2)
    energy = np.mean(-h*spin -J * spin
            * (np.roll(spin, 1, 0) + np.roll(spin, 1, 1) ))
    #         + np.roll(spin, -1, 0) + np.roll(spin, -1, 1)))
    return energy

def computeMagnetization(lattice_size, T, J, h, spin):
    return np.abs(np.mean(spin))

def updateMetropolisHastings(lattice_size, T, J, h, spin):


    x = int( lattice_size*np.random.random() )
    y = int( lattice_size*np.random.random() )

    mag_old = spin[x][y]
    energy_old = ( spin[x][(y+1)%lattice_size] + spin[x][y-1]
                  + spin[(x+1)%lattice_size][y] + spin[x-1][y] ) * spin[x][y]

    mag_new = -mag_old
    energy_new = -energy_old 

    dH = -J * (energy_new-energy_old) -h * (mag_new-mag_old)
    if dH < 0.0:
        #Flip!
        spin[x][y] = -1 * spin[x][y]
    else:
        # Like standard Heat bath,
        # flip only sometimes
        P_flip = np.exp(-dH/T)
        if np.random.random() < P_flip:
            spin[x][y] = -1 * spin[x][y]
    return spin

def updateHeatBath(lattice_size, T, J, h, spin):

    x = int( lattice_size*np.random.random() )
    y = int( lattice_size*np.random.random() )

    energy_plus  = -(1) * ( spin[x][(y+1)%lattice_size] + spin[x][y-1]
                          + spin[(x+1)%lattice_size][y] + spin[x-1][y] )
    energy_minus = -energy_plus

    mag_plus = -spin[x][y]
    mag_minus = -mag_plus

    P_plus  = np.exp( -J*energy_plus/T -h*mag_plus/T )
    P_minus = np.exp( -J*energy_minus/T -h*mag_minus/T)

    probability_plus = P_plus / (P_plus + P_minus)

    if np.random.random() < probability_plus:
       spin[x][y] = 1
    else:
       spin[x][y] = -1

    return spin

warm_up = 100000
steps = 10000 
lattice_size = 20
J = 1.0
h = 0.0
temperature = np.linspace(0.025, 5, 50)

Skip = 10
MT = np.zeros(len(temperature))
ET = np.zeros(len(temperature))

for T in temperature: 
    E = 0.0
    M = 0.0
    num_measure = 0
    #reset spins before every new temperature!
    spin = np.ones([lattice_size,lattice_size], dtype=int) 
    for k in range(0, warm_up):
        spin = updateMetropolisHastings(lattice_size, T, J, h, spin)

    for k in range(0, steps):
        spin = updateMetropolisHastings(lattice_size, T, J, h, spin)
        if steps % Skip == 0:
            E+=computeEnergy(lattice_size, T, J, h, spin)
            M+=computeMagnetization(lattice_size, T, J, h, spin)
            num_measure+=1
    #print("<E> is {}".format( E/num_measure ) )
    #print("<M> is {}".format( M/num_measure ) )
    MT[ (np.where(temperature == T)) ] = M/num_measure
    ET[ (np.where(temperature == T)) ] = E/num_measure

plt.figure()      
plt.plot(temperature, MT)
plt.xlabel('T')
plt.ylabel('M')
plt.ylim(top=1.1)
plt.savefig('Homework_M.pdf')

plt.figure()      
plt.plot(temperature, ET)
plt.xlabel('T')
plt.ylabel('E')
plt.savefig('Homework_E.pdf')

