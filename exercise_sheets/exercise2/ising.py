import numpy as np
import random as random
import matplotlib.pyplot as plt
import matplotlib.pylab as pl
from scipy.optimize import curve_fit

def computeTau(acf):
    smaller = np.where((acf/acf[0]) < np.exp(-1))[0]
    return smaller[0] if len(smaller) > 0 else len(acf)

def computeACF(X):
    mu = np.mean(X)
    X_shift = X - mu
    # X - mu = X[j] - X_1
    N = len(X)
    f = np.fft.fft(X_shift, n=2*N)
    acf = np.fft.ifft( f * np.conjugate(f) )[: len(X)].real  
    return acf/(4*N)

def computeACF_errorsc(X):
    mu = np.mean(X)
    sigma = np.var(X)
    N = len(X)
    maxlag = 70
    acf = np.zeros(maxlag)
    t_int = 0
    for it in range(0, maxlag):
        if it >= N/2: continue
        fi = 0
        av1 = 0
        av2 = 0
        nc = N-it
        for j in range(0,nc):
            fi += X[j]*X[j+it]
            av1 += X[j]
            av2 += X[j+it]
        
        fi = (fi/nc - (av1/nc)*av2/nc)/sigma
        #t_int = fi*nc/N
        acf[it] = fi
    return acf, t_int

def computeEnergy(lattice_size, T, J, h, spin):
    energy = 0.0
    energy = np.mean(-h*spin -J * spin
            * (np.roll(spin, 1, 0) + np.roll(spin, 1, 1) ))
    return energy

def computeMagnetization(lattice_size, T, J, h, spin):
    return np.abs(np.mean(spin))

def updateWolff(lattice_size, T, J, h, spin):
    x = int( lattice_size*np.random.random() )
    y = int( lattice_size*np.random.random() )
    Cluster = [[x,y]]
    Pocket = [[x,y]]
    P_add = 1. - np.exp(-2 * J/T)

    while len(Pocket) > 0:
        Pocket_new = []
        for i,j in Pocket:
            ip1 = (i+1) % lattice_size
            im1 = (i-1+lattice_size) %lattice_size
            jp1 = (j+1) % lattice_size
            jm1 = (j-1+lattice_size) %lattice_size
            nbr = [[ip1,j], 
                   [im1,j], 
                   [i,jp1], 
                   [i,jm1]]

            for l in nbr:
                if spin[l[0],l[1]] == spin[i][j] and l not in Cluster:
                    if np.random.rand() < P_add:
                        Pocket_new.append(l)
                        Cluster.append(l)

        Pocket = Pocket_new

    for i,j in Cluster:
        spin[i,j] *= -1
    return spin

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

#warm_up = 100000
#steps = 10000 
warm_up = 1000
steps = 100
lattice_size = 20
J = 1.0
h = 0.0
temperature = np.arange(4.75, 5.25, 0.125) #np.linspace(1, 5, 5)
Skip = 1
MT = np.zeros(len(temperature))
ET = np.zeros(len(temperature))
tauT = np.zeros(len(temperature))
colours = pl.cm.coolwarm(np.linspace(0.0,1.0,len(temperature)))
plt.figure()      

for T in temperature: 
    E = 0.0
    M = 0.0
    num_measure = 0
    #reset spins before every new temperature!
    # Cold
    spin = np.ones([lattice_size,lattice_size], dtype=int) 
    # Warm
    #spin = 2.0 * np.random.randint(0, 2, size=(lattice_size, lattice_size)) - 1
    for k in range(0, warm_up):
       # spin = updateMetropolisHastings(lattice_size, T, J, h, spin)
        spin = updateWolff(lattice_size, T, J, h, spin)

    M_list = []
    E_list = []
    
    for k in range(0, steps):
        #spin = updateMetropolisHastings(lattice_size, T, J, h, spin)
        spin = updateWolff(lattice_size, T, J, h, spin)
        if steps % Skip == 0:
            #E+=computeEnergy(lattice_size, T, J, h, spin)
            M+=computeMagnetization(lattice_size, T, J, h, spin)
            M_list.append(np.abs(computeMagnetization(lattice_size, T, J, h, spin)))
            E_list.append(np.abs(computeEnergy(lattice_size, T, J, h, spin)))
            num_measure+=1
    #print("<E> is {}".format( E/num_measure ) )
    #print("<M> is {}".format( M/num_measure ) )
    acf, tInt = computeACF_errorsc( np.abs(np.array(M_list)) )
    plt.plot(np.abs(acf/acf[0]), color=colours[(np.where(temperature == T))])
    MT[ (np.where(temperature == T)) ] = M/num_measure
    ET[ (np.where(temperature == T)) ] = E/num_measure
    tauT[ (np.where(temperature == T)) ] = computeTau(acf)

plt.ylabel('C(t)/C(0)')
plt.ylim([np.exp(-3) , 1.1])
plt.axhline( np.exp(-1) , linestyle='dashed')
#plt.gca().set_yscale('log',base=np.e)
plt.legend()
plt.xlim([0,70])
plt.xlabel('t')
plt.savefig('ACF.pdf')

plt.figure()
plt.plot(temperature, tauT)
print(temperature, tauT)
plt.ylabel('$\\tau$')
plt.xlabel('T')
plt.savefig('acTime.pdf')

plt.figure()
plt.plot(temperature,np.abs(MT))
plt.xlabel('T')
plt.ylabel('M')
plt.savefig('magnetization.pdf')