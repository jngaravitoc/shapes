import numpy as np
import matplotlib.pyplot as plt
import sys
from scipy import linalg
from pygadgetreader import *

tolerance = 1E-4
snap = str(sys.argv[1])
# Initial and final snapshot number
i_n = int(sys.argv[2])
i_f = int(sys.argv[3])
Nhost = 30000
Nsat = 30000

path = '../LMC-MW/data/LMCMW/MW1LMC4/a1/'

# Number of Snapshots
N_snaps = (i_f - i_n) + 1

def A(V, x_cm, y_cm, z_cm):
    R = np.sqrt((V[0,:] - x_cm)**2 + (V[1,:] - y_cm)**2 + (V[2,:]\
    - z_cm)**2)
    index = np.where(R == max(R))[0]
    return R[index]

def RIT(XYZ, q, s):
    I = np.zeros([3, 3])
    N = len(XYZ[0])
    for i in range(3):
        for j in range(3):
            XX = np.zeros(N)
            for n in range(N):
                d = np.sqrt(XYZ[0,n]**2 + XYZ[1,n]**2/q**2 \
                + XYZ[2,n]**2/s**2)
                Xi = sum(XYZ[i,n])
                Xj = sum(XYZ[j,n])
                XX[n] = Xi * Xj / d**2
            I[i][j] = sum(XX)
    return I

def one_tensor(x, y, z):
    N = len(x)
    XYZ = zeros([3,N])
    XYZ[0,:] = x
    XYZ[1,:] = y
    XYZ[2,:] = z
    return XYZ

def Shape(XYZ, tol):
    old_q = 1.2
    old_s = 1.2
    new_q = 1.0
    new_s = 1.0

    while((abs(new_s - old_s) > tol) & (abs(new_q - old_q) > tol)):
        old_s = new_s
        old_q = new_q
        I_test = RIT(XYZ, old_q, old_s)
        eival, evec = eig(I_test)
        oeival = sort(eival)
        XYZ = dot(evec.T, XYZ)
        #print oeival
        la = oeival[2]
        lb = oeival[1]
        lc = oeival[0]
        new_s = np.sqrt(lc/la)
        new_q = np.sqrt(lb/la)
        #print Ixy, Ixz, Iyx, Iyz, Izx, Izy
    return new_s, new_q, # evec

def Ellipsoid(a, b, c):
    theta = np.random.rand(100) * 2 - 1
    phi = (np.random.rand(100) * 2 * pi)
    x = a * np.sin(np.arccos(theta)) * np.cos(phi)
    y = b * np.sin(np.arccos(theta)) * np.sin(phi)
    z = c * np.cos(np.arccos(theta))
    return x, y, z

time = np.zeros(N_snaps)
#positions = np.zeros(N_snaps)
#velocities = np.zeros(N_snaps)


for i in range(i_n, i_f + 1):
    if i<10:
        time[i-i_n] = readheader(path + snap + "_00" + str(i),'time')
        positions = readsnap(path + snap + "_00" + str(i),'pos', 'dm')
        velocities = readsnap(path + snap + "_00" + str(i), 'vel','dm')
        particles_ids = readsnap(path + snap + "_00" + str(i),'pid', 'dm')
    elif ((i>=10) & (i<100)):
        time[i-i_n] = readheader(path + snap + "_0" +str(i),'time')
        positions = readsnap(path + snap + "_0" + str(i),'pos','dm')
        velocities = readsnap(path + snap + "_0" + str(i), 'vel','dm')
        particles_ids = readsnap(path + snap + "_0" + str(i),'pid', 'dm')
    else:
        time[i-i_n] = readheader(path + snap + "_" +str(i),'time')
        positions = readsnap(path + snap + "_" + str(i),'pos','dm')
        velocities = readsnap(path + snap + "_" + str(i), 'vel','dm')
        particles_ids = readsnap(path + snap + "_" + str(i),'pid', 'dm')
    ID = np.sort(particles_ids)
    # The first set of particles are from the host DM halo, the
    # second set are from the satellite DM halo, the limit is known by
    # the number of particles in the host halo.
    idcut = ID[Nhost-1]
    index_mw = np.where(particles_ids<=idcut)
    index_LMC = np.where(particles_ids>idcut)

    Rmw = positions[index_mw[0],:]
    print np.shape(Rmw.T)
    print Rmw.T[0,0]
    x_mw = positions[index_mw[0],0]
    y_mw = positions[index_mw[0],1]
    z_mw = positions[index_mw[0],2]
    x_lmc = positions[index_LMC[0],0]
    y_lmc = positions[index_LMC[0],1]
    z_lmc = positions[index_LMC[0],2]

    a = A(Rmw.T, 0, 0, 0)
    #s, q, = Shape(Rmw.T, tolerance)
    #print type(RMW.T)
    print a, s, q
