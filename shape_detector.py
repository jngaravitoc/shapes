"""
Returns the values a, b, c corresponding to the length of the
principal axis.

python shape_detector.py Snapshot_name Ni Nf

Ni = initial snapshot number
Nf = final snapshot number

To do:
1. Implement CM correction.
2. Plot ellipse on the top of the halo dots.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
from scipy import linalg
from pygadgetreader import *
from movie import *

tolerance = 1E-6
snap = str(sys.argv[1])
# Initial and final snapshot number
i_n = int(sys.argv[2])
i_f = int(sys.argv[3])
Nhost = 30000
Nsat = 30000

path = '../LMC-MW/data/LMCMW/MW1LMC4/a1/'

# Number of Snapshots
N_snaps = (i_f - i_n) + 1

S = np.zeros(N_snaps)
Q = np.zeros(N_snaps)

#This function returns the length of a, takes as the largest distance
#from the CM to the Rvir?
def A(V, x_cm, y_cm, z_cm):
    R = np.sqrt((V[0,:] - x_cm)**2 + (V[1,:] - y_cm)**2 + (V[2,:] - z_cm)**2)
    index = np.where(R == max(R))[0]
    return R[index]

# Function that computes the reduced inertia tensor.
def RIT(XYZ, q, s):
    I = np.zeros([3, 3])
    N = len(XYZ[0])
    for i in range(3):
        for j in range(3):
            XX = np.zeros(N)
            for n in range(N):
                d = np.sqrt(XYZ[0,n]**2. + XYZ[1,n]**2./q**2.+ XYZ[2,n]**2./s**2.)
                Xi = XYZ[i,n]
                Xj = XYZ[j,n]
                XX[n] = Xi * Xj / d**2.
            I[i][j] = sum(XX)
    return I

# function that make a matrix of 3 vectors (should I need this?)
def one_tensor(x, y, z):
    N = len(x)
    XYZ = np.zeros([3,N])
    XYZ[0,:] = x
    XYZ[1,:] = y
    XYZ[2,:] = z
    return XYZ

#Function that iterative evaluate the inertia tensor to find the shape.

def Shape(XYZ, tol):
    old_q = 1.2
    old_s = 1.2
    new_q = 1.0
    new_s = 1.0
    while((abs(new_s - old_s) > tol) & (abs(new_q - old_q) > tol)):
        old_s = new_s
        old_q = new_q
        I_test = RIT(XYZ, old_q, old_s)
        #print I_test
        eival, evec = linalg.eig(I_test)
        oeival = np.sort(eival)
        XYZ = np.dot(evec.T, XYZ)
        la, lb, lc = oeival[2], oeival[1], oeival[0]
        new_s = np.sqrt(lc/la)
        new_q = np.sqrt(lb/la)
    return new_s, new_q, XYZ

# Function to make an ellipsoid  plot.
def Ellipsoid(a, b, c):
    theta = np.random.rand(100) * 2 - 1
    phi = (np.random.rand(100) * 2 * np.pi)
    x = a * np.sin(np.arccos(theta)) * np.cos(phi)
    y = b * np.sin(np.arccos(theta)) * np.sin(phi)
    z = c * np.cos(np.arccos(theta))
    return x, y, z

def projection(x, y, z, a, q, s):
    plt.figure(figsize=(17, 5))
    plt.subplot(1, 3, 1)
    u = np.linspace(0, 2 * np.pi, 100)
    xt = a*np.cos(u)
    yt = a*q*np.sin(u)
    plt.scatter(x, y, s=1)
    plt.plot(xt, yt, lw=2, c='r')
    #plt.xlim(-200, 200)
    #plt.ylim(-200, 200)
    plt.xlabel('$x$', fontsize=25)
    plt.ylabel('$y$', fontsize=25)

    plt.subplot(1, 3, 2)
    plt.scatter(x, z, s=1)
    #plt.xlim(-200, 200)
    #plt.ylim(-200, 200)
    yt2 = a*s*np.sin(u)
    plt.plot(xt, yt2, lw=2, c='r')
    plt.xlabel('$x$', fontsize=25)
    plt.ylabel('$z$', fontsize=25)

    plt.subplot(1, 3, 3)
    plt.scatter(y, z, s=1)
    xt2 = a*q*np.cos(u)
    plt.plot(xt2, yt2, lw=2, c='r')
    #plt.xlim(-200, 200)
    #plt.ylim(-200, 200)
    plt.xlabel('$y$', fontsize=25)
    plt.ylabel('$z$', fontsize=25)
    plt.show()

time = np.zeros(N_snaps)

f = open(snap + "shape.txt", "w")
f.write("Q, S \n")

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
    index_mw = np.where(particles_ids<=idcut)[0]
    index_LMC = np.where(particles_ids>idcut)[0]

    Rmw = positions[index_mw,:]
    #print np.shape(Rmw.T)
    #print len(Rmw.T[0])
    x_mw = positions[index_mw,0]
    y_mw = positions[index_mw,1]
    z_mw = positions[index_mw,2]
    Rmax = np.sqrt(x_mw**2 + y_mw**2 + z_mw**2)
    Rvir_cut = np.where(Rmax<261.0)[0]
    x_mw, y_mw, z_mw = x_mw[Rvir_cut], y_mw[Rvir_cut], z_mw[Rvir_cut]
    Rmw = Rmw[Rvir_cut,:]
    #x_lmc = positions[index_LMC[0],0]
    #y_lmc = positions[index_LMC[0],1]
    #z_lmc = positions[index_LMC[0],2]

    a = A(Rmw.T, 0, 0, 0)
    S[i], Q[i], D = Shape(Rmw.T, tolerance)
    f.write("%f \t %f \n"%(Q[i], S[i]))
    #print np.shape(D)
    #rint a, s, q
    #rojection(D[0,:], D[1,:], D[2,:], a, q, s)
    #movie(D[0,:], D[1,:])

f.close()
#xe, ye, ze = Ellipsoid(a, q*a, s*a)
#XYZe = one_tensor(xe, ye, ze)
#XYZ_e_rot = np.dot(evec, XYZe)
#projection(XYZe[0,:], XYZe[1,:], XYZe[2,:])
