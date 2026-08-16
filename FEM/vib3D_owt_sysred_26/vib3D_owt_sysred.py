# -*- coding: utf-8 -*-
"""
vib3D_owt_sysred
----------

Driver for vibration analysis of 3D OWT - system reduced model

Created: Feb 2026 - DTU Construct

# x = side-side (in-rotor plane)
# y = fore-aft (in wind direction)
# z = vertical (upwards , z = 0 at MSL)

# nodes  1 to 65: support structure = tower + monopile
#        66 to 69: drive-train
#        70 to 72: hub
#        73 to 141: blade 1 (vertical)
#        142 to 210: blade 2 (down + negative x, down-left from upstream view)
#        211 to 279: blade 3 (down + positive x, down-right from upstream view)

"""

# Import packages
import numpy as np
import scipy as sp
import sys as sys
import matplotlib.pyplot as plt

# if "module 'scipy' has no attribute 'linalg'" then import as:
from scipy import linalg as linalg
#sys.exit()

### INPUT - load X, K, M for 3D OWT FE model
# qn = [disp_x disp_y disp_z rot_x rot_y rot_z] in global {x,y,z}-coord sys.
npzfile = np.load('owt_XKM.npz')
X   = npzfile['X']
K   = npzfile['K']
M   = npzfile['M']

### SYSTEM REDUCTION - static condensation

# dynamic nodes
nodes_tower = np.arange(1,45+1)
nodes_rotor = np.array([72, 73, 90, 107, 124, 141, 142, 159, 176, 193, 210, 211, 228, 245, 262, 279])
#nodes_rotor = np.arange(72,279+1)
nodes_d = np.concatenate((nodes_tower,nodes_rotor))
Xd = X[:,nodes_d-1]

# dynamic dofs in Id
block = np.arange(1,6+1)                        # [1, 2, 3, 4, 5, 6]
Id_matrix = (nodes_d[:, None] - 1) * 6 + block  # shape: (len(nodes_d), 6)
Id = Id_matrix.ravel()                          # flatten to 1D

# static dofs in Is
Is = np.arange(1,6*279+1,1) # all dofs
Is = np.setdiff1d(Is,Id)    # static = remove dynamic

# block matrices
Mdd = M[np.ix_(Id-1,Id-1)]
Mds = M[np.ix_(Id-1,Is-1)]
Msd = M[np.ix_(Is-1,Id-1)]
Mss = M[np.ix_(Is-1,Is-1)]

Kdd = K[np.ix_(Id-1,Id-1)]
Kds = K[np.ix_(Id-1,Is-1)]
Ksd = K[np.ix_(Is-1,Id-1)]
Kss = K[np.ix_(Is-1,Is-1)]

# static constraint matrix
S = -np.linalg.solve(Kss,Ksd)
ST = np.transpose(S)

# reduced matrices
K = Kdd + np.dot(ST,Ksd) + np.dot(Kds,S) + np.dot(ST, np.dot(Kss,S) )
M = Mdd + np.dot(ST,Msd) + np.dot(Mds,S) + np.dot(ST, np.dot(Mss,S) )

### VIBRATION ANALYSIS - solve EVP
n_modes = 20    # Determines lowest n_modes eig solutions
[lam,U] = sp.linalg.eigh(K, M, subset_by_index=[0, n_modes - 1])
lam = np.real(lam)

omega = np.sqrt(lam)
iw = np.argsort(omega)

# Natural freqiencies and mode shapes (normalized to max = 1)
omega = omega[iw]
freq  = omega/(2*np.pi)
U = U[:,iw]
U = U / U[np.argmax(np.abs(U), axis=0), np.arange(0, n_modes)]
print(freq[0:15])

# Save results in .npz file
np.savez('vibprop_sysred',Xd=Xd,K=K,M=M,U=U,omega=omega,freq=freq,S=S,Id=Id,Is=Is)

### 3D PLOT MODE SHAPE
plt.close('all')

mode_plot = 13                  # mode to plot
mode_scl = 50                   # scaling of mode in plot
freq_plot = freq[mode_plot-1]   # frequency for mode to plot
print(freq_plot)

fig1 = plt.figure(1,figsize=[6,6])
ax = fig1.add_subplot(111, projection='3d')
ax.scatter(
    Xd[0, :], 
    Xd[1, :], 
    Xd[2, :], 
    color='k', 
    marker='.')
ax.scatter(
    Xd[0, :]+U[0::6,mode_plot-1]*mode_scl, 
    Xd[1, :]+U[1::6,mode_plot-1]*mode_scl, 
    Xd[2, :]+U[2::6,mode_plot-1]*mode_scl,  
    color='r', 
    marker='.')
plt.axis('equal')
plt.tight_layout()
plt.show()

# stop program
#sys.exit()