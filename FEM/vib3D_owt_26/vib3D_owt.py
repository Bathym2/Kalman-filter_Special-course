# -*- coding: utf-8 -*-
"""
vib3D_owt
----------

Driver for vibration analysis of 3D OWT

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

# nodes and dofs
dof  = 6
nnodes = np.size(X,1)
ndof = nnodes*dof

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

### 3D PLOT MODE SHAPE
plt.close('all')

mode_plot = 13                  # mode to plot
mode_scl = 50                   # scaling of mode in plot
freq_plot = freq[mode_plot-1]   # frequency for mode to plot
print(freq_plot)

fig1 = plt.figure(1,figsize=[6,6])
ax = fig1.add_subplot(111, projection='3d')
ax.scatter(
    X[0, :], 
    X[1, :], 
    X[2, :], 
    color='k', 
    marker='.')
ax.scatter(
    X[0, :]+U[0::6,mode_plot-1]*mode_scl, 
    X[1, :]+U[1::6,mode_plot-1]*mode_scl, 
    X[2, :]+U[2::6,mode_plot-1]*mode_scl,  
    color='r', 
    marker='.')
plt.axis('equal')
plt.tight_layout()
plt.show()

# stop program
sys.exit()

### PLOT GEOMETRY

# support structure
x_tower = X[1-1,1-1:65]
y_tower = X[2-1,1-1:65]
z_tower = X[3-1,1-1:65]

# drivetrain
x_drive = X[1-1,66-1:69]
y_drive = X[2-1,66-1:69]
z_drive = X[3-1,66-1:69]

# hub
x_hub = X[1-1,70-1:72]
y_hub = X[2-1,70-1:72]
z_hub = X[3-1,70-1:72]

# blade 1
x_b1 = X[1-1,73-1:141]
y_b1 = X[2-1,73-1:141]
z_b1 = X[3-1,73-1:141]

# blade 2
x_b2 = X[1-1,142-1:210]
y_b2 = X[2-1,142-1:210]
z_b2 = X[3-1,142-1:210]

# blade 3
x_b3 = X[1-1,211-1:279]
y_b3 = X[2-1,211-1:279]
z_b3 = X[3-1,211-1:279]


# side-side geometry
fig10 = plt.figure(10,figsize=[6,12])
plt.plot(x_tower,z_tower)
plt.plot(x_b1,z_b1)
plt.plot(x_b2,z_b2)
plt.plot(x_b3,z_b3)
plt.plot(x_drive,z_drive)
plt.plot(x_hub,z_hub,'*')
plt.axis('equal')
plt.title('side-side geometry')
plt.grid()
plt.tight_layout()
#plt.show()

# fore-aft geometry
fig11 = plt.figure(11,figsize=[6,12])
plt.plot(y_tower,z_tower)
plt.plot(y_b1,z_b1)
plt.plot(y_b2,z_b2)
plt.plot(y_b3,z_b3)
plt.plot(y_drive,z_drive)
plt.plot(y_hub,z_hub,'*')
plt.axis('equal')
plt.title('Fore-aft geometry')
plt.grid()
plt.tight_layout()
plt.show()

# sys.exit()