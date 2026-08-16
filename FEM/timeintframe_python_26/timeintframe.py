# -*- coding: utf-8 -*-
"""
Timeintframe
------------

Driver for Time Integration of Frame structure equations.
The user has the option between the following time integration algorithms:
 - Newmark
 - Linear generalized alpha

Created: June 2020

@author: Oscar Bondo Ellekvist, s163774


Input
-----
K    : Array
       Stiffness matrix.
C    : Array
       Damping matrix. (optional)
M    : Array
       Mass matrix.
iu   : Array
       Index-set with free (unconstrained) dofs.
ndof : Integer
       Number of dofs.
    
"""

# Import packages
import numpy as np
import matplotlib.pyplot as plt
import sys as sys

# Import functions
from newmark import newmark

# set font size in plots
plt.rc('font', size=10)         # controls default text sizes
plt.rc('axes', titlesize=10)    # fontsize of the axes title
plt.rc('axes', labelsize=12)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=10)   # fontsize of the tick labels
plt.rc('ytick', labelsize=10)   # fontsize of the tick labels
plt.rc('legend', fontsize=10)   # legend fontsize
plt.rc('figure', titlesize=12)  # fontsize of the figure title

### INPUT DATA

# Import input data
npzfile = np.load('vibprop_sysred.npz')
X     = npzfile['Xd']
K     = npzfile['K']
M     = npzfile['M']
S     = npzfile['S']
U     = npzfile['U']
omega = npzfile['omega']
# NB: Import extra variables to be used in analysis


# Number of dofs
ndof = np.size(M,0)

# Initial conditions
x0 = np.zeros((ndof,1),dtype=float)
v0 = x0

### LOAD --- Example: harmonic load

# Time simulation parameters
t0 = 0                 # initial time
N  = 2000              # number of time steps
dt = 0.01              # time step size

# sinusoidal load
tf = np.arange(0,N+1,1,dtype=float)*dt    # time vector
freq  = omega/(2*np.pi)                   # Hz frequencies
omegaf = omega[13-1]                      # loading frequency [rad/s]
freqf = freq[13-1]                        # loading frequency [Hz]
Tf = 1/freqf                              # loading period
f = 1e6*np.sin(omegaf*tf)                 # load vector - sinusoidal

# Load in reduced system - for load acting on "static dof"
ndof_s = np.size(S,1-1)                   # number of static dofs
nodef_s = 50 - 45                         # load at original node 50 in static system
Fs = np.zeros((ndof_s,N+1),dtype=float)   # static load vector
Fs[(nodef_s-1)*6+2-1,:] = 1e6*np.sin(omegaf*tf)
F = np.dot(np.transpose(S),Fs)            # load vector in reduced system by S'*Fs

### RAYLEIGH DAMPING MATRIX
zetamin = 0.025
omegamin = omega[13-1]

# Rayleigh parameters from minimum point condition
aR = zetamin*omegamin
bR = zetamin/omegamin

# Damping matrix
C = aR*M + bR*K

### TIME INTEGRATION

# Newmark parameters
gamma = 0.5
beta  = 0.25

# Newmark time integration
(q,v,a,t) = newmark(K,C,M,x0,v0,dt,N,F,beta=beta,gamma=gamma)

### POST-PROCESSING

# plot response at dofplot
dofplot = (41-1)*6+2

# Plot response history using matplotlib
plt.close('all')

fig1 = plt.figure(1,figsize=[12,4])      # plot of disp and acc

fig1.add_subplot(121)           # subplot 1 = disp
plt.plot(t,q[dofplot-1,:],color='blue')
#plt.title('Displacement')
plt.xlabel(r'$t$ - [s]')
plt.ylabel(r'$v_{41}$ - [m]')
plt.grid()
#plt.xlim([0, 600])
#plt.ylim([-0.2, 3.0])

fig1.add_subplot(122)           # subplot 2 = acc
plt.plot(t,a[dofplot-1,:],color='red')
plt.xlabel(r'$t$ - [s]')
plt.ylabel(r'$\ddot{v}_{41}$ - [m/s$^2$]')
plt.grid()

# Save response history to .png file
plt.savefig('timeintframe_response.png',bbox_inches='tight')
plt.show()

# stop code here - move up to break code where you like
#sys.exit()