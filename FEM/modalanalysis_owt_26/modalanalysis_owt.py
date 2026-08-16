# -*- coding: utf-8 -*-
"""
modalanalysis_owt
----------

Driver for Modal analysis of beam/Frame structure.

Created: June 2020

@author: Oscar Bondo Ellekvist, s163774
         Modified spring 2026 for owt case with ship impact
     
"""

# Import packages
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import sys

# Import functions
from REPORTS.Report_2.timestep import timestep
from scipy import linalg as linalg

### INPUT
npzfile = np.load('owt_XKM.npz')
X   = npzfile['X']
K   = npzfile['K']
M   = npzfile['M']
#sys.exit()

### NODES and GEOMETRY
# Dimensions of 3D FE model = 6 dofs per node: 
# [disp_x disp_y disp_z rot_x rot_y rot_z] in global {x,y,z}-coord sys.
dof  = 6
ndof = np.size(X,1)*dof
print(ndof)

# eigenvalue problem
n_modes = 13
[lam,U] = sp.linalg.eigh(K, M, subset_by_index=[0, n_modes - 1])
lam = np.real(lam)
omega = np.sqrt(lam)
iw = np.argsort(omega)
omega = omega[iw]
freq  = omega/(2*np.pi)
U = U[:,iw]
U = U / U[np.argmax(np.abs(U), axis=0), np.arange(0, n_modes)]
print(freq[1-1:15])

# Initial conditions
x0 = np.zeros((ndof,1),dtype=float)
v0 = x0

# Damping ratio (constant modal damping) for all modes included
zeta = 0.025

# Time simulation parameters
t0 = 0                  # initial time
N  = 20000              # number of time steps
dt = 0.001              # time step size

# Load vector - sine at node 50 in y-direction
nodef = 50
tf = np.arange(0,N+1,1,dtype=float)*dt   # time vector
omegaf = omega[13-1]
freqf = freq[13-1]
Tf = 1/freqf
F = np.zeros((ndof,N+1),dtype=float)
F[(nodef-1)*6+2-1,:] = 1e6*np.sin(omegaf*tf)

# plot force
fig10 = plt.figure(10,figsize=[6,4])
plt.plot(tf,F[(nodef-1)*6+2-1,:]*1e-6,color='blue')
plt.title('Double-impact load')
plt.xlabel(r'$t$ - [s]', fontsize=14)
plt.ylabel(r'$f$ - [MN]', fontsize=14)
plt.grid()
plt.xlim([0, 4])
#plt.ylim([-0.01, 0.01])
plt.xticks([0, 1, 2, 3, 4])
plt.yticks([0, 1, 2])
#plt.show()

### MODAL ANALYSIS

# Modal mass and stiffness
mm = np.linalg.multi_dot((np.transpose(U),M,U))
km = np.linalg.multi_dot((np.transpose(U),K,U))

# Initial conditions
r0 = np.linalg.solve(mm,np.linalg.multi_dot((np.transpose(U),M,x0)))
dr0 = np.linalg.solve(mm,np.linalg.multi_dot((np.transpose(U),M,v0)))

# Modal load
f = np.dot(np.transpose(U),F)

# Modal damping vector
zetam = zeta*np.ones(n_modes,dtype=float)

# Modal time histories
r   = np.zeros((n_modes,N+1),dtype=float)
dr  = np.zeros((n_modes,N+1),dtype=float)
ddr = np.zeros((n_modes,N+1),dtype=float)
Em  = np.zeros((n_modes,N+1),dtype=float)

# Loop over modes j
for j in range(0,n_modes):
    
    # modal mass and stiffness
    mj = mm[j,j]
    kj = km[j,j]
    
    # from zeta to modal c
    cj = 2*zetam[j]*np.sqrt(mj*kj)
    
    # Time response of mode i
    (rj,drj,ddrj,_,t) = timestep(mj,cj,kj,r0[j],dr0[j],t0,dt,N,f[j,:])
    
    # Store response in modal vector
    r[j,:]   = rj
    dr[j,:]  = drj
    ddr[j,:] = ddrj

    # modal energy
    Em[j,:] = 0.5*mj*drj**2 + 0.5*kj*rj**2


### RESPONSE HISTORY BY SUPERPOSITION

# Displacement
q = np.dot(U,r)

# Velocity
v = np.dot(U,dr)

# Acceleration
a = np.dot(U,ddr)

### POST-PROCESSING

# plot response at dofplot
dofplot = (41-1)*6+2

# Plot response history using matplotlib
plt.close('all')

# update font size
plt.rcParams.update({'font.size': 14})

fig1 = plt.figure(1,figsize=[16,4])      # plot of disp and acc

fig1.add_subplot(121)           # subplot 1 = disp
plt.plot(t,q[dofplot-1,:],color='blue')
plt.title('fore-aft disp - node 41')
plt.xlabel(r'$t$ - [s]')
plt.ylabel(r'$v_{41}$ - [m]')
plt.grid()
#plt.xlim([0, 60])
#plt.ylim([-0.01, 0.01])

fig1.add_subplot(122)           # subplot 2 = acc
plt.plot(t,a[dofplot-1,:],color='red')
plt.title('fore-aft acc - node 41')
plt.xlabel(r'$t$ - [s]')
plt.ylabel(r'$\ddot{v}_{41}$ - [m/s$^2$]')
plt.grid()
#plt.xlim([0, 60])
#plt.ylim([-5, 5])
#plt.show()

# Save response history to .png file
plt.savefig('modal_response_owt.png',bbox_inches='tight')

fig2 = plt.figure(2,figsize=[16,4])      # plot of disp and acc

fig2.add_subplot(121)           # subplot 1 = disp
plt.plot(t,Em[1-1,:],color='blue')
plt.plot(t,Em[2-1,:],color='red')
plt.plot(t,Em[13-1,:],color='magenta')
plt.title('Modal energy')
plt.xlabel(r'$t$ - [s]')
plt.ylabel(r'$E_{j}$ - [J]')
plt.grid()
#plt.xlim([0, 60])
#plt.ylim([-0.01, 0.01])
plt.show()
