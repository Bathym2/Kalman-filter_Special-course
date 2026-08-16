# -*- coding: utf-8 -*-
"""
timestep(omega0,zeta,x0,v0,t0,dt,n,f)

Created: June 2020

@author: Oscar Bondo Ellekvist, s163774

"""

# Import modules
import numpy as np

def timestep(m,c,k,x0,v0,t0,dt,n,f=np.zeros(1,dtype=float)):
    """
    timestep
    --------
    Exact time-stepping for linear, damped, forced oscillator.
    The normalized load is piecewise linear over intervals of length dt.

    Parameters
    ----------
    m      : Float
             mass
    c      : Float
             viscous coefficient
    k      : Float
             stiffness
    x0     : Float
             Initial displacement.
    v0     : Float
             Initial velocity.
    t0     : Float
             Starting time.
    dt     : Float
             Size of time step.
    n      : Integer
             Number of time steps.
    f      : Array
             Load amplitudes. (optional)

    Returns
    -------
    x : Vector
        Response history.
    v : Vector
        Velocity history.
    a : Vector
        Acceleration history.
    f : Vector
        Load history.
    t : Vector
        Discrete times.

    """
    
    # Set load vector
    if np.size(f) != n+1:
        f = np.concatenate((f,np.zeros((n+1-np.size(f)),dtype=float)))
        
    # Determine normalized parameters
    omega0 = np.sqrt(k/m)
    zeta = c/(2*omega0*m)
        
    # Compute coefficient matrices
    omegad = omega0*np.sqrt(1-zeta**2)
    alpha  = omegad*dt
    beta   = zeta*omega0*dt
    A      = np.zeros((2,2),dtype=float)
    A[0,0] = alpha*np.cos(alpha)+beta*np.sin(alpha)
    A[0,1] = np.sin(alpha)
    A[1,0] = -(alpha**2+beta**2)*np.sin(alpha)
    A[1,1] = alpha*np.cos(alpha)-beta*np.sin(alpha)
    A      = A * np.exp(-beta)/alpha
    B      = np.zeros((2,2),dtype=float)
    B[0,0] = 2*alpha*beta*(1-np.exp(-beta)*np.cos(alpha))/(alpha**2+beta**2) \
             + (alpha**2-beta**2)*np.exp(-beta)*np.sin(alpha)/(alpha**2 \
             + beta**2)-np.exp(-beta)*(alpha*np.cos(alpha)+beta*np.sin(alpha))
    B[0,1] = -2*alpha*beta*(1-np.exp(-beta)*np.cos(alpha))/(alpha**2+beta**2) \
             - (alpha**2-beta**2)*np.exp(-beta)*np.sin(alpha) \
             /(alpha**2+beta**2)+alpha
    B[1,0] = (alpha**2+beta**2+beta)*np.exp(-beta)*np.sin(alpha) \
             - alpha*(1-np.exp(-beta)*np.cos(alpha))
    B[1,1] = alpha*(1-np.exp(-beta)*np.cos(alpha)) \
           - beta*np.exp(-beta)*np.sin(alpha)
    B      =  B / (omega0**2*alpha)

    # Set initial conditions
    y      = np.zeros((2,n+1),dtype=float)
    y[0,0] = x0
    y[1,0] = dt*v0
    
    # Compute time history
    for i in range(0,n):
        y[:,i+1] = np.dot(A,y[:,i]) + np.dot(B, np.transpose(f[i:(i+2)]))/m
    
    # Set time vector
    ################################
    t = np.ones((n+1),dtype=float)*t0 + np.arange(0,(n+1),1,dtype=float)*dt
    
    # Set response and velocity history
    x = y[0,:]
    v = y[1,:]/dt
    
    # Calculate acceleration by equation of motion
    a = f/m - 2*zeta*omega0*v - omega0**2*x
    
    return(x,v,a,f,t)
 