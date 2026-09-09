# -*- coding: utf-8 -*-
"""
lingalpha(K,C,M,x0,v0,dt,N,F,alpham,alphaf)

Created: June 2020

@author: Oscar Bondo Ellekvist, s163774

"""

import numpy as np

def lingalpha(K,C,M,x0,v0,dt,N,F=None,alpham=0,alphaf=0):
    """
    lingalpha
    ---------
    Evaluates the forced dynamic response of a linear system with n degrees of
    freedom at equally spaced times by the (linear) generalized alpha procedure.

    Parameters
    ----------
    K      : Array
             Stiffness matrix.
    C      : Array
             Damping matrix.
    M      : Array
             Mass matrix.
    x0     : Array
             Initial displacements.
    v0     : Array
             Initial velocities.
    dt     : Float
             Size of time step.
    N      : Integer
             Number of time steps.
    F      : Array
             Load amplitudes. (optional)        
    alpham : Float
             Relative weight of the old inertial term, alpham=0 (default).
    alphaf : Float
             Relative weight of the old force terms from damping, stiffness and
             external damping, alphaf=0 (default).

    Returns
    -------
    x : Array
        Response history.
    v : Array
        Velocity history.
    a : Array
        Acceleration history.
    t : Array
        Discrete times.

    """
    
    # Newmark parameters
    gamma = 0.5 + alphaf - alpham
    beta = 0.25*(gamma + 0.5)**2
    
    # Number of dofs
    ndof = np.size(x0)
    
    # Set load vector
    if F is None:
        F = np.zeros((ndof,N+1),dtype=float)
    elif np.size(F,1) < (N+1):
        F = np.concatenate((F,np.zeros((ndof,N+1-np.size(F,1)),
                                       dtype=float)),axis=1)
    
    # Initialize output arrays
    x = np.zeros((ndof,N+1),dtype=float)
    v = np.zeros((ndof,N+1),dtype=float)
    a = np.zeros((ndof,N+1),dtype=float)
    t = np.zeros(N+1,dtype=float)
    
    # Modified mass matrix
    MM = (1-alpham)*M + (1-alphaf)*(gamma*dt*C+beta*dt**2*K)
    M1 = np.linalg.inv(MM)
    
    # Initial values
    t[0] = 0.0
    x[:,0] = np.ravel(x0)
    v[:,0] = np.ravel(v0)
    a[:,0] = np.linalg.solve(M,F[:,0]) - np.dot(C,v[:,0]) - np.dot(K,x[:,0])
    
    # Time incrementation loop
    for i in range(0,N):
        
        t[i+1] = t[i] + dt
        
        # Increment predictors
        dvp = dt*a[:,i]
        dxp = dt*v[:,i] + 0.5*dt**2*a[:,i]
        dF  = F[:,i+1] - F[:,i]
        
        # Acceleration increment
        da = np.dot(M1,F[:,i]
                    -(np.dot(M,a[:,i])+np.dot(C,v[:,i])+np.dot(K,x[:,i]))
                    +(1-alphaf)*(dF-np.dot(C,dvp)-np.dot(K,dxp)) )
                    
        # State vector update
        a[:,i+1] = a[:,i] + da
        v[:,i+1] = v[:,i] + dvp + gamma*dt*da
        x[:,i+1] = x[:,i] + dxp + beta*dt**2*da
        
    return(x,v,a,t)
