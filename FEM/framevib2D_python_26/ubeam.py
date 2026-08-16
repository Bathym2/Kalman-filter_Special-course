# -*- coding: utf-8 -*-
"""
ubeam(T,X,U,ns)

Created: June 2020

@author: Oscar Bondo Ellekvist, s163774

"""

import numpy as np
from nebeam import nebeam

def ubeam(T,X,U,ns):
    """
    ubeam
    -----
    Calculates displacements along the elements in a group of
    linear elastic beams.

    Parameters
    ----------
    T  : Array
         Topology matrix.
    X  : Array
         Node coordinates matrix.
    U  : Array (nnodes x dof)
         Element array of nodal displacements.
    ns : Integer
         Number of data points along each element.

    Returns
    -------
    Uen : Array
          Displacements in elements.

    """
    
    # Initialize element displacement array
    Uen = np.zeros((2,np.size(T,0),ns),dtype=float)
    
    # Loop over elements: Calculate element displacements
    for i in range(np.size(T,0)):
        
        # Define element arrays
        Xe = X[T[i,0:2]-1,:]
        
        # Element displacement 2d array
        Ue = U[T[i,0:2]-1,:]
        
        # Element displacement 1d array
        Ue = Ue.reshape((6,1))
        
        # Unit directional vector
        a0 = Xe[1,:]-Xe[0,:]        # element vector
        L  = np.sqrt(np.dot(a0,a0)) # element length
        n  = a0/L                   # unit element vector
        
        # Transformation matrix
        Ae = np.array([[ n[0] , n[1] , 0 ,  0   ,  0   , 0],
                       [-n[1] , n[0] , 0 ,  0   ,  0   , 0],
                       [  0   ,  0   , 1 ,  0   ,  0   , 0],
                       [  0   ,  0   , 0 , n[0] , n[1] , 0],
                       [  0   ,  0   , 0 ,-n[1] , n[0] , 0],
                       [  0   ,  0   , 0 ,  0   ,  0   , 1]])
        
        # Translational transformation matrix
        An = np.array([[n[0],n[1]],[-n[1],n[0]]])
        
        # Displacements in local coordinates
        Ue = np.dot(Ae,Ue)
        
        # Calculate global element displacements
        for j in range(0,ns):
            
            # normalized coordinate
            s = j/(ns-1)
            
            # element interpolation matrix - loop over elemtype:
            Ne = nebeam(Xe,s)
            
            # global element displacements
            Uen[:,i,j] = np.linalg.multi_dot((np.transpose(An),Ne,Ue)).reshape(2)
    
    return(Uen)
