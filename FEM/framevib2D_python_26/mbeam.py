# -*- coding: utf-8 -*-
"""
mbeam(T,X,H,dof)

Created: June 2020

@author: Oscar Bondo Ellekvist, s163774

"""

# Import modules
import numpy as np
from mebeam import mebeam
from assem import assem

def mbeam(T,X,H,dof):
    """
    mbeam
    -----
    Creates and assembles the global mass matrix of elastic beam elements.

    Parameters
    ----------
    T   : Array (nel x 3)
          Topology, [node1 node2 propno].
    X   : Array (nnodes x 2)
          Node coordinates.
    H   : Array (npropno x 4)
          Material properties, [E A I rho*A].
    dof : Integer
          Degrees of freedom per node.

    Returns
    -------
    M : Array (ndof x ndof)
        Global mass matrix.

    """
    
    # Dimensions of FE model
    ndof = np.size(X,0)*dof
    
    # Initialize mass matrix
    M = np.zeros((ndof,ndof))
    
    # Global mass matrix by loop over elements
    for i in range(np.size(T,0)):
        
        # Define element arrays
        Xe = X[T[i,0:2]-1,:] # element node coordinates
        He = H[T[i,2]-1,:]   # element properties
        
        # element mass matrix - loop over elemtype:
        Me = mebeam(Xe,He)
        
        # Element mass into global format
        M = assem(M,Me,T[i,0:3],dof)
        
    return(M)