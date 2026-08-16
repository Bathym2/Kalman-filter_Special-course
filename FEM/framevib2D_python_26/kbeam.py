# -*- coding: utf-8 -*-
"""
kbeam(T,X,H,dof)

Created: June 2020

@author: Oscar Bondo Ellekvist, s163774

"""

# Import modules
import numpy as np
from kebeam import kebeam
from assem import assem

def kbeam(T,X,H,dof):
    """
    kbeam
    -----
    Creates and assembles the global stiffness matrix of elastic beam elements.

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
    K : Array (ndof x ndof)
        Global stiffness matrix.

    """
    
    # Dimensions of FE model
    ndof = np.size(X,0)*dof
    
    # Initialize stiffness matrix
    K = np.zeros((ndof,ndof))
    
    # Global stiffness matrix by loop over elements
    for i in range(np.size(T,0)):
        
        # Define element arrays
        Xe = X[T[i,0:2]-1,:] # element node coordinates
        He = H[T[i,2]-1,:]   # element properties
        
        # element stiffness matrix - loop over elemtype:
        Ke = kebeam(Xe,He)
        
        # Element stiffness into global format
        K = assem(K,Ke,T[i,0:3],dof)
        
    return(K)