# -*- coding: utf-8 -*-
"""
kbeamg(T,X,mRNA,dof)

Created: June 2020

@author: Oscar Bondo Ellekvist, s163774
update 2022

"""

# Import modules
import numpy as np
from kebeamg import kebeamg
from assem import assem

def kbeamg(T,X,mRNA,dof):
    """
    kbeamg
    -----
    Creates and assembles the global stiffness matrix of elastic beam elements.

    Parameters
    ----------
    T   : Array (nel x 3)
          Topology, [node1 node2 propno].
    X   : Array (nnodes x 2)
          Node coordinates.
    mRNA: Weight of RNA in [kg]
    dof : Integer
          Degrees of freedom per node.

    Returns
    -------
    K : Array (ndof x ndof)
        Global geometric stiffness matrix.

    """
    
    # Dimensions of FE model
    ndof = np.size(X,0)*dof
    
    # Initialize stiffness matrix
    Kg = np.zeros((ndof,ndof))
    
    # Global stiffness matrix by loop over elements
    for i in range(np.size(T,0)):
        
        # Define element arrays
        Xe = X[T[i,0:2]-1,:] # element node coordinates
        
        # element stiffness matrix - loop over elemtype:
        Keg = kebeamg(Xe,mRNA)
        
        # Element stiffness into global format
        Kg = assem(Kg,Keg,T[i,0:3],dof)
        
    return(Kg)