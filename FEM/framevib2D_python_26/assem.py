# -*- coding: utf-8 -*-
"""
assem(A,Ae,Te,dof)

Created: June 2020

@author: Oscar Bondo Ellekvist, s163774

"""

# Import modules
import numpy as np

def assem(A,Ae,Te,dof):
    """
    assem
    -----
    Assembles system matrix or vector by adding element contributions from 
    elements to an existing global matrix. The system matrix may be square 
    like the stiffness mass or conductivity matrix, or may be a one-dimensional
    vector like the load vector.

    Parameters
    ----------
    A   : Array
          Global matrix.
    Ae  : Array
          Element matrix.
    Te  : Array
          Element topology vector.
    dof : Integer
          Degrees of freedom per node.

    Returns
    -------
    A : Array
        Updated global matrix.

    """
    
    # Number of element nodes MOD
    enodes = np.size(Te)-1
    
    # Define global address vector for element dofs
    ig = np.zeros(enodes*dof,dtype=int)
    for i in range(enodes):
        ig[np.arange(i*dof,(i+1)*dof)] = np.arange((Te[i]-1)*dof,Te[i]*dof)

    # Add element matrix/vector to global matrix/vector
    if np.size(A,1) != np.size(A,0):
        A[ig] = A[ig] + Ae
    else:
        A[np.ix_(ig,ig)] = A[np.ix_(ig,ig)] + Ae

        
    return(A)