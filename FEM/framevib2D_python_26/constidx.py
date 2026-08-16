# -*- coding: utf-8 -*-
"""
constidx(C,dof,ndof)

Created: June 2020

@author: Oscar Bondo Ellekvist, s163774

"""

import numpy as np

def constidx(C,dof,ndof):
    """
    constidx
    --------
    Evaluation of constrained/unconstrained index sets.

    Parameters
    ----------
    C    : Array
           Constraint matrix.
    dof  : Integer
           Degrees of freedom per node.
    ndof : Integer
           Degrees of freedom in total.

    Returns
    -------
    ic : Array
         Index set of constrained dofs.
    iu : Array
         Index set of unconstrained dofs.
    
    """
    
    # Constrained dofs
    ic = (C[:,0]-1)*dof+C[:,1]
    
    # Unconstrained dofs
    iu = np.setdiff1d(np.arange(1,ndof+1,dtype=int),ic)
    
    return(ic,iu)

