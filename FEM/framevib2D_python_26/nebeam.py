# -*- coding: utf-8 -*-
"""
nebeam(Xe,s)

Created: June 2020

@author: Oscar Bondo Ellekvist, s163774

"""

import numpy as np

def nebeam(Xe,s):
    """
    nebeam
    ------
    Calculates element displacements in an elastic beam element.

    Parameters
    ----------
    Xe : Array
         Initial coordinates, Xe = [[x1,y1],[x2,y2]].
    s  : Float
         Element coordinate.

    Returns
    -------
    Ne : Array
         Element interpolation matrix.

    """
    
    # Calculate element length
    a0 = Xe[1,:]-Xe[0,:] # element vector
    L  = np.sqrt(np.dot(a0,a0))  # element length
    
    # Define interpolation matrix
    Ne = np.array([[1-s,0,0,s,0,0],
                   [0,1-3*s**2+2*s**3,(s-2*s**2+s**3)*L,
                    0,3*s**2-2*s**3,(-s**2+s**3)*L]])
    
    return(Ne)


