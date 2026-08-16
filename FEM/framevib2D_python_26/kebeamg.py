# -*- coding: utf-8 -*-
"""
kebeamg(Xe,mRNA)

Created: June 2020

@author: Oscar Bondo Ellekvist, s163774
updated 2022

"""

# Import modules
import numpy as np

def kebeamg(Xe,mRNA):
    """
    kebeamg
    ------
    Creates the element stiffness matrix of a beam element.

    Parameters
    ----------
    Xe : Array (2 x 2)
         Nodal coordinates.
    mRNA: weight of RNA in [kg]
    
    Returns
    -------
    Ke : Array (6 x 6)
         Element geometric stiffness matrix.

    """
    
    # Unit directional vector
    a0 = Xe[1,:]-Xe[0,:]       # element vector
    L = np.sqrt(np.dot(a0,a0)) # element length
    n = a0/L                   # unit element vector
    
    
    
    # Local element geometric stiffness matrix
    # and no axial component
    N  = -mRNA*9.82
    
    # geometric element stiffness matrix
    Keg = N/(30*L)* \
         np.array([[  0 , 0 , 0 , 0 , 0 , 0], 
                   [  0 , 0 , 0 , 0 , 0 , 0],
                   [  0 , 0 , 0 , 0 , 0 , 0],
                   [  0 , 0 , 0 , 0 , 0 , 0],
                   [  0 , 0 , 0 , 0 , 0 , 0],
                   [  0 , 0 , 0 , 0 , 0 , 0]])
    
    
    # Transformation matrix
    Ae = np.array([[ n[0] , n[1] , 0 ,  0   ,  0   , 0],
                   [-n[1] , n[0] , 0 ,  0   ,  0   , 0],
                   [  0   ,  0   , 1 ,  0   ,  0   , 0],
                   [  0   ,  0   , 0 , n[0] , n[1] , 0],
                   [  0   ,  0   , 0 ,-n[1] , n[0] , 0],
                   [  0   ,  0   , 0 ,  0   ,  0   , 1]])

    # Global element stiffness matrix
    Keg = np.linalg.multi_dot((np.transpose(Ae),Keg,Ae))
    
    return(Keg)