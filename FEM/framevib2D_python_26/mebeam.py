# -*- coding: utf-8 -*-
"""
mebeam(Xe,He)

Created: June 2020

@author: Oscar Bondo Ellekvist, s163774

"""

# Import modules
import numpy as np

def mebeam(Xe,He):
    """
    mebeam
    ------
    Creates the element mass matrix of a beam element.

    Parameters
    ----------
    Xe : Array (2 x 2)
         Nodal coordinates.
    He : Array (nnodes x 2)
         Element properties, [E A I m].
    
    Returns
    -------
    Me : Array (6 x 6)
         Element mass matrix.

    """
    
    # Unit directional vector
    a0 = Xe[1,:]-Xe[0,:]       # element vector
    L = np.sqrt(np.dot(a0,a0)) # element length
    n = a0/L                   # unit element vector
    
    # Element properties
    me = He[4-1]
    
    # Local element mass matrix
    Me = me*L/420* \
         np.array([[140 ,     0 ,      0 ,  70 ,     0 ,      0], 
                   [  0 ,   156 ,   22*L ,   0 ,    54 ,  -13*L],
                   [  0 ,  22*L ,  4*L**2 ,   0 ,  13*L , -3*L**2],
                   [ 70 ,     0 ,      0 , 140 ,     0 ,      0],
                   [  0 ,    54 ,   13*L ,   0 ,   156 ,  -22*L],
                   [  0 , -13*L , -3*L**2 ,   0 , -22*L ,  4*L**2]])
    
    # Transformation matrix
    Ae = np.array([[ n[0] , n[1] , 0 ,  0   ,  0   , 0],
                   [-n[1] , n[0] , 0 ,  0   ,  0   , 0],
                   [  0   ,  0   , 1 ,  0   ,  0   , 0],
                   [  0   ,  0   , 0 , n[0] , n[1] , 0],
                   [  0   ,  0   , 0 ,-n[1] , n[0] , 0],
                   [  0   ,  0   , 0 ,  0   ,  0   , 1]])

    # Global element mass matrix
    Me = np.linalg.multi_dot((np.transpose(Ae),Me,Ae))
    
    return(Me)