# -*- coding: utf-8 -*-
"""
kebeam(Xe,He)

Created: June 2020

@author: Oscar Bondo Ellekvist, s163774

"""

# Import modules
import numpy as np

def kebeam(Xe,He):
    """
    kebeam
    ------
    Creates the element stiffness matrix of a beam element.

    Parameters
    ----------
    Xe : Array (2 x 2)
         Nodal coordinates.
    He : Array (nnodes x 2)
         Element properties, [E A I m].
    
    Returns
    -------
    Ke : Array (6 x 6)
         Element stiffness matrix.

    """
    
    # Unit directional vector
    a0 = Xe[1,:]-Xe[0,:]       # element vector
    L = np.sqrt(np.dot(a0,a0)) # element length
    n = a0/L                   # unit element vector
    
    # Element properties
    E = He[0]
    A = He[1]
    I = He[2]
    
    # soil stiffness if available
    ks = 0
    if np.size(He) > 4:
        ks = He[4]
    
    # Local element constitutive stiffness matrix
    Kec = np.array([[E*A/L  , 0            , 0           ,
                    -E*A/L , 0            ,     0],
                   [0      , 12*E*I/L**3  , 6*E*I/L**2  , 
                    0      , -12*E*I/L**3 , 6*E*I/L**2],
                   [0      , 6*E*I/L**2   , 4*E*I/L     , 
                    0      , -6*E*I/L**2  ,  2*E*I/L],
                   [-E*A/L , 0            , 0           , 
                    E*A/L  , 0            ,    0],
                   [0      , -12*E*I/L**3 , -6*E*I/L**2 , 
                    0      , 12*E*I/L**3  , -6*E*I/L**2],
                   [0      , 6*E*I/L**2   , 2*E*I/L     , 
                    0      , -6*E*I/L**2  , 4*E*I/L     ]])
    
    # Local element soil stiffness matrix = mass matrix with me = ks
    # and no axial component
    Kes = ks*L/420* \
         np.array([[  0 ,     0 ,      0 ,   0 ,     0 ,      0], 
                   [  0 ,   156 ,   22*L ,   0 ,    54 ,  -13*L],
                   [  0 ,  22*L ,  4*L**2 ,  0 ,  13*L , -3*L**2],
                   [  0 ,     0 ,      0 ,   0 ,     0 ,      0],
                   [  0 ,    54 ,   13*L ,   0 ,   156 ,  -22*L],
                   [  0 , -13*L , -3*L**2 ,  0 , -22*L ,  4*L**2]])
    
    
    # total element stiffness matrix
    Ke = Kec + Kes
    
    # Transformation matrix
    Ae = np.array([[ n[0] , n[1] , 0 ,  0   ,  0   , 0],
                   [-n[1] , n[0] , 0 ,  0   ,  0   , 0],
                   [  0   ,  0   , 1 ,  0   ,  0   , 0],
                   [  0   ,  0   , 0 , n[0] , n[1] , 0],
                   [  0   ,  0   , 0 ,-n[1] , n[0] , 0],
                   [  0   ,  0   , 0 ,  0   ,  0   , 1]])

    # Global element stiffness matrix
    Ke = np.linalg.multi_dot((np.transpose(Ae),Ke,Ae))
    
    return(Ke)