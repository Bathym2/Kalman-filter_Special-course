# -*- coding: utf-8 -*-
"""
Cantilever beam for vibration analysis.

Created: June 2020

@author: Oscar Bondo Ellekvist, s163774

"""

import numpy as np

# Lenght of beam
a = 10

# Number of elements
n = 4

# Coordinates of nodes, X = [x y]
X = np.stack((np.linspace(0,a,n+1),np.zeros((n+1))),axis=1)

# Topology matrix, T = [node1 node2 propno]
T = np.stack((np.arange(1,n+1,dtype=int),np.arange(2,n+2,dtype=int),
              np.ones(n,dtype=int)),axis=1)

# Element property matrix, H = [E  A  I rho*A]
H = np.array([[100e9 , 10e-3 , 100e-6 , 100]])

# Boundary conditions, BC = [node dof]
BC = np.array([[1 , 1],
               [1 , 2],
               [1 , 3]])
