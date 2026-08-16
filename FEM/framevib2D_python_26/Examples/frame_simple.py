# -*- coding: utf-8 -*-
"""
Simply supported portal frame for vibration analysis.

Created: June 2020

@author: Oscar Bondo Ellekvist, s163774

"""

import numpy as np

# Length of columns
a = 5
# Length of beams
b = 10

# Coordinates of nodes, X = [x y]
X = np.array([[0 ,   0],
              [b ,   0],
              [0 ,   a],
              [b ,   a],
              [0 , 2*a],
              [b , 2*a],
              [0 , 3*a],
              [b , 3*a]])

# Topology matrix, T = [node1 node2 propno]
T = np.array([[1,3,1],
              [2,4,1],
              [3,4,1],
              [3,5,1],
              [4,6,1],
              [5,6,1],
              [5,7,1],
              [6,8,1],
              [7,8,1]])

# Element property matrix, H = [E  A  I rho*A]
H = np.array([[100e9 , 10e-3 , 100e-6 , 100]])

# Boundary conditions, BC = [node dof]
BC = np.array([[1 , 1],
               [1 , 2],
               [2 , 1],
               [2 , 2]])
