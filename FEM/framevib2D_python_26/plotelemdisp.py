# -*- coding: utf-8 -*-
"""
plotelemdisp(T,X,Ue,...)

Created: June 2020

@author: Oscar Bondo Ellekvist, s163774

"""

import numpy as np
import matplotlib.pyplot as plt

def plotelemdisp(T,X,Ue,nonum=False,title=None,color=['k','b'],
                 linewidth=1.4,linestyle=['--','-'],fontsize=[16,30],
                 figsize=[10,8]):
    """
    plotelemdisp
    ------------
    Plots element displacement in Ue.
    Supported element types: 
        - 2-node beam element

    Parameters
    ----------
    T         : Array
                Topology matrix.
    X         : Array
                Node coordinates matrix.
    Ue        : Array
                Displacements along the element.
    nonum     : Boolean or array
                If nonum=False (default), node numbers are not plotted.
                If nonum=True, node numbers are plotted.        
                nonum can also be given an array of nodes that should be plotted.
                (optional)
    title     : String
                Figure title. (optional)
    color     : List of strings
                Line color of initial and deformed geometry, respectively, 
                according to Matplotlib colors, color=['k','b'] (default). 
                (optional)            
    linewidth : Float
                Line width, linewidth=1.4 (default). (optional)
    linestyle : List of strings
                Line style of initial and deformed geometry, respectively, 
                linestyle=['--','-'] (default).
                Options: '-', ':', '--', '-.'. (optional)
    fontsize  : List of floats
                Font sizes, fontsize=[16,30] (default), meaning that font
                size of node numbers will be 16 and title will be 30. (optional)
    figsize   : List
                Size of figure in inches, figsize=[10,8] (default). (optional)

    Returns
    -------
    Plot.

    """
    
    ### FIGURE (WINDOW)
    
    plt.figure(figsize=(figsize[0],figsize[1]))
    
    
    ### PLOT PARTS
    
    # Element type
    nnodes = np.size(T,1)-1
    
    if nnodes == 2:
        order = np.array([0,1],dtype=int)
    # Plot elements
    ndata = np.size(Ue,2) # data points per element
    for i in range(np.size(T,0)):
        
        # Initial geometry
        plt.plot(X[T[i,order]-1,0],X[T[i,order]-1,1],
                 color=color[0],linestyle=linestyle[0],linewidth=linewidth)
        
        # Deformed geometry
        X1 = X[T[i,0]-1,:]
        a0 = X[T[i,1]-1,:]-X[T[i,0]-1,:]
        Xd = np.zeros((ndata,np.size(X1)),dtype=float)
        for j in range(0,ndata):
            s = j/(ndata-1)
            Xd[j,:] = X1+a0*s+np.array([Ue[0,i,j],Ue[1,i,j]])
        plt.plot(Xd[:,0],Xd[:,1],color=color[1],linestyle=linestyle[1],
                 linewidth=linewidth)
    
    
    ### AXIS
    
    plt.axis('equal')
    plt.axis('off')
    xlim = plt.xlim()
    ylim = plt.ylim()
    
    
    ### PLOT TEXT
    
    # Plot node numbers
    if nonum is True:
        nonum = np.arange(np.size(X,0))
    if nonum is not False:
        xoffset = 0.01*max((xlim[1]-xlim[0]),(ylim[1]-ylim[0]))
        yoffset = 0.01*max((xlim[1]-xlim[0]),(ylim[1]-ylim[0]))
        for i in range(np.size(nonum)):
            plt.text(X[nonum[i],0]+xoffset,X[nonum[i],1]+yoffset,nonum[i],
                     fontsize=fontsize[0])
    
    
    ### FIGURE LAYOUT
    
    # Title
    if title is not None:
        plt.suptitle(title,fontsize=fontsize[1])
    
    plt.savefig('fig_element_disp.png')
    plt.show(block=False)
    plt.pause(3)
    