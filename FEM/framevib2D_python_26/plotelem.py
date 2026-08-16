# -*- coding: utf-8 -*-
"""
plotelem(T,X,...)

Created: June 2020

@author: Oscar Bondo Ellekvist, s163774

"""

import numpy as np
import matplotlib.pyplot as plt

def plotelem(T,X,nonum=True,elnum=False,title=None,color='b',
             linewidth=1.4,linestyle='-',fontsize=[16,30],figsize=[10,8]):
    """
    plotelem
    --------
    Plots elements in topology matrix T with coordinate matrix X. 
    Uses linear line segment between all nodes.
    Supported element types: 
        - 2-node beam element

    Parameters
    ----------
    T         : Array
                Topology matrix.
    X         : Array
                Node coordinates matrix.
    nonum     : Boolean or array
                If nonum=True (default), node numbers are plotted.
                If nonum=False, node numbers are not plotted.
                nonum can also be given an array of nodes that should be
                plotted. (optional)
    elnum     : Boolean or array
                If elnum=True (default), element numbers are plotted.
                If elnum=False, element numbers are not plotted.
                elnum can also be given an array of elements that should be 
                plotted. (optional)
    title     : String
                Figure title. (optional)
    color     : String
                Line color according to Matplotlib colors, color='b' (default). 
                (optional)
    linewidth : Float
                Line width, linewidth=1.4 (default). (optional)
    linestyle : String
                Line style, linestyle='-' (default).
                Options: '-', ':', '--', '-.'. (optional)
    fontsize  : List of floats
                Font sizes, fontsize=[16,30] (default), meaning that font
                size of node/element numbers will be 16 and title will be 30. 
                (optional)
    figsize   : List
                Size of figure in inches, figsize=[10,8] (default). (optional)

    Returns
    -------
    Plot.

    """
    
    ### PROPERTIES
            
    # Node diameter
    nodediam = 100
    

    ### FIGURE (WINDOW)
    
    plt.figure(figsize=(figsize[0],figsize[1]))
    
    
    ### PLOT PARTS
    
    # Element type
    nnodes = np.size(T,1)-1
    if nnodes == 2:
        order = np.array([0,1],dtype=int)
    else:
        return
    # Plot elements
    for i in range(np.size(T,0)):
        plt.plot(X[T[i,order]-1,0],X[T[i,order]-1,1],
                 color=color,linestyle=linestyle,linewidth=linewidth)
        
    # Plot nodes
    plt.scatter(X[:,0],X[:,1],
                s=nodediam,facecolors='none',edgecolors=color,
                linewidth=linewidth)
    
    
    ### AXIS
    
    plt.axis('equal')
    plt.axis('off')
    xlim = plt.xlim()
    ylim = plt.ylim()

    
    ### PLOT TEXT
    
    # Plot node numbers
    if nonum is True:
        nonum = np.arange(np.size(X,0))+1
    if nonum is not False:
        xoffset = 0.015*max((xlim[1]-xlim[0]),(ylim[1]-ylim[0]))
        yoffset = 0.015*max((xlim[1]-xlim[0]),(ylim[1]-ylim[0]))
        for i in range(np.size(nonum)):
            plt.text(X[nonum[i]-1,0]+xoffset,X[nonum[i]-1,1]+yoffset,nonum[i],
                     fontsize=fontsize[0])
            
    # Plot element numbers
    if elnum is True:
        elnum = np.arange(np.size(T,0))+1
    if elnum is not False:
        for i in range(np.size(elnum)):
            plt.text(np.sum(X[T[elnum[i]-1,order]-1,0])/nnodes,
                     np.sum(X[T[elnum[i]-1,order]-1,1])/nnodes,
                     elnum[i],bbox=dict(facecolor='w',edgecolor=color),
                     fontsize=fontsize[0])
    
    
    ### FIGURE LAYOUT
    
    # Title
    if title is not None:
        plt.suptitle(title,fontsize=fontsize[1])
    
    plt.savefig('fig_element.png')
    plt.show(block=False)
    plt.pause(3)
    
        