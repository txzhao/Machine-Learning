#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 11:56:42 2017

"""

from cvxopt.solvers import qp
from cvxopt.base import matrix
import numpy as np
import pylab, math
import testdata as td

# define kernel function
def LinKernel(x, y):
    return np.dot(x, y) + 1

def PolyKernel(x, y, p):
    return pow(np.dot(x, y) + 1, p)
    
def RBFKernel(x, y, theta):
    return math.exp(-np.dot(x - y, x - y)/(2*pow(theta, 2)))
    
def SigKernel(x, y, k, eta):
    return math.tanh(k*np.dot(x, y) - eta)

# build P matrix from data points
def MatrixForm(x):
    P = np.zeros((len(x), len(x)))
    for i in range(len(x)):
        for j in range(len(x)):
            P[i, j] = x[i][2]*x[j][2]*PolyKernel(x[i][0:2], x[j][0:2], 4)
    return P

# solve optimization problem    
def QuadOpt(x, P):
    # intialize parameters
    q = -1*np.ones((len(x), 1))
    h = np.zeros((len(x), 1))
    G = np.diag([-1.0]*len(x))

    # call qp
    r = qp(matrix(P), matrix(q), matrix(G), matrix(h))
    alpha = list(r['x'])
    
    # pick out non-zero alpha
    alphalist = list()
    for i in range(len(alpha)):
        if alpha[i] > 10e-5:
            alphalist.append((x[i][0], x[i][1], x[i][2], alpha[i])) 
    return alphalist

# implement indicator function
def Indicator(xstar, alphalist):
    ind = 0
    for i in range(len(alphalist)):
        ind += alphalist[i][3]*alphalist[i][2]*PolyKernel(xstar, alphalist[i][0:2], 4)
    return ind

# plot the decision boundary 
def BoundaryPlot(alphalist):
    xrange = np.arange(-4, 4, 0.05)
    yrange = np.arange(-4, 4, 0.05)
    grid = matrix([[Indicator([x, y], alphalist) for y in yrange] for x in xrange])
    pylab.contour(xrange, yrange, grid, (-1.0, 0.0, 1.0), colors = ('red', 'black', 'blue'), linewidths = (1, 3, 1))
    pylab.show()
    
def Main():
    P = MatrixForm(td.data)
    alphalist = QuadOpt(td.data, P)    
    BoundaryPlot(alphalist)

    
Main()
 