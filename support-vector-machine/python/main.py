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
            P[i, j] = x[i][2]*x[j][2]*PolyKernel(x[i][0:2], x[j][0:2], 3)
    return P

# solve optimization problem    
def QuadOpt(x, P, slack, C):
    # intialize parameters
    # slack variable introduced
    if slack == 1:
        h = np.row_stack((np.zeros((len(x), 1)), C*np.ones((len(x), 1))))
        G = np.row_stack((np.diag([-1.0]*len(x)), np.diag([1.0]*len(x))))
    else:
        h = np.zeros((len(x), 1))
        G = np.diag([-1.0]*len(x))
    q = -1*np.ones((len(x), 1))

    # call qp
    r = qp(matrix(P), matrix(q), matrix(G), matrix(h))
    alpha = list(r['x'])
    
    # pick out non-zero alpha
    # slack variable introduced
    support = list()
    for i in range(len(alpha)):
        if slack == 1:
            if (alpha[i] > 10e-5) and (alpha[i] < C):
                support.append((x[i][0], x[i][1], x[i][2], alpha[i])) 
        else:
            if alpha[i] > 10e-5:
                support.append((x[i][0], x[i][1], x[i][2], alpha[i]))
    return support

# implement indicator function
def Indicator(xstar, support):
    ind = 0
    for i in range(len(support)):
        ind += support[i][3]*support[i][2]*PolyKernel(xstar, support[i][0:2], 3)
    return ind

# plot the decision boundary 
def BoundaryPlot(support):
    xrange = np.arange(-4, 4, 0.05)
    yrange = np.arange(-4, 4, 0.05)
    grid = matrix([[Indicator([x, y], support) for y in yrange] for x in xrange])
    pylab.contour(xrange, yrange, grid, (-1.0, 0.0, 1.0), colors = ('red', 'black', 'blue'), linewidths = (1, 3, 1))
    pylab.show()
    
def Main():
    slack = 1
    C = 10
    P = MatrixForm(td.data)
    support = QuadOpt(td.data, P, slack, C)
    BoundaryPlot(support)

    
Main()
 