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
import matplotlib.pyplot as plt

# define kernel function
def LinKernel(x, y):
    return np.dot(x, y) + 1

def PolyKernel(x, y, p):
    return pow(np.dot(x, y) + 1, p)
    
def RBFKernel(x, y, theta):
    sub = [a[0] - a[1] for a in zip(x, y)]
    return math.exp(-(np.dot(sub, sub))/(2*pow(theta, 2)))
    
def SigKernel(x, y, k, eta):
    return math.tanh(k*np.dot(x, y) - eta)

# build P matrix from data points
def MatrixForm(x):
    P = np.zeros((len(x), len(x)))
    for i in range(len(x)):
        for j in range(len(x)):
            #P[i, j] = x[i][2]*x[j][2]*PolyKernel(x[i][0:2], x[j][0:2], 4)
            #P[i, j] = x[i][2]*x[j][2]*LinKernel(x[i][0:2], x[j][0:2])
            P[i, j] = x[i][2]*x[j][2]*RBFKernel(x[i][0:2], x[j][0:2], 1)
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
            if (alpha[i] > 1e-5) and (alpha[i] < C):
                support.append((x[i][0], x[i][1], x[i][2], alpha[i])) 
        else:
            if alpha[i] > 1e-5:
                support.append((x[i][0], x[i][1], x[i][2], alpha[i]))
    return support

# implement indicator function
def Indicator(xstar, support):
    ind = 0
    for i in range(len(support)):
        #ind += support[i][3]*support[i][2]*PolyKernel(xstar, support[i][0:2], 3)
        #ind += support[i][3]*support[i][2]*LinKernel(xstar, support[i][0:2])
        ind += support[i][3]*support[i][2]*RBFKernel(xstar, support[i][0:2], 1)
    return ind

# plot the decision boundary 
def BoundaryPlot(support):
    xrange = np.arange(-4, 4, 0.05)
    yrange = np.arange(-4, 4, 0.05)
    grid = matrix([[Indicator([x, y], support) for y in yrange] for x in xrange])
    pylab.contour(xrange, yrange, grid, (-1.0, 0.0, 1.0), colors = ('red', 'black', 'blue'), linewidths = (1, 3, 1))
    
    # plot data point
    pylab.hold(True)
    pylab.plot([p[0] for p in td.classA], [p[1] for p in td.classA], 'bo')
    pylab.plot([p[0] for p in td.classB], [p[1] for p in td.classB], 'ro')
    pylab.show()

def Main():
    slack = 1
    C1 = 0.1
    C2 = 10
    C3 = 1000
    
    P = MatrixForm(td.data)
    support1 = QuadOpt(td.data, P, slack, C1)
    support2 = QuadOpt(td.data, P, slack, C2)
    support3 = QuadOpt(td.data, P, slack, C3)
    
    plt.figure(1)
    BoundaryPlot(support1)
    plt.figure(2)
    BoundaryPlot(support2)
    plt.figure(3)
    BoundaryPlot(support3)

    
Main()
 