#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 22:33:49 2017

@author: MMGF2
"""
import numpy, random

# generate the same dataset over and over again
numpy.random.seed(100)
classA = [(random.normalvariate(-1.5, 0.5), random.normalvariate(0.5, 1), 1.0) for i in range(30)] + \
          [(random.normalvariate(1.5, 0.3), random.normalvariate(0.5, 1), 1.0) for i in range(30)]
           
classB = [(random.normalvariate(0.0, 0.5), random.normalvariate(-0.5, 0.5), -1.0) for i in range(60)]

data = classA + classB
random.shuffle(data) 
