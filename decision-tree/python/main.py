#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 12:58:18 2017

"""

import numpy as np
import monkdata as m
import dtree as d
import drawtree_qt5 as dqt
import matplotlib.pyplot as plt


## Assignment 1: calculate entropy of a dataset
en_m1 = d.entropy(m.monk1)
en_m2 = d.entropy(m.monk2)
en_m3 = d.entropy(m.monk3)

# output print
print '-------- Assignment 1 --------'
print 'entropy:'
print 'monk 1: ' + str(en_m1)
print 'monk 2: ' + str(en_m2)
print 'monk 3: ' + str(en_m3)
print ''


## Assignment 3: calculate information gain
Ga_m1 = np.empty([6,1], dtype = float)
Ga_m2 = np.empty([6,1], dtype = float)
Ga_m3 = np.empty([6,1], dtype = float)

for i in range(0,6):
    Ga_m1[i] = d.averageGain(m.monk1, m.attributes[i])
    Ga_m2[i] = d.averageGain(m.monk2, m.attributes[i])
    Ga_m3[i] = d.averageGain(m.monk3, m.attributes[i])

# output print
print '-------- Assignment 3 --------'
print 'information gain:'
print 'monk 1 (a1->a6): ' + str(np.transpose(Ga_m1))
print 'monk 2 (a1->a6): ' + str(np.transpose(Ga_m2))
print 'monk 3 (a1->a6): ' + str(np.transpose(Ga_m3))
print ''


## Assignment 5: build decision tree
Ga_m11 = np.empty([6,4], dtype = float)
en_m11 = np.empty([1,4], dtype = float)

for i in range(0,4):
    en_m11[0,i] = d.entropy(d.select(m.monk1, m.attributes[4], (i+1)))
    for j in range(0,6):
        Ga_m11[j,i] = d.averageGain(d.select(m.monk1, m.attributes[4], (i+1)), m.attributes[j])

# majority class
#mc = d.mostCommon(d.select(m.monk1, m.attributes[4], 1))

t1 = d.buildTree(m.monk1, m.attributes)
t2 = d.buildTree(m.monk2, m.attributes)
t3 = d.buildTree(m.monk3, m.attributes)

# output print
print '-------- Assignment 5 --------'
print 'decision tree of monk1:'
print(t1)
print 'train set error: ' + str(round((1-d.check(t1, m.monk1))*100, 2)) + '%'
print 'test set error: ' + str(round((1-d.check(t1, m.monk1test))*100, 2)) + '%'
print ''
print 'decision tree of monk2:'
print(t2)
print 'train set error: ' + str(round((1-d.check(t2, m.monk2))*100, 2)) + '%'
print 'test set error: ' + str(round((1-d.check(t2, m.monk2test))*100, 2)) + '%'
print ''
print 'decision tree of monk3:'
print(t3)
print 'train set error: ' + str(round((1-d.check(t3, m.monk3))*100, 2)) + '%'
print 'test set error: ' + str(round((1-d.check(t3, m.monk3test))*100, 2)) + '%'
print ''

#dqt.drawTree(t1)
#dqt.drawTree(t2)
#dqt.drawTree(t3)


## Assignment 7: reduced error pruning
fraction = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
err_test1, var_test1 = d.errReducedPruned(m.monk1, m.monk1test, m.attributes, fraction, 500)
err_test3, var_test3 = d.errReducedPruned(m.monk3, m.monk3test, m.attributes, fraction, 500)

# plot results
plt.figure(4)
plt.subplot(2, 1, 1)
plt.title('Mean Error vs. Fraction - 500 runs')
plt.xlabel('Fraction')
plt.ylabel('Mean Error')
line1, = plt.plot(fraction, err_test1, 'bo-', label="monk 1")
line3, = plt.plot(fraction, err_test3, 'ro-', label="monk 3")
plt.legend(handles = [line1, line3])
plt.subplot(2, 1, 2)
plt.title('Variance vs. Fraction - 500 runs')
plt.xlabel('Fraction')
plt.ylabel('Variance')
line1, = plt.plot(fraction, var_test1, 'b*--', label="monk 1")
line3, = plt.plot(fraction, var_test3, 'r*--', label="monk 3")
plt.legend(handles = [line1, line3])
plt.show()

