#!/usr/bin/python
# coding: utf-8

import numpy as np
from scipy import misc
from imp import reload
from labfuns import *
import random

#============= Bayes classifier =============#

# in: labels - N vector of class labels
#          W - N x 1 weight matrix
# out: prior - C x 1 vector of class priors
def computePrior(labels, W):
    Npts = labels.shape[0]
    if W is None:
        W = np.ones((Npts, 1))/float(Npts)
    else:
        assert(W.shape[0] == Npts)
    classes = np.unique(labels)
    Nclasses = np.size(classes)

    prior = np.zeros((Nclasses, 1))

    for jdx, c in enumerate(classes):
        idx = np.where(labels == c)[0]
        wlc = W[idx, :]
        prior[jdx, :] = np.sum(wlc)/np.sum(W)

    return prior

# in:      X - N x d matrix of N data points
#     labels - N vector of class labels
#          W - N x 1 weight matrix
# out:    mu - C x d matrix of class means (mu[i] - class i mean)
#      sigma - C x d x d matrix of class covariances (sigma[i] - class i sigma)
def mlParams(X, labels, W):
    assert(X.shape[0] == labels.shape[0])
    Npts, Ndims = np.shape(X)
    classes = np.unique(labels)
    Nclasses = np.size(classes)

    if W is None:
        W = np.ones((Npts, 1))/float(Npts)

    mu = np.zeros((Nclasses, Ndims))
    sigma = np.zeros((Nclasses, Ndims, Ndims))

    for jdx, c in enumerate(classes):
        idx = np.where(labels == c)[0]
        xlc = X[idx, :]
        wlc = W[idx, :]

        # using matrix multiplication to get equivalent result
        # and avoid loops
        mu[jdx, :] = np.dot(np.transpose(wlc), xlc)/np.sum(wlc)
        sigma[jdx, :, :] = np.diag((np.dot(np.transpose(wlc), (xlc - mu[jdx, :])**2)/np.sum(wlc))[0])
      
    return mu, sigma

# in:      X - N x d matrix of N data points
#      prior - C x 1 matrix of class priors
#         mu - C x d matrix of class means (mu[i] - class i mean)
#      sigma - C x d x d matrix of class covariances (sigma[i] - class i sigma)
# out:     h - N vector of class predictions for test points
def classifyBayes(X, prior, mu, sigma):

    Npts = X.shape[0]
    Nclasses, Ndims = np.shape(mu)
    logProb = np.zeros((Nclasses, Npts))

    for jdx in range(Nclasses):
        u = mu[jdx, :]
        delta = X - u
        sig = sigma[jdx, :, :]
        d_sig = np.diag(sig)
        inv_sig = np.diag(1.0/d_sig)
        
        t1 = -0.5*np.log(np.linalg.det(sig))
        t2 = -0.5*np.diag(np.dot(np.dot(delta, inv_sig), np.transpose(delta)))
        t3 = np.log(prior[jdx, :])
        
        logProb[jdx, :] = t1 + t2 + t3

    # finding max a-posteriori after computing the log posterior
    h = np.argmax(logProb, axis = 0)
    return h

class BayesClassifier(object):
    def __init__(self):
        self.trained = False

    def trainClassifier(self, X, labels, W = None):
        rtn = BayesClassifier()
        rtn.prior = computePrior(labels, W)
        rtn.mu, rtn.sigma = mlParams(X, labels, W)
        rtn.trained = True
        return rtn

    def classify(self, X):
        return classifyBayes(X, self.prior, self.mu, self.sigma)
        

        
#=============== Boosting =================#

# in: base_classifier - a classifier of the type that we will boost, e.g. BayesClassifier
#                   X - N x d matrix of N data points
#              labels - N vector of class labels
#                   T - number of boosting iterations
# out:    classifiers - (maximum) length T Python list of trained classifiers
#              alphas - (maximum) length T Python list of vote weights
def trainBoost(base_classifier, X, labels, T = 10):
    # these will come in handy later on
    Npts, Ndims = np.shape(X)

    classifiers = []
    alphas = [] 

    # The weights for the first iteration
    wCur = np.ones((Npts, 1))/float(Npts)

    for i_iter in range(0, T):
        # a new classifier can be trained like this, given the current weights
        classifiers.append(base_classifier.trainClassifier(X, labels, wCur))

        # do classification for each point
        vote = classifiers[-1].classify(X)
        delta = np.zeros((Npts, 1))
        para = np.zeros((Npts, 1))
        
        for i in range(Npts):
            if vote[i] == labels[i]:
                delta[i] = 1
        
        error = np.sum(wCur*(1 - delta))
        alpha = 0.5*(np.log(1 - error) - np.log(error))
        alphas.append(alpha)
        
        for i in range(Npts):
            if delta[i] == 1:
                para[i] = np.exp(-alpha)
            else:
                para[i] = np.exp(alpha)
        
        wCur = (wCur*para)/float(np.sum(wCur*para))
        
    return classifiers, alphas

# in:       X - N x d matrix of N data points
# classifiers - (maximum) length T Python list of trained classifiers as above
#      alphas - (maximum) length T Python list of vote weights
#    Nclasses - the number of different classes
# out:  yPred - N vector of class predictions for test points
def classifyBoost(X, classifiers, alphas, Nclasses):
    Npts = X.shape[0]
    Ncomps = len(classifiers)

    # if we only have one classifier, we may just classify directly
    if Ncomps == 1:
        return classifiers[0].classify(X)
    else:
        votes = np.zeros((Npts, Nclasses))

        # implement classificiation when we have trained several classifiers
        for t in range(Ncomps):
            h = classifiers[t].classify(X)
            for i in range(Npts):
                votes[i, h[i]] += alphas[t]

        # compute yPred after accumulating the votes
        return np.argmax(votes, axis = 1)

class BoostClassifier(object):
    def __init__(self, base_classifier, T = 10):
        self.base_classifier = base_classifier
        self.T = T
        self.trained = False

    def trainClassifier(self, X, labels):
        rtn = BoostClassifier(self.base_classifier, self.T)
        rtn.nbr_classes = np.size(np.unique(labels))
        rtn.classifiers, rtn.alphas = trainBoost(self.base_classifier, X, labels, self.T)
        rtn.trained = True
        return rtn

    def classify(self, X):
        return classifyBoost(X, self.classifiers, self.alphas, self.nbr_classes)

        

#========== Test the Maximum Likelihood estimates ==========#
#X, labels = genBlobs(centers = 5)
#mu, sigma = mlParams(X, labels, None)
#plotGaussian(X, labels, mu, sigma)



#============= Test with 'iris' & 'vowel' ===============#
## Bayesian & 'iris'
#testClassifier(BayesClassifier(), dataset = 'iris', split = 0.7)
#plotBoundary(BayesClassifier(), dataset = 'iris', split = 0.7)

## Bayesian & boost & 'iris'
#testClassifier(BoostClassifier(BayesClassifier(), T = 10), dataset = 'iris', split = 0.7)
#plotBoundary(BoostClassifier(BayesClassifier()), dataset = 'iris', split = 0.7)

## Bayesian & 'vowel'
#testClassifier(BayesClassifier(), dataset = 'vowel', split = 0.7)
#plotBoundary(BayesClassifier(), dataset = 'vowel', split = 0.7)

## Bayesian & boost & 'vowel'
#testClassifier(BoostClassifier(BayesClassifier(), T = 10), dataset = 'vowel', split = 0.7)
#plotBoundary(BoostClassifier(BayesClassifier()), dataset = 'vowel', split = 0.7)

## decision tree & 'iris'
#testClassifier(DecisionTreeClassifier(), dataset = 'iris', split = 0.7)
#plotBoundary(DecisionTreeClassifier(), dataset = 'iris', split = 0.7)

## decision tree & boost & 'iris'
#testClassifier(BoostClassifier(DecisionTreeClassifier(), T = 10), dataset = 'iris', split = 0.7)
#plotBoundary(BoostClassifier(DecisionTreeClassifier(), T = 10), dataset = 'iris', split = 0.7)

# decision tree & 'vowel'
#testClassifier(DecisionTreeClassifier(), dataset = 'vowel', split = 0.7)
#plotBoundary(DecisionTreeClassifier(), dataset = 'vowel', split = 0.7)

# decision tree & boost & 'vowel'
#testClassifier(BoostClassifier(DecisionTreeClassifier(), T = 10), dataset = 'vowel', split = 0.7)
#plotBoundary(BoostClassifier(DecisionTreeClassifier(), T = 10), dataset = 'vowel', split = 0.7)



#================== Classifying faces ===================#
# first check how a boosted decision tree classifier performs on the olivetti data
testClassifier(BayesClassifier(), dataset = 'olivetti', split = 0.7, dim = 20)
#testClassifier(DecisionTreeClassifier(), dataset = 'olivetti', split = 0.7, dim = 20)
#testClassifier(BoostClassifier(BayesClassifier(), T = 10), dataset = 'olivetti', split = 0.7, dim = 20)
#testClassifier(BoostClassifier(DecisionTreeClassifier(), T = 10), dataset = 'olivetti', split = 0.7, dim = 20)

# fetch the olivetti data
X,y,pcadim = fetchDataset('olivetti')

# split into training and testing
xTr,yTr,xTe,yTe,trIdx,teIdx = trteSplitEven(X, y, 0.7)

# use PCA to reduce the dimension to 20
pca = decomposition.PCA(n_components = 20)

# use training data to fit the transform
pca.fit(xTr)

# apply on training and test data
xTrpca = pca.transform(xTr)
xTepca = pca.transform(xTe)

# use our pre-defined decision tree classifier together with the implemented
# boosting to classify data points in the training data
classifier = BayesClassifier().trainClassifier(xTrpca, yTr)
#classifier = DecisionTreeClassifier().trainClassifier(xTrpca, yTr)
#classifier = BoostClassifier(BayesClassifier(), T = 10).trainClassifier(xTrpca, yTr)
#classifier = BoostClassifier(DecisionTreeClassifier(), T = 10).trainClassifier(xTrpca, yTr)
yPr = classifier.classify(xTepca)

# choose a test point to visualize
testind = random.randint(0, xTe.shape[0] - 1)

# visualize the test point together with the training points used to train
# the class that the test point was classified to belong to
visualizeOlivettiVectors(xTr[yTr == yPr[testind], :], xTe[testind, :])

