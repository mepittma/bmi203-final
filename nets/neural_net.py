# Endless thanks to Stephen C Welch: https://github.com/stephencwelch/Neural-Networks-Demystified

# Classes that follow are mostly his, except that I have added cross-validation
# functionality, alternative activation and optimization functions (instead of
# stealing scipy's), learning rate, and a bias node.

# I also initialize random start weights according to the best practices outlined
# here: https://stats.stackexchange.com/questions/47590/what-are-good-initial-weights-in-a-neural-network

# Tests to check functions appearing in this script are in test/test_algs.py

import numpy as np
from scipy import optimize

# Class for a neural network
class Neural_Network(object):
    def __init__(self,inS,outS,hS,n_iter=200,actFunction="sigmoid",learningRate=1.0, optim="BFGS"):

        #Define Hyperparameters
        self.inputLayerSize = inS
        self.outputLayerSize = outS
        self.hiddenLayerSize = hS
        self.actFunction = actFunction
        self.learningRate = learningRate
        self.optimizer = optim

        #Weights (parameters)
        # Random initial weights
        r0 = math.sqrt(2.0/(inputLayerSize))
        r1 = math.sqrt(2.0/(hiddenLayerSize))
        self.W1 = np.random.uniform(size=(self.inputLayerSize, self.hiddenLayerSize),low=-r0,high=r0)
        self.W2 = np.random.uniform(size=(self.hiddenLayerSize,self.outputLayerSize),low=-r1,high=r1)

    def forward(self, X):

        # add a bias unit to the input layer
        X = np.concatenate((np.atleast_2d(np.ones(X.shape[0])).T, X), axis=1)

        #Propogate inputs though network
        self.z2 = np.dot(X, self.W1)
        self.a2 = self.sigmoid(self.z2)
        self.z3 = np.dot(self.a2, self.W2)
        yHat = self.sigmoid(self.z3)
        return yHat

    def sigmoid(self, z):

        if actFunction == "sigmoid":
            return 1/(1+np.exp(-z))

        if actFunction == "softMax":
            return

        if actFunction =="tanh":
            return

    def sigmoidPrime(self,z):

        if actFunction == "sigmoid":
            return np.exp(-z)/((1+np.exp(-z))**2)

        if actFunction == "softMax":
            return

        if actFunction =="tanh":
            return

    def costFunction(self, X, y):
        #Compute cost for given X,y, use weights already stored in class.
        self.yHat = self.forward(X)
        J = 0.5*sum((y-self.yHat)**2)
        return J

    def costFunctionPrime(self, X, y):
        #Compute derivative with respect to W1 and W2 for a given X and y:
        self.yHat = self.forward(X)

        delta3 = np.multiply(-(y-self.yHat), self.sigmoidPrime(self.z3))
        dJdW2 = np.dot(self.a2.T, delta3)

        delta2 = np.dot(delta3, self.W2.T)*self.sigmoidPrime(self.z2)
        dJdW1 = np.dot(X.T, delta2)

        return dJdW1, dJdW2

    #Helper Functions for interacting with other classes:
    def getParams(self):
        #Get W1 and W2 unrolled into vector:
        params = np.concatenate((self.W1.ravel(), self.W2.ravel()))
        return params

    def setParams(self, params):
        #Set W1 and W2 using single paramater vector.
        W1_start = 0
        W1_end = self.hiddenLayerSize * self.inputLayerSize
        self.W1 = np.reshape(params[W1_start:W1_end], (self.inputLayerSize, self.hiddenLayerSize))
        W2_end = W1_end + self.hiddenLayerSize*self.outputLayerSize
        self.W2 = np.reshape(params[W1_end:W2_end], (self.hiddenLayerSize, self.outputLayerSize))

    def computeGradients(self, X, y):
        dJdW1, dJdW2 = self.costFunctionPrime(X, y)
        return np.concatenate((dJdW1.ravel(), dJdW2.ravel()))

def computeNumericalGradient(N, X, y):
        paramsInitial = N.getParams()
        numgrad = np.zeros(paramsInitial.shape)
        perturb = np.zeros(paramsInitial.shape)
        e = 1e-4

        for p in range(len(paramsInitial)):
            #Set perturbation vector
            perturb[p] = e
            N.setParams(paramsInitial + perturb)
            loss2 = N.costFunction(X, y)

            N.setParams(paramsInitial - perturb)
            loss1 = N.costFunction(X, y)

            #Compute Numerical Gradient
            numgrad[p] = (loss2 - loss1) / (2*e)

            #Return the value we changed to zero:
            perturb[p] = 0

        #Return Params to original value:
        N.setParams(paramsInitial)

        return numgrad


from scipy import optimize


class trainer(object):
    def __init__(self, N):
        #Make Local reference to network:
        self.N = N

    def callbackF(self, params):
        self.N.setParams(params)
        self.J.append(self.N.costFunction(self.X, self.y))

    def costFunctionWrapper(self, params, X, y):
        self.N.setParams(params)
        cost = self.N.costFunction(X, y)
        grad = self.N.computeGradients(X,y)
        return cost, grad

    def train(self, X, y):
        #Make an internal variable for the callback function:
        self.X = X
        self.y = y

        #Make empty list to store costs:
        self.J = []

        params0 = self.N.getParams()

        options = {'maxiter': n_iter, 'disp' : True}
        _res = optimize.minimize(self.costFunctionWrapper, params0, jac=True, method=optimizer, \
                                 args=(X, y), options=options, callback=self.callbackF)

        self.N.setParams(_res.x)
        self.optimizationResults = _res
