# Endless thanks to Stephen C Welch: https://github.com/stephencwelch/Neural-Networks-Demystified

# Classes that follow are mostly his, except that I have added cross-validation
# functionality, alternative activation functions, homebrewed SGD (instead of
# stealing scipy's), alternative score metrics, learning rate, and a bias node.

# I also initialize random start weights according to the best practices outlined
# here: https://stats.stackexchange.com/questions/47590/what-are-good-initial-weights-in-a-neural-network
# Thanks, Alan Richmond http://python3.codes/a-neural-network-in-python-part-2-activation-functions-bias-sgd-etc/

# Tests to check functions appearing in this script are in test/test_algs.py

import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, roc_auc_score

# Class for a neural network
class Neural_Network(object):
    def __init__(self,inS,outS,hS,actFunction="sigmoid"):

        #Define Hyperparameters
        self.inputLayerSize = inS
        self.outputLayerSize = outS
        self.hiddenLayerSize = hS
        self.actFunction = actFunction

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
        if actFunction == "ReLU":
            return z * (z > 0) #0 if z<=0; otherwise, = z
        if actFunction =="tanh":
            return np.tanh(z)

    def sigmoidPrime(self,z):

        if actFunction == "sigmoid":
            return np.exp(-z)/((1+np.exp(-z))**2)
        if actFunction == "ReLU":
            return z > 0 #1 if z>0, 0 if z<=0
        if actFunction =="tanh":
            return 1 - z**2

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


class trainer(object):
    def __init__(self, N, X, y, epochs = 400, batch_size = 100, metric = "roc_auc",learningRate="default"):

        #Make Local reference to network, data, parameters:
        self.N = N
        self.X = X
        self.y = y

        self.epochs = epochs
        self.batch_size = batch_size
        self.metric = metric

        #Unless otherwise specified, define learning rate based on which activation function was chosen
        if learningRate == "default":
            if actFunction == "sigmoid":
                self.learningRate = 0.45
            if actFunction == "ReLU":
                self.learningRate = 0.0005
            if actFunction == "tanh":
                self.learningRate = 0.005

    # Function to train the NN in batches, using stratified sampling
    def next_batch(X, y):
    for i in np.arange(0, X.shape[0], batchSize):

        # Row indices that indicate a positive example
        pos_list =

        # Row indices that indicate a negative example
        neg_list =

        # Draw a sample from one of the two groups with equal probability
        idx_list = []
        for j in range(batchSize):
            if bool(random.getrandbits(1)) == True:
                idx_list.append(random.choice(pos_list))
            else:
                idx_list.append(random.choice(neg_list))

        yield (X[idx_list], y[idx_list])


    def train(self, X, y):
        lossHistory = []

        for i in range(epochs):
            epochLoss = []

            for (Xb, Yb) in next_batch(X, y):

                H = sigmoid(np.dot(Xb, Wh))            # hidden layer results
                Z = activate(np.dot(H,  Wz))            # output layer results
                E = Yb - Z                              # how much we missed (error)

                # Implement error metric of choice
                if metric == "mse":
                    epochLoss.append(np.sum(E**2))
                if metric == "roc_auc":
                    false_positive_rate, true_positive_rate, thresholds = roc_curve(Yb, Z)
                    epochLoss.append(1 - roc_auc_score(y, est.predict(X)))

                dZ = E * activatePrime(Z)               # delta Z
                dH = dZ.dot(Wz.T) * activatePrime(H)    # delta H
                Wz += H.T.dot(dZ) * learningRate        # update output layer weights
                Wh += Xb.T.dot(dH) * learningRate       # update hidden layer weights

            error = np.average(epochLoss)
            lossHistory.append(error)

        H = activate(np.dot(X, Wh))
        Z = activate(np.dot(H, Wz))

        return(Z)
