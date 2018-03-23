# Code modeled heavily off the follwing two sources:
# Stephen C Welch: https://github.com/stephencwelch/Neural-Networks-Demystified
# Alan Richmond http://python3.codes/a-neural-network-in-python-part-2-activation-functions-bias-sgd-etc/

# Classes that follow are mostly theirs, except that I have added cross-validation
# functionality, alternative activation functions, homebrewed SGD (instead of
# stealing scipy's), alternative score metrics, learning rate, and a bias node.

# I also initialize random start weights according to the best practices outlined
# here: https://stats.stackexchange.com/questions/47590/what-are-good-initial-weights-in-a-neural-network

# Tests to check functions appearing in this script are in test/test_algs.py

import math, random
import numpy as np
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
        r0 = math.sqrt(2.0/(self.inputLayerSize))
        r1 = math.sqrt(2.0/(self.hiddenLayerSize))
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

    def activate(self, z):

        if self.actFunction == "sigmoid":
            return 1/(1+np.exp(-z))
        if self.actFunction == "ReLU":
            return z * (z > 0) #0 if z<=0; otherwise, = z
        if self.actFunction =="tanh":
            return np.tanh(z)

    def activatePrime(self,z):

        if self.actFunction == "sigmoid":
            return np.exp(-z)/((1+np.exp(-z))**2)
        if self.actFunction == "ReLU":
            return z > 0 #1 if z>0, 0 if z<=0
        if self.actFunction =="tanh":
            return 1 - z**2


class trainer(object):
    def __init__(self, N, epochs = 400, batch_size = 100, metric = "roc_auc",learningRate="default"):

        #Make Local reference to network, data, parameters:
        self.N = N

        self.epochs = epochs
        self.batch_size = batch_size
        self.metric = metric

        #Unless otherwise specified, define learning rate based on which activation function was chosen
        if learningRate == "default":
            if self.N.actFunction == "sigmoid":
                self.learningRate = 0.45
            if self.N.actFunction == "ReLU":
                self.learningRate = 0.0005
            if self.N.actFunction == "tanh":
                self.learningRate = 0.005
        else:
            self.learningRate = learningRate

    # Function to train the NN in batches, using stratified sampling
    def next_batch(self, X, y):

        for i in np.arange(0, X.shape[0], self.batch_size):

            # Row indices that indicate a positive example
            pos_list = [ i for i, response in enumerate(y) if response == [1] ]

            # Row indices that indicate a negative example
            neg_list = [ i for i, response in enumerate(y) if response == [0] ]

            # Initialize the batch with one positive and one negative example
            # (ensuring that there is always at least one of each class - AUROC gets mad otherwise)
            idx_list = []
            idx_list.append(random.choice(pos_list))
            idx_list.append(random.choice(neg_list))

            # Draw a sample from one of the two groups with equal probability
            for j in range(self.batch_size-2):
                if bool(random.getrandbits(1)) == True:
                    idx_list.append(random.choice(pos_list))
                else:
                    idx_list.append(random.choice(neg_list))

        yield (X[idx_list], y[idx_list])

    # Function to keep track of which parameters yielded a given result
    #def print_params(self):


    def train(self, X, y):
        lossHistory = []

        for i in range(self.epochs):
            epochLoss = []

            for (Xb, Yb) in self.next_batch(X, y):

                H = self.N.activate(np.dot(Xb, self.N.W1))            # hidden layer results
                Z = self.N.activate(np.dot(H,  self.N.W2))            # output layer results
                E = Yb - Z                              # how much we missed (error)

                # Implement error metric of choice
                if self.metric == "mse":
                    epochLoss.append(np.sum(E**2))
                if self.metric == "roc_auc":
                    false_positive_rate, true_positive_rate, thresholds = roc_curve(Yb, Z)
                    epochLoss.append(1 - roc_auc_score(Yb, Z))

                dZ = E * self.N.activatePrime(Z)                    # delta Z - change in output error wrt W2
                dH = dZ.dot(self.N.W2.T) * self.N.activatePrime(H)  # delta H - change in hidden error wrt W1
                self.N.W2 += H.T.dot(dZ) * self.learningRate        # update output layer weights
                self.N.W1 += Xb.T.dot(dH) * self.learningRate       # update hidden layer weights

            error = np.average(epochLoss)
            lossHistory.append(error)

        H = self.N.activate(np.dot(X, self.N.W1))
        Z = self.N.activate(np.dot(H, self.N.W2))

        self.errorHistory = lossHistory
        self.prediction = Z
