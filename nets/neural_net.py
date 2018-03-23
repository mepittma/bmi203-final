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
    def __init__(self,inS,outS,hS,depth,actFunction="sigmoid"):

        #Define Hyperparameters
        self.inputLayerSize = inS
        self.outputLayerSize = outS
        self.hiddenLayerSize = hS
        self.depth = depth
        self.actFunction = actFunction

        #Weights (parameters)
        # Random initial weights
        r0 = math.sqrt(2.0/(self.inputLayerSize))
        r1 = math.sqrt(2.0/(self.hiddenLayerSize))
        self.W1 = np.random.uniform(size=(self.inputLayerSize, self.hiddenLayerSize, self.depth),low=-r0,high=r0)
        self.W2 = np.random.uniform(size=(self.hiddenLayerSize,self.outputLayerSize),low=-r1,high=r1)


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

        yield(np.take(X, idx_list, axis=0), np.take(y, idx_list, axis=0))

    def train(self, X, y):
        lossHistory = []

        for i in range(self.epochs):
            epochLoss = []

            for (Xa, Ya) in self.next_batch(X, y):

                #Xb = np.reshape(Xa,[self.N.inputLayerSize,-1])
                # Multiply each response by four in place
                #Yb = []
                #for j in Ya:
                #    Yb = Yb + [j,j,j,j]

                H = self.N.activate(np.tensordot(Xa, self.N.W1, axes=([1,2],[0,2])))          # hidden layer results
                Z = self.N.activate(np.dot(H,  self.N.W2))            # output layer results
                E = Ya - Z                              # how much we missed (error)                            # how much we missed (error)

                # Implement error metric of choice
                if self.metric == "mse":
                    epochLoss.append(np.sum(E**2))
                if self.metric == "roc_auc":
                    false_positive_rate, true_positive_rate, thresholds = roc_curve(Ya, Z)
                    epochLoss.append(1 - roc_auc_score(Ya, Z))

                dZ = E * self.N.activatePrime(Z)                    # delta Z - change in output error wrt W2
                dH = dZ.dot(self.N.W2.T) * self.N.activatePrime(H)  # delta H - change in hidden error wrt W1
                self.N.W2 += H.T.dot(dZ) * self.learningRate        # update output layer weights
                self.N.W1 += np.einsum('ijk,il->jlk', Xa, dH)       # update hidden layer weights


            error = np.average(epochLoss)
            lossHistory.append(error)

        # reshape X and predict
        #X = np.reshape(X, [self.N.inputLayerSize,-1])
        H = self.N.activate(np.tensordot(X, self.N.W1, axes = ([1,2],[0,2])))
        Z = self.N.activate(np.dot(H, self.N.W2))

        self.errorHistory = lossHistory
        self.prediction = Z

        self.params = {"Input Size":self.N.inputLayerSize, "Output Size":self.N.outputLayerSize,\
                        "Hidden Layers": self.N.hiddenLayerSize, "Activation Function":self.N.actFunction,\
                        "Number Epochs":self.epochs,"Training Batch Size":self.batch_size,\
                        "Optimization Metric":self.metric, "Learning Rate":self.learningRate}

    # Function to output the predicted results of test/unknown data
    def forward(self, X):

        #Propogate inputs though network
        self.z2 = np.tensordot(X, self.N.W1, axes=([1,2],[0,2]))
        self.a2 = self.N.activate(self.z2)
        self.z3 = np.dot(self.a2, self.N.W2)
        self.yHat = self.N.activate(self.z3)
