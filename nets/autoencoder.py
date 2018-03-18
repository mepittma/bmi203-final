# Implement an 8x3x8 autoencoder. This neural network should take a matrix
# input and returns the same matrix as an output.

# First, represent the neural network as a list of layers, where each layer in
# the network is represented as a class with a weight matrix, bias vector,
# activation function, function's derivative.

import numpy as np
np.random.seed(1)

# Sigmoid function (from https://iamtrask.github.io/2015/07/12/basic-python-network/)
def sigmoid(x, deriv = False):
    if deriv == True:
        return x*(1-x)
    return 1/(1+np.exp(-x))
