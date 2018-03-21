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

# Train a neural network with three layers, given input and output
def create_nn(X, y, gamma, n_iter=60000):

    ncol = len(X[0])
    nrow = len(X)

    # Initialize weights connecting layer 1 to layer 2, 2 to 3
    w1_2 = 2*np.random.random((ncol,nrow)) - 1
    w2_3 = 2*np.random.random((nrow,1)) - 1

    # Initialize biases
    bias_1 = 1.0
    bias_2 = 1.0

    # Initialize output nodes
    l0 = np.array(X)
    l1 = sigmoid(np.dot(l0,w1_2))
    l2 = sigmoid(np.dot(l1,w2_3))

    for j in range(int(n_iter)):

        # Forward propogation: equal to the sigmoid of the dot product of previous layer and weights
        l1 = sigmoid(np.dot(l0,w1_2)) #+ bias_1
        l2 = sigmoid(np.dot(l1,w2_3)) #+ bias_2

        # Calculate the error and amount to alter weights
        l2_error = y - l2
        l2_delta = l2_error*sigmoid(l2,deriv=True)

        l1_error = y - l1
        l1_delta = l1_error*sigmoid(l1,deriv=True)

        # Update weights and biases
        w1_2 -= gamma * l0.T.dot(l1_delta)
        w2_3 -= gamma * l1.T.dot(l2_delta)
        #bias_1 -= gamma * l1_delta
        #bias_2 -= gamma * l2_delta

        # Print error value every 10,000 iterations
        if j%10000 == 0:
            print( "Error after {} iterations: {}".format(j,l2_error))


    # Return the output layer
    return(l2)

# Function to test the input/output of a binary test case
def auto_test(X,y,gamma=0.1,n_iter=60000):
    print("Input vector: ", X)

    l2 = create_nn(X,y,n_iter,gamma)

    # Round each value in the output layer to 0 or 1
    output = [[round(number) for number in row] for row in l2]

    print("Output vector: ", output)
    return(output)

test_vec = [[0],[0],[0],[0],[0],[0],[1],[0]]
output = create_nn(test_vec, test_vec, gamma = .01)
auto_test(test_vec, test_vec)
