from nets import neural_net as nn
import numpy as np

NN = nn.Neural_Network(inS=1, outS=1, hS=3)

# 8x3x8 test vector
test_vec = [[0],[0],[0],[0],[0],[1],[0]]

X = np.array(test_vec, dtype=float)
y = np.array(test_vec, dtype=float)

#Train network with new data:
T = nn.trainer(NN)
T.train(X,y)

# Print the result of the training
results = NN.forward(X)
print(results)
# Round each value in the output layer to 0 or 1
output = [[round(number) for number in row] for row in results]
print(output)
