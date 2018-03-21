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


"""
plt.figure(figsize=(12, 9))
plt.subplot(311)
plt.plot(lossHistory)
plt.subplot(312)
plt.plot(H, '-*')
plt.subplot(313)
plt.plot(x, Y, 'ro')    # training data
plt.plot(X[:, 1], Z, 'bo')   # learned
plt.show()

print('[', inputLayerSize, hiddenLayerSize, outputLayerSize, ']',
      'Activation:', activation, 'Iterations:', epochs,
      'Learning rate:', L, 'Final loss:', mse, 'Time:', end - start)
"""
