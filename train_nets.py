from nets import neural_net as nn
#import matplotlib.pyplot as plt
import numpy as np

# Instantiate neural network and training classes
NN = nn.Neural_Network(inS=17, outS=1, hS=3, actFunction="sigmoid")
T = nn.trainer(NN, epochs = 400, batch_size = 8, metric = "roc_auc",learningRate="default")

# Ensure that the 8x3x8 encoder problem can be solved by this NN
test_vec = [[0],[0],[0],[0],[0],[1],[0]]
X = np.array(test_vec, dtype=float)
y = np.array(test_vec, dtype=float)
T.train(X,y)
results = NN.forward(X)
print(results)
print([[round(number) for number in row] for row in results])

# Read in and encode the positive and negative sequences to train on Rap1 binding
# files:
"""
nfile = "seqs/filt-negative.txt"
pfile = "seqs/filt-positive.txt"

def encode_17(seq):
    sample = []
    for char in seq:
        gsub(char,)
    return sample

# Function to get kmers of 17 from the negative example file
def choose_17(long_seq):
"""


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
