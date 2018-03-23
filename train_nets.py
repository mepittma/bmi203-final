from nets import neural_net as nn
import matplotlib.pyplot as plt
import numpy as np

# Instantiate neural network and training classes
NN = nn.Neural_Network(inS=1, outS=1, hS=3, actFunction="sigmoid")
T = nn.trainer(NN, epochs = 400, batch_size = 8, metric = "roc_auc",learningRate="default")

# Ensure that the 8x3x8 encoder problem can be solved by this NN
test_vec = [[0],[0],[0],[0],[0],[0],[1],[0]]
X = np.array(test_vec, dtype=float)
y = np.array(test_vec, dtype=float)
T.train(X,y)
results = T.prediction
print(results)
print([[round(number) for number in row] for row in results])
#print(T.params)

# Read in and encode the positive and negative sequences to train on Rap1 binding
nfile = "seqs/filt-negative.txt"
pfile = "seqs/filt-positive.txt"

# Function to get kmers of 17 from the negative example file
def get_kmers(long_seq, k=17):
    kmers = []
    for x in range(len(long_seq)+1-k):
        kmers.append(long_seq[x:x+k])
    return kmers

# Function to convert a DNA sequence into a list in which each element is an
# encoded nucleotide (A,C,T,G->1000,0100,0010,0001)
def encode(seq):
    fro, to = ['A','C','T','G'], [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
    translationdict = dict(zip(fro, to))

    sample = []
    for n in seq:
        sample.append(translationdict.get(n))
    return sample


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
