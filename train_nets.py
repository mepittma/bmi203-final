from nets import neural_net as nn
import matplotlib.pyplot as plt
from operator import itemgetter
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.model_selection import train_test_split
import numpy as np

"""
# AUTOENCODER
# Instantiate neural network and training classes
NN = nn.Neural_Network(inS=1, outS=1, hS=3, depth = 1, actFunction="sigmoid")
T = nn.trainer(NN, epochs = 400, batch_size = 8, metric = "roc_auc",learningRate="default")

# Ensure that the 8x3x8 encoder problem can be solved by this NN
test_vec = [[0],[0],[0],[0],[0],[0],[1],[0]]
print("Input: ", test_vec)
X = np.array(test_vec, dtype=float)
y = np.array(test_vec, dtype=float)
T.train(X,y)
results = T.prediction
print("Prediction: ", results)
"""

# Function to get kmers of 17 from the negative example file
# Step size = 8 because any shorter and the sequences will share >50% identity...
# not as interesting as new sequences
def get_kmers(long_seq, k=17):
    kmers = []
    for x in range(0,len(long_seq)+1-k, 16): #Create an overlap of 1bp
        seq = long_seq[x:x+k]
        if seq not in kmers:
            kmers.append(seq)
    return kmers

# Function to convert a DNA sequence into a list in which each element is an
# encoded nucleotide (A,C,T,G->1000,0100,0010,0001)
def encode(seq):
    fro, to = ['A','C','T','G'], [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
    translationdict = dict(zip(fro, to))

    sample = []
    for n in seq:
        sample.append(translationdict.get(n))
    return np.asarray(sample)

# Read in and encode the positive and negative sequences to train on Rap1 binding
nfile = "seqs/filt-negative.txt"
pfile = "seqs/filt-positive.txt"

# Read in and encode the DNA sequences
with open(pfile, 'r') as pf:
    pos_seqs = pf.read().splitlines()
pos_list = []
for seq in pos_seqs:
    pos_list.append(encode(seq))

with open(nfile, 'r') as nf:
    neg_seqs = nf.read().splitlines()
kmers = []

for long_seq in neg_seqs[0:2]:
    neg_kmers = get_kmers(long_seq)
    kmers = kmers + neg_kmers
neg_list = []
for seq in kmers:
    neg_list.append(encode(seq))

#print(neg_list)
# Combine into X and y, adding a column of ones to X to represent the bias node
x = np.concatenate( (np.asarray(pos_list), np.asarray(neg_list)), axis=0 )
y = [[1] for i in range(0,len(pos_list))] + [[0] for i in range(0,len(neg_list))]
test = np.ones((x.shape[0],1,4))
X = np.hstack((test, x)) #bias node: one for each sample(?)

# Parameter grid search
act_opts = ["sigmoid","ReLU","tanh"]
epoch_opts = np.arange(300,500,50)
batch_opts = np.arange(50,250,25)
metric_opts = ["roc_auc", "mse"]
LR_opts = {"sigmoid": np.arange(.20,.80,.05),\
            "ReLU":   np.arange(.0002,.0008,.00005),\
            "tanh":   np.arange(.002,.008,.0005)}

# Test each possible parameter combination
for seed in [1,42,7]:

    print("Random seed: ", seed)

    # Split the test/train data by random seed
    print("Length of first axis: ", len(X[0]))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)

    param_comparison = []

    for actFunction in act_opts:
        for epochs in epoch_opts:
            for batch_size in batch_opts:
                for metric in metric_opts:
                    for learningRate in LR_opts[actFunction]:

                        # Initialize and train the network
                        NN = nn.Neural_Network(inS=18, outS=1, hS=3, depth=4, actFunction=actFunction)
                        T = nn.trainer(NN, epochs, batch_size, metric,learningRate)
                        T.train(X_train,y_train)

                        # Now try it on the testing data
                        T.forward(X_test)
                        Z = T.yHat
                        score = roc_auc_score(y_test, Z)

                        # Save parameter dict and accuracy result in a list of tuples
                        param_comparison.append(tuple([T.params, score]))


    # Print out best parameters and their scores
    best = max(param_comparison,key=itemgetter(1))
    worst = min(param_comparison, key=itemgetter(1))
    print("Best parameters: ", best[0])
    print("AUROC score: ", best[1])
    print("\nWorst parameters: ", worst[0])
    print("AUROC score: ", worst[1])

    # Save parameter results to a text file
    with open('output/ParameterSearchResults_{}.tsv'.format(seed), 'w') as f:
        for tup in param_comparison:
            f.write("\t".join(tup[0],tup[1]))
            f.write("\n")

    # Plot, coloring by parameter of interest
    fig, axs = plt.subplots(1, 3, figsize=(9, 3), sharey=True)
    params = param_comparison[0]
    axs[0].scatter(params["Number Epochs"], param_comparison[1], c=params["Activation Function"],label=params["Activation Function"])
    axs[1].scatter(params["Number Epochs"], param_comparison[1], c=params["Optimization Metric"],label=params["Activation Function"])
    axs[2].scatter(params["Training Batch Size"], param_comparison[1], c=params["Activation Function"], label=params["Activation Function"])
    fig.suptitle('Parameter Grid Search Results')
    axs.legend()
    axs.grid(True)

    fig.savefig("output/ParameterGrid_{}".format(seed))


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
