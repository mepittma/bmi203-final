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
    for x in range(0,len(long_seq)+1-k): # to create an overlap of 1 bp, specify step size of 16
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

# Inverse function to decode the one-hot examples
def decode(seq):
    to, fro  = ['A','C','T','G'], [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]
    translationdict = dict(zip(fro,to))

    sample = []
    for n in seq:
        sample.append(translationdict.get(n))
    return str(sample)

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

for long_seq in neg_seqs[0:10]:
    neg_kmers = get_kmers(long_seq)
    kmers = kmers + neg_kmers
neg_list = []
for seq in kmers:
    neg_list.append(encode(seq))

# Combine into X and y, adding a column of ones to X to represent the bias node
x = np.concatenate( (np.asarray(pos_list), np.asarray(neg_list)), axis=0 )
y = [[1] for i in range(0,len(pos_list))] + [[0] for i in range(0,len(neg_list))]
test = np.ones((x.shape[0],1,4))
X = np.hstack((test, x)) #bias node: one for each sample(?)

"""
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

    act_list = list()
    epoch_list = list()
    batch_list = list()
    metric_list = list()
    learning_list = list()
    score_list = list()

    param_comparison = []

    for actFunction in act_opts:
        print("Activation fucntion: ", actFunction)
        for epochs in epoch_opts:
            print("Epoch ",epochs)
            for batch_size in batch_opts:
                for metric in metric_opts:
                    for learningRate in LR_opts[actFunction]:

                        # Append parameters to lists
                        act_list.append(actFunction)
                        epoch_list.append(epochs)
                        batch_list.append(batch_size)
                        metric_list.append(metric)
                        learning_list.append(learningRate)

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
                        score_list.append(score)


    # Print out best parameters and their scores
    best = max(param_comparison,key=itemgetter(1))
    worst = min(param_comparison, key=itemgetter(1))
    print("Best parameters: ", best[0])
    print("AUROC score: ", best[1])
    print("\nWorst parameters: ", worst[0])
    print("AUROC score: ", worst[1])

    # Save parameter results to a text file
    with open('output/ParameterSearchResults_{}.tsv'.format(seed), 'w') as f:
        for i in range(len(act_list)):
            value_list = [act_list[i],epoch_list[i],batch_list[i],metric_list[i],learning_list[i],score_list[i]]
            f.write("\t".join(str(v) for v in value_list))
            f.write("\n")
"""

"""
    #SCREW THIS, I'M USING R
    # Plot, coloring by parameter of interest
    print("Lengths of lists:\n",len(epoch_list))
    print("\n",len(act_list))
    print("\n",len(score_list))
    fig, axs = plt.subplots(1, 3, figsize=(9, 3), sharey=True)
    axs[0].scatter(epoch_list, score_list, c=act_list,label=act_list)
    axs[1].scatter(epoch_list, score_list, c=metric_list,label=metric_list)
    axs[2].scatter(batch_list, score_list, c=act_list, label=act_list)
    fig.suptitle('Parameter Grid Search Results')
    axs.legend()
    axs.grid(True)

    fig.savefig("output/ParameterGrid_{}".format(seed))
"""

"""
# RUN FOR IDEAL PARAMETERS

def ideal_run(NN, suff):
    T = nn.trainer(NN, epochs=300, batch_size=50, metric='roc_auc',learningRate=0.5)
    T.train(X,y)

    # Now try it on the testing data
    T.forward(X)
    Z = T.yHat
    lossHistory = T.errorHistory

    # Print out the performance given by AUROC
    score = roc_auc_score(y, Z)
    print("Score of final training: ", score)

    # Plot a figure looking at the error over time
    plt.plot(lossHistory)
    plt.title('Error history')
    plt.xlabel('Epoch Number')
    plt.ylabel('Error')
    plt.savefig("output/LossHistory_{}".format(suff))

    return score

# Initialize and train the network for the ideal case
NN = nn.Neural_Network(inS=18, outS=1, hS=3, depth=4, actFunction='sigmoid')
ideal_run(NN, "3Layer")

# Test on different values of hidden layer size
score_list = []
for i in range(1,20):
    NN = nn.Neural_Network(inS=18, outS=1, hS=i, depth=4, actFunction='sigmoid')
    score_list.append(ideal_run(NN, suff=i))
print(len(score_list))
# create a simple scatterplot to look at this
plt.scatter(range(1,20), score_list)
plt.title('Model score dynamics with respect to changing hiddenLayerSize')
plt.xlabel('Hidden Layer Size')
plt.ylabel('AUROC score')
plt.show()
"""


# OUT-OF-SAMPLE DATA

# Run on the test data, saving out in file with format seq\tscore\n
tfile = "seqs/rap1-lieb-test.txt"
seq_list = []

# Read in and encode the DNA sequences
with open(tfile, 'r') as tf:
    seqs = tf.read().splitlines()
pos_list = []
for seq in seqs:
    seq_list.append(encode(seq))

# Train the neural net
NN = nn.Neural_Network(inS=18, outS=1, hS=3, depth=4, actFunction='sigmoid')
T = nn.trainer(NN, epochs=300, batch_size=50, metric='roc_auc',learningRate=0.5)
T.train(X,y)

# make a prediction for the data, adding a bias vector
encseqs = np.asarray(seq_list)
bias = np.ones((encseqs.shape[0],1,4))
OOS = np.hstack((bias, encseqs))
T.forward(OOS)
Z = T.yHat

"""# Undo DNA encoding
seq_list = T.seq_list
nuc_list = []
for seq in seq_list:
    nuc_list.append(decode(seq))"""

# Print out to file
outfile = 'output/predictions.txt'
with open(outfile,'w') as fh:
    for i in range(len(seq_list)):
        string = "{}\t{}\n".format(seqs[i],Z[i])
        fh.write(string)
