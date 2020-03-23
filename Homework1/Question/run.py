# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import numpy as np
import time
import random

## Network architecture
NUM_INPUT = 784  # Number of input neurons
NUM_OUTPUT = 10  # Number of output neurons
NUM_CHECK = 5  # Number of examples on which to check the gradient

## Hyperparameters
NUM_HIDDEN = 100
LEARNING_RATE = 0.05
BATCH_SIZE = 64
NUM_EPOCH = 400

print("NUM_HIDDEN: ", NUM_HIDDEN)
print("LEARNING_RATE: ", LEARNING_RATE)
print("BATCH_SIZE: ", BATCH_SIZE)
print("NUM_EPOCH: ", NUM_EPOCH)

# Given a vector w containing all the weights and biased vectors, extract
# and return the individual weights and biases W1, b1, W2, b2.
def unpack (w):
    W1 = np.reshape(w[:NUM_INPUT * NUM_HIDDEN],(NUM_INPUT,NUM_HIDDEN))
    w = w[NUM_INPUT * NUM_HIDDEN:]
    b1 = np.reshape(w[:NUM_HIDDEN], NUM_HIDDEN)
    w = w[NUM_HIDDEN:]
    W2 = np.reshape(w[:NUM_HIDDEN*NUM_OUTPUT], (NUM_HIDDEN,NUM_OUTPUT))
    w = w[NUM_HIDDEN*NUM_OUTPUT:]
    b2 = np.reshape(w,NUM_OUTPUT)
    return W1, b1, W2, b2

# Given individual weights and biases W1, b1, W2, b2, concatenate them and
# return a vector w containing all of them.
def pack (W1, b1, W2, b2):
    W1_ = np.reshape(W1,NUM_INPUT*NUM_HIDDEN)
    # print(W1_.shape)
    W2_ = np.reshape(W2,NUM_HIDDEN*NUM_OUTPUT)
    # print(W2_.shape)
    w = np.concatenate((W1_,b1, W2_, b2))
    # print(w.shape)
    return w

# Load the images and labels from a specified dataset (train or test).
def loadData (which):
    images = np.load("./data/mnist_{}_images.npy".format(which))
    labels = np.load("./data/mnist_{}_labels.npy".format(which))
    return images, labels

## 1. Forward Propagation
# Given training images X, associated labels Y, and a vector of combined weights
# and bias terms w, compute and return the cross-entropy (CE) loss.
def ReLU(X):
   return np.maximum(0,X)

def Softmax(X):
    # expo = np.exp(X)
    # expo_sum = np.sum(np.exp(X))
    # return expo/expo_sum
    ### Compute the softmax in a numerically stable way.
    X = X - np.max(X)
    exp_x = np.exp(X)
    softmax_x = exp_x / np.sum(exp_x)
    return softmax_x
    
def Sgn(X):
    return np.signbit(X).astype(int)

def fCE (X, Y, w):
    # print(X.shape)
    W1, b1, W2, b2 = unpack(w)
    loss = 0.0
    ## your code here
    z1 = X.dot(W1) + b1
    h1 = ReLU(z1)
    z2 = h1.dot(W2) + b2
    Yhat = Softmax(z2)
    loss = -np.sum(Y * np.log(Yhat+1e-7))/Y.shape[0]
    return loss,Yhat,h1,z1

## 2. Backward Propagation
# Given training images X, associated labels Y, and a vector of combined weights
# and bias terms w, compute and return the gradient of fCE. 
def gradCE (X, Y, w, Yhat, h1, z1):
    W1, b1, W2, b2 = unpack(w)
    ## your code here
    delta_W_2 = np.dot(h1.T, Yhat - Y) / Y.shape[0]
    delta_b_2 = np.sum(Yhat - Y, axis=0) / Y.shape[0]
    delta_W_1 = np.dot(X.T, np.dot(Yhat - Y, W2.T) * Sgn(z1)) / Y.shape[0]
    delta_b_1 = np.sum(np.dot(Yhat - Y, W2.T) * Sgn(z1), axis=0) / Y.shape[0]
    delta = pack(delta_W_1, delta_b_1, delta_W_2, delta_b_2)
    return delta

## 3. Parameter Update
# Given training and testing datasets and an initial set of weights/biases,
# train the NN.
def train(trainX, trainY, testX, testY, w):
    ## your code here
    learnRate = LEARNING_RATE
    for epochRound in range(NUM_EPOCH):
        batchIter = get_data_batch(trainX, trainY, BATCH_SIZE, True)
        while True:
            batch_X, batch_Y = next(batchIter)
            if len(batch_X) == 0:
                break
            loss,Yhat,h1,z1 = fCE(batch_X,batch_Y,w)
           
            delta = gradCE(batch_X,batch_Y,w,Yhat,h1,z1)
            W1, b1, W2, b2 = unpack(w)
            delta_W_1, delta_b_1, delta_W_2, delta_b_2 = unpack(delta)
            W1 = W1 - delta_W_1 * learnRate
            b1 = b1 - delta_b_1 * learnRate
            W2 = W2 - delta_W_2 * learnRate
            b2 = b2 - delta_b_2 * learnRate
            w = pack(W1, b1, W2, b2)
            # print(loss)

    lossTr,YhatTr,_,_ = fCE(trainX,trainY,w)
    lossTe,YhatTe,_,_ = fCE(testX,testY,w)
    PredictTr = list(map(lambda x: x == max(x), YhatTr)) * np.ones(shape=YhatTr.shape)
    PredictTe = list(map(lambda x: x == max(x), YhatTe)) * np.ones(shape=YhatTe.shape)
    accTr = sum(sum(PredictTr * trainY)) / len(trainY)
    accTe = sum(sum(PredictTe * testY)) / len(testY)
    print("Train Acc:{0}; Test Acc:{1}".format(accTr,accTe))

def get_data_batch(X, Y, batch_size=None, shuffle=False):
    size = len(X[0])
    indices = list(range(size))
    if shuffle:
        random.shuffle(indices)
    while True:
        batch_indices = np.asarray(indices[0:batch_size])
        indices = indices[batch_size:] #+ indices[:batch_size]
        if len(indices) == 0:
            yield [],[]
        batch_X = X[batch_indices]
        batch_Y = Y[batch_indices]
        yield batch_X,batch_Y


if __name__ == "__main__":
    # Load data
    start_time = time.time()
    trainX, trainY = loadData("train")
    testX, testY = loadData("test")

    print("len(trainX): ", len(trainX))
    print("len(testX): ", len(testX))

    # Initialize weights randomly
    W1 = 2*(np.random.random(size=(NUM_INPUT, NUM_HIDDEN))/NUM_INPUT**0.5) - 1./NUM_INPUT**0.5
    b1 = 0.01 * np.ones(NUM_HIDDEN)
    W2 = 2*(np.random.random(size=(NUM_HIDDEN, NUM_OUTPUT))/NUM_HIDDEN**0.5) - 1./NUM_HIDDEN**0.5
    b2 = 0.01 * np.ones(NUM_OUTPUT)

    w = pack(W1, b1, W2, b2)
    print("Shape of w:",w.shape)

    # # Train the network and report the accuracy on the training and test set.
    train(trainX, trainY, testX, testY, w)


# %%



# %%



# %%



# %%


