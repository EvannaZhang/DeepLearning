from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from mlp_numpy import MLP
from modules import CrossEntropy, Linear, ReLU, SoftMax
from sklearn.datasets import make_moons, load_breast_cancer
from numpy import array
from numpy import argmax
import math
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, TensorDataset
from pytorch_mlp import MLP as PMLP
import torch.nn as nn
import torch

# Default constants
DNN_HIDDEN_UNITS_DEFAULT = '20'
LEARNING_RATE_DEFAULT = 1e-4
MAX_EPOCHS_DEFAULT = 1500
EVAL_FREQ_DEFAULT = 20

train_X = None
train_y = None
test_X = None
test_y = None
train_y_encoded = None
test_y_encoded = None

FLAGS = None

relu = ReLU()
softmax = SoftMax()
cross = CrossEntropy()

plot_train = []
plot_test = []
train_loss = []
test_loss = []
plot_x = []

p_plot_train = []
p_plot_test = []
p_train_loss = []
p_test_loss = []
p_plot_x = []

descent = 0

def encode(label):
    encoded = np.zeros((len(label), 2))
    for i in range(len(label)):
        encoded[i][0] = label[i]
        encoded[i][1] = 1-label[i]
    return encoded


def decode(encoded):
    # one hot decoding
    inverted = np.zeros(len(encoded), dtype=np.int16)
    for i in range(len(encoded)):
        if encoded[i][0] >= 0.5:
            inverted[i] = 1
        else:
            inverted[i] = 0
    return inverted

def accuracy(predictions, targets):
    """
        Computes the prediction accuracy, i.e., the average of correct predictions
        of the network.
        Args:
        predictions: 2D float array of size [number_of_data_samples, n_classes]
        labels: 2D int array of size [number_of_data_samples, n_classes] with one-hot encoding of ground-truth labels
        Returns:
        accuracy: scalar float, the accuracy of predictions.
        """
    all = len(predictions)
    hit = 0
    for i in range(all):
        if(predictions[i] == targets[i]):
            hit += 1
    accuracy = hit/all*100
    return accuracy


def training_accuracy(mlp):
    answer = np.zeros((len(train_X), 2))
    for index in range(len(train_X)):
        solution = mlp.forward(train_X[index])
        answer[index] = solution
    decode_answer = decode(answer)
    plot_train.append(accuracy(decode_answer, train_y))
    loss = cross.forward(answer, train_y_encoded)
    train_loss.append(loss/len(train_X))


def test_accuracy(mlp):
    answer = np.zeros((len(test_X), 2))
    for index in range(len(test_X)):
        solution = mlp.forward(test_X[index])
        answer[index] = solution
    decode_answer = decode(answer)
    plot_test.append(accuracy(decode_answer, test_y))
    testloss = cross.forward(answer, test_y_encoded)
    test_loss.append(testloss/len(test_X))


def sto_gd(mlp):
    for tttt in range(FLAGS.max_steps):
        count = -1
        for data in train_X:
            count+=1
            mlp.forward(data)
            temp = cross.backward(mlp.outlayer.params['after_a'], train_y_encoded[count])
            mlp.backward(temp)
            mlp.outlayer.params['weight'] = mlp.outlayer.params['weight'] - FLAGS.learning_rate * mlp.outlayer.grads['weight']
            mlp.outlayer.params['bias'] = mlp.outlayer.params['bias'] - FLAGS.learning_rate * mlp.outlayer.grads['bias']
            for i in range(mlp.num_of_hidden):
                mlp.hiddenlayers[i].params['weight'] = mlp.hiddenlayers[i].params['weight'] -FLAGS.learning_rate * mlp.hiddenlayers[i].grads['weight']
                mlp.hiddenlayers[i].params['bias'] = mlp.hiddenlayers[i].params['bias'] - FLAGS.learning_rate * mlp.hiddenlayers[i].grads['bias']
            
        if tttt%FLAGS.eval_freq == 0:
            plot_x.append(tttt)
            training_accuracy(mlp)
            test_accuracy(mlp)

def train():
    """
    Performs training and evaluation of MLP model.
    NOTE: You should the model on the whole test set each eval_freq iterations.
    """
    # YOUR TRAINING CODE GOES HERE
    
    hidden_units = [int(i) for i in FLAGS.dnn_hidden_units.split(',')]
    num_of_hidden_layer = len(hidden_units)
    mlp = MLP(30, hidden_units, 2)
    sto_gd(mlp)


    fig1 = plt.subplot(2,1,1)
    fig2 = plt.subplot(2,1,2)
    fig1.plot(plot_x, plot_train,  c='red', label='training data accuracy')
    fig1.plot(plot_x, plot_test, c='blue', label='test data accuracy')
    fig1.legend()
    fig2.plot(plot_x, train_loss, c='green', label='train CE loss')
    fig2.plot(plot_x, test_loss, c='yellow', label='test CE loss')
    fig2.legend()
    plt.show()


def main():
    """
    Main function
    """
    print('NUMPY MLP WORKING...')
    train()

def test_eva(mlp, testloader):
    total = 0
    correct = 0
    with torch.no_grad():
        for data in testloader:
            x, y = data
            outputs = mlp(x)
            _, predict = torch.max(outputs.data, 1)
            total += y.size(0)
            correct += (predict == y).sum().item()
    p_plot_test.append(100 * correct / total)


def train_eva(mlp, trainloader):
    total = 0
    correct = 0
    with torch.no_grad():
        for data in trainloader:
            x, y = data
            outputs = mlp(x)
            _, predict = torch.max(outputs.data, 1)
            total += y.size(0)
            correct += (predict == y).sum().item()
    p_plot_train.append(100 * correct / total)

def compute_loss(mlp, loader, optimizer, loss):
    celoss = 0
    for i, data in enumerate(loader, 0):
        inputs, labels = data
        #inputs = inputs.reshape(-1, 2)
        optimizer.zero_grad()
        outputs = mlp(inputs)
        l = loss(outputs, labels)
        celoss += l.item()
        l.backward()
        optimizer.step()
    return celoss/len(loader)

def pymlp():
    print('PYTORCH MLP WORKING...')
    hidden_units = [int(i) for i in FLAGS.dnn_hidden_units.split(',')]
    mlp = PMLP(30, hidden_units, 2)
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(mlp.parameters(), lr=FLAGS.learning_rate)
    trainset = TensorDataset(torch.FloatTensor(train_X), torch.LongTensor(train_y))
    trainloader = torch.utils.data.DataLoader(trainset, batch_size = 1, shuffle=True)
    testset = TensorDataset(torch.FloatTensor(test_X), torch.LongTensor(test_y))
    testloader = torch.utils.data.DataLoader(testset, batch_size = 1, shuffle=True)
    
    for tttt in range(FLAGS.max_steps):
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = mlp(inputs)
            l = loss(outputs, labels)
            l.backward()
            optimizer.step()
        
        if tttt % FLAGS.eval_freq == 0:
            p_plot_x.append(tttt)
            test_eva(mlp, testloader)
            train_eva(mlp, trainloader)
            p_train_loss.append(compute_loss(mlp, trainloader, optimizer, loss))
            p_test_loss.append(compute_loss(mlp, testloader, optimizer, loss))


    fig1 = plt.subplot(2,1,1)
    fig2 = plt.subplot(2,1,2)
    fig1.plot(p_plot_x, p_plot_train,  c='red', label='training data accuracy')
    fig1.plot(p_plot_x, p_plot_test, c='blue', label='test data accuracy')
    fig1.legend()
    fig2.plot(p_plot_x, p_train_loss, c='green', label='train data loss')
    fig2.plot(p_plot_x, p_test_loss, c='yellow', label='test data loss')
    fig2.legend()
    plt.show()


if __name__ == '__main__':
    print('Both modules trained and tested on the same data.')
    sample = load_breast_cancer()
    data = sample.data
    label = sample.target
    train_X, test_X = train_test_split(data, test_size=0.2, random_state=0)
    train_X = np.mat(train_X)
    test_X = np.mat(test_X)
    train_y, test_y = train_test_split(label, test_size=0.2, random_state=0)
    train_y_encoded = encode(train_y)
    test_y_encoded = encode(test_y)

    # Command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dnn_hidden_units', type = str, default = DNN_HIDDEN_UNITS_DEFAULT,
                      help='Comma separated list of number of units in each hidden layer')
    parser.add_argument('--learning_rate', type = float, default = LEARNING_RATE_DEFAULT,
                      help='Learning rate')
    parser.add_argument('--max_steps', type = int, default = MAX_EPOCHS_DEFAULT,
                      help='Number of epochs to run trainer.')
    parser.add_argument('--eval_freq', type=int, default=EVAL_FREQ_DEFAULT,
                          help='Frequency of evaluation on the test set')
    FLAGS, unparsed = parser.parse_known_args()
    main()
    pymlp()

