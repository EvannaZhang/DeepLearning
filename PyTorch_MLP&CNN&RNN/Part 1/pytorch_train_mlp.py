from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
import torch.nn as nn
import torch
from pytorch_mlp import MLP
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, TensorDataset
import math
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator

# Default constants
DNN_HIDDEN_UNITS_DEFAULT = '20'
LEARNING_RATE_DEFAULT = 1e-2
MAX_EPOCHS_DEFAULT = 1500
EVAL_FREQ_DEFAULT = 10

train_X = None
test_X = None
train_y = None
test_y = None

FLAGS = None

plot_train = []
plot_test = []
train_loss = []
test_loss = []
plot_x = []


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
    return targets.size(0), (predictions == targets).sum().item()

def test_eva(mlp, testloader):
    total = 0
    correct = 0
    with torch.no_grad():
        for data in testloader:
            x, y = data
            x = x.reshape(-1, 2)
            outputs = mlp(x)
            _, predict = torch.max(outputs.data, 1)
            total += y.size(0)
            correct += (predict == y).sum().item()
    plot_test.append(100 * correct / total)


def train_eva(mlp, trainloader):
    total = 0
    correct = 0
    with torch.no_grad():
        for data in trainloader:
            x, y = data
            x = x.reshape(-1, 2)
            outputs = mlp(x)
            _, predict = torch.max(outputs.data, 1)
            total += y.size(0)
            correct += (predict == y).sum().item()
    plot_train.append(100 * correct / total)

def compute_loss(mlp, loader, optimizer, loss):
    celoss = 0
    for i, data in enumerate(loader, 0):
        inputs, labels = data
        inputs = inputs.reshape(-1, 2)
        optimizer.zero_grad()
        outputs = mlp(inputs)
        l = loss(outputs, labels)
        celoss += l.item()
        l.backward()
        optimizer.step()
    return celoss/len(loader)

def train():
    """
    Performs training and evaluation of MLP model.
    NOTE: You should the model on the whole test set each eval_freq iterations.
    """
    # YOUR TRAINING CODE GOES HERE
    hidden_units = [int(i) for i in FLAGS.dnn_hidden_units.split(',')]
    mlp = MLP(2, hidden_units, 2)
    
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(mlp.parameters(), lr=FLAGS.learning_rate)
    trainset = TensorDataset(torch.FloatTensor(train_X), torch.LongTensor(train_y))
    trainloader = torch.utils.data.DataLoader(trainset, batch_size = 1, shuffle=True)
    testset = TensorDataset(torch.FloatTensor(test_X), torch.LongTensor(test_y))
    testloader = torch.utils.data.DataLoader(testset, batch_size = 1, shuffle=True)

    for tttt in range(FLAGS.max_steps):
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs = inputs.reshape(-1, 2)
            optimizer.zero_grad()
            outputs = mlp(inputs)
            l = loss(outputs, labels)
            l.backward()
            optimizer.step()
        
        if tttt % FLAGS.eval_freq == 0:
            plot_x.append(tttt)
            test_eva(mlp, testloader)
            train_eva(mlp, trainloader)
            train_loss.append(compute_loss(mlp, trainloader, optimizer, loss))
            test_loss.append(compute_loss(mlp, testloader, optimizer, loss))

    

    fig1 = plt.subplot(2,1,1)
    fig2 = plt.subplot(2,1,2)
    fig1.plot(plot_x, plot_train,  c='red', label='training data accuracy')
    fig1.plot(plot_x, plot_test, c='blue', label='test data accuracy')
    fig1.legend()
    fig2.plot(plot_x, train_loss, c='green', label='train data loss')
    fig2.plot(plot_x, test_loss, c='yellow', label='test data loss')
    fig2.legend()
    plt.show()


def main():
    """
    Main function
    """
    print('WORKING...')
    train()

if __name__ == '__main__':
    sample = make_moons(n_samples=100, shuffle=True, noise=None, random_state=None)
    data = sample[0]
    label = sample[1]
    train_X, test_X, train_y, test_y = train_test_split(data, label, test_size=0.2, random_state=0)

    
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
