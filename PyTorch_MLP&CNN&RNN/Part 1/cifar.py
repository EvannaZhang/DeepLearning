from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch
import torch.nn as nn
import os
import argparse
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

# Constants
DNN_HIDDEN_UNITS_DEFAULT = '20'
LEARNING_RATE_DEFAULT = 1e-2
MAX_EPOCHS_DEFAULT = 300
EVAL_FREQ_DEFAULT = 10

FLAGS = None

plot_train = []
plot_test = []
train_loss = []
test_loss = []
plot_x = []

class MLP(nn.Module):
    def __init__(self, node, keep_rate=0):
        super(MLP, self).__init__()
        self.n_hidden_nodes = node
        self.n_hidden_layers = 2
        if not keep_rate:
            keep_rate = 0.5
        self.keep_rate = keep_rate
        # Set up perceptron layers and add dropout
        self.fc1 = torch.nn.Linear(32 * 32 * 3, node)
        self.fc1_drop = torch.nn.Dropout(1 - keep_rate)
        self.fc2 = torch.nn.Linear(self.n_hidden_nodes, self.n_hidden_nodes)
        self.fc2_drop = torch.nn.Dropout(1 - keep_rate)
        self.out = torch.nn.Linear(self.n_hidden_nodes, 10)

    def forward(self, x):
        x = x.view(-1, 32 * 32 * 3)
        x = nn.functional.relu(self.fc1(x))
        x = self.fc1_drop(x)
        if self.n_hidden_layers == 2:
            x = torch.nn.functional.relu(self.fc2(x))
        x = self.fc2_drop(x)
        return nn.functional.log_softmax(self.out(x), dim=0)


def main():
    cuda = torch.cuda.is_available()
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=4,shuffle=True, num_workers=0, pin_memory=False)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=4,shuffle=False, num_workers=0, pin_memory=False)
    validation_loader = torch.utils.data.DataLoader(testset, batch_size=4,shuffle=False, num_workers=0, pin_memory=False)
    
    model = MLP(100)
    if cuda:
        model.cuda()
    optimizer = torch.optim.SGD(model.parameters(), lr=FLAGS.learning_rate)
    for epoch in range(1, FLAGS.max_steps + 1):
        model.train()
        correct = 0
        train_accuracy = 0
        count = 0
        loss_train = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            count += 1
            if cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            optimizer.zero_grad()
            output = model(data)
            pred = output.data.max(1)[1]
            correct += pred.eq(target.data).cpu().sum()
            loss = nn.functional.nll_loss(output, target)
            loss_train += loss
            loss.backward()
            optimizer.step()
        if epoch % 20 == 0:
            train_accuracy = 100. * correct / len(train_loader.dataset)
            plot_x.append(epoch)
            plot_train.append(train_accuracy)
            train_loss.append(loss_train/count)
            test_accuracy = 0
            t_loss = 0
            correct = 0
            test_count = 0
            for batch_idx, (data, target) in enumerate(test_loader):
                test_count += 1
                if cuda:
                    data, target = data.cuda(), target.cuda()
                data, target = Variable(data), Variable(target)
                optimizer.zero_grad()
                output = model(data)
                predi = output.data.max(1)[1] # get the index of the max log-probability
                correct += predi.eq(target.data).cpu().sum()
                t_loss += nn.functional.nll_loss(output, target)
            test_accuracy = 100. * correct / len(test_loader.dataset)
            plot_test.append(test_accuracy)
            test_loss.append(t_loss/test_count)

    fig1 = plt.subplot(2,1,1)
    fig2 = plt.subplot(2,1,2)
    fig1.plot(plot_x, plot_train,  c='red', label='training data accuracy')
    fig1.plot(plot_x, plot_test, c='blue', label='test data accuracy')
    fig1.legend()
    fig2.plot(plot_x, train_loss, c='green', label='train data loss')
    fig2.plot(plot_x, test_loss, c='yellow', label='test data loss')
    fig2.legend()
    plt.savefig("/home/evanna/mlp/300_20.jpg")
    plt.show()
                                                            


if __name__ == '__main__':
    os.environ['KMP_DUPLICATE_LIB_OK']='True'
    parser = argparse.ArgumentParser()
    parser.add_argument('--dnn_hidden_units', type = str, default = DNN_HIDDEN_UNITS_DEFAULT, help='Comma separated list of number of units in each hidden layer')
    parser.add_argument('--learning_rate', type = float, default = LEARNING_RATE_DEFAULT,
                                            help='Learning rate')
    parser.add_argument('--max_steps', type = int, default = MAX_EPOCHS_DEFAULT,
                                            help='Number of epochs to run trainer.')
    parser.add_argument('--eval_freq', type=int, default=EVAL_FREQ_DEFAULT,
                                            help='Frequency of evaluation on the test set')
    FLAGS, unparsed = parser.parse_known_args()
    main()
