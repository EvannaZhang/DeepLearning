from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt
from dataset import PalindromeDataset
from vanilla_rnn import VanillaRNN

plot_step = []
plot_loss = []
plot_accuracy = []

def train(config):
    print('Vanilla RNN is WORKING...')

    model = VanillaRNN(config.input_length, config.input_dim, config.num_hidden, config.num_classes, config.batch_size)
    dataset = PalindromeDataset(config.input_length+1)
    data_loader = DataLoader(dataset, config.batch_size, num_workers=1)

    # Setup the loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.RMSprop(model.parameters(), config.learning_rate, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0, centered=False)
    #optimizer = torch.optim.SGD(model.parameters(), config.learning_rate)

    # model.train()
    for step, (batch_inputs, batch_targets) in enumerate(data_loader):
        # step: epoch
        # Add more code here ...
        # the following line is to deal with exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.max_norm)
        optimizer.zero_grad()
        batch_inputs = batch_inputs.unsqueeze(0)
        output= model(batch_inputs)[0]
        loss = criterion(output, batch_targets)
        loss.backward()
        optimizer.step()
        _, pred = torch.max(output, 1)
        all = len(pred)
        correct = 0
        for i in range(len(pred)):
            if batch_targets[i] == pred[i]:
                correct += 1

        # Add more code here ...
        accuracy = correct/all

        if step % 50 == 0:
            plot_step.append(step)
            plot_loss.append(loss)
            plot_accuracy.append(accuracy*100)
            # print acuracy/loss here

        if step == config.train_steps:
            # If you receive a PyTorch data-loader error, check this bug report:
            # https://github.com/pytorch/pytorch/pull/9655
            break

    fig1 = plt.subplot(2,1,1)
    fig2 = plt.subplot(2,1,2)
    fig1.plot(plot_step, plot_accuracy,  c='red', label='accuracy')
    fig1.legend()
    fig2.plot(plot_step, plot_loss, c='green', label='loss')
    fig2.legend()
    plt.show()
    print('Done training.')

if __name__ == "__main__":
    # Following line needed on Mac
    os.environ['KMP_DUPLICATE_LIB_OK']='True'
    parser = argparse.ArgumentParser()
    # Model params
    parser.add_argument('--input_length', type=int, default=10, help='Length of an input sequence')
    parser.add_argument('--input_dim', type=int, default=1, help='Dimensionality of input sequence')
    parser.add_argument('--num_classes', type=int, default=10, help='Dimensionality of output sequence')
    parser.add_argument('--num_hidden', type=int, default=128, help='Number of hidden units in the model')
    parser.add_argument('--batch_size', type=int, default=128, help='Number of examples to process in a batch')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--train_steps', type=int, default=1500, help='Number of training steps')
    parser.add_argument('--max_norm', type=float, default=10.0)

    config = parser.parse_args()
    train(config)
