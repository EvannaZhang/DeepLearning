from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import time
import numpy as np
import os
import torch
from torch.utils.data import DataLoader
from dataset import PalindromeDataset
from lstm import LSTM
import matplotlib.pyplot as plt

plot_epoch = []
plot_test_loss = []
plot_test_accuracy = []

config = None

def train():
    train_on_gpu = torch.cuda.is_available()
    # Initialize the model that we are going to use
    model = LSTM(config.input_length, config.input_dim, config.num_hidden, config.num_classes, config.batch_size)
    if train_on_gpu:
        model.cuda()
    
    # Initialize the dataset and data loader (leave the +1)
    dataset = PalindromeDataset(config.input_length+1)
    data_loader = DataLoader(dataset, config.batch_size, num_workers=1)

    # Setup the loss and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.RMSprop(model.parameters(), lr=config.learning_rate)

    # Adjust learning rate
    lrs = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.96)

    for step, (batch_inputs, batch_targets) in enumerate(data_loader):
        # Add more code here ...
        if train_on_gpu:
            batch_inputs, batch_targets = batch_inputs.cuda().float(), batch_targets.cuda()
        prediction = model(batch_inputs)
        loss = criterion(prediction, batch_targets)
        optimizer.zero_grad()
        loss.backward(retain_graph=True)
        optimizer.step()
        lrs.step()

        # the following line is to deal with exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.max_norm)

        if step % 25 == 0:
            # print acuracy/loss here
            plot_epoch.append(step)
            correct = 0
            total = 0
            with torch.no_grad():
                for i, (test_input, test_target) in enumerate(data_loader):
                    if train_on_gpu:
                        test_input, test_target = test_input.cuda(), test_target.cuda()
                    eval_prediction = model(test_input)
                    num = len(test_target)
                    total += num
                    _, target = torch.max(eval_prediction.data, 1)
                    correct_batch = (target == test_target).sum().item()
                    correct += correct_batch
                    if i == len(test_target) - 1:
                        break;

            accuracy = correct/total
            test_loss = loss.item()
            plot_test_accuracy.append(accuracy*100)
            plot_test_loss.append(test_loss)

        if step == config.train_steps:
            break

    fig1 = plt.subplot(2,1,1)
    fig2 = plt.subplot(2,1,2)
    fig1.plot(plot_epoch, plot_test_accuracy,  c='red', label='accuracy')
    fig1.legend()
    fig2.plot(plot_epoch, plot_test_loss, c='green', label='loss')
    fig2.legend()
    plt.show()
    print('Done training.')

if __name__ == "__main__":
    # To deal with plt on Mac
    os.environ['KMP_DUPLICATE_LIB_OK']='True'
    # Parse training configuration
    parser = argparse.ArgumentParser()
    print('LSTM is WORKING...')
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
    # Train the model
    train()
