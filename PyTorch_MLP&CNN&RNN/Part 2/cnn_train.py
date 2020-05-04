from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from cnn_model import CNN
from torchvision import datasets
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
import torch.optim as optim

# Default constants
LEARNING_RATE_DEFAULT = 1e-4
BATCH_SIZE_DEFAULT = 32
MAX_EPOCHS_DEFAULT = 150
EVAL_FREQ_DEFAULT = 5
OPTIMIZER_DEFAULT = 'ADAM'
DATA_DIR_DEFAULT = 'data'
FLAGS = None

plot_epoch = []
plot_train_accuracy = []
plot_test_accuracy = []
plot_train_loss = []
plot_test_loss = []


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
    correct = 0
    total = 0
    for i in range(len(predictions)):
        correct += predictions[i]
        total += targets[i]
    accuracy = correct/total
    return accuracy

def train():
    """
    Performs training and evaluation of MLP cnn.
    NOTE: You should the cnn on the whole test set each eval_freq iterations.
    """
    # YOUR TRAINING CODE GOES HERE
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    train_data = datasets.CIFAR10('data', train=True, download=True, transform=transform)
    test_data = datasets.CIFAR10('data', train=False, download=True, transform=transform)
    train_on_gpu = torch.cuda.is_available()
    num_train = len(train_data)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=FLAGS.batch_size, shuffle=True, num_workers=0)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=FLAGS.batch_size, shuffle=False, num_workers=0)
    classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    cnn = CNN(3, 10)
    if train_on_gpu:
        cnn.cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(cnn.parameters(), lr=FLAGS.learning_rate)
    
    for epoch in range(1, FLAGS.max_steps):
        class_correct = list(0. for i in range(10))
        class_total = list(0. for i in range(10))
        train_loss = 0.0
        test_loss = 0.0
        cnn.train()
        for data, target in train_loader:
            if train_on_gpu:
                data, target = data.cuda(), target.cuda()
            optimizer.zero_grad()
            output = cnn(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            _, pred = torch.max(output, 1)
            correct_tensor = pred.eq(target.data.view_as(pred))
            train_loss += loss.item()*data.size(0)
            correct = np.squeeze(correct_tensor.numpy()) if not train_on_gpu else np.squeeze(correct_tensor.cpu().numpy())
            for i in range(len(target.data)):
                label = target.data[i]
                class_correct[label] += correct[i].item()
                class_total[label] += 1
    
    
        if epoch % FLAGS.eval_freq == 0:
            test_correct = list(0. for i in range(10))
            test_total = list(0. for i in range(10))
            cnn.eval()
            for data, target in test_loader:
                if train_on_gpu:
                    data, target = data.cuda(), target.cuda()
                output = cnn(data)
                _, pred = torch.max(output, 1)
                correct_tensor = pred.eq(target.data.view_as(pred))
                correct = np.squeeze(correct_tensor.numpy()) if not train_on_gpu else np.squeeze(correct_tensor.cpu().numpy())
                loss = criterion(output, target)
                test_loss += loss.item()*data.size(0)
                for i in range(len(target.data)):
                    label = target.data[i]
                    test_correct[label] += correct[i].item()
                    test_total[label] += 1

            train_loss = train_loss/len(train_loader.dataset)
            test_loss = test_loss/len(test_loader.dataset)
            plot_epoch.append(epoch)
            plot_train_loss.append(train_loss)
            plot_test_loss.append(test_loss)
            print('Epoch: {} \tTraining Loss: {:.6f} \tTest Loss: {:.6f}'.format(epoch, train_loss, test_loss))

            percent_train = accuracy(class_correct, class_total)*100
            percent_test = accuracy(test_correct, test_total)*100
            plot_train_accuracy.append(percent_train)
            plot_test_accuracy.append(percent_test)
            print('train accuracy: ', percent_train, 'test accuracy: ', percent_test)

    fig1 = plt.subplot(2,1,1)
    fig2 = plt.subplot(2,1,2)
    fig1.plot(plot_epoch, plot_train_accuracy,  c='red', label='training data accuracy')
    fig1.plot(plot_epoch, plot_test_accuracy, c='blue', label='test data accuracy')
    fig1.legend()
    fig2.plot(plot_epoch, plot_train_loss, c='green', label='train CE loss')
    fig2.plot(plot_epoch, plot_test_loss, c='yellow', label='test CE loss')
    fig2.legend()
    plt.show()


def main():
    """
    Main function
    """
    train()


if __name__ == '__main__':
  # Command line arguments
  os.environ['KMP_DUPLICATE_LIB_OK']='True'
  parser = argparse.ArgumentParser()
  parser.add_argument('--learning_rate', type = float, default = LEARNING_RATE_DEFAULT,
                      help='Learning rate')
  parser.add_argument('--max_steps', type = int, default = MAX_EPOCHS_DEFAULT,
                      help='Number of steps to run trainer.')
  parser.add_argument('--batch_size', type = int, default = BATCH_SIZE_DEFAULT,
                      help='Batch size to run trainer.')
  parser.add_argument('--eval_freq', type=int, default=EVAL_FREQ_DEFAULT,
                        help='Frequency of evaluation on the test set')
  parser.add_argument('--data_dir', type = str, default = DATA_DIR_DEFAULT,
                      help='Directory for storing input data')
  FLAGS, unparsed = parser.parse_known_args()

  main()
