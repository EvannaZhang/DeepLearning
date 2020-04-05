from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import os
from mlp_numpy import MLP
from modules import CrossEntropy, Linear, ReLU, SoftMax
from sklearn.datasets import make_moons, make_circles, make_classification
from numpy import array
from numpy import argmax
import math
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator

# Default constants
DNN_HIDDEN_UNITS_DEFAULT = '20'
LEARNING_RATE_DEFAULT = 1e-2
MAX_EPOCHS_DEFAULT = 1500
EVAL_FREQ_DEFAULT = 20

training_data = None
training_label = None
test_data = None
test_label = None
training_label_encoded = None
test_label_encoded = None

FLAGS = None

relu = ReLU()
softmax = SoftMax()
cross = CrossEntropy()

plot_train = []
plot_test = []
plot_loss = []
test_loss = []
x = []
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
    answer = np.zeros((len(training_data), 2))
    for index in range(len(training_data)):
        solution = mlp.forward(training_data[index])
        answer[index] = solution
    decode_answer = decode(answer)
    plot_train.append(accuracy(decode_answer, training_label))
    loss = cross.forward(answer, training_label_encoded)
    plot_loss.append(loss/len(training_data))


def test_accuracy(mlp):
    answer = np.zeros((len(test_data), 2))
    for index in range(len(test_data)):
        solution = mlp.forward(test_data[index])
        answer[index] = solution
    decode_answer = decode(answer)
    plot_test.append(accuracy(decode_answer, test_label))
    testloss = cross.forward(answer, test_label_encoded)
    test_loss.append(testloss/len(test_data))


def sto_gd(mlp):
    for tttt in range(FLAGS.max_steps):
        count = -1
        for data in training_data:
            count+=1
            mlp.forward(data)
            temp = cross.backward(mlp.outlayer.params['after_a'], training_label_encoded[count])
            mlp.backward(temp)
            mlp.outlayer.params['weight'] = mlp.outlayer.params['weight'] - FLAGS.learning_rate * mlp.outlayer.grads['weight']
            mlp.outlayer.params['bias'] = mlp.outlayer.params['bias'] - FLAGS.learning_rate * mlp.outlayer.grads['bias']
            #print(mlp.outlayer.params['weight'])
            for i in range(mlp.num_of_hidden):
                mlp.hiddenlayers[i].params['weight'] = mlp.hiddenlayers[i].params['weight'] -FLAGS.learning_rate * mlp.hiddenlayers[i].grads['weight']
                mlp.hiddenlayers[i].params['bias'] = mlp.hiddenlayers[i].params['bias'] - FLAGS.learning_rate * mlp.hiddenlayers[i].grads['bias']
            
        if tttt%FLAGS.eval_freq == 0:
            x.append(tttt)
            training_accuracy(mlp)
            test_accuracy(mlp)


def batch_gd(mlp):
    for tttt in range(FLAGS.max_steps):
        count = -1
        mlp.outlayer.dw = np.zeros(mlp.outlayer.dw.shape)
        mlp.outlayer.db = np.zeros(mlp.outlayer.db.shape)
        for i in range(mlp.num_of_hidden):
            mlp.hiddenlayers[i].dw = np.zeros(mlp.hiddenlayers[i].dw.shape)
            mlp.hiddenlayers[i].db = np.zeros(mlp.hiddenlayers[i].db.shape)
        for data in training_data:
            count+=1
            mlp.forward(data)
            temp = cross.backward(mlp.outlayer.params['after_a'], training_label_encoded[count])
            mlp.backward(temp)
            mlp.outlayer.dw += mlp.outlayer.grads['weight']
            mlp.outlayer.db += mlp.outlayer.grads['bias']
            for i in range(mlp.num_of_hidden):
                mlp.hiddenlayers[i].dw += mlp.hiddenlayers[i].grads['weight']
                mlp.hiddenlayers[i].db += mlp.hiddenlayers[i].grads['bias']
        mlp.outlayer.params['weight'] = mlp.outlayer.params['weight'] - FLAGS.learning_rate * mlp.outlayer.dw / len(training_data)
        mlp.outlayer.params['bias'] = mlp.outlayer.params['bias'] - FLAGS.learning_rate * mlp.outlayer.db /len(training_data)
        for i in range(mlp.num_of_hidden):
            mlp.hiddenlayers[i].params['weight'] = mlp.hiddenlayers[i].params['weight'] - FLAGS.learning_rate * mlp.hiddenlayers[i].dw / len(training_data)
            mlp.hiddenlayers[i].params['bias'] =  mlp.hiddenlayers[i].params['bias'] - FLAGS.learning_rate * mlp.hiddenlayers[i].db / len(training_data)
    
        if tttt%FLAGS.eval_freq == 0:
            x.append(tttt)
            training_accuracy(mlp)
            test_accuracy(mlp)

def train(task):
    """
    Performs training and evaluation of MLP model.
    NOTE: You should the model on the whole test set each eval_freq iterations.
    """
    # YOUR TRAINING CODE GOES HERE
    
    hidden_units = [int(i) for i in FLAGS.dnn_hidden_units.split(',')]
    num_of_hidden_layer = len(hidden_units)
    mlp = MLP(2, hidden_units, 2)
    if int(task) == 2:
        batch_gd(mlp)
    else:
        print('Please input the optimizer type')
        descent = input('1 for SGD and others for BGD: ')
        if int(descent) == 1:
            sto_gd(mlp)
        else:
            batch_gd(mlp)


    fig1 = plt.subplot(2,1,1)
    y_major_locator=MultipleLocator(10)
    ax=plt.gca()
    ax.yaxis.set_major_locator(y_major_locator)
    plt.ylim(30,100)
    fig2 = plt.subplot(2,1,2)
    fig1.plot(x, plot_train,  c='red', label='training data accuracy')
    fig1.plot(x, plot_test, c='blue', label='test data accuracy')
    fig1.legend()
    fig2.plot(x, plot_loss, c='green', label='train CE loss')
    fig2.plot(x, test_loss, c='yellow', label='test CE loss')
    fig2.legend()
    plt.show()


def main():
    """
    Main function
    """
    print('WORKING...')
    train(FLAGS.task)


if __name__ == '__main__':
    sample = make_moons(n_samples=1000, shuffle=True, noise=None, random_state=None)

    data = sample[0]
    label = sample[1]
    training_data = np.mat(data[0:800])
    training_label = label[0:800]
    test_data = np.mat(data[800:])
    test_label = label[800:]
    training_label_encoded = encode(training_label)
    test_label_encoded = encode(test_label)

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
    parser.add_argument('--task', type=int, default=3,
                            help='Specify task 2 or 3')
    FLAGS, unparsed = parser.parse_known_args()
    main()
