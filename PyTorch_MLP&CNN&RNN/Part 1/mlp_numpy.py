from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from modules import * 

softmax = SoftMax()
cross = CrossEntropy()

class MLP(object):

    def __init__(self, n_inputs, n_hidden, n_classes):
        """
        Initializes multi-layer perceptron object.    
        Args:
            n_inputs: number of inputs (i.e., dimension of an input vector).
            n_hidden: list of integers, where each integer is the number of units in each linear layer
            n_classes: number of classes of the classification problem (i.e., output dimension of the network)
        """
        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.num_of_hidden = len(self.n_hidden)
        self.n_classes = n_classes
        temp = Linear(n_inputs, self.n_hidden[0])
        self.hiddenlayers = [temp]
        for i in range(self.num_of_hidden-1):
            temp = Linear(self.n_hidden[i], self.n_hidden[i+1])
            self.hiddenlayers.append(temp)
        self.outlayer = Linear(self.n_hidden[self.num_of_hidden-1], self.n_classes)

    def forward(self, x):
        """
        Predict network output from input by passing it through several layers.
        Args:
            x: input to the network
        Returns:
            out: output of the network
        """
        
        first = self.hiddenlayers[0].forward(x)
        self.hiddenlayers[0].params['before_a'] = first
        self.hiddenlayers[0].params['after_a'] = self.hiddenlayers[0].relu.forward(first)
        for i in range(1, self.num_of_hidden):
            t = self.hiddenlayers[i].forward(self.hiddenlayers[i-1].params['after_a'])
            self.hiddenlayers[i].params['before_a'] = t
            self.hiddenlayers[i].params['after_a'] = self.hiddenlayers[i].relu.forward(t)
        last = self.outlayer.forward(self.hiddenlayers[self.num_of_hidden-1].params['after_a'])
        self.outlayer.params['before_a'] = last
        self.outlayer.params['after_a'] = softmax.forward(last)
        out = self.outlayer.params['after_a']
        return out

    def backward(self, dout):
        """
        Performs backward propagation pass given the loss gradients. 
        Args:
            dout: gradients of the loss
        """
        
        next = self.outlayer.backward(dout)
        for i in range(0, self.num_of_hidden): # 0~num-1
            j = self.num_of_hidden-1-i # num-1~0
            next = self.hiddenlayers[j].relu.backward(next)
            next = self.hiddenlayers[j].backward(next)
        
        return
