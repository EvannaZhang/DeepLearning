from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import torch.nn as nn
import torch

class MLP(nn.Module):
    
    def __init__(self, n_inputs, n_hidden, n_classes):
        """
        Initializes multi-layer perceptron object.    
        Args:
            n_inputs: number of inputs (i.e., dimension of an input vector).
            n_hidden: list of integers, where each integer is the number of units in each linear layer
            n_classes: number of classes of the classification problem (i.e., output dimension of the network)
        """
        super(MLP, self).__init__()
        self.n_inputs = n_inputs
        self.n_classes = n_classes
        self.n_hidden = n_hidden
        self.input_layer = nn.Linear(n_inputs, n_hidden[0])
        self.hidden = nn.ModuleList([])
        for i in range(len(n_hidden)):
            if i == len(n_hidden)-1:
                self.hidden.append(nn.Linear(n_hidden[i], n_classes))
            else:
                self.hidden.append(nn.Linear(n_hidden[i], n_hidden[i+1]))

    
    def forward(self, x):
        """
        Predict network output from input by passing it through several layers.
        Args:
            x: input to the network
        Returns:
            out: output of the network
        """
        x = self.input_layer(x)
        for i in range(len(self.hidden)):
            x = torch.relu(x)
            x = self.hidden[i](x)
        return x

