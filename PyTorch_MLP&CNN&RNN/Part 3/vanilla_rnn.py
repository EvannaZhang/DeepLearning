from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils

class VanillaRNN(nn.Module):

    def __init__(self, seq_length, input_dim, hidden_dim, output_dim, batch_size):
        super(VanillaRNN, self).__init__()
        # Initialization here ...
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.seq_length = seq_length
        self.batch_size = batch_size
        
        self.rnn = nn.LSTM(seq_length, hidden_dim, 1)
        self.outlayer = nn.Linear(hidden_dim, output_dim)
        self.softmax = nn.LogSoftmax(dim=1)
        self.h0 = torch.zeros(1, batch_size, self.hidden_dim)
        self.c0 = torch.zeros(1, batch_size, self.hidden_dim)

    def forward(self, x):
        # Implementation here ...
        output, _ = self.rnn(x, (self.h0, self.c0))
        #outputs = self.outlayer(output[:, -1, :])
        outputs = self.outlayer(output)
        return outputs
        
    # add more methods here if needed
