from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import numpy
import torch.nn.functional as F

class LSTM(nn.Module):

    def __init__(self, seq_length, input_dim, hidden_dim, output_dim, batch_size):
        super(LSTM, self).__init__()
        
        # Initialization here ...
        self.seq_length = seq_length
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.batch_size = batch_size
        
        # Activation function
        self.tanh = torch.tanh
        self.sig = torch.sigmoid
        
        # Linear Layers
        self.gx = nn.Linear(input_dim, hidden_dim)
        self.gh = nn.Linear(hidden_dim, hidden_dim)
        self.ix = nn.Linear(input_dim, hidden_dim)
        self.ih = nn.Linear(hidden_dim, hidden_dim)
        self.fx = nn.Linear(input_dim, hidden_dim)
        self.fh = nn.Linear(hidden_dim, hidden_dim)
        self.ox = nn.Linear(input_dim, hidden_dim)
        self.oh = nn.Linear(hidden_dim, hidden_dim)
        self.ph = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Implementation here ...
        x = x.unsqueeze(2)
        # parameters
        self.h = torch.zeros(self.hidden_dim, self.hidden_dim)
        self.c = torch.zeros(self.hidden_dim, self.hidden_dim)
        self.bg = torch.zeros(self.batch_size, self.batch_size)
        self.bi = torch.zeros(self.batch_size, self.batch_size)
        self.bf = torch.zeros(self.batch_size, self.batch_size)
        self.bo = torch.zeros(self.batch_size, self.batch_size)
        self.bp = torch.zeros(self.batch_size, self.output_dim)
        
        # use for to forward
        train_on_gpu = torch.cuda.is_available()
        if train_on_gpu:
            self.h, self.c = self.h.cuda(), self.c.cuda()
        for t in range(self.seq_length):
            temp = self.gx(x[:, t]) + self.gh(self.h) + self.bg
            g = self.tanh(temp)
            temp = self.ix(x[:, t]) + self.ih(self.h) + self.bi
            i = self.sig(temp)
            temp = self.fx(x[:, t]) + self.fh(self.h) + self.bf
            f = self.sig(temp)
            temp = self.ox(x[:, t]) + self.oh(self.h) + self.bo
            o = self.sig(temp)
            self.c = self.c*f + g*i
            self.h = self.tanh(self.c)*o
    
        self.p = self.ph(self.h) + self.bp
        out = F.softmax(self.p, dim=1)
        return out

    # add more methods here if needed
