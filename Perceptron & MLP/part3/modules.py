import numpy as np
import math

class Linear(object):
    def __init__(self, in_features, out_features):
        """
        Module initialisation.
        Args:
            in_features: input dimension
            out_features: output dimension
        TODO:
        1) Initialize weights self.params['weight'] using normal distribution with mean = 0 and 
        std = 0.0001.
        2) Initialize biases self.params['bias'] with 0. 
        3) Initialize gradients with zeros.
        """
        self.params = {}
        self.grads = {}
        self.params['weight'] = np.random.normal(loc=0.0, scale = 0.5, size = (in_features,out_features))
#        self.params['weight'] = np.random.default_rng().normal(0, (2/out_features)**0.5, (in_features, out_features))
        self.params['bias'] = np.zeros((1,out_features))
        self.x = None
        self.dx = None
        self.dout = None
        self.params['before_a'] = None
        self.params['after_a'] = None
        self.params['dout'] = None
        self.grads['weight'] = None
        self.grads['bias'] = None
        self.dw = np.zeros((in_features, out_features))
        self.db = np.zeros((1, out_features))
        self.relu = ReLU()

    def forward(self, x):
        """
        Forward pass (i.e., compute output from input).
        Args:
            x: input to the module
        Returns:
            out: output of the module
        Hint: Similarly to pytorch, you can store the computed values inside the object
        and use them in the backward pass computation. This is true for *all* forward methods of *all* modules in this class
        """
        self.x = x
        out = np.dot(x, self.params['weight']) + self.params['bias']
        self.params['before_a'] = out
        return out

    def backward(self, dout):
        """
        Backward pass (i.e., compute gradient).
        Args:
            dout: gradients of the previous module
        Returns:
            dx: gradients with respect to the input of the module
        TODO:
        Implement backward pass of the module. Store gradient of the loss with respect to 
        layer parameters in self.grads['weight'] and self.grads['bias']. 
        """
        self.dout = dout
        self.grads['weight'] = np.dot(self.x.T, dout)
        self.grads['bias'] = dout
        self.dx = np.dot(dout, self.params['weight'].T)
        dx = self.dx
        return dx

class ReLU(object):
    def forward(self, x):
        """
        Forward pass.
        Args:
            x: input to the module
        Returns:
            out: output of the module
        """
        self.x = np.mat(x)
        self.mask = (x > 0)
        out = np.maximum(x, 0)
        return out
#        relu_forward = np.maximum(x, 0)
#        return relu_forward

    def backward(self, dout):
        """
        Backward pass.
        Args:
            dout: gradients of the previous module
        Returns:
            dx: gradients with respect to the input of the module
        """
        dx = np.where(self.mask, dout, 0)
        self.dx = dx
        return dx
#        dx = np.maximum(dout, 0)
#        self.dx = dx
#        return dx

class SoftMax(object):
    def forward(self, x):
        """
        Forward pass.
        Args:
            x: input to the module
        Returns:
            out: output of the module
    
        TODO:
        Implement forward pass of the module. 
        To stabilize computation you should use the so-called Max Trick
        https://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/
        
        """
        y = np.exp(x-np.max(x))
        out = y/y.sum()
        return out

    def backward(self, dout):
        """
        Backward pass. 
        Args:
            dout: gradients of the previous module
        Returns:
            dx: gradients with respect to the input of the module
        """
        dx = dout
        return dx

class CrossEntropy(object):
    def forward(self, x, y):
        """
        Forward pass.
        Args:
            x: input to the module
            y: labels of the input
        Returns:
            out: cross entropy loss
        """
        out = -np.sum(y*np.log(x))
#        x[x==0]=1
#        out = -np.sum(y*np.log(x))
        return out

    def backward(self, x, y):
        """
        Backward pass.
        Args:
            x: input to the module
            y: labels of the input
        Returns:
            dx: gradient of the loss with respect to the input x.
        """
        softmax = SoftMax();
        dx = softmax.backward(np.subtract(x, y))
        return dx
