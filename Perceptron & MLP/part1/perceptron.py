import numpy as np
import matplotlib.pyplot as plt
'''
mean = (1, 1)
cov = [[1, 0], [0, 1]]
x = np.random.multivariate_normal(mean, cov, (1,2), 'raise')
print(x)
'''

def gausian_generator(mean, cov):
    x, y = np.random.multivariate_normal(mean, cov, 100).T
    data = np.zeros((100, 2))
    for i in range(100):
        data[i][0] = x[i]
        data[i][1] = y[i]
    return data

class Perceptron(object):
    def __init__(self, n_inputs, max_epochs=1e2, learning_rate=1e-2):
        self.n_inputs = n_inputs
        self.size_inputs = 2
        self.max_epochs = max_epochs
        self.learning_rate = learning_rate
        self.weights = np.zeros(self.size_inputs)
        self.bias = 0
        """
        Initializes perceptron object.
        Args:
            n_inputs: number of inputs.
            max_epochs: maximum number of training cycles.
            learning_rate: magnitude of weight changes at each training cycle
        """

    def forward(self, input):
        """
        Predict label from input 
        Args:
            input: array of dimension equal to n_inputs.
        """
        summation = np.dot(self.weights, input) + self.bias
        if summation > 0:
            label = 1
        else:
            label = -1
        return label

    def train(self, training_inputs, labels):
        """
        Train the perceptron
        Args:
            training_inputs: list of numpy arrays of training points.
            labels: arrays of expected output value for the corresponding point in training_inputs.
        """

        '''
        for _ in range(int(self.max_epochs)):
            for inputs, label in zip(training_inputs, labels):
                prediction = self.forward(inputs)
                self.weights[1:] += self.learning_rate * (label - prediction) * inputs
                self.weights[0] += self.learning_rate * (label - prediction)
        '''
        m = len(training_inputs)
        for _ in range(int(self.max_epochs)):
            for i in range(m):
                if np.any(labels[i] * (np.dot(self.weights, training_inputs[i]) + self.bias) <= 0):
                    self.weights = self.weights[1:] + self.learning_rate * labels[i] * training_inputs[i].T
                    self.bias = self.bias + self.learning_rate * labels[i]


if __name__=='__main__':
    p = Perceptron(80)
    print('Generating first distribution...')
    print('Please input the mean(Example: 5 5): ')
    mean1 = [int(n) for n in input().split()]
    print('Please input the cov(Example: 2 0 \Enter 0 2): ')
    cov1 = [[0]*2]*2
    for i in range(2):
        cov1[i] = input().split(" ")
    a = gausian_generator(mean1, cov1)
    print('First distribution generated')

    print('Generating second distribution...')
    print('Please input the mean(Example: -5 -5): ')
    mean2 = [int(n) for n in input().split()]
    print('Please input the cov(Example: 1 0 \Enter 0 1): ')
    cov2 = [[0]*2]*2
    for i in range(2):
        cov2[i] = input().split(" ")
    b = gausian_generator(mean2, cov2)
    print('Second distribution generated')
    a_train = a[0:80]
    a_test = a[80:100]
    a_label = np.ones(80, dtype=np.int16)
    b_train = b[0:80]
    b_test = b[80:100]
    b_label = -a_label
    p.train(a_train, a_label)
    p.train(b_train, b_label)
    accuracy = 0
    print('Begin prediction')
    for i in range(20):
        print(a_test[i])
        print('From distribution 1, predicted as ', p.forward(a_test[i-1]))
        if(p.forward(a_test[i-1])==1):
            accuracy += 1
        print(b_test[i])
        print('From distribution -1, predicted as ', p.forward(b_test[i-1]))
        if(p.forward(b_test[i-1])==-1):
            accuracy += 1
    print('Accuracy of test: ', accuracy/40*100, '%')


