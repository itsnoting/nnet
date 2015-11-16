from re import T

import numpy as np

class Neural_Network(object):
    def __init__(self):
        #Define HyperParameters
        self.inputLayerSize = 7
        self.outputLayerSize = 1
        #Using (training)/arb(input size + output size)
        self.hiddenLayerSize = 10

        self.W1 = np.random.randn(self.inputLaterSize,
                                 self.hiddenLayerSize)
        self.W2 = np.random.randn(self.hiddenLayerSize,
                                  self.outputLayerSize)

    def forward(self, X):
        #Propagate inputs through network
        self.z2 = np.dot(X, self.W1)
        self.a2 = self.sigmoid(self.z2)
        self.z3 = np.dot(self.a2, self.W2)
        yHat = self.sigmoid(self.z3)
        return yHat

    def costFunctionPrime(self, X, y):
        #Compute derivative with   respect to W1 and W2
        self.yHat = self.forward(X)

        delta3 = np.multiply(-(y-self.yHat), self.sigmoidPrime(self.z3))
        dJdW2 = np.dot(self.a2.T, delta3)

        delta2 = np.dot(delta3, self.W2.T) * self.sigmoidPrime(self.z2)
        dJdW1 = np.dot(X.T, delta2)

        return dJdW1,

    def sigmoid(self, z):
        #Apply sigmoid activation function
        return 1/(1+np.exp(-z))

    def sigmoidPrime(self, z):
        return np.exp(-z)/((1+np.exp(-z))**2)

