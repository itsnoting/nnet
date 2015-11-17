import new

import numpy as np
from PIL import Image
from features import *
from nnet import Neural_Network
from trainer import trainer
from ubyte_unpack import ubyte_unpack

def pred_perc(actual, pred):
    count = 0
    for i in range(len(actual)):
        if actual[i] == pred[i]:
            count += 1
    return float(count/float(len(actual)))

def matrix_to_list(matrix):
    return [float(round(n, 0)) for n in list(np.array(matrix).reshape(-1))]

def main():

    trainer_set = ubyte_unpack("./ubyte", "training")
    inputs, solutions = trainer_set.get_dataset(100)

    NN = Neural_Network()
    T = trainer(NN)
    T.train(inputs, solutions)
    yHat = matrix_to_list(NN.forward(inputs))
    solutions = matrix_to_list(solutions)
    percent = pred_perc(solutions, yHat)
    print "Prediction percentage:", percent * 100

    tester_set = ubyte_unpack("./ubyte", "testing")
    X, y = tester_set.get_dataset(1000)

    test_yHat = matrix_to_list(NN.forward(X))
    y = matrix_to_list(y)
    test_percent = pred_perc(y, test_yHat)
    print "Test prediction percentage:", test_percent * 100
    print "Current first level weights:", NN.W1
    print "Current second level weights:", NN.W2

if __name__ == "__main__":
    main()