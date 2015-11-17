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

    #Training set initiated
    trainer_set = ubyte_unpack("./ubyte", "training")
    inputs, solutions = trainer_set.get_dataset(100)

    NN = Neural_Network()

    print "TRAINING CASES"

    #solutions = matrix_to_list(solutions)
    #Running with training cases before training
    yHat = matrix_to_list(NN.forward(inputs))
    before_solutions = matrix_to_list(solutions)
    percent = pred_perc(before_solutions, yHat)
    print "Prediction percentage before training:", percent * 100

    #Training Neural Network
    T = trainer(NN)
    T.train(inputs, solutions)

    #Running with training cases after training
    yHat = matrix_to_list(NN.forward(inputs))
    solutions = matrix_to_list(solutions)
    percent = pred_perc(solutions, yHat)
    print "Prediction percentage after training:", percent * 100

    print "\nTESTING CASES"

    #Running with testing cases after training
    tester_set = ubyte_unpack("./ubyte", "testing")
    X, y = tester_set.get_dataset(1000)

    test_yHat = matrix_to_list(NN.forward(X))
    y = matrix_to_list(y)
    test_percent = pred_perc(y, test_yHat)
    print "Test prediction percentage:", test_percent * 100
    print "\n\nCurrent first level weights:", NN.W1
    print "Current second level weights:", NN.W2

if __name__ == "__main__":
    main()