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
    inputs, solutions = trainer_set.get_dataset(200)

    NN = Neural_Network()

    print "TRAINING CASES"

    #solutions = matrix_to_list(solutions)
    #Running with training cases before training
    yHat = matrix_to_list(NN.forward(inputs))
    before_solutions = matrix_to_list(solutions)
    percent = pred_perc(before_solutions, yHat)

    best_percent = 0
    best_weights = []
    T = trainer(NN)
    #Training Neural Network
    for i in range(100):
        yHat = matrix_to_list(NN.forward(inputs))
        before_solutions = matrix_to_list(solutions)
        percent = pred_perc(before_solutions, yHat)
        T.train(inputs, solutions)
        if percent > best_percent:
            print percent
            best_percent = percent
            best_W1, best_W2 = NN.get_weights()
            best_weights = [best_W1, best_W2]
        NN.set_weights(best_weights[0], best_weights[1])



    #Running with training cases after training
    yHat = matrix_to_list(NN.forward(inputs))
    solutions = matrix_to_list(solutions)
    percent = pred_perc(solutions, yHat)
    print "Prediction percentage after training:", percent * 100

    print "\nTESTING CASES"

    #Running with testing cases after training
    tester_set = ubyte_unpack("./ubyte", "testing")
    X, y = tester_set.get_dataset(2000)

    test_yHat = matrix_to_list(NN.forward(X))
    y = matrix_to_list(y)
    test_percent = pred_perc(y, test_yHat)
    print "Test prediction percentage:", test_percent * 100
    print "\n\nCurrent first level weights:"
    for weight in NN.W1:
        print weight
    print
    print "Current second level weights:"
    for weight in NN.W2:
        print weight

if __name__ == "__main__":
    main()