import new

import numpy as np
from PIL import Image
from features import *
from nnet import Neural_Network
from trainer import trainer
import ubyte_unpack

def main():
    inputs = []
    solutions = []
    count = 0
    generator = ubyte_unpack.read()
    for img in generator:
        if img[0] == 5 or img[0] == 1:
            im = Image.fromarray(img[1])
            im.save('./test_cases/test_' + str(count) + '.png')
            count += 1
            if img[0] == 5:
                solutions.append(1)
            else:
                solutions.append(0)
        if count == 10:
            break
    print solutions

    for i in range(10):
        cur_inputs = []
        i = Image.open("./test_cases/test_" + str(i) + ".png")
        cur_inputs.append(float(pix_density(i)))
        cur_inputs.append(float(density_ul(i)))
        cur_inputs.append(float(density_ur(i)))
        cur_inputs.append(float(density_bl(i)))
        cur_inputs.append(float(density_br(i)))
        cur_inputs.append(float(hgt_to_wdth(i)))
        cur_inputs.append(float(num_holes(i)))
        inputs.append(cur_inputs)

    inputs = np.array(inputs, dtype=float)
    solutions = np.matrix(solutions, dtype=float).T
    solutions = np.array(solutions, dtype=float)
    inputs = inputs/np.amax(inputs, axis = 0)

    print solutions

    NN = Neural_Network()
    T = trainer(NN)
    T.train(inputs, solutions)
    yHat = NN.forward(inputs)
    count = 0
    b_yHat = []
    for i in range(len(yHat)):
        b_yHat.append(float(round(yHat[i][0])))
        if solutions[i] == b_yHat[i]:
            count += 1
    print b_yHat
    print "Prediction percentage:", count/len(yHat) * 100

if __name__ == "__main__":
    main()