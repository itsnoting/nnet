import new

import numpy as np
from PIL import Image
from features import *
from nnet import Neural_Network as neuron
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
        if count == 80:
            break
    print solutions

    for i in range(80):
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

    def computeNumericalGradient(N, X, y):
        paramsInitial = N.getParams()
        numgrad = np.zeros(paramsInitial.shape)
        perturb = np.zeros(paramsInitial.shape)
        e = 1e-4

        for p in range(len(paramsInitial)):
            #Set perturbation vector
            perturb[p] = e
            N.setParams(paramsInitial + perturb)
            loss2 = N.costFunction(X, y)

            N.setParams(paramsInitial - perturb)
            loss1 = N.costFunction(X, y)
            print "numgrad", numgrad, len(numgrad)
            #Compute Numerical Gradient
            numgrad[p] = (loss2 - loss1) / (2*e)

            #Return the value we changed to zero:
            perturb[p] = 0

        #Return Params to original value:
        N.setParams(paramsInitial)

        return numgrad

    NN = neuron()
    numgrad = computeNumericalGradient(NN, inputs, solutions)
    print numgrad

if __name__ == "__main__":
    main()