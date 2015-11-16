import new

import numpy as np
from PIL import Image
from features import *
from nnet import neuron
import ubyte_unpack

def main():
    inputs = []
    solutions = []
    count = 0
    img = None
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
        if count == 1000:
            break
    print solutions

    i = Image.open("./test_cases/test_0.png")
    i2 = Image.open("./test_cases/test_1.png")
    i.show
    for i in range(1000):
        cur_inputs = []
        i = Image.open("./test_cases/test_" + str(i) + ".png")
        cur_inputs.append(pix_density(i))
        cur_inputs.append(density_ul(i))
        cur_inputs.append(density_ur(i))
        cur_inputs.append(density_bl(i))
        cur_inputs.append(density_br(i))
        cur_inputs.append(hgt_to_wdth(i))
        cur_inputs.append(num_holes(i))
        inputs.append(cur_inputs)
    nrn = neuron(inputs, np.array(solutions).T)
    #print nrn.forward_prop()

if __name__ == "__main__":
    main()