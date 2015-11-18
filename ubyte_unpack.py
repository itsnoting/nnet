from PIL import Image
import os
import struct
import numpy as np
from features import *


class ubyte_unpack(object):

    def __init__(self, path, dataset="training"):
        self.dataset = dataset
        self.path = path

    def _read(self):
        """
        Python function for importing the MNIST data set.  It returns an iterator
        of 2-tuples with the first element being the label and the second element
        being a numpy.uint8 2D array of pixel data for the given image.
        """
        if self.dataset == "training":
            fname_img = os.path.join(self.path, 'train-images.idx3-ubyte')
            fname_lbl = os.path.join(self.path, 'train-labels.idx1-ubyte')
        elif self.dataset == "testing":
            fname_img = os.path.join(self.path, 't10k-images.idx3-ubyte')
            fname_lbl = os.path.join(self.path, 't10k-labels.idx1-ubyte')
        else:
            raise ValueError, "dataset must be 'testing' or 'training'"
        img_set = []
        # Load everything in some numpy arrays
        with open(fname_lbl, 'rb') as flbl:
            magic, num = struct.unpack(">II", flbl.read(8))
            lbl = np.fromfile(flbl, dtype=np.int8)

        with open(fname_img, 'rb') as fimg:
            magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
            img = np.fromfile(fimg, dtype=np.uint8).reshape(len(lbl), rows, cols)

        get_img = lambda idx: (lbl[idx], img[idx])

        # Create an iterator which returns each image in turn
        for i in xrange(len(lbl)):
            #yield get_img(i)
            img_set.append(get_img(i))
        self._imgset = img_set

    def get_dataset(self, num_of_sets):
        self._read()
        inputs = []
        solutions = []
        file_cat = ""
        count = 0
        if self.dataset == "training":
            file_cat = "train"
        elif self.dataset == "testing":
            file_cat = "test"
        for i, img in enumerate(self._imgset):
            if img[0] == 5 or img[0] == 1:
                im = Image.fromarray(img[1])
                im.save('./' + file_cat + '_cases/' + file_cat + '_' + str(count) + '.png')
                if img[0] == 5:
                    solutions.append(1)
                if img[0] == 1:
                    solutions.append(0)
                count += 1
            if len(solutions) == num_of_sets:
                break

        for i in range(num_of_sets):
            cur_inputs = []
            i = Image.open("./" + file_cat + "_cases/" + file_cat + "_" + str(i) + ".png")
            cur_inputs.append(float(pix_density(i)))
            cur_inputs.append(float(density_ul(i)))
            cur_inputs.append(float(density_ur(i)))
            cur_inputs.append(float(density_bl(i)))
            cur_inputs.append(float(density_br(i)))
            cur_inputs.append(float(hgt_to_wdth(i)))
            cur_inputs.append(float(horiz_symmetry(i)))
            cur_inputs.append(float(num_holes(i)))
            cur_inputs.append(float(num_intersections(i)))
            inputs.append(cur_inputs)

        inputs = np.array(inputs, dtype=float)
        solutions = np.matrix(solutions, dtype=float).T
        solutions = np.array(solutions, dtype=float)
        inputs = inputs/np.amax(inputs, axis = 0)
        return inputs, solutions



