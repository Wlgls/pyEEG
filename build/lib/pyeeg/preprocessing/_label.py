

import numpy as np

def label_binarizer(label, threshold=5):
    tmp = np.copy(label)
    tmp[label<=threshold] = 0
    tmp[label> threshold] = 1
    return tmp