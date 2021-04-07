


def label_binarizer(label, threshold=5):
    label[label<=threshold] = 0
    label[label> threshold] = 1
    return label