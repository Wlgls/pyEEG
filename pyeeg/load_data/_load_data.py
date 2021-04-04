# -*- encoding: utf-8 -*-
'''
@File        :load_deap.py
@Time        :2021/03/28 16:16:17
@Author      :wlgls
@Version     :1.0
@Desc        :Only for DEAP dataset
'''


import numpy as np
import pickle
from collections.abc import Iterable


def _load(path, EEG=True, target=0):
    with open(path, "rb") as f:
        subject = pickle.load(f, encoding='latin1')

    data = subject['data']
    label = subject['labels']

    if EEG:
        data = data[:, :32]

    if isinstance(target, int):
        lindex = np.array([target])
    elif isinstance(target, Iterable):
        lindex = target
    label = label[:, lindex]

    return data, label


def load_deap(path, subjects=0, EEG=True, target=0):
    """Getting data from DEAP dataset, only for DEAP.

    Parameters
    ----------
    path ： str
        The directory where you store your data。
    subjects : Interger / list
        The person you need
    EEG : bool, optional
        The deap data set contains non EEG signals. If it is true, only EEG signals are included, by default True
    target : Interger, optional
        There are many groups of tags in deap dataset, and the order of tags is ("valence", "arousal", "dominance", "liking")，enter an interger or list to get the tag you need, by default 0
    
    Return
    ----------
    data, label

    Example
    ----------
    In [3]: d, l = load_deap(path, 0)

    In [4]: d.shape, l.shape
    Out[4]: ((40, 32, 8064), (40, 1))

    In [5]: d, l = load_deap(path, subjects=(1, 2))

    In [6]: d.shape, l.shape
    Out[6]: ((2, 40, 32, 8064), (2, 40, 1))

    In [7]: d, l = load_deap(path, subjects=0, target=(0, 1))

    In [8]: d.shape, l.shape
    Out[8]: ((40, 32, 8064), (40, 2))
    """

    path = path+"/s{}.dat"

    subjectList = np.array(('01','02','03','04','05','06','07','08',
               '09','10','11','12','13','14','15','16',
               '17','18','19','20','21','22','23','24',
               '25','26','27','28','29','30','31','32'))
    if isinstance(subjects, int):
        fileList = [subjectList[subjects]]
    elif isinstance(subjects, Iterable):
        fileList = subjectList[np.array(subjects)]
    
    data = []
    label = []
    for subject in fileList:
        d, l = _load(path.format(subject), EEG=EEG, target=target)
        data.append(d)
        label.append(l)
    
    if isinstance(subjects, int):
        return data[0], label[0]
    
    return np.stack(data), np.stack(label)


