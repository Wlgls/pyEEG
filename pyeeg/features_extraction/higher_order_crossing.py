# -*- encoding: utf-8 -*-
'''
@File        :hoc.py
@Time        :2021/03/28 19:28:06
@Author      :wlgls
@Version     :1.0
'''

import numpy as np


def hoc(data, k=10, combined=True):
    """Solving the feature of hoc. Hoc is a high order zero crossing quantity.

    Parameters
    ----------
    data : array
        data, for DEAP dataset, It's shape may be (n_trials, n_channels, points) 
    k : int, optional
        Order, by default 10
    
    Return
    ----------
    nzc:
        Solved feature, It's shape is similar to the shape of your input data.
        e.g. for input.shape is (n_trials, n_channels, points), the f.shape is (n_trials, n_channels, n_features)

    Example
    ----------
    In [4]: d, l = load_deap(path, 0)

    In [5]: hoc(d, k=10).shape
    Out[5]: (40, 32, 10)

    In [6]: hoc(d, k=5).shape
    Out[6]: (40, 32, 5)
    """
    nzc = []
    for i in range(k):
        curr_diff = np.diff(data, n=i)
        x_t = curr_diff >= 0
        x_t = np.diff(x_t)
        x_t = np.abs(x_t)

        count = np.count_nonzero(x_t, axis=-1)
        nzc.append(count)
    nzc = np.stack(nzc, axis=-1)
    return nzc