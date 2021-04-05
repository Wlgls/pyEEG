# -*- encoding: utf-8 -*-
'''
@File        :sevcik_FD.py
@Time        :2021/03/28 19:22:34
@Author      :wlgls
@Version     :1.0
'''


import numpy as np

def sevcik_fd(data):
    """Fractal dimension feature is solved, which is used to describe the shape information of EEG time series data. It seems that this feature can be used to judge the electrooculogram and EEG.The calculation methods include Sevcik, fractal Brownian motion, box counting, Higuchi and so on.

    Sevcik method: fast calculation and robust analysis of noise
    Higuchi: closer to the theoretical value than box counting

    The Sevick method is used here because it is easier to implement
    Parameters
    ----------
    Parameters
    ----------
    data array
        data, for DEAP dataset, It's shape may be (n_trials, n_channels, points)
    
    Return
    ----------
    f:
        Solved feature, It's shape is similar to the shape of your input data.
        e.g. for input.shape is (n_trials, n_channels, points), the f.shape is (n_trials, n_channels, n_features)
    
    Example
    ----------
    In [7]: d.shape, l.shape
    Out[7]: ((40, 32, 8064), (40, 1))

    In [8]: sevcik_fd(d).shape
    Out[8]: (40, 32, 1)

    """

    points = data.shape[-1]

    x = np.arange(1, points+1)
    x_ = x / np.max(x)

    miny = np.expand_dims(np.min(data, axis=-1), axis=-1)
    maxy = np.expand_dims(np.max(data, axis=-1), axis=-1)
    y_ = (data-miny) / (maxy-miny)

    L = np.expand_dims(np.sum(np.sqrt(np.diff(y_, axis=-1)**2 + np.diff(x_)**2), axis=-1), axis=-1)
    FD = 1 + np.log(L) / np.log(2 * (points-1))
    # print(FD.shape)
    return FD


