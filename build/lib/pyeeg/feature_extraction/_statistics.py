# -*- encoding: utf-8 -*-
'''
@File        :statistics.py
@Time        :2021/03/28 19:13:26
@Author      :wlgls
@Version     :1.0
'''

import numpy as np

def statistics(data):
    """Statistical features， include Power, Mean, Std, 1st differece, Normalized 1st difference, 2nd difference,  Normalized 2nd difference.

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
    In [13]: d.shape, l.shape
    Out[13]: ((40, 32, 8064), (40, 1))

    In [14]: statistics_feature(d).shape
    Out[14]: (40, 32, 7)
    """
    # Power
    power = np.mean(data**2, axis=-1)
    # Mean
    ave = np.mean(data, axis=-1)
    # Standard Deviation
    std = np.std(data, axis=-1)
    # the mean of the absolute values of 1st differece mean
    diff_1st = np.mean(np.abs(np.diff(data,n=1, axis=-1)), axis=-1)
    # the mean of the absolute values of Normalized 1st difference
    normal_diff_1st = diff_1st / std
    # the mean of the absolute values of 2nd difference mean 
    diff_2nd = np.mean(np.abs(data[..., 2:] - data[..., :-2]), axis=-1)
    # the mean of the absolute values of Normalized 2nd difference
    normal_diff_2nd = diff_2nd / std
    # Features.append(np.concatenate((Power, Mean, Std, diff_1st, normal_diff_1st, diff_2nd, normal_diff_2nd), axis=2))
    
    f = np.stack((power, ave, std, diff_1st, normal_diff_1st, diff_2nd, normal_diff_2nd), axis=-1)
    return f
