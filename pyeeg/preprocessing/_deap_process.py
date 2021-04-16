# -*- encoding: utf-8 -*-
'''
@File        :_base_process.py
@Time        :2021/04/05 16:15:55
@Author      :wlgls
@Version     :1.0
'''

import numpy as np

def split_signal(data, label, windows=1, fs=128):
    """In the study, We divided a trial into 60 seconds. 

    Parameters
    ----------
    data : array
        data, for DEAP dataset, It's shape may be (n_trials, n_channels, points) 
    label : array
        In order to correspond with data
    windows : int, optional
        Window size of segmentation, by default 1
    sf : int, optional
        sampling frequency, by default 128

    Returns
    -------
    tmpData:
        Sliced data, If your input's shape is (n_trials, n_channels, points), The tmpData.shape is (n_trials, points//(windows*fs), n_channels, windows*fs)
    tmpLabel:
        Corresponding with data. It's shape maybe (n_trials, points//(windows*fs))
    
    Examples:
    ----------
    In [16]: data.shape, label.shape
    Out[16]: ((40, 32, 8064), (40, 1))

    In [17]: d, l = split_signal(data, label)

    In [18]: d.shape, l.shape
    Out[18]: ((40, 63, 32, 128), (40, 63))
    """
    if len(data.shape) != 3:
        raise ValueError

    sp = data.shape[-1] // fs // windows
    tmpData = np.stack(np.split(data, sp, axis=2), axis=1)
    tmpLabel = np.repeat(label, tmpData.shape[1], axis=1)
    return tmpData, tmpLabel


def remove_baseline(data, label, baseline=3):
    """For the deap dataset, the first three seconds are the baseline, we need to get rid of it.

    Parameters
    ----------
    data : array
        data, for sliced DEAP dataset, It's shape may be (n_trials, n_slices,  n_channels, points) 
    label : array
        In order to correspond with data
    baseline : int, optional
        Baseline time, by default 3

    Returns
    -------
    base:
        Baseline
    signal:
        Remove baseline data.It's shape maybe (n_trials, n_slices-baseline, n_channels, points)
    
    Examples:
    ----------

    In [24]: d.shape, l.shape
    Out[24]: ((40, 63, 32, 128), (40, 63))
    
    In [21]: base, baselabel, signal, signallabel = remove_baseline(d, l)

    In [22]: base.shape, baselabel.shape
    Out[22]: ((40, 3, 32, 128), (40, 3))

    In [23]: base.shape, signallabel.shape
    Out[23]: ((40, 3, 32, 128), (40, 60))

    """
    base = data[:, :baseline, ...]
    signal = data[:, baseline:, ...]

    base_label = label[:, :baseline]
    signal_label = label[:, baseline:]

    return base, base_label, signal, signal_label
