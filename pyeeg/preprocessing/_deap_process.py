# -*- encoding: utf-8 -*-
'''
@File        :_base_process.py
@Time        :2021/04/05 16:15:55
@Author      :wlgls
@Version     :1.0
'''

import numpy as np

def remove_baseline(data, fs=128, baseline=3):
    """[summary]

    Parameters
    ----------
    data : [type]
        [description]
    fs : int, optional
        [description], by default 128
    baseline : int, optional
        [description], by default 3

    Returns
    -------
    [type]
        [description]
    """
    f = fs * baseline
    return data[..., :f], data[..., f:]

def split_signal(data, label, windows=1, fs=128):
    """[summary]

    Parameters
    ----------
    data : [type]
        [description]
    label : [type]
        [description]
    windows : int, optional
        [description], by default 1
    fs : int, optional
        [description], by default 128

    Returns
    -------
    [type]
        [description]

    Raises
    ------
    ValueError
        [description]
    """
    if len(data.shape) != 3:
        raise ValueError

    sp = data.shape[-1] // fs // windows
    tmpData = np.stack(np.split(data, sp, axis=2), axis=1)
    tmpLabel = np.repeat(label, tmpData.shape[1], axis=1)
    return tmpData, tmpLabel