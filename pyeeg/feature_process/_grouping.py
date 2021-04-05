# -*- encoding: utf-8 -*-
'''
@File        :_grouping.py
@Time        :2021/04/05 16:22:29
@Author      :wlgls
@Version     :1.0
'''

import numpy as np

def group_by_trial(data, label, shuffle=False):
    """[summary]

    Parameters
    ----------
    data : [type]
        [description]
    label : [type]
        [description]
    shuffle : bool, optional
        [description], by default False

    Returns
    -------
    [type]
        [description]
    """
    trials, blocks, *features_shape = data.shape
    data = data.reshape(-1, *features_shape)
    label = label.reshape(-1)

    groups = np.arange(1, trials+1)
    groups = np.repeat(groups, blocks)
    
    if shuffle:
        index = np.arange(len(groups))
        np.random.shuffle(index)
        return groups[index], data[index], label[index]

    return groups, data, label

def group_by_time(data, label, shuffle=False):
    """[summary]

    Parameters
    ----------
    data : [type]
        [description]
    label : [type]
        [description]
    shuffle : bool, optional
        [description], by default False

    Returns
    -------
    [type]
        [description]
    """
    trials, blocks, *features_shape = data.shape
    data = data.reshape(-1, *features_shape)
    label = label.reshape(-1)

    groups = np.arange(1, blocks+1)
    groups = np.title(groups, trials)

    if shuffle:
        index = np.arange(len(groups))
        np.random.shuffle(index)
        return groups[index], data[index], label[index]
    
    return groups, data, label