# -*- encoding: utf-8 -*-
'''
@File        :_grouping.py
@Time        :2021/04/05 16:22:29
@Author      :wlgls
@Version     :1.0
'''

import numpy as np

def group_by_trial(data, label, shuffle=False):
    """For the deap dataset, in the research, we usually make a sample of each person's trial, so that we can divide the data set according to the trial in the future. And in the experiment, we will slice each trial according to the window sometimes, so we need to group each slice.

    Parameters
    ----------
    data : array
        data, for sliced DEAP dataset, It's shape may be (n_trials, n_slices,  n_features) 
    label : array
        Similarly, we need to map labels to data
    shuffle : bool, optional
        shuffle, by default False

    Returns
    -------
    groups: array
        Your group
    data: array
        Data after grouping. If your input's shape is (n_trials*n_slices,  n_features) 
    label: array
        Label after grouping.
    """
    trials, slices, *features_shape = data.shape
    data = data.reshape(-1, *features_shape)
    label = label.reshape(-1, label.shape[-1])

    groups = np.arange(1, trials+1)
    groups = np.repeat(groups, slices)
    
    if shuffle:
        index = np.arange(len(groups))
        np.random.shuffle(index)
        return groups[index], data[index], label[index]

    return groups, data, label

def group_by_time(data, label, shuffle=False):
    """It may be wrong. In the study, We divided a trial into 60 seconds. For 40 trials, We may choose some time for the training set, the rest is for test set. So we need to group data by time.

    Parameters
    ----------
    data : array
        data, for sliced DEAP dataset, It's shape may be (n_trials, n_slices,  n_features) 
    label : array
        Similarly, we need to map labels to data
    shuffle : bool, optional
        shuffle, by default False

    Returns
    -------
    groups: array
        Your group
    data: array
        Data after grouping. If your input's shape is (n_trials*n_slices,  n_features) 
    label: array
        Label after grouping.
    """
    trials, slices, *features_shape = data.shape
    data = data.reshape(-1, *features_shape)
    label = label.reshape(-1)

    groups = np.arange(1, slices+1)
    groups = np.tile(groups, trials)

    if shuffle:
        index = np.arange(len(groups))
        np.random.shuffle(index)
        return groups[index], data[index], label[index]
    
    return groups, data, label