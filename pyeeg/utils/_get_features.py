# -*- encoding: utf-8 -*-
'''
@File        :get_features.py
@Time        :2021/04/07 15:23:41
@Author      :wlgls
@Version     :1.0
'''

import numpy as np
from ..preprocessing import remove_baseline, split_signal
from ..feature_process import combined_electrode, group_by_trial
from ..preprocessing import label_binarizer

def get_features(data, label, feature_func):
    """A boring code, just don't want to write repeated code every time
    """
    data, label = split_signal(data, label)
    label = label_binarizer(label)

    base, _, signal, signal_label = remove_baseline(data)

    basef = feature_func(base)
    signalf = feature_func(signal)

    base_mean = np.mean(basef, axis=1)[:, np.newaxis, :]

    X = signalf - base_mean
    Y = signal_label

    G, X, Y = group_by_trial(X, Y)

    return G, X, Y


