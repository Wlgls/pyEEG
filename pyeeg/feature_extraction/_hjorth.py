# -*- encoding: utf-8 -*-
'''
@File        :hjorth.py
@Time        :2021/03/28 19:19:01
@Author      :wlgls
@Version     :1.0
'''

import numpy as np

def hjorth(data):
    """Solving Hjorth features， include activity, mobility, complexity

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
    In [15]: d.shape, l.shape
    Out[15]: ((40, 32, 8064), (40, 1))

    In [16]: hjorth_features(d).shape
    Out[16]: (40, 32, 3)
    """
    data = np.array(data)
    ave = np.mean(data, axis=-1)[..., np.newaxis]
    diff_1st = np.diff(data, n=1, axis=-1)
    # print(diff_1st.shape)
    diff_2nd = data[..., 2:] - data[..., :-2]
    # Activity
    activity = np.mean((data-ave)**2, axis=-1)
    # print(Activity.shape)
    # Mobility
    varfdiff = np.var(diff_1st, axis=-1)
    # print(varfdiff.shape)
    mobility = np.sqrt(varfdiff / activity)

    # Complexity
    varsdiff = np.var(diff_2nd, axis=-1)
    complexity = np.sqrt(varsdiff/varfdiff) / mobility

    f = np.stack((activity, mobility, complexity), axis=-1)
    return f
