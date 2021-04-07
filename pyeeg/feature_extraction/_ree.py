# -*- encoding: utf-8 -*-
'''
@File        :_ree.py
@Time        :2021/04/05 13:14:10
@Author      :wlgls
@Version     :1.0
'''

import pywt
import numpy as np

def recoursing_energy_efficiency(data):
    """Time-Frequency feature. It is based on 《Classification of human emotion from EEG using discrete wavelet transform》.
    In this function, we use "db4" mother wavelet to decompose the signal into 4 layers.
    Warning: maybe It can only be used in deap datasets.

    Parameters
    ----------
    data : array
        data, for DEAP dataset, It's shape may be (n_trials, n_channels, points) 

    Returns
    -------
    f:
        Solved feature, It's shape is similar to the shape of your input data.
        e.g. for input.shape is (n_trials, n_channels, points), the f.shape is (n_trials, n_channels, n_features)

    Examples:
    In [6]: data.shape, label.shape
    Out[6]: ((40, 32, 8064), (40, 1))

    In [7]: ree(data).shape
    Out[7]: (40, 32, 9)
    """
    wave = pywt.wavedec(data, "db4", level=4)
    _, _ , cD3, cD2, cD1 = wave
    E_alpha = np.sum(cD3**2, axis=-1)
    E_beta = np.sum(cD2**2, axis=-1)
    E_gamma = np.sum(cD1**2, axis=-1)
    E_totel = E_alpha + E_beta + E_gamma

    REE_alpha = E_alpha / E_totel
    REE_beta = E_beta / E_totel
    REE_gamma = E_gamma / E_totel

    REE = np.stack((REE_alpha, REE_beta, REE_gamma), axis=-1)

    log_REE = np.log(REE)
    abs_log_REE = np.abs(log_REE)

    f = np.concatenate((REE, log_REE, abs_log_REE), axis=-1)
    return f



    