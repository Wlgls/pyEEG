# -*- encoding: utf-8 -*-
'''
@File        :power_spectra.py
@Time        :2021/03/28 15:17:28
@Author      :wlgls
@Version     :1.0
'''


import numpy as np
from scipy import signal

def power_spectra(data, sf=128, nperseg=128, band=(0, 4, 8, 14, 31, 65)):
    """The power of each frequency band is calculated according to the frequency band division，and then it combines the frequency band power into a feature vector. It mainly uses Welch method.
    

    Parameters
    ----------
    data : array
        data, for DEAP dataset, It's shape may be (n_trials, n_channels, points)
    sf : int, optional
        sampling frequency, by default 128
    nperseg : int, optional
        for Welch method, According to scipy.signal.welch , by default 1
    band : tuple, optional
        boundary frequencies of bands, by default (0, 4, 8, 14, 31, 65)
        e.g. for (0, 4, 8, 14, 31, 65), It will calculate the power spectrum of delta(0~3Hz),theta(4~7Hz),alpha(8~13Hz),beta(14~30Hz) and gamma(31~64Hz).

    Returns
    -------
    f:
        Solved feature, It's shape is similar to the shape of your input data.
        e.g. for input.shape is (n_trials, n_channels, points), the f.shape is (n_trials, n_channels, n_features)
    
    Example
    ------
    >>> data, label
    """
    band = np.array(band)

    freqs, power = signal.welch(data, sf, nperseg=nperseg)
    
    freqband = np.hsplit(freqs, band)[1:-1] # Remove the beginning and the end

    # Get the index of the corresponding frequency band
    pindex = []
    for fb in freqband:
        pindex.append(np.where(np.in1d(freqs, fb))[0])
    
    # Get features
    f = []
    for index in pindex:
        f.append(np.mean(power[..., index]), axis=-1)
    
    f = np.stack(f, axis=-1)
    return f