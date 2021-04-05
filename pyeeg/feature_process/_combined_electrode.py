# -*- encoding: utf-8 -*-
'''
@File        :_combined_electrode.py
@Time        :2021/04/05 16:35:05
@Author      :wlgls
@Version     :1.0
'''


def combined_electrode(features):
    
    return features.reshape((*features.shape[:-2], -1))
