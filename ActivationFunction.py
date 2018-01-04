# -*- coding: utf-8 -*-

import numpy as np

sigmoid_range = 34.538776394910684

def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -sigmoid_range, sigmoid_range)))

def derivative_sigmoid(o):
    return o * (1.0 - o)

def softmax(x):
    c = np.max(x)
    exp_x = np.exp(x - c)
    return exp_x / np.sum(exp_x)