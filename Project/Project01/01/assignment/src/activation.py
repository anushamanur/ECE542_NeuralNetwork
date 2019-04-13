#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: activation.py

import numpy as np


def sigmoid(z):
    """The sigmoid function."""
    return 1/(1+np.exp(-z))

def sigmoid_prime(z):
    """Derivative of the sigmoid function."""
    sigma_prime=1/(1+np.exp(-z))*(1-(1/(1+np.exp(-z))))
    return sigma_prime
