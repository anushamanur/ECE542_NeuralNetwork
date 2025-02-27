#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: bp.py

import numpy as np
from src.activation import sigmoid, sigmoid_prime


def backprop(x, y, biases, weightsT, cost, num_layers):
    """ function of backpropagation
        Return a tuple ``(nabla_b, nabla_w)`` representing the
        gradient of all biases and weights.

        Args:
            x, y: input image x and label y
            biases, weights (list): list of biases and transposed weights of entire network
            cost (CrossEntropyCost): object of cost computation
            num_layers (int): number of layers of the network

        Returns:
            (nabla_b, nabla_wT): tuple containing the gradient for all the biases
                and weightsT. nabla_b and nabla_wT should be the same shape as 
                input biases and weightsT
    """
    # initial zero list for store gradient of biases and weights
    nabla_b = [np.zeros(b.shape) for b in biases]
    nabla_wT = [np.zeros(wT.shape) for wT in weightsT]

    ### Implement here
    # feedforward
    # Here you need to store all the activations of all the units
    # by feedforward pass
    ###
    
    activations=[x]
    i=1
    for b, wT in zip(biases, weightsT):
            activations.append(sigmoid(np.dot(wT, activations[i-1])+b))
            i+=1
    #############activations=sigmoid(input*weight+bias)
    # compute the gradient of error respect to output
    # activations[-1] is the list of activations of the output layer
    delta = (cost).df_wrt_a(activations[-1], y)
    
    ### Implement here
    # backward pass
    # Here you need to implement the backward pass to compute the
    # gradient for each weight and bias
    ###  h(k)
    z = np.dot(weightsT[-1],activations[-2])+biases[-1]
    delt = np.multiply(delta,sigmoid_prime(z))
    for i in range(num_layers-1,0,-1):
        j=0
        for wT,b,d in zip(weightsT[i-1], biases[i-1],delt):
            nabla_wT[i-1][j] = np.multiply(np.transpose(activations[i-1]),d)
            nabla_b[i-1][j]= d
            j+=1
        if i!=1:
            delt = np.multiply(np.dot(np.transpose(weightsT[i-1]),delt),sigmoid_prime(np.dot(weightsT[i-2],activations[i-2])+biases[i-2]))
         
    return (nabla_b, nabla_wT)

