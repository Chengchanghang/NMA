# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 14:23:13 2017

@author: Administrator
"""

import numpy as np
import random

#load data
IR = np.load('E:/NMA/IR_spectra_G_15.npy')[:10000]
Structure = np.load('E:/NMA/structs.npy')[:10000]
training_x = IR[:9000]
test_x = IR[9000:]
training_y = Structure[:9000] / 6
test_y = Structure[9000:]
tr_data = zip(training_x,training_y)
te_data = zip(test_x,test_y)

eta = 0.3
epcoes = 30

def sigmoid(z):
    return 1.0 / ( 1.0 + np.exp(-z))
    
def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))
def cost_derivative(output_activations,y):
    return (output_activations - y )

network = ([1201,30,51])
num_layers = len(network)
biases = [np.random.randn(y,1) for y in network[1:]]
weights = [np.random.randn(y,x) for x,y in zip(network[:-1],network[1:])]
           
nabla_b = [np.zeros(b.shape) for b in biases]
nabla_w = [np.zeros(w.shape) for w in weights]

#mnist
x_labels = []
y_labels = []
n = len(tr_data)
for j in xrange(epcoes):
    random.shuffle(tr_data)
    mnist_batchs =[tr_data[k : k + sizes] for k in xrange(0,n,sizes)] 
    for mnist_batch in mnist_batchs:
        for x, y in mnist_batch:
            activation = x
            y = y
            activations = [x] 
            zs = []
            for b, w in zip(biases, weights):
                z = np.dot(w, activation)+b
                zs.append(z)
                activation = sigmoid(z)
                activations.append(activation)
            delta = cost_derivative(activations[-1], y) * sigmoid_prime(zs[-1]) 
            nabla_b[-1] = delta
            nabla_w[-1] = np.dot(delta, activations[-2].transpose())
            for l in xrange(2,num_layers):
                z = zs[-l]
                sp = sigmoid_prime(z)
                delta = np.dot(weights[-l+1].transpose(), delta) * sp
                nabla_b[-l] = delta
                nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
            delta_nabla_b, delta_nabla_w = nabla_b,nabla_w
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]  
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
            weights = [w-(eta)*nw for w, nw in zip(weights, nabla_w)]