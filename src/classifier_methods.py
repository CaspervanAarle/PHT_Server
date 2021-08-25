# -*- coding: utf-8 -*-
"""
Created on Sat Mar 27 13:54:33 2021

@author: Casper
"""
import numpy as np


class LinReg():
    def __init__(self, hidden_units):
        self.hidden_units = hidden_units
        self.m = np.random.normal(loc=0.0, scale=1.0, size=hidden_units)
        self.c = 0

    def call(self, inputs):
        return np.dot(inputs, self.m) + self.c

    def gradient(self, X, Y):
        Y_pred = np.dot(X, self.m) + self.c  # The current predicted value of Y=
        D_m = (-2/float(len(X))) * np.dot((Y - Y_pred), X)  # Derivative wrt m
        D_c = (-2/float(len(X))) * sum(Y - Y_pred)  # Derivative wrt c
        return D_m, D_c
    
    def get_weights(self):
        return (self.m, self.c)
    
    def set_weights(self, value):
        self.m = value[0]
        self.c = value[1]
        return 
    
    
class LogReg():
    def __init__(self, hidden_units):
        self.m = np.random.normal(loc=0.0, scale=1.0, size=hidden_units)
        self.c = 0

    def call(self, X):
        return self._sigmoid(np.dot(X, self.m) + self.c)

    def gradient(self, X, Y):
        X = np.array(X)
        Y = np.array(Y)
        Y_sig = self.call(X)
        return (1 / len(X)) * np.dot(X.T, Y_sig - Y)
    
    def get_weights(self):
        return (self.m, self.c)
    
    def set_weights(self, value):
        self.m = value[0]
        self.c = value[1]
        return 
    
    def cost(self, X, Y):
        X = np.array(X)
        Y = np.array(Y)
        predictions = self.call(X)
    
        class1_cost = -Y*np.log(predictions)
        class2_cost = (1-Y)*np.log(1-predictions)
        cost = class1_cost - class2_cost
        return cost.sum() / len(X)
        
    def _sigmoid(self, x):
        # Activation function used to map any real value between 0 and 1
        return 1 / (1 + np.exp(-x))