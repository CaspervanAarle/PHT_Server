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
        self.L = 0.05

    def call(self, inputs):
        return np.dot(inputs, self.m) + self.c

    def gradient(self, X, Y):
        Y_pred = np.dot(X, self.m) + self.c  # The current predicted value of Y=
        D_m = (-2/float(len(X))) * np.dot((Y - Y_pred), X)  # Derivative wrt m
        D_c = (-2/float(len(X))) * sum(Y - Y_pred)  # Derivative wrt c
        #self.m = self.m - self.L * D_m  # Update m
        #self.c = self.c - self.L * D_c  # Update c
        return D_m, D_c
    
    def get_weights(self):
        return (self.m, self.c)
    
    def set_weights(self, value):
        self.m = value[0]
        self.c = value[1]
        return 
    
    
class LogReg():
    def __init__(self, hidden_units):
        pass

    def call(self, inputs):
        pass

    def gradient(self, X, Y):
        pass
    
    def get_weights(self):
        pass
    
    def set_weights(self, value):
        pass