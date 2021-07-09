# -*- coding: utf-8 -*-
"""
Created on Fri Jul  9 12:07:01 2021

@author: Casper
"""
import numpy as np

class Scaler():
    def __init__(self, X, d):
        mu = X[:int(len(X)/2)]/len(d)
        sigma_nosquare = 1/(len(d)) * ( X[int(len(X)/2):] - (len(d))*mu**2 )
        sigma = np.sqrt([abs(v) for v in sigma_nosquare])
        self.mu=mu
        self.sigma=sigma
        
    def scale(self, x):
        return (x - self.mu) / self.sigma
    
    def get_mu(self):
        return self.mu
    
    def get_sigma(self):
        return self.sigma