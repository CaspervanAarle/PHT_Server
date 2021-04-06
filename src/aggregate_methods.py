# -*- coding: utf-8 -*-
"""
Created on Sat Mar 27 17:38:41 2021

@author: Casper
"""
import numpy as np

class FedSGD_Server():
    def __init__(self, model):
        self.learning_rate = 0.05
        self.model = model
        self.L = 0.05
        
    def update(self, l):   
        # gradient averaging:
        d_weights = np.sum([la[0] for la in l], axis=0)/len(l)
        d_bias = sum([la[1] for la in l])/len(l)
        
        # update current weights with gradient:
        new_weights = self.model.get_weights()[0] - self.L * d_weights
        new_bias = self.model.get_weights()[1] - self.L * d_bias
        
        # set the new weights:
        self.model.set_weights((new_weights, new_bias))
        
        # terminal output:
        print("[INFO] aggregated gradients and update weights:")
        print(self.model.get_weights())
        
        # return copy of the weights for reference:
        return self.model.get_weights()


class FedAvg_Server():
    def __init__(self, model):
        self.model = model
        self.agg_type = "FedAvg"
        
    def update(q):
        pass