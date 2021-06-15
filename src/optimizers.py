# -*- coding: utf-8 -*-
"""
Created on Sat Mar 27 17:38:41 2021

@author: Casper
"""
import numpy as np
import abc


class CustomOptimizer(abc.ABC):
    @abc.abstractmethod
    def update_weights(self, l):
        pass


class SGD_Optimizer(CustomOptimizer):
    def __init__(self, model, learning_rate=0.05, reg=0.0):
        self.model = model
        self.eta = learning_rate
        self.reg=reg
        
        
    def update_weights(self, l):   
        # gradient averaging:
        d_weights = (np.sum([la[0] for la in l], axis=0) + self.reg*self.model.get_weights()[0])/len(l)
        d_bias = sum([la[1] for la in l])/len(l) + 2*self.reg*self.model.get_weights()[1]
        
        # update current weights with gradient:
        new_weights = self.model.get_weights()[0] - self.eta * d_weights
        new_bias = self.model.get_weights()[1] - self.eta * d_bias
        
        # set the new weights:
        self.model.set_weights((new_weights, new_bias))
        
        # terminal output:
        print("[INFO] aggregated gradients and update weights: {}".format(self.model.get_weights()))
        
        # return copy of the weights for reference:
        return self.model.get_weights()
        
        
class AdaGrad_Optimizer(CustomOptimizer):
    # adagrad parameters:
    def __init__(self, model, learning_rate=50):
        self.model = model
        self.eta = learning_rate
        self.v_w, self.v_b, self.eps = 0, 0, 1e-8
        
    def update_weights(self, l):
        # gradient averaging:
        dw = np.sum([la[0] for la in l], axis=0)/len(l)
        db = sum([la[1] for la in l])/len(l)
                
        self.v_w = self.v_w + dw**2
        self.v_b = self.v_b + db**2
            
        w = self.model.get_weights()[0] - (self.eta/np.sqrt(self.v_w + self.eps)) * dw
        b = self.model.get_weights()[1] - (self.eta/np.sqrt(self.v_b + self.eps)) * db
        
        self.model.set_weights((w, b))
        
        # terminal output:
        print("[INFO] aggregated gradients and update weights: {}".format(self.model.get_weights()))
        
        # return copy of the weights for reference:
        return self.model.get_weights()