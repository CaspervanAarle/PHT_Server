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
    
    
class Adam_Optimizer(CustomOptimizer):
    def __init__(self, model, learning_rate=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.model = model
        self.m_dw, self.v_dw = 0, 0
        self.m_db, self.v_db = 0, 0
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.t = 1
    def update_weights(self, l):
        # get current weights:
        w,b = self.model.get_weights()
        # get current gradients:
        dw = np.sum([la[0] for la in l], axis=0)/len(l)
        db = np.sum([la[1] for la in l])/len(l)
        
        ## dw, db are from current minibatch
        ## momentum beta 1
        # *** weights *** #
        self.m_dw = self.beta1*self.m_dw + (1-self.beta1)*dw
        # *** biases *** #
        self.m_db = self.beta1*self.m_db + (1-self.beta1)*db

        ## rms beta 2
        # *** weights *** #
        self.v_dw = self.beta2*self.v_dw + (1-self.beta2)*(dw**2)
        # *** biases *** #
        self.v_db = self.beta2*self.v_db + (1-self.beta2)*(db**2)

        ## bias correction
        m_dw_corr = self.m_dw/(1-self.beta1**self.t)
        m_db_corr = self.m_db/(1-self.beta1**self.t)
        v_dw_corr = self.v_dw/(1-self.beta2**self.t)
        v_db_corr = self.v_db/(1-self.beta2**self.t)

        ## update weights and biases
        w = w - self.learning_rate*(m_dw_corr/(np.sqrt(v_dw_corr)+self.epsilon))
        b = b - self.learning_rate*(m_db_corr/(np.sqrt(v_db_corr)+self.epsilon))
        
        self.model.set_weights((w, b))
        
        # terminal output:
        print("[INFO] aggregated gradients and update weights: {}".format(self.model.get_weights()))
        self.t = self.t+1
        return self.model.get_weights()