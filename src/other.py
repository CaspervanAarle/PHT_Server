# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 12:44:22 2021

@author: Casper
"""

import time

class EarlyStopper():
    def __init__(self):
        self.TOL = 10 ** -7
        self.TOL_COUNT_THRESHOLD = 4
        self.tol_count = 0
        
        self.previous_loss = 10 ** 7 #=infinity
        
    
    def check_terminate(self, loss):
        if( (self.previous_loss - loss) < self.TOL):
            self.previous_loss = loss
            self.tol_count +=1
            print("loss minimum found")
            time.sleep(1)
            if self.tol_count >= self.TOL_COUNT_THRESHOLD:
                return True
            else:
                return False
        else:
            self.tol_count = 0
            self.previous_loss = loss
            return False
        