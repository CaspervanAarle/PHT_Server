# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 09:48:18 2021

@author: Casper
"""

import datetime
import numpy as np
import os

def save_result(i, weights, loss=None, accuracy=None):
    if not os.path.exists("..//results//"):
        os.makedirs("..//results//")
    f = open("..//results//all_results.txt", "a")
    
    f.write("\n\n-- Results Experiment: {}".format(datetime.datetime.now()))
    
    f.write("\nCoefficients: ")
    f.write(np.array_str(weights[0]))
    
    f.write("\nIntercept: ")
    f.write(str(weights[1]))
    
    if loss:
        f.write("\nLoss: ")
        f.write(str(loss))
    
    if accuracy:
        f.write("\nAccuracy: ")
        f.write(str(accuracy))
    
    f.write("\nIterations: ")
    f.write(str(i))
    
    f.close()
    
    
def save_multi_result(loss):
    if not os.path.exists("..//results//"):
        os.makedirs("..//results//")
    f = open("..//results//all_results.txt", "a")
    
    f.write("\n\n-- Results Multi Experiment: {}".format(datetime.datetime.now()))
    
    f.write("\nLoss: ")
    f.write(str(loss))
    
    avg_loss = sum(loss)/len(loss)
    f.write("\nAverage Loss: ")
    f.write(str(avg_loss))
    
    f.close()