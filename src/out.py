# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 09:48:18 2021

@author: Casper
"""

import datetime
import numpy as np
import os
import pickle
import json

from classifier_methods import LinReg, LogReg
from sklearn.linear_model import LogisticRegression, LinearRegression


def save_result(i, weights=None, train_loss=None, test_loss=None, test_accuracy=None, mu=None, sigma=None):
    if not os.path.exists("..//results//"):
        os.makedirs("..//results//")
    f = open("..//results//all_results.txt", "a")

    f.write("\n\n-- Results Experiment: {}".format(datetime.datetime.now()))

    if weights is not None:
        f.write("\nCoefficients: ")
        f.write(np.array_str(weights[0]))
        f.write("\nIntercept: ")
        f.write(str(weights[1]))

    if train_loss is not None:
        f.write("\nTrain loss: ")
        f.write(str(train_loss))

    if test_loss is not None:
        f.write("\nTest loss: ")
        f.write(str(test_loss))

    if test_accuracy is not None:
        f.write("\nTest accuracy: ")
        f.write(str(test_accuracy))

    if mu is not None:
        f.write("\nFeatures mean: ")
        f.write(str(mu))

    if sigma is not None:
        f.write("\nFeatures sigma: ")
        f.write(str(sigma))

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


def build_scikit_model(custom_model):
    if(type(custom_model) == LinReg):
        scikit_model = LinearRegression()
    if(type(custom_model) == LogReg):
        scikit_model = LogisticRegression()
        scikit_model.classes_ = np.array([0,1])
    scikit_model.coef_ = np.array([custom_model.m])
    scikit_model.intercept_ = np.array(custom_model.c)
    return scikit_model

def save_model(name, custom_model, *hyperparams):
    save_dir = "../results/"
    # save the model to disk
    pickle.dump(build_scikit_model(custom_model), open(save_dir + name + '.sav', 'wb'))
    
    # save hyperparameters to disk
    with open(save_dir + name + '.json','w') as jsonFile:
        json.dump(hyperparams[0], jsonFile)

if __name__ == "__main__":
    lr = LinReg(5)
    save_model("experiment_model", lr, {'iter' : 100, 'lr' : 0.1})
