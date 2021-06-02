# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 16:00:45 2021

@author: Casper
"""
    
import saggregator
import numpy as np
import config_setup
import datetime
import out
import time
from random import randint
import random

# import the correct algorithm:
from classifier_methods import LinReg, LogReg
from optimizers import SGD_Optimizer, AdaGrad_Optimizer
from other import EarlyStopper

# homomorphic encryption for mean and average
from phe import paillier

# import socket protocol
from ipc_client import IPC_Client  

# ADJUSTABLE PARAMETERS:
MAX_ITER = 200
model_class = LinReg

var_list = ["F1",	"F2",	"F3",	"F4",	"F5",	"F6",	"F7",	"F8",	"F9"]
target_list = ["RMSD"]
test_split = 0
optimizer = AdaGrad_Optimizer
#learning_rate=0.05
is_experiment = True
# lr: SGD=0.05, AdaGra=5/50
lr = 5000

def learning_loop():   
    connections = []
    config = config_setup.setup()
    # connect with corresponding lockers
    for locker in config['lockers']:
        conn = IPC_Client(locker['locker_ip'], locker['host_port'])
        if(conn.connect()):
            connections.append(conn)
            print("Locker connected: " + locker['locker_ip'] + " " + locker['host_port'])
    assert len(connections) > 0
    
    # shuffle connections
    if(test_split > 0):
        if(is_experiment):
            conn_training, conn_test = split(connections, 1)
        else:
            conn_training, conn_test = split(connections, randint(0, 1000))
    else:
        conn_training, conn_test = connections, []
    model = model_class([len(var_list)])
    otm = optimizer(model, lr)
    
    
    # main learning loop:
    weights = model.get_weights()
    # termination object
    earlystopper = EarlyStopper()
    print("[INFO] starting learning loop")
    
    ### request encrypted means
    wt = set_request_message(["",""], 3)
    l_n = saggregator.request(conn_training, wt)
    print(l_n)
    enc_means = [sum(i)/len(conn_training) for i in zip(*l_n)]
    wt = set_request_message([enc_means,""], 4)
    l_n = saggregator.request(conn_training, wt)
    
    ### request encrypted stdev
    wt = set_request_message(["",""], 5)
    l_n = saggregator.request(conn_training, wt)
    enc_stdev = [sum(i)/len(conn_training) for i in zip(*l_n)]
    wt = set_request_message([enc_stdev,""], 6)
    l_n = saggregator.request(conn_training, wt)
    
    i = 0
    while i < MAX_ITER:
        ### lockers request
        wt = set_request_message(weights, 1)
        print(wt)
        l_n = saggregator.request(conn_training, wt)
        ### data aggregeren
        weights = otm.update_weights(l_n)
        
        i += 1
        
        ### check training loss
        if(len(conn_training) > 0):
            wt = set_request_message(weights, 0)
            loss_list = saggregator.request(conn_training, wt)
            metric_train = sum(loss_list)/len(conn_training)
            print("train loss: {}".format(metric_train))
        
        
        ### check validation loss
        if(len(conn_test) > 0):
            wt = set_request_message(weights, 0)
            loss_list = saggregator.request(conn_test, wt)
            metric_test = sum(loss_list)/len(conn_test)
            print("test loss: {}".format(metric_test))
            if earlystopper.check_terminate(metric_test):
                break
        else:
            if earlystopper.check_terminate(metric_train):
                break
        
        ### accuracy request (only for logistic regression)
        #wt = set_request_message(weights, 2)
        #accuracy = sum(saggregator.request(conn_training, wt))/len(conn_training)
        #print("accuracy: {}".format(accuracy))
        
    ### loss request (use test data if available)
    wt = set_request_message(weights, 0)
    if(len(conn_test) > 0):
        loss = sum(saggregator.request(conn_test, wt))/len(conn_test)
    else:
        loss = sum(saggregator.request(conn_training, wt))/len(conn_training)
    
    ### accuracy request (only for logistic regression)
    wt = set_request_message(weights, 2)
    #accuracy = sum(saggregator.request(conn_training, wt))/len(conn_training)
    
    out.save_result(i+1, weights, loss)
    

def set_request_message(message, request):
    w = [*message]
    w.append(request)
    wt = *w,
    return wt

def split(connections, seed):
    locker_amount = len(connections)
    indices = list(range(locker_amount))
    random.Random(seed).shuffle(indices)
    print(indices)
    training_indices, testing_indices = indices[:int(test_split*locker_amount)], indices[int(test_split*locker_amount):]
    return [connections[i] for i in training_indices], [connections[i] for i in testing_indices]

   
    
if __name__ == "__main__" :
        print("[INFO] server is starting...")
        learning_loop()
        
        
        