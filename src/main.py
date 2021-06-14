# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 16:00:45 2021

@author: Casper
"""
import sys
current_module = sys.modules[__name__]
import saggregator
import numpy as np
import config_setup
import out
import time

# import the correct algorithm:
from classifier_methods import LinReg, LogReg
from optimizers import SGD_Optimizer, AdaGrad_Optimizer
from earlystopper import EarlyStopper
import other
from sklearn.model_selection import ShuffleSplit, train_test_split

# homomorphic encryption for mean and average
from phe import paillier

# import socket protocol
from ipc_client import IPC_Client 

# regularization constant
REG_CONSTANT = 1
# train-test splitting:
TEST_SPLIT = 0.5
# shufflesplit settings:
AMOUNT_OF_ITERATIONS = 5 # only used in shufflesplit optimization
VALIDATION_SPLIT = 0.2   # only used in shufflesplit optimization

def start():   
    # setup
    connections = []
    config, learn_settings = config_setup.setup()
    print(str(config).replace(', ',',\n '))
    print(str(learn_settings).replace(', ',',\n '))
    
    # connect with corresponding lockers
    for locker in config['lockers']:
        conn = IPC_Client(locker['locker_ip'], locker['host_port'])
        if(conn.connect()):
            connections.append(conn)
            print("Locker connected: " + locker['locker_ip'] + " " + locker['host_port'])
    #print(connections)
    assert len(connections) > 0
    
    ### train-test split       
    if(TEST_SPLIT > 0):
        conn_train_valid, conn_test = train_test_split(connections, test_size=TEST_SPLIT, random_state=0)
    else:
        conn_train_valid, conn_test = connections, []
    print("test size: {}".format(len(conn_test)))
            
    ### shufflesplit optimization         
    if(learn_settings["mode"] == "SHUFFLESPLIT"):
        rs = ShuffleSplit(n_splits=AMOUNT_OF_ITERATIONS, test_size=VALIDATION_SPLIT, random_state=0)
        loss_list = []
        for train_index, validation_index in rs.split(list(range(len(conn_train_valid)))):
            print("TRAIN:", train_index, "\nVALIDATION:", validation_index)
            loss = train_model(np.array(conn_train_valid)[train_index], np.array(conn_train_valid)[validation_index], learn_settings)
            loss_list.append(loss)
        out.save_multi_result(loss_list)
        
    ### normal training    
    if(learn_settings["mode"] == "NORMAL"):
        loss = train_model(conn_train_valid, conn_test, learn_settings)
    
def train_model(conn_training, conn_test, learn_settings):
    print("Training set size: {}".format(len(conn_training)))
    print("validation set size: {}".format(len(conn_test)))
    
    model_class = getattr(current_module, learn_settings["regressor"])
    model = model_class([len(learn_settings["var_list"])])
    
    optimizer = getattr(current_module, learn_settings["optimizer"] + '_Optimizer')
    otm = optimizer(model, learn_settings["learning_rate"])
    
    # main learning loop:
    weights = model.get_weights()
    # termination object
    earlystopper = EarlyStopper()
    print("[INFO] starting learning loop")
    
    ### request encrypted means
    wt = other.set_request_message(["",""], 3)
    l_n = saggregator.request(conn_training, wt)
    print(l_n)
    enc_means = [sum(i)/len(conn_training) for i in zip(*l_n)]
    wt = other.set_request_message([enc_means,""], 4)
    l_n = saggregator.request(conn_training, wt)
    l_n = saggregator.request(conn_test, wt)
    
    ### request encrypted stdev
    wt = other.set_request_message(["",""], 5)
    l_n = saggregator.request(conn_training, wt)
    enc_stdev = [sum(i)/len(conn_training) for i in zip(*l_n)]
    wt = other.set_request_message([enc_stdev,""], 6)
    l_n = saggregator.request(conn_training, wt)
    l_n = saggregator.request(conn_test, wt)
    
    i = 0
    while i < learn_settings["max_iter"]:
        print("Iteration: {}".format(i+1))
        ### lockers request
        wt = other.set_request_message(weights, 1)
        print(wt)
        l_n = saggregator.request(conn_training, wt)
        ### data aggregeren
        weights = otm.update_weights(l_n)
        
        i += 1
        
        ### check training loss
        if(len(conn_training) > 0):
            wt = other.set_request_message(weights, 0)
            loss_list = saggregator.request(conn_training, wt)
            metric_train = sum(loss_list)/len(conn_training)
            print("train loss: {}".format(metric_train))
        
        
        ### check validation loss
        """
        if(len(conn_test) > 0):
            wt = set_request_message(weights, 0)
            loss_list = saggregator.request(conn_test, wt)
            metric_test = sum(loss_list)/len(conn_test)
            print("test loss: {}".format(metric_test))
        """
            
        # check if minimum loss is reached
        if earlystopper.check_terminate(metric_train):
            break
        
        
    ### loss request (use test data if available)
    
    wt = other.set_request_message(weights, 0)
    if(len(conn_test) > 0):
        loss = sum(saggregator.request(conn_test, wt))/len(conn_test)
    else:
        loss = sum(saggregator.request(conn_training, wt))/len(conn_training)
   
    ### accuracy request (only for logistic regression)
    #wt = set_request_message(weights, 2)
    #accuracy = sum(saggregator.request(conn_training, wt))/len(conn_training)
    #print("accuracy: {}".format(accuracy))
    
    
    ### accuracy request (only for logistic regression)
    # wt = set_request_message(weights, 2)
    #accuracy = sum(saggregator.request(conn_training, wt))/len(conn_training)
    
    out.save_result(i+1, weights, loss)
    return loss
    

   
    
if __name__ == "__main__" :
        print("[INFO] server is starting...")
        try:
            start()
        except Exception as e:
            print("[ERROR] {}".format(e))
            time.sleep(20)
        
        
        