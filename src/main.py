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
from random import sample
import math

# import the correct algorithm:
from classifier_methods import LinReg, LogReg
from optimizers import SGD_Optimizer, AdaGrad_Optimizer, Adam_Optimizer
from earlystopper import EarlyStopper
import other
from sklearn.model_selection import ShuffleSplit, train_test_split
import data_scaler
import time
from datetime import datetime

# homomorphic encryption for mean and average
from phe import paillier

# import socket protocol
from ipc_client import IPC_Client 

# train-test splitting:
TEST_SPLIT = 0

# shufflesplit settings:
#AMOUNT_OF_ITERATIONS = 5 # only used in shufflesplit optimization
#VALIDATION_SPLIT = 0.2   # only used in shufflesplit optimization


def start():   
    ### setup
    connections = []
    config, learn_settings = config_setup.setup()
    print(str(config).replace(', ',',\n '))
    print(str(learn_settings).replace(', ',',\n '))
    
    ### connect with corresponding lockers
    for locker in config['lockers']:
        conn = IPC_Client(locker['locker_ip'], locker['host_port'])
        if(conn.connect()):
            connections.append(conn)
            print("Locker connected: " + locker['locker_ip'] + " " + locker['host_port'])
    assert len(connections) > 0
    
    ### create train-test split       
    if(TEST_SPLIT > 0):
        conn_train_valid, conn_test = train_test_split(connections, test_size=TEST_SPLIT, random_state=0)
    else:
        conn_train_valid, conn_test = connections, []
    print("test size: {}".format(len(conn_test)))
                
    ### training    
    scaler, model = train_model(conn_train_valid, conn_test, learn_settings)
        
    ### save model
    print("[INFO] saving model")
    name = "experiment_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    if scaler is None:
        out.save_model(name, model, {"n_nodes":len(config['lockers']),
                                            "input_vars":learn_settings["var_list"] ,
                                            "target_vars":learn_settings["target_list"]})        
    else:
        out.save_model(name, model, {"n_nodes":len(config['lockers']),
                                            "input_vars":learn_settings["var_list"] ,
                                            "target_vars":learn_settings["target_list"],
                                            "mu_list":scaler.get_mu(),
                                            "sigma_list":scaler.get_sigma()})
    print("[INFO] model saved as:\n" + name)
    
    print("[INFO] ended")
    time.sleep(60)
    
    
    
def train_model(conn_training, conn_test, learn_settings):
    print("Training set size: {}".format(len(conn_training)))
    print("validation set size: {}".format(len(conn_test)))
    
    # load correct regressor
    regressor_class = getattr(current_module, learn_settings["regressor"])
    model = regressor_class([len(learn_settings["var_list"])])
    
    # load optimizer with or without regularization
    optimizer = getattr(current_module, learn_settings["optimizer"] + '_Optimizer')
    if isinstance(model,LogReg) and isinstance(optimizer, SGD_Optimizer):
        otm = optimizer(model, learn_settings["learning_rate"], learn_settings["regularization"])
    else:
        otm = optimizer(model, learn_settings["learning_rate"])
    
    if(learn_settings["standardization"]):
        ### request encrypted means for standardization
        """
        wt = other.set_request_message(["",""], 3)
        l_n = saggregator.request(conn_training, wt)
        enc_means = [sum(i)/len(conn_training) for i in zip(*l_n)]
        wt = other.set_request_message([enc_means,""], 4)
        l_n = saggregator.request(conn_training, wt)
        l_n = saggregator.request(conn_test, wt)
        """
        
        ### request encrypted stdev for standardization
        
        """
        wt = other.set_request_message(["",""], 5)
        l_n = saggregator.request(conn_training, wt)
        enc_stdev = [sum(i)/len(conn_training) for i in zip(*l_n)]
        wt = other.set_request_message([enc_stdev,""], 6)
        l_n = saggregator.request(conn_training, wt)
        l_n = saggregator.request(conn_test, wt)
        """
        
        ### request secure aggregation for scaling
        wt = other.set_request_message(["",""], 7)
        l_n = saggregator.request(conn_training, wt)
        wt = other.set_request_message([l_n,""], 8)
        l_n = saggregator.request(conn_training, wt)
        wt = other.set_request_message(["",""], 10)
        l_n = saggregator.request(conn_training, wt)
        # calculate mu sigma
        X = np.sum(l_n,axis=0)
        scaler = data_scaler.Scaler(X, conn_training)
        # scale homomorphically
        wt = other.set_request_message(["",""], 11)
        features_and_connections = saggregator.request2(conn_training, wt)
        for f, c in features_and_connections:
            f = scaler.scale(f)
            wt = other.set_request_message([f,""], 12)
            l_n = saggregator.request([c], wt)
    else:
        scaler = None
        
        
        
    print("[INFO] starting learning loop")
    i = 0
    while i < learn_settings["max_iter"]:
        print("Iteration: {}".format(i+1))
        weights = model.get_weights()
        
        ### lockers request without secure aggregation
        wt = other.set_request_message(weights, 1)
        l_n = saggregator.request(conn_training, wt)
        
        ### lockers request with secure aggregation
        """
        wt = other.set_request_message(["",""], 7)
        l_n = saggregator.request(conn_training, wt)
        wt = other.set_request_message([l_n,""], 8)
        l_n = saggregator.request(conn_training, wt)
        wt = other.set_request_message(weights, 9)
        l_n = saggregator.request(conn_training, wt)
        """
        
        ### only update if no errors
        if(len(l_n) == len(conn_training)):
            weights = otm.update_weights(l_n)
        i += 1
        
        ### check training loss
        if learn_settings["calc_train_loss"]:
            wt = other.set_request_message(weights, 0)
            loss_list = saggregator.request(conn_training, wt)
            metric_train = sum(loss_list)/len(conn_training)
            print("Train loss: {}".format(metric_train))

        
        ### check test loss
        if len(conn_test) > 0 and learn_settings["calc_test_loss"]:
            wt = other.set_request_message(weights, 0)
            loss_list = saggregator.request(conn_test, wt)
            metric_test = sum(loss_list)/len(conn_test)
            print("Test loss: {}".format(metric_test))
            
        
    ### accuracy request
    test_accuracy = None
    wt = other.set_request_message(weights, 2)
    if optimizer == LogReg:
        test_accuracy = sum(saggregator.request(conn_test, wt))/len(conn_test)

    ### train loss
    wt = other.set_request_message(weights, 0)
    train_loss = sum(saggregator.request(conn_training, wt))/len(conn_training)
    
    ### test loss
    if(len(conn_test) > 0):
        test_loss = sum(saggregator.request(conn_test, wt))/len(conn_test)
    
    
    return scaler, model
    

if __name__ == "__main__" :
        print("[INFO] server is starting...")
        try:
            start()
        except Exception as e:
            print("[ERROR] {}".format(e))
            time.sleep(60)
        
        
        