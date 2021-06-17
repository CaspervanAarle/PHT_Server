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

# train-test splitting:
TEST_SPLIT = 0.4
# shufflesplit settings:
AMOUNT_OF_ITERATIONS = 5 # only used in shufflesplit optimization
VALIDATION_SPLIT = 0.2   # only used in shufflesplit optimization


def start():   
    ### 1) setup
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
    
    for i in range(2,12):
        ### 2) train-test split       
        if(TEST_SPLIT > 0):
            conn_train_valid, conn_test = train_test_split(connections, test_size=TEST_SPLIT, random_state=i)
        else:
            conn_train_valid, conn_test = connections, []
        print("test size: {}".format(len(conn_test)))
                
        ### 2.1) shufflesplit optimization         
        if(learn_settings["mode"] == "SHUFFLESPLIT"):
            rs = ShuffleSplit(n_splits=AMOUNT_OF_ITERATIONS, test_size=VALIDATION_SPLIT, random_state=0)
            loss_list = []
            for train_index, validation_index in rs.split(list(range(len(conn_train_valid)))):
                print("TRAIN:", train_index, "\nVALIDATION:", validation_index)
                loss = train_model(np.array(conn_train_valid)[train_index], np.array(conn_train_valid)[validation_index], learn_settings)
                loss_list.append(loss)
            out.save_multi_result(loss_list)
            
        ### 2.2) normal training    
        if(learn_settings["mode"] == "NORMAL"):
            loss = train_model(conn_train_valid, conn_test, learn_settings)
    
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
    
    # termination object
    earlystopper = EarlyStopper()
    
    if(learn_settings["standardization"]):
        ### request encrypted means for standardization
        wt = other.set_request_message(["",""], 3)
        l_n = saggregator.request(conn_training, wt)
        enc_means = [sum(i)/len(conn_training) for i in zip(*l_n)]
        wt = other.set_request_message([enc_means,""], 4)
        l_n = saggregator.request(conn_training, wt)
        l_n = saggregator.request(conn_test, wt)
        
        ### request encrypted stdev for standardization
        wt = other.set_request_message(["",""], 5)
        l_n = saggregator.request(conn_training, wt)
        enc_stdev = [sum(i)/len(conn_training) for i in zip(*l_n)]
        wt = other.set_request_message([enc_stdev,""], 6)
        l_n = saggregator.request(conn_training, wt)
        l_n = saggregator.request(conn_test, wt)
    
    print("[INFO] starting learning loop")
    i = 0
    while i < learn_settings["max_iter"]:
        print("Iteration: {}".format(i+1))
        ### lockers request
        weights = model.get_weights()
        wt = other.set_request_message(weights, 1)
        l_n = saggregator.request(conn_training, wt)
        ### data aggregeren
        weights = otm.update_weights(l_n)
        
        i += 1
        
        ### check training loss
        if learn_settings["calc_train_loss"]:
            wt = other.set_request_message(weights, 0)
            loss_list = saggregator.request(conn_training, wt)
            metric_train = sum(loss_list)/len(conn_training)
            print("Train loss: {}".format(metric_train))
            # earlystopping:
            if earlystopper.check_terminate(metric_train):
                break
        
        ### check test loss
        if len(conn_test) > 0 and learn_settings["calc_test_loss"]:
            wt = other.set_request_message(weights, 0)
            loss_list = saggregator.request(conn_test, wt)
            metric_test = sum(loss_list)/len(conn_test)
            print("Test loss: {}".format(metric_test))
            
        # check if minimum loss is reached
        if earlystopper.check_terminate(metric_train):
            break

    ### accuracy request
    test_accuracy = None
    wt = other.set_request_message(weights, 2)
    if optimizer == LogReg:
        test_accuracy = sum(saggregator.request(conn_test, wt))/len(conn_test)

    ### train loss
    wt = other.set_request_message(weights, 0)
    train_loss = sum(saggregator.request(conn_training, wt))/len(conn_training)
    
    ### test loss
    test_loss = None
    if(len(conn_test) > 0):
        test_loss = sum(saggregator.request(conn_test, wt))/len(conn_test)
        
    ### save results
    out.save_result(i+1, weights, train_loss, test_loss, test_accuracy)
    
    ### return valid loss
    if(len(conn_test) > 0):
        return test_loss
    else:
        return train_loss

   
    
if __name__ == "__main__" :
        print("[INFO] server is starting...")
        try:
            start()
        except Exception as e:
            print("[ERROR] {}".format(e))
            time.sleep(20)
        
        
        