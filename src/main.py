# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 16:00:45 2021

@author: Casper
"""
    
import saggregator
import numpy as np
import config_setup

# import the correct algorithm:
from classifier_methods import LinReg
from aggregate_methods import FedSGD_Server

# import socket protocol
from ipc_client import IPC_Client  

#connection_info = [("192.168.0.13", 5050), ("192.168.0.13", 5051), ("192.168.0.13", 5052)]
connections = []


def learning_loop():      
    config = config_setup.setup()
    
    # connect with corresponding lockers
    for locker in config['lockers']:
        conn = IPC_Client(locker['locker_ip'], locker['host_port'])
        if(conn.connect()):
            connections.append(conn)
            print("Locker connected: " + locker['locker_ip'] + " " + locker['host_port'])
    assert len(connections) > 0
    
    # input and output variable names
    var_list = ["var_1", "var_2", "var_3", "var_4", "var_5", "var_6"]
    target_list = ["var_9"]
    
    model = LinReg([len(var_list)])
    server = FedSGD_Server(model)
    
    # main learning loop:
    weights = model.get_weights()
    print("[INFO] starting learning loop")
    i = 0
    while i < 20:
        ### lockers request
        l_n = saggregator.request(connections, weights)
        
        ### data aggregeren
        weights = server.update(l_n)
        
        i += 1
    
    
    
if __name__ == "__main__" :
        print("[INFO] server is starting...")
        learning_loop()
        
        
        