# -*- coding: utf-8 -*-
"""
Created on Mon Apr 19 10:44:11 2021

@author: Casper
"""
import json
import os
import time
import subprocess
from os import path

settings_directory = "..//settings//config_{}.json"
config_name = "experiment"
locker_count = 100

# For this experiment, the lockers all reside on one location
STATIC_LOCKER_IP = "192.168.0.24"

# The lockers will be generated from the following port interatively
STATIC_LOCKER_PORT = 5050

def new_config():
    out = {}
    out['lockers'] = []
    locker = {}
    
    out['config_name'] = config_name
    if(path.exists(settings_directory.format(config_name))):
        print("[WARNING] Overwriting existing config")
    
    request_input = True
    for i in range(locker_count):
        locker['locker_ip'] = STATIC_LOCKER_IP
        
        locker['host_port'] = str(STATIC_LOCKER_PORT+i)
        
        out['lockers'].append(locker.copy())
    
    
    with open(settings_directory.format(config_name),'w') as f:
        json.dump(out, f)
    return out


new_config()
subprocess.run('start python main.py -c config_{}.json'.format(config_name), shell=True)

