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

# edit:
locker_count = 20


SETTINGS_DIRECTORY = "..//settings//config_{}.json"
CONFIG_NAME = "experiment"
# For this experiment, the lockers all reside on one location
STATIC_LOCKER_IP = "192.168.0.24"
# The lockers will be generated from the following port interatively
STATIC_LOCKER_PORT = 5050

def new_config():
    out = {}
    out['lockers'] = []
    locker = {}
    
    out['config_name'] = CONFIG_NAME
    if(path.exists(SETTINGS_DIRECTORY.format(CONFIG_NAME))):
        print("[WARNING] Overwriting existing config")
    
    for i in range(locker_count):
        locker['locker_ip'] = STATIC_LOCKER_IP
        
        locker['host_port'] = str(STATIC_LOCKER_PORT+i)
        
        out['lockers'].append(locker.copy())
    
    
    with open(SETTINGS_DIRECTORY.format(CONFIG_NAME),'w') as f:
        json.dump(out, f)
    return out


new_config()
subprocess.run('start python main.py -c config_{}.json'.format(CONFIG_NAME), shell=True)

