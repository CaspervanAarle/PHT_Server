# -*- coding: utf-8 -*-
"""
Created on Tue Mar 30 16:37:47 2021

@author: Casper
"""

import argparse
import json
import inquirer
import os

from os import path

def new_config():
    out = {}
    out['lockers'] = []
    locker = {}
    
    config_name = input("config name:")
    out['config_name'] = config_name
    goal_dir = "../settings/config_" + config_name + ".json"
    if(path.exists(goal_dir)):
        print("[WARNING] Overwriting existing config")
    
    request_input = True
    while request_input:
        locker_ip = input("locker-ip:")
        locker['locker_ip'] = locker_ip
        
        host_port = input("locker-port:")
        locker['host_port'] = host_port
        while True: 
            query = input('add another locker? [y/n]') 
            Fl = query[0].lower() 
            if query == '' or not Fl in ['y','n']: 
                print('Please answer with yes or no!') 
            else: 
                break 
        if Fl == 'y': 
            request_input = True
        if Fl == 'n': 
            request_input = False
        out['lockers'].append(locker.copy())
    
    
    with open(goal_dir,'w') as f:
        json.dump(out, f)
    return out


    
def setup():
    if not os.path.exists('../settings'):
        os.makedirs('../settings')
    parser = argparse.ArgumentParser(description="Simple argument parser")
    # add a new command line option, call it '-c' and set its destination to 'config'
    parser.add_argument("-c", action="store", dest="config_file", help='choose your configuration without a settings menu')
    
    # get the result
    result = parser.parse_args()
    #print(result)
    # since we set 'config_file' as destination for the argument -c, 
    # we can fetch its value like this (and print it, for example):    
        
    ### if argument is given
    if(result.config_file):
        config_file = result.config_file
    
    ### if no argument is given
    else:
        config_files = ["<new_config>"]
        for file in os.listdir("../settings"):
            if file.endswith(".json"):
                config_files.append(file) 
                
        questions = [
            inquirer.List(
                "config",
                message="What config do you need?",
                choices=config_files,
            ),
        ]
        answer = inquirer.prompt(questions)
        
        ### if user wants to create new config file:
        if answer["config"] == "<new_config>":
            return new_config()
        ### if user chooses existing config file
        config_file = answer["config"]
    try: 
        with open("../settings/" + config_file) as json_file:
            return json.load(json_file)
    except:
        print("[ERROR] load error")
        return
    
