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

def new_learnconfig():
    pass

    
def setup():
    if not os.path.exists('../settings'):
        os.makedirs('../settings')
    parser = argparse.ArgumentParser(description="Simple argument parser")
    # add a new command line option, call it '-c' and set its destination to 'config'
    parser.add_argument("-c", action="store", dest="config_file", help='choose your configuration without a settings menu')
    
    parser.add_argument("-l", action="store", dest="learnconfig_file", help='choose your learn configuration without a settings menu')
    
    # get the result
    result = parser.parse_args()
    
    #print(result)
    # since we set 'config_file' as destination for the argument -c, 
    # we can fetch its value like this (and print it, for example)    
        
    """ setup config file """    
    ### if argument config is given
    if(result.config_file):
        chosen_config = open_file(result.config_file)
    ### if no argument is given for config
    else:
        config_files = ["<new_config>"]
        for file in os.listdir("../settings"):
            if file.endswith(".json") and file.startswith("config"):
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
            chosen_config = new_config()
        ### if user chooses existing config file
        else:
            chosen_config = open_file(answer["config"])
            
    """ setup learnconfig file """        
    ### if argument learnconfig is given    
    if(result.learnconfig_file):
            chosen_learnconfig = open_file(result.learnconfig_file)
    ### if no argument is given for learnconfig
    else:
        config_files = ["<new_learnconfig>"]
        for file in os.listdir("../settings"):
            if file.endswith(".json") and file.startswith("learnconfig"):
                config_files.append(file) 
        questions = [
            inquirer.List(
                "learnconfig",
                message="What learnconfig do you need?",
                choices=config_files,
            ),
        ]
        answer = inquirer.prompt(questions)
        ### if user wants to create new config file:
        if answer["learnconfig"] == "<new_learnconfig>":
            chosen_learnconfig = new_learnconfig()
        ### if user chooses existing config file
        else:
            chosen_learnconfig = open_file(answer["config"])

    return chosen_config, chosen_learnconfig
    


def open_file(config_file):
    try: 
        with open("../settings/" + config_file) as json_file:
            return json.load(json_file)
    except:
        print("[ERROR] load error")
        return