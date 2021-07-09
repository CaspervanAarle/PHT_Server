# -*- coding: utf-8 -*-
"""
Created on Tue Apr 20 12:44:22 2021

@author: Casper
"""
import random

def set_request_message(message, request):
    w = [*message]
    w.append(request)
    wt = *w,
    return wt

def split(connections, seed, test_split):
    locker_amount = len(connections)
    indices = list(range(locker_amount))
    random.Random(seed).shuffle(indices)
    training_indices, testing_indices = indices[:int(test_split*locker_amount)], indices[int(test_split*locker_amount):]
    print(training_indices)
    print(testing_indices)
    return [connections[i] for i in training_indices], [connections[i] for i in testing_indices]

def get_requests_list():
    return ['calc_loss', 'learn', 'acc', 'mean_get', 'mean_set', 'std_get', 'std_set', 'secret_share1', 'secret_share2', 'derivative_sec_agg', 'standardize_sec_agg', 'scale_get', 'scale_set']