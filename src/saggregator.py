# -*- coding: utf-8 -*-
"""
Created on Thu Mar 25 00:36:32 2021

@author: Casper
"""

import threading, queue
import time
import other

TIMEOUT = 25

finished = False

    
def request(connections, message):
    print("[INFO] Sending {} requests to {} nodes".format(other.get_requests_list()[message[2]],len(connections)))
    q = queue.Queue()
    thread = threading.Thread(target=aggregate_results, args=(q, connections, message))
    thread.start()
    
    t_ = time.time()
    while( thread.is_alive() and (time.time()-t_ < TIMEOUT)):
        time.sleep(0.0001)
    if(not time.time()-t_ < TIMEOUT):
        print("[WARNING] time-out connection")
        return
    print(f"[INFO] Server data from {len(list(q.queue))} nodes after finishing")
    return list(q.queue)
    
def aggregate_results(q, connections, message):
    threads = [threading.Thread(target=data_request, args=(q, c, message)) for c in connections]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
        
        
def data_request(q, connection, message):
    msg = connection.request(message)
    q.put(msg)
    return
    
    
    
    
    