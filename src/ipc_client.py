# -*- coding: utf-8 -*-
"""
Created on Fri Apr  2 17:14:13 2021

@author: Casper
"""

# CENTRAL AGGREGATOR!

import socket
import pickle

HEADER = 30
#HOST = '192.168.0.13'  # The server's hostname or IP address
#PORT = 9898        # The port used by the server
FORMAT = 'utf-8'

class IPC_Client():
    def __init__(self, host, port):
        self.host = host
        self.port = port
    
    def connect(self):
        """ Check if able to connect and initialize connection if possible """
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.connect((self.host, int(self.port)))
            print("[INFO] Locker connected: " + self.host + " " + self.port)
            return True
        except:
            print("[WARNING] Failed to connect locker: " + self.host + " " + self.port)
            return False
        
    def request(self, message):
        """ Send a request to the lockers """
        self._send_bytes(message)
        return self._receive_bytes()
    
    
    def _send_bytes(self, msg):
        message = pickle.dumps(msg)
        msg_length = len(message)
        send_length = str(msg_length).encode(FORMAT)
        send_length += b' ' * (HEADER - len(send_length))
        self.socket.send(send_length)
        self.socket.send(message)
    
    
    def _receive_bytes(self):
        msg_length = self.socket.recv(HEADER).decode(FORMAT)
        if msg_length:
            msg_length = int(msg_length)
            msg = self.socket.recv(msg_length)
            msg = pickle.loads(msg)
            return msg



