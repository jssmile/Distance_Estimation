'''
 * Copyright 2017 Distance Measurement, EE, NCKU. All rights reserved. 
 * File : udp_recv.py 
 * User : Syuan Jhao 
 * Date : 2017/3/16 
 * Version : 1.0
 * OS : Ubuntu Mate 16.04 LTS
 * Tools : Python 2.7 + Opencv 3.2.0
 * Introduction : Receive the camera frame and display through UDP protocol.
'''

import socket
import numpy
import time
import cv2

UDP_IP = "140.116.164.8"
UDP_PORT = 5005

sock = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
sock.bind ((UDP_IP, UDP_PORT))

s=""

while True:

    data, addr = sock.recvfrom(46080)

    s += data

    if len(s) == (46080*20):

        frame = numpy.fromstring (s,dtype=numpy.uint8)
        frame = frame.reshape (480,640,3)

        cv2.imshow('frame',frame)

        s=""

    if cv2.waitKey(1) & 0xFF == ord ('q'):
        break