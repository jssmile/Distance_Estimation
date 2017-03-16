'''
 * Copyright 2017 Distance Measurement, EE, NCKU. All rights reserved. 
 * File : udp_send.py 
 * User : Syuan Jhao 
 * Date : 2017/3/16 
 * Version : 1.0
 * OS : Ubuntu Mate 16.04 LTS
 * Tools : Python 2.7 + Opencv 3.2.0
 * Introduction : sned the camera frame through UDP protocol.
'''

import numpy as np
import cv2
import socket
import datetime

UDP_IP = "140.116.164.8"
UDP_PORT = 5005
cnt = 0
fps = 0

cap = cv2.VideoCapture(0)

while(True):
    ret, frame = cap.read()
    cnt = cnt + 1
    if cnt == 1:
        start = datetime.datetime.now()
    cv2.putText(frame,
                    "fps" + str(fps),
                    (0, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (150,0,255),
                    2)
    if cnt == 10:
    	end = datetime.datetime.now()
    	period = end - start
    	period = period.total_seconds()
    	fps = 10 / period
    	print(str(fps))
    	cnt = 0
    	cv2.putText(frame,
                    "fps" + str(fps),
                    (0, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (150,0,255),
                    2)

    cv2.imshow('frame',frame)


    sock = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)

    d = frame.flatten ()
    s = d.tostring ()



    for i in xrange(16):

        sock.sendto (s[i*46080:(i+1)*46080],(UDP_IP, UDP_PORT))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()