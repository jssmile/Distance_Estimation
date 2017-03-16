'''
 * Copyright 2017 Distance Measurement, EE, NCKU. All rights reserved. 
 * File : server.py 
 * User : Syuan Jhao 
 * Date : 2017/3/16 
 * Version : 1.0
 * OS : Ubuntu Mate 16.04 LTS
 * Tools : Python 2.7 + Opencv 3.2.0
 * Introduction : Receive the camera frames from client and send them back.
'''

import socket
import cv2
import numpy

# Receive the image data
def recvall(sock, count):
    buf = b''
    while count:
        newbuf = sock.recv(count)
        if not newbuf: return None
        buf += newbuf
        count -= len(newbuf)
    return buf

# IP and port number. must same as server's
TCP_IP = '140.116.164.7'
TCP_PORT = 5001

# AF_INET -> IPv4, SOCK_STREAM -> TCP
server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind((TCP_IP, TCP_PORT))
server.listen(True)
conn, addr = server.accept()

while (True):
    # Receive the data
	length = recvall(conn,16)
	stringData = recvall(conn, int(length))
	data = numpy.fromstring(stringData, dtype='uint8')
    
	decimg=cv2.imdecode(data,1)
	cv2.imshow('SERVER', decimg)
	cv2.waitKey(1)

server.close()
cv2.destroyAllWindows() 