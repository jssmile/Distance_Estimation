import socket
import numpy
import time
import cv2

UDP_IP = "127.0.0.1"
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