import socket
import numpy
import time
import cv2

def recvall(sock, count):
    buf = b''
    while count:
        newbuf = sock.recv(count)
        if not newbuf: return None
        buf += newbuf
        count -= len(newbuf)
    return buf

UDP_IP = "140.116.164.8"
UDP_PORT = 5005

sock = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)
sock.bind ((UDP_IP, UDP_PORT))

s=""

while True:

    length = recvall(conn,16)
	stringData = recvall(conn, int(length))
	data = numpy.fromstring(stringData, dtype='uint8')

	decimg=cv2.imdecode(data,1)
	cv2.imshow('SERVER', decimg)
	cv2.waitKey(1)

    if cv2.waitKey(1) & 0xFF == ord ('q'):
        break