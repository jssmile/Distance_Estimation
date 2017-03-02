import socket
import cv2
import numpy

def recvall(sock, count):
    buf = b''
    while count:
        newbuf = sock.recv(count)
        if not newbuf: return None
        buf += newbuf
        count -= len(newbuf)
    return buf

TCP_IP = '140.116.164.7'
TCP_PORT = 5001

# AF_INET -> IPv4, SOCK_STREAM -> TCP
server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind((TCP_IP, TCP_PORT))
server.listen(True)
conn, addr = server.accept()

while (True):
	length = recvall(conn,16)
	stringData = recvall(conn, int(length))
	data = numpy.fromstring(stringData, dtype='uint8')

	decimg=cv2.imdecode(data,1)
	cv2.imshow('SERVER', decimg)
	cv2.waitKey(1)

server.close()
cv2.destroyAllWindows() 