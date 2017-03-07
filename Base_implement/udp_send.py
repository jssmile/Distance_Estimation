import numpy as np
import cv2
import socket

def recvall(sock, count):
    buf = b''
    while count:
        newbuf = sock.recv(count)
        if not newbuf: return None
        buf += newbuf
        count -= len(newbuf)
    return buf

UDP_IP = "127.0.0.1"
UDP_PORT = 5005

cap = cv2.VideoCapture(0)

while(True):
    ret, frame = cap.read()

    encode_param=[int(cv2.IMWRITE_JPEG_QUALITY),30]
	_, imgencode = cv2.imencode('.jpg', frame, encode_param)
	data_send = numpy.array(imgencode)
	stringData_send = data_send.tostring()

	sock.send(str(len(stringData_send)).ljust(16));
	sock.send(stringData_send);
	
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
