import socket
import cv2
import numpy

TCP_IP = '140.116.164.8'
TCP_PORT = 5001

sock = socket.socket()
sock.connect((TCP_IP, TCP_PORT))

capture = cv2.VideoCapture(0)

def recvall(sock, count):
    buf = b''
    while count:
        newbuf = sock.recv(count)
        if not newbuf: return None
        buf += newbuf
        count -= len(newbuf)
    return buf

while (True):
	_, frame = capture.read()

	encode_param=[int(cv2.IMWRITE_JPEG_QUALITY),30]
	_, imgencode = cv2.imencode('.jpg', frame, encode_param)
	data_send = numpy.array(imgencode)
	stringData_send = data_send.tostring()

	sock.send(str(len(stringData_send)).ljust(16));
	sock.send(stringData_send);

	length = recvall(sock,16)
	stringData_recv = recvall(sock, int(length))
	data_recv = numpy.fromstring(stringData_recv, dtype='uint8')

	frame_recv = cv2.imdecode(data_recv, 1)
	cv2.imshow("ssd", frame_recv)
	cv2.waitKey(1)

sock.close()
cv2.destroyAllWindows()