import numpy as np
import cv2
import socket

UDP_IP = "140.116.164.8"
UDP_PORT = 5005

cap = cv2.VideoCapture(0)

while(True):
    ret, frame = cap.read()

    cv2.imshow('frame',frame)


    sock = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)

    d = frame.flatten ()
    s = d.tostring ()



    for i in xrange(20):

        sock.sendto (s[i*46080:(i+1)*46080],(UDP_IP, UDP_PORT))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()