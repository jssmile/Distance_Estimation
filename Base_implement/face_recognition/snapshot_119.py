'''
 * Copyright 2017 Distance Measurement, EE, NCKU. All rights reserved. 
 * File : snapshot.py 
 * User : Syuan Jhao 
 * Date : 2017/3/16 
 * Version : 1.0
 * OS : Ubuntu Mate 16.04 LTS
 * Tools : Python 2.7 + Opencv 3.2.0
 * Introduction : Take a snapshot
 * Guide : while running the program, press 's' to save the snapshot.
'''

import cv2
import datetime
import numpy as np

cap = cv2.VideoCapture(1)
count = 0
pos = ''
pic_format = '.jpg'

faceCascade = cv2.CascadeClassifier('haar_files/haarcascade_frontalface_default.xml')

cnt = 0
fps = 0

# Take a snapshot
def Snap(img):
    global count
    NAME = pos + str(count) + pic_format
    cv2.imwrite(NAME, img)
    print("Saved ", NAME )
    count += 1


while (True):
    _, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Detect faces in the image
    faces = faceCascade.detectMultiScale(
                                        gray,
                                        scaleFactor=1.2,
                                        minNeighbors=5,
                                        minSize=(30, 30),
                                        flags = cv2.cv.CV_HAAR_SCALE_IMAGE
)
    for(x,y,w,h) in faces:
        if len(faces) == 1:
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,255), 2)
            gray = gray[y:y+h, x:x+w]
            s = cv2.resize(gray, (92,112))
        elif len(faces) > 1 : 
            print("Damn")
            cv2.rectangle(frame, (x,y), (x+w, y+h), (0,0,255), 2)
    cv2.imshow("frame", frame)

    
    # Ket event
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        print("quit")
        break
    elif key & 0xFF == ord('s'):
        Snap(s)
cap.release()
cv2.destroyAllWindows() 
