'''
 * Copyright 2017 Distance Measurement, EE, NCKU. All rights reserved. 
 * File : car_capture.py 
 * User : Syuan Jhao 
 * Date : 2017/3/16 
 * Version : 1.0
 * OS : Ubuntu Mate 16.04 LTS
 * Tools : Python 2.7 + Opencv 3.2.0
 * Introduction : Capture the video and press 's' to save the video frame.
'''

import cv2
import numpy as np

cap = cv2.VideoCapture(0)
count = 0
pos = 'car_capture/'
pic_format = '.jpg'

#define the video codec
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('test.avi', fourcc, 30.0, (640, 480))

# Take the snap shot
def Snap(img):
    global count
    NAME = str(count) + pic_format
    cv2.imwrite(NAME, img)
    print("Saved ", NAME )
    count += 1


while (True):
    _, frame = cap.read()
    cv2.imshow("frame", frame)
    out.write(frame)
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        print("quit")
        break
    elif key & 0xFF == ord('s'):
        Snap(frame)
cap.release()
out.release()
cv2.destroyAllWindows() 
