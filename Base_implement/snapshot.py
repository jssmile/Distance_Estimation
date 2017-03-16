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
import numpy as np

cap = cv2.VideoCapture(0)
count = 0
pos = ''
pic_format = '.jpg'

# Take a snapshot
def Snap(img):
    global count
    NAME = pos + str(count) + pic_format
    cv2.imwrite(NAME, img)
    print("Saved ", NAME )
    count += 1


while (True):
    _, frame = cap.read()
    cv2.imshow("frame", frame)
    
    # Ket event
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        print("quit")
        break
    elif key & 0xFF == ord('s'):
        Snap(frame)
cap.release()
cv2.destroyAllWindows() 
