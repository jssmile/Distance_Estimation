'''
 * Copyright 2017 Distance Measurement, EE, NCKU. All rights reserved. 
 * File : edge.py 
 * User : Syuan Jhao 
 * Date : 2017/3/16 
 * Version : 1.0
 * OS : Ubuntu Mate 16.04 LTS
 * Tools : Python 2.7 + Opencv 3.2.0
 * Introduction : detect the edge of the image
 * Guide : $ python edge.py /path/to/image
'''

import numpy as np
import cv2

# convert the image to grayscale, blur it, and detect edges    
frame = cv2.imread(sys.argv[1])
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(gray, 35, 125)
cv2.imshow("edge", edged)
cv2.waitKey(0)
