import numpy as np
import cv2
     # convert the image to grayscale, blur it, and detect edges
    
frame = cv2.imread("images/90.png")
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (5, 5), 0)
edged = cv2.Canny(gray, 35, 125)
cv2.imshow("edge", edged)
cv2.waitKey(0)
