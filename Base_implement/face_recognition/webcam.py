'''
 * Copyright 2017 Distance Measurement, EE, NCKU. All rights reserved. 
 * File : Webcam.py 
 * User : Syuan Jhao 
 * Date : 2017/3/16 
 * Version : 1.0
 * OS : Ubuntu Mate 16.04 LTS
 * Tools : Python 2.7 + Opencv 3.2.0
 * Introduction : Display the webcam frame and compute the fps
'''

import cv2
import datetime

cap = cv2.VideoCapture(0)
cnt = 0
fps = 0

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    cnt = cnt + 1
    if cnt == 1:
    	start = datetime.datetime.now()
    cv2.putText(frame,
                    "fps" + str(fps),
                    (0, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (150,0,255),
                    2)
    cv2.putText(frame,
                "x",
                (320, 240),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (150,0,255),
                2)
    # Display the resulting frame
    cv2.imshow('frame',frame)

    #cv2.imwrite("capture.png", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if cnt == 10:
    	end = datetime.datetime.now()
    	period = end - start
    	period = period.total_seconds()
    	fps = 10 / period
    	print(str(fps))
    	cnt = 0
    	cv2.putText(frame,
                    "fps" + str(fps),
                    (0, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (150,0,255),
                    2)
    print str(fps)
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
