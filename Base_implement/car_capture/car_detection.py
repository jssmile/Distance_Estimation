import cv2
import numpy as np
import sys
from scipy.spatial import distance as dist


#focal length
focal_length = 818.0

#car's real width(cm)
car_width = 170.0

#car pic detection
#   Get User supplied values
imgPath = sys.argv[2]
cascPath  = sys.argv[1]

img = cv2.imread(imgPath)
car_cascade = cv2.CascadeClassifier(cascPath)
height, width = img.shape[:2]
print(height)
#faceCascade = cv2.CascadeClassifier(cascPath)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cars = car_cascade.detectMultiScale(gray, 1.3, 1)
print("Founded {0} cars!".format(len(cars)))
for(x, y, w, h) in cars:
	cv2.rectangle(img, (x,y), (x+w,y+h), (0, 0, 255), 2)
	distance = (focal_length * car_width)/w
	print(distance, "cm")
	
cv2.imshow('car detection', img)
cv2.imwrite("cars_detection.jpg", img)
cv2.waitKey(0)
cv2.destroyAllWindows()


"""
#Video capture

cascPath  = sys.argv[1]
vidPath = sys.argv[2]
#	Capture video from camera
cap = cv2.VideoCapture(vidPath)
car_Cascade = cv2.CascadeClassifier(cascPath)
while(True):
	ret, img = cap.read()
	if(type(img) == type(None)):
		break

	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	cars = car_Cascade.detectMultiScale(gray, 1.2, 1)
	print("Found {0} cars!".format(len(cars)))
	
	for (x, y, w, h) in cars:
		cv2.rectangle(img, (x,y), (x+w, y+h), (0, 0, 255), 2)
		distance = (focal_length * car_width)/w
		print(distance, "cm")
		cv2.putText(img, "%.2fcm" % distance, (x+w, y+h), cv2.FONT_HERSHEY_SIMPLEX,2.0, (0, 255, 0), 2)
	cv2.imshow('car_detection', img)

	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
cv2.waitKey(0)
cv2.destroyAllWindows()
"""