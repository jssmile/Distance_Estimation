# import the necessary packages
import numpy as np
import cv2

def distance_to_camera(knownWidth, focalLength, perWidth):
	# compute and return the distance from the maker to the camera
	return (knownWidth * focalLength) / perWidth

# initialize the known object width, which in this case, the piece of
# paper is 12 inches wide
KNOWN_WIDTH = 29.7

#focal length of C310
focalLength = 546.3

# loop over the images
cap = cv2.VideoCapture(0)

while(True):
    # load the image, find the marker in the image, then compute the
    # distance to the marker from the camera
    _, frame = cap.read()
    # convert the image to grayscale, blur it, and detect edges
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(gray, 35, 125)

    # find the contours in the edged image and keep the largest one;
    # we'll assume that this is our piece of paper in the image
    (_, cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        c = max(cnts, key = cv2.contourArea)
        # compute the bounding box of the of the paper region and return it
        # draw a bounding box around the image and display it
        marker = cv2.minAreaRect(c)
        box = cv2.boxPoints(marker)
        cv2.drawContours(frame, [box.astype(np.int)], -1, (0, 255, 0), 2)
        distance = distance_to_camera(KNOWN_WIDTH, focalLength, marker[1][0])
        print(distance," cm")
        cv2.putText(frame, "%.2fcm" % distance, (50, 50), cv2.FONT_HERSHEY_SIMPLEX,2.0, (0, 255, 0), 3)
    cv2.imshow("distance", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

