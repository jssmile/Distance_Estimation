# USAGE
# python distance_to_camera.py

# import the necessary packages
import numpy as np
import cv2
import sys

def find_marker(frame):
	# convert the image to grayscale, blur it, and detect edges
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(gray, 35, 125)
    cv2.imshow("edge", edged)
	# find the contours in the edged image and keep the largest one;
	# we'll assume that this is our piece of paper in the image
    (_, cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    c = max(cnts, key = cv2.contourArea)

	# compute the bounding box of the of the paper region and return it
    return cv2.minAreaRect(c)

def distance_to_camera(knownWidth, focalLength, perWidth):
	# compute and return the distance from the maker to the camera
	return (knownWidth * focalLength) / perWidth

# initialize the known distance from the camera to the object, which
# in this case is 24 inches
KNOWN_DISTANCE = 90.0

# initialize the known object width, which in this case, the piece of
# paper is 12 inches wide
KNOWN_WIDTH = 29.7

#Focal length of C310
focalLength = 806.0

if sys.argv[1] == "image":
    # initialize the list of images that we'll be using
    #IMAGE_PATHS = ["images/2ft.png", "images/3ft.png", "images/4ft.png"]
    IMAGE_PATHS = sys.argv[2]
    # load the furst image that contains an object that is KNOWN TO BE 2 feet
    # from our camera, then find the paper marker in the image, and initialize
    # the focal length
    image = cv2.imread(IMAGE_PATHS)
    marker = find_marker(image)
    print(focalLength)
    #focal length of C310
    #print(marker)
    # loop over the images

if sys.argv[1] == "video":
    cap = cv2.VideoCapture(0)
    #for imagePath in IMAGE_PATHS:
    while(True):
	   # load the image, find the marker in the image, then compute the
	   # distance to the marker from the camera
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        edged = cv2.Canny(gray, 35, 125)

        (_, cnts, _) = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            c = max(cnts, key = cv2.contourArea)
            # compute the bounding box of the of the paper region and return it
            # draw a bounding box around the image and display it
            marker = cv2.minAreaRect(c)
            box = cv2.boxPoints(marker)
            cv2.drawContours(frame, [box.astype(np.int)], -1, (0, 255, 0), 2)
            if marker[1][0] > marker[1][1]:
                distance = distance_to_camera(KNOWN_WIDTH, focalLength, marker[1][0])
            else:
                distance = distance_to_camera(KNOWN_WIDTH, focalLength, marker[1][1])
            #print(distance," cm")
            print(marker[1][0],"\n\n", marker[1][1])
            cv2.putText(frame, "%.2fcm" % distance, (50, 50), cv2.FONT_HERSHEY_SIMPLEX,2.0, (0, 255, 0), 3)
        cv2.imshow("distance", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
