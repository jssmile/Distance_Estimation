import cv2
import numpy as np

cap = cv2.VideoCapture(0)
count = 0
pos = 'new_calibrate/'
pic_format = '.jpg'

def Snap(img):
    global count
    NAME = pos + str(count) + pic_format
    cv2.imwrite(NAME, img)
    print("Saved ", NAME )
    count += 1


while (True):
    _, frame = cap.read()
    cv2.imshow("frame", frame)
    key = cv2.waitKey(1)
    if key & 0xFF == ord('q'):
        print("quit")
        break
    elif key & 0xFF == ord('s'):
        Snap(frame)
cap.release()
cv2.destroyAllWindows() 
