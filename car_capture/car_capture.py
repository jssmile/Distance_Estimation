import cv2
import numpy as np

cap = cv2.VideoCapture(0)
count = 0
pos = 'car_capture/'
pic_format = '.jpg'
#define the codec
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('test.avi', fourcc, 30.0, (640, 480))
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
