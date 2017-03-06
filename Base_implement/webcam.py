import cv2
import datetime

cap = cv2.VideoCapture(0)
cnt = 0
fps = 0
cv2.namedWindow("frame", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("frame", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

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
