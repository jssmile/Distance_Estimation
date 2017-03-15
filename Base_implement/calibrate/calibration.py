import numpy as np
import cv2
import glob

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points
objp = np.zeros((6*8, 3), np.float32)
objp[:,:2] = np.mgrid[0:8,0:6].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

images = glob.glob('*.jpg')

for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (8,6),None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)

        cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        imgpoints.append(corners)
        # Draw and display the corners
        cv2.drawChessboardCorners(img, (8,6), corners, ret)
        ret, mtx, dist, rvecs,tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
        print(mtx, "\n")
        print(dist, "\n")
        np.savez("pose/calibrate_out", ret=ret, mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)
        # Draw and display the corners
        cv2.imshow('img', img)
        cv2.waitKey(500)
np.savez("pose/out", ret=ret, mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)
cv2.destroyAllWindows()
