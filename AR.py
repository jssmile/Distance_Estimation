import cv2
import numpy as np
import glob

# Load previously saved data
with np.load('pose/out.npz') as X:
    dist, mtx, _, rvecs , tvecs= [X[i] for i in ('dist', 'mtx', 'ret', 'rvecs','tvecs')]

def draw(img, corners, imgpts):
    corner = tuple(corners[0].ravel())
    img = cv2.line(img, corner, tuple(imgpts[0].ravel()), (255, 0,0), 5)
    img = cv2.line(img, corner, tuple(imgpts[1].ravel()), (255, 0,0), 5)
    img = cv2.line(img, tuple(imgpts[1].ravel()), tuple(imgpts[2].ravel()), (255, 0,0), 5)
    img = cv2.line(img, tuple(imgpts[0].ravel()), tuple(imgpts[2].ravel()), (255, 0,0), 5)

    img = cv2.line(img, tuple(imgpts[0].ravel()), tuple(imgpts[3].ravel()), (255, 0,0), 5)
    img = cv2.line(img, tuple(imgpts[1].ravel()), tuple(imgpts[3].ravel()), (255, 0,0), 5)
    img = cv2.line(img, tuple(imgpts[2].ravel()), tuple(imgpts[3].ravel()), (255, 0,0), 5)
    img = cv2.line(img, corner, tuple(imgpts[3].ravel()), (255, 0,0), 5)
    return img
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objp = np.zeros((6*8,3), np.float32)
objp[:,:2] = np.mgrid[0:8,0:6].T.reshape(-1,2)

axis = np.float32([[2,0,0], [0,2,0], [2,2,0], [0,0,2]]).reshape(-1,3)

for fname in glob.glob('calibrate/*.jpg'):
    img = cv2.imread(fname)

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (8,6), None)
    if ret == True:
        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)

        # Find the rotation and translation vectors.
        ret, rvecs, tvecs= cv2.solvePnP(objp, corners, mtx, dist)

        # project 3D points to image plane
        imgpts, jac = cv2.projectPoints(axis, rvecs, tvecs, mtx, dist)

        img = draw(img,corners,imgpts)
        cv2.imshow('img',img)
        cv2.imwrite('AR.jpg', img)
        cv2.waitKey(500)

cv2.destroyAllWindows()
