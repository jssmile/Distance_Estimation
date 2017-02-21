import numpy as np
import cv2

data = np.load('pose/calibrate_out.npz')
print(data['mtx'])
