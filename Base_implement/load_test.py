import numpy as np
import cv2

data = np.load('parameter_output/calibrate_out.npz')
print(data['mtx'])
