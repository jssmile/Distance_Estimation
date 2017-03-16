'''
 * Copyright 2017 Distance Measurement, EE, NCKU. All rights reserved. 
 * File : thread.py 
 * User : Syuan Jhao 
 * Date : 2017/3/16 
 * Version : 1.0
 * OS : Ubuntu Mate 16.04 LTS
 * Tools : Python 2.7 + Opencv 3.2.0
 * Introduction : Create two threads for capturing frames and displaying.
'''
import multiprocessing
import cv2
import sys
 
# Capture the camera
def cam_loop(the_q):
	cap = cv2.VideoCapture(0)
 
	while True:
		_ , img = cap.read()
		if img is not None:
			the_q.put(img)

# Diaplay the frame
def show_loop(the_q):
	cv2.namedWindow('pepe')
 
 	while True:
		from_queue = the_q.get()
		cv2.imshow('pepe', from_queue)
		cv2.waitKey(1)
 
if __name__ == '__main__':
	logger = multiprocessing.log_to_stderr()
	logger.setLevel(multiprocessing.SUBDEBUG)
 
	the_q = multiprocessing.Queue()
 
	cam_process = multiprocessing.Process(target=cam_loop,args=(the_q, ))
	cam_process.start()
 
	show_process = multiprocessing.Process(target=show_loop,args=(the_q, ))
	show_process.start()
 
	cam_process.join()
	show_loop.join()