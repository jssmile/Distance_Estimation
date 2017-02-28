
import multiprocessing
import cv2
import sys
 
def cam_loop(the_q):
	cap = cv2.VideoCapture(0)
 
	while True:
		_ , img = cap.read()
		if img is not None:
			the_q.put(img)
 
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