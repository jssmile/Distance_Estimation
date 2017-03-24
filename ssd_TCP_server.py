'''
 * Copyright 2017 Distance Measurement, EE, NCKU. All rights reserved. 
 * File : ssd_TCP_server.py 
 * User : Syuan Jhao 
 * Date : 2017/3/15 
 * Version : 2.0
 * OS : Ubuntu Mate 16.04 LTS
 * Tools : Python 2.7 + Caffe + CUDA 8.0 + cuDNN v5.1 + Opencv 3.2.0
'''

# Import the necessary library 
from functions import *

# flag for terminating
EXIT = False
pickle.dump(EXIT, open("exit_server.txt", "w"))

# Continuous showing the frame from main loop
def show_loop(the_q):

	global EXIT
	lst = (cv2.__version__).split('.')
	major_name = int(lst[0])
	if major_name > 2:
		#define the codec
		fourcc = cv2.VideoWriter_fourcc(*'XVID')
		# define the frame size to full screen
		cv2.namedWindow('image_display', cv2.WND_PROP_FULLSCREEN)
		cv2.setWindowProperty('image_display', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
	else:
		#define the codec
		fourcc = cv2.cv.CV_FOURCC(*'XVID')
		# define the frame size to full screen
		cv2.namedWindow('image_display', cv2.WND_PROP_FULLSCREEN)
		cv2.setWindowProperty('image_display', cv2.WND_PROP_FULLSCREEN, cv2.cv.CV_WINDOW_FULLSCREEN)

	out = cv2.VideoWriter('test.avi', fourcc, 15, (640, 480))
	cv2.createTrackbar('Quality', 'image_display', 50, 100, nothing)

	# frames counter
	fps = 0
	cnt = 0

	while (True):
		image = the_q.get()
		if image is None:	
			print("Client disconnect")
			EXIT = True
			pickle.dump(EXIT, open("exit_server.txt", "w"))
			os.exit()
		cnt += 1

		if cnt == 1:
			start = datetime.datetime.now()
		# count 10 frames and calculated the frames per seconds(fps) 
		if cnt == 10:
			end = datetime.datetime.now()
			fps = 10 / ((end-start).total_seconds())
			cnt = 0
		cv2.putText(image,
                "fps : %f"%fps,
                (0, 30),
                font,
                1,
                (150,0,255),
                2)
		quality = cv2.getTrackbarPos('Quality', 'image_display')
		encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality] # quality from 0 - 100, higher means more clear
		_, imgencode = cv2.imencode('.jpg', image, encode_param)
		data_send = np.array(imgencode)
		stringData_send = data_send.tostring()

		conn.send(str(len(stringData_send)).ljust(16))
		conn.send(stringData_send)
		
		# Save the images as a video file
		out.write(np.uint8(image))
		cv2.imshow('image_display', image)

		# if press 'q' then exit the program
		if cv2.waitKey(1) & 0xFF == ord('q'):
			print("Exit the program")
			EXIT = True
			pickle.dump(EXIT, open("exit_server.txt", "w"))
			os._exit()

		cv2.waitKey(1)

# Detect the object and computing the distance
def main():

	# define the multiprocess
	the_q = multiprocessing.Queue()
	show_process = multiprocessing.Process(target=show_loop,args=(the_q, ))
	show_process.start()

	while(True):
		# Receive data from client
		length = recvall(conn, 16)
		if length is not None:
			# Decode the received data
			stringData_recv = recvall(conn, int(length))
			data_recv = np.fromstring(stringData_recv, dtype = 'uint8')
			frame = cv2.imdecode(data_recv, 1)

			# Normalize the frame data
			image = frame.astype(np.float32)/255
			image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
			image = image[...,::-1]
			transformed_image = transformer.preprocess('data', image)
			net.blobs['data'].data[...] = transformed_image

			# Forward pass
			detections = net.forward()['detection_out']

			# Parse the output
			det_label = detections[0,0,:,1]
			det_conf  = detections[0,0,:,2]
			det_xmin  = detections[0,0,:,3]
			det_ymin  = detections[0,0,:,4]
			det_xmax  = detections[0,0,:,5]
			det_ymax  = detections[0,0,:,6]

			# Get detections with confidence higher than 0.6.
			top_indices = [i for i, conf in enumerate(det_conf) if conf >= 0.6]
			top_conf = det_conf[top_indices]
			top_label_indices = det_label[top_indices].tolist()
			top_labels = get_labelname(labelmap, top_label_indices)
			top_xmin = det_xmin[top_indices]
			top_ymin = det_ymin[top_indices]
			top_xmax = det_xmax[top_indices]
			top_ymax = det_ymax[top_indices]
		
			for i in xrange(top_conf.shape[0]):
				# Get the coordinate of the detected object
				xmin = int(round(top_xmin[i] * image.shape[1]))
				ymin = int(round(top_ymin[i] * image.shape[0]))
				xmax = int(round(top_xmax[i] * image.shape[1]))
				ymax = int(round(top_ymax[i] * image.shape[0]))
				score = top_conf[i]
				label = int(top_label_indices[i])
				label_name = top_labels[i]

				if len(sys.argv) < 2:
					mode = 'w'
				else:
					mode = sys.argv[1]
				if mode == 'w':
					# Classify the object and compute the distance
					if label_name == 'car':
						show_object(frame, label_name, car_width, xmax, xmin, ymax, ymin)
			
					if label_name == 'person':
						show_object(frame, label_name, person_width, xmax, xmin, ymax, ymin)

					if label_name == 'motorbike':
						show_object(frame, label_name, motorbike_width, xmax, xmin, ymax, ymin)			

					if label_name == 'bus':
						show_object(frame, label_name, bus_width, xmax, xmin, ymax, ymin)

					else :
						cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (150,0,255), 2)
						cv2.putText(frame,
            		        label_name,
                		    (xmin, ymin),
                		    font,
                	    	1,
                    		(150,0,255),
                    		2)
				elif mode =='p':
					
					C = 94
					y = 480-ymax
					theta_i = math.degrees(math.atan((ymax-240)/focal_length))
					D = C * math.tan(math.radians(90-theta_i))
					cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (150,0,255), 2)
					cv2.putText(frame,
            		    "%.2fcm" % D,
                		(xmax, ymax),
                		font,
                	    1,
                    	(150,0,255),
                    	2)
					cv2.putText(frame,
            		    label_name,
                		(xmin, ymin),
                		font,
                	    1,
                    	(150,0,255),
                    	2)


			# Send the processed frame to the child-process
			the_q.put(frame)

			# Check if we exit the program
			Leaving = pickle.load(open("exit_server.txt", "r"))
			if Leaving:
				sys.exit(0)
		else:
			print("Client disconnect!")
			the_q.put(None)
			Leaving = pickle.load(open("exit_server.txt", "r"))
			if Leaving:
				sys.exit(0)

if __name__ == '__main__':
	conn, addr = connection(TCP_IP, TCP_PORT)
	net, transformer, labelmap = load_model()
	main()

	# Release the buffer
	the_q.put(None)
	show_process.join()
	server.close()
	out.release()
	cv2.DestroyAllWindows()