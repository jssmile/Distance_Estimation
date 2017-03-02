import caffe
import cv2
import datetime
import time
import multiprocessing
import numpy as np
import os
import socket
import sys
import Queue
from google.protobuf import text_format
from caffe.proto import caffe_pb2


# Make sure that caffe is on the python path:
caffe_root = '/home/jssmile/Distance_Estimation/caffe'  # this file is expected to be in {caffe_root}/examples
os.chdir(caffe_root)
sys.path.insert(0, 'python')
caffe.set_device(0)
caffe.set_mode_gpu()

# focal length
focal_length = 816

# car's real width(cm)
car_width = 180

# image size
width = 640
height = 480
scale = 2

# frames counter
cnt = 0
fps = 0

# Socket setting
connect = None
TCP_IP = '140.116.164.7'
TCP_PORT = 5001
client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.settimeout(3.0)

try:
	client.connect((TCP_IP, TCP_PORT))
except socket.error:
	print("Connect fail!")
	connect = False
else:
	print("Connect Success!")
	connect = True
finally:
	pass

# load PASCAL VOC labels
labelmap_file = 'data/VOC0712/labelmap_voc.prototxt'
file = open(labelmap_file, 'r')
labelmap = caffe_pb2.LabelMap()
text_format.Merge(str(file.read()), labelmap)

model_def = 'models/VGGNet/VOC0712/SSD_300x300/deploy.prototxt'
model_weights = 'models/VGGNet/VOC0712/SSD_300x300/VGG_VOC0712_SSD_300x300_iter_120000.caffemodel'

net = caffe.Net(model_def,      # defines the structure of the model
                model_weights,  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't perform dropout)

# input preprocessing: 'data' is the name of the input blob == net.inputs[0]
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2, 0, 1))
transformer.set_mean('data', np.array([104,117,123]))   # mean pixel
transformer.set_raw_scale('data', 255)                  # the reference model operates on images in [0,255] range instead of [0,1]
transformer.set_channel_swap('data', (2,1,0))           # the reference model has channels in BGR order instead of RGB

# set net to batch size of 1
image_resize = 300
net.blobs['data'].reshape(1, 3, image_resize, image_resize)	

def get_labelname(labelmap, labels):
    num_labels = len(labelmap.item)
    labelnames = []
    if type(labels) is not list:
        labels = [labels]
    for label in labels:
        found = False
        for i in xrange(0, num_labels):
            if label == labelmap.item[i].label:
                found = True
                labelnames.append(labelmap.item[i].display_name)
                break
        assert found == True
    return labelnames

def show_loop(the_q):

	global cnt, fps, connect

	while (True):
		image = the_q.get()
		if image is None:	break
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
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (150,0,255),
                2)
		if connect == True:
			encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 20] # quality from 0 - 100, higher means bigger size
			_, imgencode = cv2.imencode('.jpg', image, encode_param)
			data = np.array(imgencode)
			stringData = data.tostring()

			client.send(str(len(stringData)).ljust(16))
			client.send(stringData)
		cv2.imshow('image_display', image)
		if cv2.waitKey(1) & 0xFF == ord('q'):
			print("fuck")

		cv2.waitKey(1)
		continue

def main():

	# Open the webcam
	cap = cv2.VideoCapture(0)

	#define the codec
	fourcc = cv2.VideoWriter_fourcc(*'XVID')
	out = cv2.VideoWriter('test.avi', fourcc, 15, (640, 480))
	
	the_q = multiprocessing.Queue()
	show_process = multiprocessing.Process(target=show_loop,args=(the_q, ))
	show_process.start()

	while(True):
		_, frame = cap.read()

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
			xmin = int(round(top_xmin[i] * image.shape[1]))
			ymin = int(round(top_ymin[i] * image.shape[0]))
			xmax = int(round(top_xmax[i] * image.shape[1]))
			ymax = int(round(top_ymax[i] * image.shape[0]))
			score = top_conf[i]
			label = int(top_label_indices[i])
			label_name = top_labels[i]

			if label_name == 'car':
				img_car_width = xmax-xmin
				distance = (focal_length * car_width)/img_car_width
				#color = colors[label].astype(np.float8)*255
				cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (150,0,255), 2)
				cv2.putText(frame,
                        label_name,
                        (xmin, ymin),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (150,0,255),
                        2)
				cv2.putText(frame,
                        "%.2fcm" % distance,
                        (xmax, ymax),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2)
			else :
				cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (150,0,255), 2)
				cv2.putText(frame,
                    label_name,
                    (xmin, ymin),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (150,0,255),
                    2)
		# Save the images as a video file
		out.write(np.uint8(frame))
		the_q.put(frame)
if __name__ == '__main__':
	main()

# Release the buffer
the_q.put(None)
show_process.join()
cap.release()
out.release()
cv2.DestroyAllWindows()