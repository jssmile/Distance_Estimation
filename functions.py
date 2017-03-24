'''
 * Copyright 2017 Distance Measurement, EE, NCKU. All rights reserved. 
 * File : functions.py 
 * User : Syuan Jhao 
 * Date : 2017/3/23 
 * Version : 1.0
 * Usage : Store the finctions used in ssd_TCP_server.py
 * OS : Ubuntu Mate 16.04 LTS
 * Tools : Python 2.7 + Caffe + CUDA 8.0 + cuDNN v5.1 + Opencv 3.2.0
'''

# Import the necessary library 
import caffe
import cv2
import datetime
import math
import multiprocessing
import numpy as np
import os
import pickle
import Queue
import socket
import sys
import time

from caffe.proto import caffe_pb2
from google.protobuf import text_format
from Tkinter import *

# Make sure that caffe is on the python path:
caffe_root = '/home/jssmile/Distance_Estimation/caffe'  # this file is expected to be in {caffe_root}/examples
os.chdir(caffe_root)
sys.path.insert(0, 'python')
caffe.set_device(0)
caffe.set_mode_gpu()

# focal length of Logitech C310
focal_length = 816.0

# real width of obstacle(cm)
car_width = 180
bus_width = 230
motorbike_width = 70
person_width = 53

# TCP ip and port
TCP_IP = None 
TCP_PORT = None

# Text Type
font = cv2.FONT_HERSHEY_SIMPLEX

# pass function
def nothing():
	pass

# Login GUI
class Login:
  def __init__(self, master):
    frame = Frame(master)
    frame.pack()

    self.label_1 = Label(frame, text="IP_address")
    self.label_2 = Label(frame, text="Port")
    self.label_1.grid(row=0, sticky=E)
    self.label_2.grid(row=1, sticky=E)

    self.entry_1 = Entry(frame)
    self.entry_1.insert(END, '140.116.164.8')
    self.entry_2 = Entry(frame)
    self.entry_2.insert(END, '5001')
    self.entry_1.grid(row=0, column=1)
    self.entry_2.grid(row=1, column=1)

    self.Connect_btn = Button(frame, text ="Connect", command = self.write_slogan)
    self.Connect_btn.grid(row=3, column=0)
    self.Cancel_btn = Button(frame, text="Exit", fg="red", command=quit)
    self.Cancel_btn.grid(row=3, column = 1)
  
  def write_slogan(self):
    global TCP_IP, TCP_PORT
    TCP_IP = self.entry_1.get()
    TCP_PORT = int(self.entry_2.get())
    print(TCP_IP, TCP_PORT)
    root.destroy()
root = Tk()
root.wm_title("Server")
app = Login(root)
root.mainloop()

# Socket setting
def connection(ip, Port):
	server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
	server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
	server.bind((ip, Port))
	server.listen(True)
	print("Listening")
	conn, addr = server.accept()
	print("Connected!!!")
	return conn, addr

# Receive the socket data
def recvall(sock, count):
	buf = b''
	while count:
		newbuf = sock.recv(count)
		if not newbuf: return None
		buf += newbuf
		count -= len(newbuf)
	return buf

# get the name of the detectd object
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

# Boxing the detected object
def show_object(frame, label_name, real_width, x_max, x_min, y_max, y_min):
	img_width = x_max-x_min
	distance = (focal_length * real_width)/img_width
	cv2.rectangle(frame, (x_min,y_min), (x_max,y_max), (150,0,255), 2)
	cv2.putText(frame, label_name, (x_min, y_min), font, 1, (150,0,255), 2)
	cv2.putText(frame, "%.2fcm" % distance, (x_max, y_max), font, 1, (0, 255, 0), 2)
	return np.uint8(frame)

# Load pre-trained SSD model
def load_model():
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
	return net, transformer, labelmap