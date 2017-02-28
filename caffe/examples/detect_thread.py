import numpy as np
import matplotlib.pyplot as plt
import cv2
import datetime
import threading
import time
import Queue

frames = Queue.Queue(10)

class ImageGrabber(threading.Thread):
    def __init__(self, ID):
        threading.Thread.__init__(self)
        self.ID=ID
        self.cap=cv2.VideoCapture(ID)

    def Run(self):
        global frames
        while True:
            _,frame=self.cap.read()
            frames.put(frame)
            time.sleep(0.1)


class Main(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)

    def run(self):
        global frames
        while True:
            if(not frames.empty()):
                self.Currframe=frames.get()
            ##------------------------##
            ## your opencv logic here ##
            ## -----------------------##
           	cv2.imshow('frame', frames)


grabber = ImageGrabber(0)
main = Main()

grabber.start()
main.start()
main.join()
grabber.join()