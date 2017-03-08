from Tkinter import *
import socket
import cv2
import numpy

def recvall(sock, count):
    buf = b''
    while count:
        newbuf = sock.recv(count)
        if not newbuf: return None
        buf += newbuf
        count -= len(newbuf)
    return buf

class App:
  def __init__(self, master):
    frame = Frame(master)
    frame.pack()

    self.label_1 = Label(frame, text="IP_address")
    self.label_2 = Label(frame, text="Port")
    self.label_1.grid(row=0, sticky=E)
    self.label_2.grid(row=1, sticky=E)

    self.entry_1 = Entry(frame)
    self.entry_2 = Entry(frame)
    self.entry_1.grid(row=0, column=1)
    self.entry_2.grid(row=1, column=1)

    self.Connect_btn = Button(frame, text ="Connect", command = self.write_slogan)
    self.Connect_btn.grid(row=3, column=0)
    self.Cancel_btn = Button(frame, 
                         text="Exit", fg="red",
                         command=quit)
    self.Cancel_btn.grid(row=3, column = 1)
  def write_slogan(self):
    ip_add = self.entry_1.get()
    port = self.entry_2.get()
    print(ip_add, port)
    root.quit()

root = Tk()
app = App(root)
root.mainloop()

TCP_IP = app.entry_1.get()
TCP_PORT = int(app.entry_2.get())

sock = socket.socket()
sock.connect((TCP_IP, TCP_PORT))

capture = cv2.VideoCapture(0)

cv2.namedWindow("ssd", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("ssd", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while (True):
	_, frame = capture.read()

	encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
	_, imgencode = cv2.imencode('.jpg', frame, encode_param)
	data_send = numpy.array(imgencode)
	stringData_send = data_send.tostring()

	sock.send(str(len(stringData_send)).ljust(16));
	sock.send(stringData_send);

	length = recvall(sock,16)
	stringData_recv = recvall(sock, int(length))
	data_recv = numpy.fromstring(stringData_recv, dtype='uint8')

	frame_recv = cv2.imdecode(data_recv, 1)
	
	cv2.imshow("ssd", frame_recv)
	cv2.waitKey(1)

sock.close()
cv2.destroyAllWindows()
