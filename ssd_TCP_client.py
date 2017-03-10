from Tkinter import *
import socket
import cv2
import numpy

# TCP ip and port
TCP_IP = None
TCP_PORT = None

# The login interface
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
    self.Cancel_btn = Button(frame, 
                         text="Exit", fg="red",
                         command=quit)
    self.Cancel_btn.grid(row=3, column = 1)
  
  def write_slogan(self):
    global TCP_IP, TCP_PORT
    TCP_IP = self.entry_1.get()
    TCP_PORT = int(self.entry_2.get())
    print(TCP_IP, TCP_PORT)
    root.destroy()

root = Tk()
root.wm_title("Client")
app = Login(root)
root.mainloop()

def Nothing():
	pass

# Receive the image package
def recvall(sock, count):
    buf = b''
    while count:
        newbuf = sock.recv(count)
        if not newbuf: return None
        buf += newbuf
        count -= len(newbuf)
    return buf

def connection():
	sock = socket.socket()
	sock.connect((TCP_IP, TCP_PORT))
	return sock

def main():
	while (True):
		_, frame = capture.read()

		quality = cv2.getTrackbarPos('Quality', 'ssd')
		encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
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
		if cv2.waitKey(1) & 0xFF == ord('q'):
			os.exit()


if __name__ == '__main__':
	capture = cv2.VideoCapture(0)

	cv2.namedWindow("ssd", cv2.WND_PROP_FULLSCREEN)
	cv2.setWindowProperty("ssd", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
	cv2.createTrackbar('Quality', 'ssd', 50, 100, Nothing)
	
	sock = connection()
	main()
	sock.close()
	cv2.destroyAllWindows()