'''
 * Copyright 2017 Distance Measurement, EE, NCKU. All rights reserved. 
 * File : login.py 
 * User : Syuan Jhao 
 * Date : 2017/3/16 
 * Version : 1.0
 * OS : Ubuntu Mate 16.04 LTS
 * Tools : Python 2.7 + Opencv 3.2.0
 * Introduction : Create the login graphic interface
'''

from Tkinter import *
class App:
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
    ip_add = self.entry_1.get()
    port = self.entry_2.get()
    print(ip_add, port)
    root.quit()

root = Tk()
app = App(root)
root.mainloop()

ip_add = app.entry_1.get()
port = app.entry_2.get()
print(ip_add, port)