import tkinter as tk
import cv2
from PIL import Image, ImageTk
# from functools import partial

CAMERA_URI = 1
RECORDING = False

width, height = 300, 300
cap = cv2.VideoCapture(CAMERA_URI)

#
# Set camera read property
#
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

def onRecordClick():
    global RECORDING
    print('Hello world')
    if RECORDING:
        recordButtonText.set('Start record')
        RECORDING = False
    else:
        RECORDING = True 
        print('Turn on recording', RECORDING)
        recordButtonText.set('Recording ... (Press to stop)')

root = tk.Tk()
root.title('SmartLock')
#
# Press escape
# 
root.bind('q', lambda e: root.quit())

recordButtonText = tk.StringVar()
recordButtonText.set('Start record')
recordButton = tk.Button(root, textvariable=recordButtonText, command=onRecordClick, padx = 10, pady = 10, width=50)
recordButton.grid(column=1, row=0)

lmain = tk.Label(root)
lmain.grid(column=0, row=0)

def show_frame():
    global RECORDING
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)
    if RECORDING:
        frame_copy = frame.copy()
        frame = cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 255, 0), 5)
    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    img = Image.fromarray(cv2image)
    imgtk = ImageTk.PhotoImage(image=img)
    lmain.imgtk = imgtk
    lmain.configure(image=imgtk)
    lmain.after(33, show_frame)

show_frame()
root.mainloop()