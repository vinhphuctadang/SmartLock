import tkinter as tk
import cv2
import time
from PIL import Image, ImageTk
# from functools import partial

CAMERA_URI = 1
RECORDING = False
FRAME_COUNT = 0
SAVE_INTERVAL = 5

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
        FRAME_COUNT = 0
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
    global RECORDING, FRAME_COUNT
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)
    if RECORDING:
        FRAME_COUNT += 1
        if FRAME_COUNT % SAVE_INTERVAL == 0:
            # frame_copy = frame.copy()
            filename = f'{int(time.time()*1000)}.jpg'
            print(f'Going to save image as {filename}')
            cv2.imwrite(filename, frame)
            FRAME_COUNT = 0
            frame = cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 255, 0), 5)
    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    img = Image.fromarray(cv2image)
    imgtk = ImageTk.PhotoImage(image=img)
    lmain.imgtk = imgtk
    lmain.configure(image=imgtk)
    lmain.after(33, show_frame)

show_frame()
root.mainloop()