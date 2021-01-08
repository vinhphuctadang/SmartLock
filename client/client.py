import tkinter as tk
import cv2
import time
import os
import requests
import base64
from PIL import Image, ImageTk
# from functools import partial
BASE_URL = 'http://localhost:8080/'
TRAIN_PATH = 'train/'
CAMERA_URI = 1
RECORDING = False
FRAME_COUNT = 0
SAVE_INTERVAL = 5
IMAGE_LABEL = 'phuc'
MAX_TIME = 20
startTime = 0

def on_record_click():
    global RECORDING, startTime
    print('Hello world')
    if RECORDING:
        recordButtonText.set('Start record')
        RECORDING = False
    else:
        startTime = time.time()
        RECORDING = True 
        FRAME_COUNT = 0
        recordButtonText.set('Recording ... (Press to stop)')

def show_frame():
    global RECORDING, FRAME_COUNT
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)
    if RECORDING:
        FRAME_COUNT += 1
        if FRAME_COUNT % SAVE_INTERVAL == 0:
            # frame_copy = frame.copy()
            filename = f'{TRAIN_PATH}{int(time.time()*1000)}.jpg'
            print(f'Going to save image as {filename}')
            cv2.imwrite(filename, frame)
            FRAME_COUNT = 0
            frame = cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 255, 0), 5)
        if time.time() - startTime > MAX_TIME:
            # call this function to forcing recording to be ended
            onRecordClick()
            print('Auto end recording')


    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    img = Image.fromarray(cv2image)
    imgtk = ImageTk.PhotoImage(image=img)
    lmain.imgtk = imgtk
    lmain.configure(image=imgtk)
    lmain.after(33, show_frame)

def send_data():
    _, __, files = next(os.walk(TRAIN_PATH))
    for filename in files:
        with open(f'{TRAIN_PATH}{filename}', 'rb') as f:
            image_bin = f.read()
        image_label = IMAGE_LABEL
        image_base64 = base64.b64encode(image_bin)
        myobj = { 
            'image': image_base64,
            'label': image_label
        }
        url = BASE_URL+'upload'
        res = requests.post(url, data=myobj)
        print(f'Sent file {TRAIN_PATH}{filename}. Result: {res.text}')
def invoke_train():
    url = BASE_URL+'train'
    res = requests.post(url)
    print(f'Train invoked. Result: {res.text}')

def get_status():
    url = BASE_URL+'status'
    res = requests.get(url)
    print(f'Train invoked. Result: {res.text}')

def download_model():    
    model_name = 'example.model'
    url = BASE_URL+f'model/{model_name}'
    r = requests.get(url, allow_redirects=True)
    with open(model_name, 'wb') as f:
        f.write(r.content)

def main():
    #
    # prepare environment first
    #
    if os.path.isdir(TRAIN_PATH):
        os.mkdir(TRAIN_PATH)

    # set up UI    
    root = tk.Tk()
    root.title('SmartLock')
    width, height = 300, 300
    cap = cv2.VideoCapture(CAMERA_URI)

    #
    # Set camera read property
    #
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    #
    # Press 'q' to escape
    # 
    root.bind('q', lambda e: root.quit())

    recordButtonText = tk.StringVar()
    recordButtonText.set('Start record')
    recordButton = tk.Button(root, textvariable=recordButtonText, command=on_record_click, padx = 10, pady = 10, width=50, height=2)
    recordButton.grid(column=1, row=0)

    lmain = tk.Label(root)
    lmain.grid(column=0, row=0)
    show_frame()
    root.mainloop()

# send_data()
# main()
# get_status()
# download_model()