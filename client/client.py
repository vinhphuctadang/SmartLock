import tkinter as tk
import cv2
import time, threading
import os, sys
import requests
import base64
import json
from PIL import Image, ImageTk

BASE_URL = 'http://localhost:8080/'
TRAIN_PATH = 'train/'
CAMERA_URI = 0
SAVE_INTERVAL = 20
IMAGE_LABEL = 'phuc'
MAX_TIME = 5

frame_count = 0
isRecording = False
startTime = 0
# timer instance for listening to model changes 
timerInstance = None
cap          = None 
camera_view  = None
recordButtonText = None

def on_record_click():
    global isRecording, startTime, recordButtonText
    print('Hello world')
    if isRecording:
        recordButtonText.set('Start record')
        isRecording = False
    else:
        startTime = time.time()
        print(f'Set start time to {startTime}')
        isRecording = True 
        frame_count = 0
        recordButtonText.set('Recording ... (Press to stop)')

def invoke_train():
    url = BASE_URL+'train'
    res = json.loads(requests.post(url).text)
    print('Train invoked. Result:', res)
    return res

def get_status():
    url = BASE_URL+'status'
    res = json.loads(requests.get(url).text)
    return res 

def download_model(model_name):  
    url = BASE_URL+f'model/{model_name}'
    print(f'Going to download model {model_name} at {url}')
    r = requests.get(url, allow_redirects=True)
    with open(model_name, 'wb') as f:
        f.write(r.content)
    print('Download completed')

def listen_for_model_change():
    global timerInstance
    result = get_status()
    if result['status'] == 'trained':
        print(f'Model {result["model_name"]} trained, going to download')
        download_model(result["model_name"])
    else:
        print('Still not receive trained status, waiting with retrieved status:', result)
        threading.Timer(2.0, listen_for_model_change).start()

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
    print('All files sent')
    invoke_train()

    print('Going to start listening for model changes')
    threading.Timer(2.0, listen_for_model_change).start()
    # timerInstance.start()

def show_frame():
    global isRecording, frame_count
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)
    if isRecording:
        frame_count += 1
        if frame_count % SAVE_INTERVAL == 0:
            # frame_copy = frame.copy()
            filename = f'{TRAIN_PATH}{int(time.time()*1000)}.jpg'
            print(f'Going to save image as {filename}')
            cv2.imwrite(filename, frame)
            frame_count = 0
            frame = cv2.rectangle(frame, (0, 0), (frame.shape[1], frame.shape[0]), (0, 255, 0), 5)
        if time.time() - startTime > MAX_TIME:
            print('Auto end recording')
            # call this function to forcing recording to be ended
            on_record_click()
            # start new thread
            # to send data to server
            threading.Thread(target=send_data, args=()).start()

    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    img = Image.fromarray(cv2image)
    imgtk = ImageTk.PhotoImage(image=img)
    camera_view.imgtk = imgtk
    camera_view.configure(image=imgtk)
    camera_view.after(33, show_frame)

def main():
    #
    # prepare environment first
    #
    if not os.path.isdir(TRAIN_PATH):
        os.mkdir(TRAIN_PATH)

    # set up UI    
    root = tk.Tk()
    root.title('SmartLock')
    width, height = 300, 300

    global cap
    cap = cv2.VideoCapture(CAMERA_URI)
    #
    # Set camera read property
    #
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    #
    # Press 'q' to escape
    # 
    root.bind('q', lambda e: sys.exit(0))

    global camera_view, recordButtonText
    recordButtonText = tk.StringVar()
    recordButtonText.set('Start record')
    recordButton = tk.Button(root, textvariable=recordButtonText, command=on_record_click, padx = 10, pady = 10, width=50, height=2)
    recordButton.grid(column=1, row=0)

    camera_view = tk.Label(root)
    camera_view.grid(column=0, row=0)
    show_frame()
    root.mainloop()

# entry point
main()