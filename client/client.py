import tkinter as tk
import cv2
import time, threading
import os, sys
import requests
import base64
import json
import numpy as np
from joblib import load
from skimage import feature
import face_recognition as fr
from scipy.spatial import distance as dist
from PIL import Image, ImageTk

BASE_URL = 'http://localhost:8080/'
TRAIN_PATH = 'train/'
CAMERA_URI = 0
SAVE_INTERVAL = 20
IMAGE_LABEL = 'phuc'
MAX_TIME = 5
FONT_FACE = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.75
THICKNESS = 2
THRESHOLE = 0.9

frame_count = 0
isRecording = False
startTime = 0
# timer instance for listening to model changes 
timerInstance = None
cap          = None 
camera_view  = None
recordButtonText = None
statusText = None

# face status
isClosed = isOpened = False

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
    statusText.set('Invoking server train function')
    res = json.loads(requests.post(url).text)
    print('Train invoked. Result:', res)
    statusText.set('Invoked training')
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
        statusText.set(f'Model {result["model_name"]} trained, going to download')
        download_model(result["model_name"])
    else:
        statusText.set(f'Still not receive trained status, waiting with retrieved status: {result["status"]}' )
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
        try:
            res = requests.post(url, data=myobj)
            statusString = f'Sent file {TRAIN_PATH}{filename}. Result: {res.text}'
            # print(statusString)
            statusText.set(statusString)
        except Exception as err:
            statusText.set(f'Error happened: {str(err)}')

    print('All files sent')
    try:
        invoke_train()
    except Exception as err:
        statusText.set(f'Error happened: {str(err)}')
        return 
    
    statusText.set('Going to start listening for model changes')
    threading.Timer(2.0, listen_for_model_change).start()

def extract_features(img):
    # resize to TARGET_SIZE
    # to use with face_recognition faster
    ratio = 6
    ORG_SIZE = img.shape
    img = cv2.resize(img, (ORG_SIZE[1]//ratio, ORG_SIZE[0]//ratio))
    try:
        face_bounding_boxes = fr.face_locations(img)
        # If detecting image contains exactly one face
        if len(face_bounding_boxes) == 1:
            feature_vector = fr.face_encodings(img, face_bounding_boxes)
            face_landmarks = fr.face_landmarks(img, face_bounding_boxes)
            box = np.array(face_bounding_boxes[0])
            box = box * ratio
            # box: int required
            return feature_vector, face_landmarks, np.array(box, dtype='int64')
        else:
            return [], [], []
    except:
        return [], [], []

def reload_model():
    pass 

def get_ear(eye):
    # compute the euclidean distances between the two sets of
    # vertical eye landmarks (x, y)-coordinates
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    # compute the euclidean distance between the horizontal
    # eye landmark (x, y)-coordinates
    C = dist.euclidean(eye[0], eye[3])

    # compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)

    # return the eye aspect ratio
    return ear

def detect_face(frame):
    features, face_landmarks, box = extract_features(frame)
    global isClosed, isOpened 
    if len(features) == 1 and len(face_landmarks) == 1:
        top, right, bottom, left = box
        face_landmarks = face_landmarks[0]
        try:
            # Draw a box around the face
            cv2.rectangle(frame, (left, top),
                            (right, bottom), (0, 255, 0), 2)
            left_eye = face_landmarks['left_eye']
            right_eye = face_landmarks['right_eye']
            ear_left = get_ear(left_eye)
            ear_right = get_ear(right_eye)
            
            closed = ear_left <= 0.2 and ear_right <= 0.2
            if closed:
                isClosed = True
            else:
                isOpened = True

            # print('Ears:', ear_left, ear_right, 'Close, sopen status:', isClosed, isOpened)
            # Human Verification: just eye blink 2 times
            if (isClosed and isOpened):
                label = 'phuc-test' # predict(clf, features)
            else:
                label = 'Fa-ke'
            # Draw a label with a name below the face
            labelSize = cv2.getTextSize(
                label, FONT_FACE, FONT_SCALE, THICKNESS)[0]
            cv2.rectangle(
                frame, 
                (left-1, top), 
                (left+labelSize[0], top-labelSize[1]-20), 
                (0, 255, 0), 
                cv2.FILLED
            )
            cv2.putText(frame, label, (left, top - 10),
                        FONT_FACE, fontScale=FONT_SCALE, color=(0, 0, 0), thickness=THICKNESS)
        except Exception as e:
            print('Error happened:', e)
            pass
    else:
        isClosed = isOpened = False
    return frame

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
    
    frame = detect_face(frame)
    # display frame
    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    img = Image.fromarray(cv2image)
    imgtk = ImageTk.PhotoImage(image=img)
    camera_view.imgtk = imgtk
    camera_view.configure(image=imgtk)
    camera_view.after(33, show_frame)

def predict(clf, features):
    label = clf.predict(features)[0]
    if label == 'unknown':
        return 'Unknown~'
    proba = clf.predict_proba(features)
    acc_max = np.max(proba[0])
    if acc_max < threshold:
        return 'Unknown~'
    return '%s %.2f' % (label, acc_max*100)
def main():
    #
    # prepare environment first
    #
    if not os.path.isdir(TRAIN_PATH):
        os.mkdir(TRAIN_PATH)

    # set up UI    
    root = tk.Tk()
    root.title('SmartLock')
    # root.geometry("600x400")

    width, height = 400, 400

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

    global camera_view, recordButtonText, statusText

    statusText = tk.StringVar()
    statusText.set('Status text goes here')
    statusLabel = tk.Label(root, textvariable=statusText, anchor='w', width=50, height=1)
    statusLabel.grid(column=1, row=1)

    recordButtonText = tk.StringVar()
    recordButtonText.set('Start record')
    recordButton = tk.Button(root, textvariable=recordButtonText, command=on_record_click, width=50, height=3) # , padx = 10, pady = 10)
    recordButton.grid(column=1, row=0)

    camera_view = tk.Label(root, width=width, height=height)
    camera_view.grid(column=0, row=0, rowspan=2)
    show_frame()
    root.mainloop()

# entry point
main()