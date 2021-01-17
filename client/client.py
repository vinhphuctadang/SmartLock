import tkinter as tk
import tkinter.ttk as ttk
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
import tkinter.simpledialog
from PIL import Image, ImageTk
import elock
# from concurrent.futures import ThreadPoolExecutor
import multiprocessing

context = multiprocessing
if 'forkserver' in multiprocessing.get_all_start_methods():
    context = multiprocessing.get_context('forkserver')

# job_count_lock = threading.Lock()
# _job_count = 0
# job_result = Queue()
# executor = ThreadPoolExecutor(max_workers=2)

# def get_job_count():
#     job_count_lock.acquire()
#     job_cnt = _job_count
#     job_count_lock.release()
#     return job_cnt
# def add_job(inc):
#     job_count_lock.acquire()
#     global _job_count
#     _job_count += inc
#     job_count_lock.release()

BASE_URL        = 'http://localhost:8080/'
TRAIN_PATH      = 'train'
CAMERA_URI      = 0
SAVE_INTERVAL   = 5 # Deprecated
IMAGE_LABEL     = 'phuc'
# MAX_TIME        = 5 # Deprecated
MAX_SAMPLE      = 50

FONT_FACE       = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE      = 0.75
THICKNESS       = 2
THRESHOLE       = 0.85
DEFAULT_MODEL_NAME  = 'default.model'
DEFAULT_CONFIG_FILE = 'default.json'
DEFAULT_CONFIG  = {}

frame_count     = 0 
sample_count    = 0
is_recording    = False
start_time      = 0
unlockStatus    = 0

# timer instance for listening to model changes 
timer_instance  = None
cap             = None 
camera_view     = None
recordButtonText= None
recordButton    = None
lockButton      = None

statusText      = None
progressBar     = None
imageLabelTextEdit   = None
imageLabelText       = None

marked_time = None
# face status
isClosed        = isOpened = False
runningModel    = None
mutex           = threading.Lock()
width, height   = 400, 400

is_on_detecting = False 

def on_lock_click():
    global isClosed, isOpened 
    elock.setLock(False)
    recordButton['state']   = 'disabled'
    lockButton['state']     = 'disabled'
    isClosed = isOpened     = False

def parse_config():
    global DEFAULT_CONFIG
    if not os.path.isfile(DEFAULT_CONFIG_FILE):
        DEFAULT_CONFIG = { 
            'granted_people' : [],
            'all_people': []
        }
        update_config() 
        return 

    with open(DEFAULT_CONFIG_FILE, 'r', encoding='utf8') as f:
        DEFAULT_CONFIG = json.loads(f.readline())

def update_config():
    with open(DEFAULT_CONFIG_FILE, 'w', encoding='utf8') as f:
        f.write(json.dumps(DEFAULT_CONFIG))

def on_record_click():
    global is_recording, sample_count, recordButtonText, IMAGE_LABEL
    print('Hello world')
    if is_recording:
        recordButtonText.set('Add granted person')
        is_recording = False
        progressBar.grid_forget()
        progressBar['value'] = 0
    else:
        IMAGE_LABEL = tk.simpledialog.askstring('Person name', 'Please input person name (without space):')
        if not IMAGE_LABEL:
            return
        # IMAGE_LABEL = imageLabelTextEdit.get(1.0, 'end').replace('\n', '')
        # start_time = time.time()
        progressBar.grid(column=0, row=1, ipady=5, columnspan= 2, padx=20, pady=20)        
        progressBar['value'] = 0
        if not os.path.isdir(TRAIN_PATH):
            os.mkdir(TRAIN_PATH)
        sample_count = frame_count = 0
        is_recording = True
        recordButtonText.set('Recording ... (Press to stop)')
        statusText.set('Collect face of ' + IMAGE_LABEL)

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

def reload_model():
    global runningModel
    model_name = DEFAULT_MODEL_NAME
    if not os.path.isfile(model_name):
        runningModel = None
        return
    # in case loading model takes long time
    statusText.set('(Re)loading ' + model_name)
    loadingModel = load(model_name)
    mutex.acquire()
    runningModel = loadingModel
    mutex.release()
    statusText.set('Model reloaded')

def download_model(model_name):  
    url = BASE_URL+'model/'+model_name
    print('Going to download model %s at %s' % (model_name, url))
    r = requests.get(url, allow_redirects=True)
    try:
        with open(model_name, 'wb') as f:
            f.write(r.content)
        if os.path.isfile(DEFAULT_MODEL_NAME):
            os.remove(DEFAULT_MODEL_NAME)

        if IMAGE_LABEL not in DEFAULT_CONFIG['all_people']:
            DEFAULT_CONFIG['granted_people'].append(IMAGE_LABEL)
            DEFAULT_CONFIG['all_people'].append(IMAGE_LABEL)
            update_config()

        # download complete then (re)load model
        if os.path.isfile(model_name):
            os.rename(model_name, DEFAULT_MODEL_NAME)
        reload_model()
    except Exception as err:
        if os.path.isfile(model_name):
            os.remove(model_name)
        statusText.set('Error happened: ' + str(err))

def listen_for_model_change():
    global timer_instance
    try:
        result = get_status()
        if result['status'] == 'failed':
            statusText.set('Failed training face in server')
        elif result['status'] == 'trained':
            statusText.set('Model %s trained, going to download' % result["model_name"])
            download_model(result["model_name"])
        else:
            statusText.set('Still not receive trained status, waiting with retrieved status: %s' % result["status"])
            threading.Timer(2.0, listen_for_model_change).start()
    except Exception as err:
        statusText.set('Error happened: ' + str(err))

def send_data():
    _, __, files = next(os.walk(TRAIN_PATH))
    for filename in files:
        image_path = TRAIN_PATH+'/'+filename
        with open(image_path, 'rb') as f:
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
            statusString = 'Sent file %s%s. Result: %s' % (TRAIN_PATH, filename, res.text)
            # print(statusString)
            statusText.set(statusString)
        except Exception as err:
            statusText.set('Error happened: ' + str(err))
            print('Error happened: '+ str(err))
        if os.path.isfile(image_path):
            os.remove(image_path)
    print('All files sent')
    try:
        invoke_train()
    except Exception as err:
        statusText.set('Error happened: ' + str(err))
        return
    statusText.set('Going to start listening for model changes')
    threading.Timer(2.0, listen_for_model_change).start()

def extract_features(img, need_feature=True, need_landmark=True):
    # resize to TARGET_SIZE
    # to use with face_recognition faster
    ratio = 3
    ORG_SIZE = img.shape
    img = cv2.resize(img, (ORG_SIZE[1]//ratio, ORG_SIZE[0]//ratio))
    try:
        face_bounding_boxes = fr.face_locations(img)
        # If detecting image contains exactly one face
        if len(face_bounding_boxes) == 1:
            feature_vector = face_landmarks = [0]
            if need_feature:
                feature_vector = fr.face_encodings(img, face_bounding_boxes)
            if need_landmark:
                face_landmarks = fr.face_landmarks(img, face_bounding_boxes)
            box = np.array(face_bounding_boxes[0])
            box = box * ratio
            # box: int required
            return feature_vector, face_landmarks, np.array(box, dtype='int64')
        else:
            return [], [], []
    except:
        return [], [], []

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

def detect_face(frame, need_labeling=True):
    face_box = ()
    label = None
    features, face_landmarks, box = extract_features(frame, need_feature=need_labeling, need_landmark=need_labeling)
    # if not need labeling, then just return box for short
    if not need_labeling:
        if len(box): 
            top, right, bottom, left = box
            cv2.rectangle(frame, (left, top),
                    (right, bottom), (0, 255, 0), 2)
        return frame, box, None

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
            acc = None
            # Human Verification: just eye blink 2 times
            if need_labeling:
                if (isClosed and isOpened):
                    mutex.acquire()
                    try:
                        if runningModel:
                            label, acc = predict(runningModel, features)
                            if label in DEFAULT_CONFIG['granted_people']:
                                if recordButton['state'] == 'disabled': 
                                    recordButton['state'] = 'active'
                                    lockButton['state'] = 'active'
                                    elock.setLock(True)
                                    statusText.set('Face verified: %s' % label)
                        else:
                            label = 'unknown'
                    except Exception as err: 
                        print(err)
                        label = 'Blink your eye'
                    mutex.release()
                else:
                    label = 'Blink your eye'
            else:
                label = 'Detected face'

            if acc:
                label = '%s %.2f' % (label, acc)
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
            face_box = box
        except Exception as e:
            statusText.set('Error happened: ' + str(e))
    else:
        isClosed = isOpened = False
    return frame, face_box, label

executor_result = context.Queue()
executor_input  = context.Queue()

def detect_face_task():
    while True:
        # block until a frame exists
        frame = executor_input.get()
        # predict
        frame, face_box, label = detect_face(frame, need_labeling=True)
        print('Detected:', label)
        if not label:
            label = 'unknown'
        executor_result.put(label)
executor = context.Process(target=detect_face_task, args=())

def show_frame():
    global is_recording, sample_count, frame_count, marked_time, executor, is_on_detecting
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)
    
    if is_recording:
        sub_frame = frame.copy()
        sub_frame, face_box, _ = detect_face(sub_frame, need_labeling=False)
        if len(face_box):
            frame_count += 1
            if frame_count % SAVE_INTERVAL == 0:
                sample_count += 1
                filename = '%s/%d.jpg' % (TRAIN_PATH, int(time.time()*1000))
                print('Going to save image as ' + filename)
                cv2.imwrite(filename, frame)
                # replace frame for displaying
                frame = sub_frame
                progressBar['value'] = min(100, (sample_count / MAX_SAMPLE) * 100)
                if sample_count == MAX_SAMPLE:
                    print('Auto end recording')
                    # call this function to forcing recording to be ended
                    on_record_click()
                    # start new thread
                    # to send data to server
                    threading.Thread(target=send_data, args=()).start()
    else:
        if not is_on_detecting:
            is_on_detecting = True
            # mock status
            executor_input.put(frame.copy())

        frame, face_box, _ = detect_face(frame, need_labeling=False)     
        if not executor_result.empty():
            try:
                result = executor_result.get()
                cv2.putText(frame, result[1], (frame.shape[1]//2, frame.shape[0]//2),
                    FONT_FACE, fontScale=FONT_SCALE, color=(0, 255, 0), thickness=THICKNESS)
                is_on_detecting = False
            except Exception as e:
                print('Error happened: ' + str(e))
                pass

    if not marked_time:
        marked_time = time.time()
    else:
        tmp = time.time()
        fps = 1 / (tmp-marked_time)
        cv2.putText(frame, '%.2f' % fps, (frame.shape[1]//2, frame.shape[0]//2),
                    FONT_FACE, fontScale=FONT_SCALE, color=(0, 255, 0), thickness=THICKNESS)
        marked_time = tmp
    # display frame
    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    img = Image.fromarray(cv2image)
    imgtk = ImageTk.PhotoImage(image=img)
    camera_view.imgtk = imgtk
    camera_view.configure(image=imgtk)
    camera_view.after(1, show_frame)

def predict(clf, features):
    label = clf.predict(features)[0]
    proba = clf.predict_proba(features)
    acc_max = np.max(proba[0])
    if acc_max < THRESHOLE:
        return 'Unknown~', acc_max
    return label, acc_max

def main():
    parse_config()
    # set up UI    
    root = tk.Tk()
    root.title('SmartLock')

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

    global camera_view, recordButtonText, statusText, progressBar, imageLabelTextEdit, recordButton, lockButton

    camera_view = tk.Label(root, width=width, height=height)
    camera_view.grid(column=0, row=0, columnspan=2)

    style = ttk.Style()
    style.theme_use('clam')
    style.configure("red.Horizontal.TProgressbar", foreground='red', background='red')
    progressBar = ttk.Progressbar(root, orient=tk.HORIZONTAL, style="red.Horizontal.TProgressbar", length=300, mode='determinate', maximum=100, value=0)
    progressBar.grid(column=0, row=1, ipady=5, columnspan= 2, padx=20, pady=20)
    progressBar['value'] = 0
    progressBar.grid_forget()

    recordButtonText = tk.StringVar()
    recordButtonText.set('Add granted person')
    recordButton = tk.Button(root, textvariable=recordButtonText, command=on_record_click, width=25, height=3) # , padx = 10, pady = 10)
    recordButton.grid(column=0, row=2, padx=20, pady=10)
    
    lockButton = tk.Button(root, text='Lock the door', width=25, height=3, command=on_lock_click)
    lockButton.grid(column=1, row=2, padx=20)

    statusText = tk.StringVar()
    statusText.set('Status: Idle')
    statusLabel = tk.Label(root, textvariable=statusText, anchor='s', width=50, height=1)
    statusLabel.grid(column=0, row=3, columnspan=2, pady=20)

    show_frame()
    reload_model()

    # post init
    if len(DEFAULT_CONFIG['granted_people']) > 0:
        recordButton['state'] = 'disable'
        lockButton['state'] = 'disable'
        elock.setLock(True)
        statusText.set('Verify who you are before adding granted person/unlock the door')
    else:
        statusText.set('Add first granted person to start using the lock')
    root.mainloop()

# entry point
main()