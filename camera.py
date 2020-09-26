'''
Reads camera from a REMOTE IP
'''

import cv2
import numpy as np

# change your camera HERE !!! 
# or set it to 0 (i.e CAMERA_URL=0) in order to access built-in camera
CAMERA_URL = 'rtsp://192.168.0.103:8080/h264_ulaw.sdp'

# init capturer
cap = cv2.VideoCapture(CAMERA_URL)

while(True):
    ret, frame = cap.read()
    cv2.imshow('frame',frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# destruction
cap.release()
cv2.destroyAllWindows()
