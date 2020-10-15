import cv2
import numpy as np
from face_detector import FaceDetector
import logger 
import os 

# CONFIG: replace video URI here
VIDEO_URI = './data/phucb1709618.mov'
FACE_SIZE = (128, 128)

# utils
def getFilename(dir):
    return os.path.basename(dir).split('.')[0]

def area(rect):
    return rect[2] * rect[3]

# logics
VIDEO_NAME = getFilename(VIDEO_URI)
logger.debug('Video name: %s' % VIDEO_NAME)

def main():
    frame_count = 0

    DATA_PATH = 'faces/' + VIDEO_NAME + '/'

    # if path not exists then create
    if not os.path.exists(DATA_PATH):
        os.system('mkdir -p %s' % DATA_PATH)

    # new detector and frame capturer
    detector = FaceDetector()
    cap = cv2.VideoCapture(VIDEO_URI)

    while(True):
        ret, frame = cap.read()
        if not ret:
            break 
        print("%d read" % frame_count)
        if frame_count % 5 != 0: 
            frame_count += 1
            continue

        # frame = cv2.rotate(frame, cv2.cv2.ROTATE_90_CLOCKWISE)
        people, rectedImage = detector.detect(frame, returnNewImage=True)

        if len(people):
            biggest = None 
            for person in people:
                # get box having biggest area, and consider it the main person
                if not biggest or area(biggest) < area(person):
                    biggest = person
                
            # scale        
            region = frame[biggest[1]:biggest[1]+biggest[3], biggest[0]:biggest[0]+biggest[2]].copy()
            region = cv2.resize(region, FACE_SIZE)

            # write to file
            cv2.imwrite(DATA_PATH+VIDEO_NAME+'_%d.jpg' % (frame_count), region)

        frame_count += 1
        cv2.imshow('frame', rectedImage)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # destruction
    cap.release()
    cv2.destroyAllWindows()

main()