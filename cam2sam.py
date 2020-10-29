import cv2
# import numpy as np
# from face_detector import FaceDetector
# from face_wrapper import FaceWrapper
import logger
import os
import time
import sys

# PERSON NAME
"""PERSON NAME, which work as an ID
PLEASE CONFIG PERSON_NAME BEFORE RUNNING THE SCRIPT!!!
"""
if len(sys.argv) > 1:
    PERSON_NAME = sys.argv[1]
else:
    raise('Enter your name of sample.')

# CONFIG: replace video URI here
VIDEO_URI = 0
# This config is based on your camera qual
FACE_SIZE = (400, 400)
# Expected sample size
TARGET_SIZE = (128, 128)

print(PERSON_NAME)
# SAVE PATH
PATH = './faces/%s' % PERSON_NAME
if not os.path.isdir(PATH):
    os.mkdir(PATH)

WRITE_TO_DISK = False


def text(image, p, txt):
    # font
    font = cv2.FONT_HERSHEY_SIMPLEX

    # org
    org = p

    # fontScale
    fontScale = 1

    # Blue color in BGR
    color = (255, 0, 0)

    # Line thickness of 2 px
    thickness = 2

    # Using cv2.putText() method
    image = cv2.putText(image, txt, org, font,
                        fontScale, color, thickness, cv2.LINE_AA)
    return image


def convTime(second):
    return '%02d:%02d' % (second // 60, second % 60)


def main():
    global WRITE_TO_DISK

    frame_count = 0
    # new detector and frame capturer
    # detector = FaceDetector()
    cap = cv2.VideoCapture(VIDEO_URI)

    startTime = cnt = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            logger.debug("End video")
            break

        pivotY = (frame.shape[0] - FACE_SIZE[0]) // 2
        pivotX = (frame.shape[1] - FACE_SIZE[0]) // 2

        frame = cv2.flip(frame[pivotY:pivotY+FACE_SIZE[1],
                               pivotX:pivotX+FACE_SIZE[0]], 1)
        if WRITE_TO_DISK:
            frame_count += 1
            if frame_count == 5:
                frame_count = 0
                savedFrame = cv2.resize(frame, TARGET_SIZE)
                cv2.imwrite('%s/%s_%05d.jpg' % (PATH, PERSON_NAME,
                                                cnt), savedFrame)
                cv2.circle(frame, (15, 15), 10, (0x0, 0x0, 0xff), -1)
                cnt += 1

            frame = text(frame, (5, 60), convTime(time.time()-startTime))

        cv2.imshow('frame', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        if key == ord(' '):
            WRITE_TO_DISK = not WRITE_TO_DISK
            frame_count = cnt = 0
            startTime = time.time()

    # destruction
    cap.release()
    cv2.destroyAllWindows()


main()
