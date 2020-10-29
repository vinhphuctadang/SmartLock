import os
import cv2
import logger
import numpy as np
from joblib import load
import face_recognition as fr
# CONFIG: replace video URI here
VIDEO_URI = 0

# This config is based on your camera qual
FACE_SIZE = (400, 400)

# Expected sample size
TARGET_SIZE = (128, 128)
ratio = FACE_SIZE[0] / TARGET_SIZE[0]

# utils
fontFace = cv2.FONT_HERSHEY_SIMPLEX
fontScale, thickness = 1, 2


def extract_features(img):
    # resize to TARGET_SIZE
    # to use with face_recognition faster
    img = cv2.resize(img, TARGET_SIZE)
    try:
        face_bounding_boxes = fr.face_locations(img)
        # If detecting image contains exactly one face
        if len(face_bounding_boxes) == 1:
            face_enc = fr.face_encodings(img, face_bounding_boxes)[0]
            box = np.array(face_bounding_boxes[0])
            box = box * ratio
            return [face_enc], np.array(box, dtype='int64')
        else:
            return [], []
    except:
        return [], []


def main():
    process_this_frame = True
    # logger.debug("Loading saved model ...")
    clf = load('faces.model')
    classes = sorted(os.listdir('faces'))
    if '.DS_Store' in classes:
        del classes[0]
    # frame capturer
    cap = cv2.VideoCapture(VIDEO_URI)
    while True:
        ret, frame = cap.read()
        if not ret:
            logger.debug("End video")
            break

        pivotY = (frame.shape[0] - FACE_SIZE[0]) // 2
        pivotX = (frame.shape[1] - FACE_SIZE[0]) // 2

        frame = cv2.flip(frame[pivotY:pivotY+FACE_SIZE[1],
                               pivotX:pivotX+FACE_SIZE[0]], 1)
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Only process every other frame of video to save time
        if process_this_frame:
            features, box = extract_features(frame)
            if len(features) == 1:
                idx = clf.predict(features)[0]
                # print(clf.predict_proba(features))
                top, right, bottom, left = box
                # Draw a box around the face
                cv2.rectangle(frame, (left, top),
                              (right, bottom), (0, 255, 0), 2)

                # Draw a label with a name below the face
                labelSize = cv2.getTextSize(
                    classes[idx], fontFace, fontScale, thickness)[0]
                cv2.rectangle(frame, (left-1, top),
                              (left+labelSize[0], top-labelSize[1]-20), (0, 255, 0), cv2.FILLED)
                cv2.putText(frame, classes[idx], (left, top - 10),
                            fontFace, fontScale=fontScale, color=(0, 0, 0), thickness=thickness)

        # # accelerate display frame
        # process_this_frame = not process_this_frame

        # Display the resulting image
        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # destruction
    cap.release()
    cv2.destroyAllWindows()


main()
