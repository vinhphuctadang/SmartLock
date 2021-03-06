import os
import sys
import cv2
import logger
import numpy as np
from joblib import load
import face_recognition as fr
from scipy.spatial import distance as dist
# CONFIG: replace video URI here
VIDEO_URI = 0
<<<<<<< HEAD
FACE_SIZE = (512, 512)

TARGET_SIZE = (128, 128)
=======

# This config is based on your camera qual
if sys.platform == 'darwin':
    FACE_SIZE = (400, 400)
    # Expected sample size
    TARGET_SIZE = (128, 128)
else:
    # Configure camera on Raspberry Pi
    FACE_SIZE = (320, 280)
    # Expected sample size
    TARGET_SIZE = (96, 96)


ratio = FACE_SIZE[0] / TARGET_SIZE[0]
>>>>>>> 39fb2fdaa0605e07461edce6004ff5303668152d

# utils
fontFace = cv2.FONT_HERSHEY_SIMPLEX
fontScale, thickness = 0.75, 2
threshold = 0.9


<<<<<<< HEAD
    frame_count = 0
    logger.debug("Loading saved model ...")
    face = FaceWrapper()
    face.load('sample')
    
    detector = FaceDetector()
    cap = cv2.VideoCapture(VIDEO_URI)

    print(face.classes)
    while True:
        # ret, frame = cap.read()
        # if not ret:
        #     logger.debug("End video")
        #     break 
        
        # frame = cv2.flip(frame, 1)
        # people, rectedImage = detector.detect(frame, returnNewImage=True)
        # if len(people):
        #     biggest = None 
        #     for person in people:
        #         # get box having biggest area, and consider it the main person
        #         box = person
                
        #         # scale        
        #         region = frame[box[1]:box[1]+box[3], box[0]:box[0]+box[2]].copy()
        #         region = cv2.resize(region, FACE_SIZE)

        #         # predict 
        #         label = face.classes[np.argmax(face.predict([region])[0])]
                
        #         # draw label
        #         rectedImage = cv2.putText(
        #             rectedImage, # canvas
        #             label, # text
        #             (box[0], box[1] + 20), # bottom-left corner
        #             cv2.FONT_HERSHEY_SIMPLEX, # font   
        #             1, # font scale
        #             (0x0, 0xff, 0x0), 
        #             2, # thickness
        #             cv2.LINE_AA # line type
        #         )
                                
        # cv2.imshow('frame', rectedImage)
=======
def extract_features(img):
    # resize to TARGET_SIZE
    # to use with face_recognition faster
    img = cv2.resize(img, TARGET_SIZE)
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


def predict(clf, features):
    label = clf.predict(features)[0]
    if label == 'unknown':
        return 'Unknown~'

    proba = clf.predict_proba(features)
    # print(proba)
    acc_max = np.max(proba[0])
    if acc_max < threshold:
        return 'Unknown~'

    return '%s %.2f' % (label, acc_max*100)


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


def main():
    process_this_frame = True
    # logger.debug("Loading saved model ...")
    filename = sorted(os.listdir('results'))[-1]
    model_path = 'results/' + filename + '/faces.model'
    clf = load(model_path)

    # frame capturer
    cap = cv2.VideoCapture(VIDEO_URI)
    isClosed = isOpened = False
    while True:
>>>>>>> 39fb2fdaa0605e07461edce6004ff5303668152d
        ret, frame = cap.read()
        if not ret:
            logger.debug("End video")
            break
<<<<<<< HEAD
        
        pivotY = (frame.shape[0] - FACE_SIZE[0]) // 2
        pivotX = (frame.shape[1] - FACE_SIZE[0]) // 2

        frame = cv2.flip(frame[pivotY:pivotY+FACE_SIZE[1], pivotX:pivotX+FACE_SIZE[0]], 1)
        predictFrame = cv2.resize(frame, TARGET_SIZE) * 1.0/255

        # print(predictFrame)
        preds = face.predict([predictFrame])[0]

        logger.debug("Predict value:", preds)
        label = face.classes[np.argmax(preds)]
        
        rectedImage = cv2.putText(
                    frame, # canvas
                    label, # text
                    (10, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, # font   
                    1, # font scale
                    (0x0, 0xff, 0x0), 
                    2, # thickness
                    cv2.LINE_AA # line type
        )
        cv2.imshow('classification', frame)

=======

        pivotY = (frame.shape[0] - FACE_SIZE[0]) // 2
        pivotX = (frame.shape[1] - FACE_SIZE[0]) // 2
        if sys.platform == 'darwin':
            frame = cv2.flip(frame[pivotY:pivotY+FACE_SIZE[1],
                                   pivotX:pivotX+FACE_SIZE[0]], 1)
        else:
            # Configure camera on Raspberry Pi
            frame = cv2.flip(frame[pivotY:pivotY+FACE_SIZE[1],
                                   pivotX:pivotX+FACE_SIZE[0]], 1)
            # frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Only process every other frame of video to save time
        if process_this_frame:
            features, face_landmarks, box = extract_features(frame)
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

                    closed = ear_left < 0.2 and ear_right < 0.2
                    if closed:
                        isClosed = True
                    else:
                        isOpened = True

                    # Human Verification: just eye blink 2 times
                    if (isClosed and isOpened):
                        label = predict(clf, features)
                    else:
                        label = 'Fa-ke'

                    # Draw a label with a name below the face
                    labelSize = cv2.getTextSize(
                        label, fontFace, fontScale, thickness)[0]
                    cv2.rectangle(frame, (left-1, top),
                                  (left+labelSize[0], top-labelSize[1]-20), (0, 255, 0), cv2.FILLED)
                    cv2.putText(frame, label, (left, top - 10),
                                fontFace, fontScale=fontScale, color=(0, 0, 0), thickness=thickness)
                except:
                    pass
            else:
                isClosed = isOpened = False

        # # accelerate display frame
        # process_this_frame = not process_this_frame

        # Display the resulting image
        cv2.imshow('Smart Home', frame)
        # cv2.imshow('Video2', roi)
>>>>>>> 39fb2fdaa0605e07461edce6004ff5303668152d
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # destruction
    cap.release()
    cv2.destroyAllWindows()


main()
