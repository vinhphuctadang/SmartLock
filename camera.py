import os
import cv2
import logger
import numpy as np
from joblib import load
from skimage import feature
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
fontScale, thickness = 0.75, 2
threshold = 0.9


def extract_features(img):
    # resize to TARGET_SIZE
    # to use with face_recognition faster
    img = cv2.resize(img, TARGET_SIZE)
    try:
        face_bounding_boxes = fr.face_locations(img)
        # If detecting image contains exactly one face
        if len(face_bounding_boxes) == 1:
            feature_vector = fr.face_encodings(img, face_bounding_boxes)
            # top, right, bottom, left = face_bounding_boxes[0]
            # roi = img[top:bottom, left:right]
            # img_ycrcb = cv2.cvtColor(roi, cv2.COLOR_BGR2YCR_CB)
            # img_luv = cv2.cvtColor(roi, cv2.COLOR_BGR2LUV)

            # ycrcb_hist = calc_hist(img_ycrcb)
            # luv_hist = calc_hist(img_luv)
            # feature_vector = np.append(ycrcb_hist.ravel(), luv_hist.ravel())
            # feature_vector = feature_vector.reshape(1, len(feature_vector))
            # feature_vector = LBP(img, 24, 8)
            box = np.array(face_bounding_boxes[0])
            box = box * ratio

            # box: int required
            return feature_vector, np.array(box, dtype='int64')
        else:
            return [], []
    except:
        return [], []


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


def calc_hist(img):
    histogram = [0] * 3
    for j in range(3):
        histr = cv2.calcHist([img], [j], None, [256], [0, 256])
        histr *= 255.0 / histr.max()
        histogram[j] = histr
    return np.array(histogram)


def LBP(image, numPoints, radius, eps=1e-7):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # compute the Local Binary Pattern representation
    # of the image, and then use the LBP representation
    # to build the histogram of patterns
    lbp = feature.local_binary_pattern(image, numPoints,
                                       radius, method="uniform")
    (hist, _) = np.histogram(lbp.ravel(),
                             bins=np.arange(0, numPoints + 3),
                             range=(0, numPoints + 2))
    # normalize the histogram
    hist = hist.astype("float")
    hist /= (hist.sum() + eps)
    # return the histogram of Local Binary Patterns
    return np.array([hist])


def main():
    process_this_frame = True
    # logger.debug("Loading saved model ...")
    clf = load('faces2.model')
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
                top, right, bottom, left = box
                try:
                    # Draw a box around the face
                    cv2.rectangle(frame, (left, top),
                                  (right, bottom), (0, 255, 0), 2)

                    # Draw a label with a name below the face
                    label = predict(clf, features)
                    labelSize = cv2.getTextSize(
                        label, fontFace, fontScale, thickness)[0]
                    cv2.rectangle(frame, (left-1, top),
                                  (left+labelSize[0], top-labelSize[1]-20), (0, 255, 0), cv2.FILLED)
                    cv2.putText(frame, label, (left, top - 10),
                                fontFace, fontScale=fontScale, color=(0, 0, 0), thickness=thickness)
                except:
                    pass

        # # accelerate display frame
        # process_this_frame = not process_this_frame

        # Display the resulting image
        cv2.imshow('Video', frame)
        # cv2.imshow('Video2', roi)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # destruction
    cap.release()
    cv2.destroyAllWindows()


main()
