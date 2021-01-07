import os
import cv2
import logger
import numpy as np
from joblib import load
from skimage import feature
import face_recognition as fr
from scipy.spatial import distance as dist
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
            face_landmarks = fr.face_landmarks(img, face_bounding_boxes)
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


def calc_hist(img):
    '''
        Caculate Histogram of Image
    '''
    histogram = [0] * 3
    for j in range(3):
        histr = cv2.calcHist([img], [j], None, [256], [0, 256])
        histr *= 255.0 / histr.max()
        histogram[j] = histr
    return np.array(histogram)


def LBP(image, numPoints, radius, eps=1e-7):
    '''
        Caculate Local Binary Pattern Variance
    '''
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # compute the Local Binary Pattern representation
    # of the image, and then use the LBP representation
    # to build the histogram of patterns
    lbp = feature.local_binary_pattern(image, numPoints,
                                       radius, method="var")
    (hist, _) = np.histogram(lbp.ravel(),
                             bins=np.arange(0, numPoints + 3),
                             range=(0, numPoints + 2))
    # normalize the histogram
    hist = hist.astype("float")
    hist /= (hist.sum() + eps)
    # return the histogram of Local Binary Patterns
    return np.array([hist])


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
        ret, frame = cap.read()
        if not ret:
            logger.debug("End video")
            break

        pivotY = (frame.shape[0] - FACE_SIZE[0]) // 2
        pivotX = (frame.shape[1] - FACE_SIZE[0]) // 2

        frame = cv2.flip(frame[pivotY:pivotY+FACE_SIZE[1],
                               pivotX:pivotX+FACE_SIZE[0]], 1)

        # Configure camera on Raspberry Pi
        # frame = cv2.flip(frame[pivotY:pivotY+FACE_SIZE[1],
        #                        pivotX:pivotX+FACE_SIZE[0]], 0)
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
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # destruction
    cap.release()
    cv2.destroyAllWindows()


main()
