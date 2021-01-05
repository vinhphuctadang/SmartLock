import cv2
import numpy as np
from joblib import load
from skimage import feature
import face_recognition as fr

threshold = 0.


def LBP(image, numPoints, radius, eps=1e-7):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # compute the Local Binary Pattern representation
    # of the image, and then use the LBP representation
    # to build the histogram of patterns
    lbp = feature.local_binary_pattern(image, numPoints,
                                       radius, method="var")  # uniform | var
    (hist, _) = np.histogram(lbp.ravel(),
                             bins=np.arange(0, numPoints + 3),
                             range=(0, numPoints + 2))
    # normalize the histogram
    hist = hist.astype("float")
    hist /= (hist.sum() + eps)
    # return the histogram of Local Binary Patterns
    return np.array([hist])


def extract_features(img):
    try:
        face_bounding_boxes = fr.face_locations(img)
        # If detecting image contains exactly one face
        if len(face_bounding_boxes) == 1:
            feature_vector = LBP(img, 24, 8)
            return feature_vector
        else:
            return []
    except:
        return []


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


clf = load('faces_LBP.model')
img = cv2.imread('/Users/dcongtinh/Downloads/IMG_4940.jpg')
features = extract_features(img)
print(predict(clf, features))
