import os
import cv2
import numpy as np
from joblib import load, dump
import face_recognition as fr


TARGET_SIZE = (128, 128)


def extract_features(img):
    # resize to TARGET_SIZE
    # to use with face_recognition faster
    img = cv2.resize(img, TARGET_SIZE)
    try:
        face_bounding_boxes = fr.face_locations(img)
        # If detecting image contains exactly one face
        if len(face_bounding_boxes) == 1:
            feature_vector = fr.face_encodings(img, face_bounding_boxes)
            return feature_vector
        else:
            return []
    except:
        return []


def valid(name):
    return name != '.DS_Store' and name != 'Icon\r'


def load_dataset(pathname):
    print("Loading data...\n")
    X, y = [], []
    idx = 3
    for label in sorted(os.listdir(pathname)):
        if valid(label):
            print(label + ' in processing...')
            path_img = os.path.join(pathname, label)
            for img in os.listdir(path_img):
                if valid(img):
                    img = cv2.imread(os.path.join(path_img, img))
                    features = extract_features(img)
                    # print(features)
                    if len(features) == 1:
                        X.append(features[0])
                        y.append(idx)

    return np.array(X), np.array(y)


X_new,  y_new = load_dataset('data_new')


filename = sorted(os.listdir('results'))[-1]
model_path = 'results/' + filename + '/faces.model'
print(model_path)
clf = load(model_path)

clf.partial_fit(X_new, y_new)
dump(clf, 'faces_new.model')
