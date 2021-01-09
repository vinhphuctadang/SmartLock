from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import face_recognition as fr
from joblib import dump
from sklearn import svm
import numpy as np
import time
import cv2
import os

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
    print("\nLoading data...\n")
    X, y = [], []
    for label in sorted(os.listdir(pathname)):
        if valid(label):
            print(label + ' in processing...')
            path_img = os.path.join(pathname, label)
            for img in os.listdir(path_img):
                if valid(img):
                    img = cv2.imread(os.path.join(path_img, img))
                    features = extract_features(img)
                    if len(features) == 1:
                        X.append(features[0])
                        y.append(label)

    return np.array(X), np.array(y)


def train(path_to_train='train', path_to_save='models'):
    model_name = time.strftime('%Y%m%d_%H%M%S') + '.model'
    model_path = path_to_save + '/' + model_name

    # Load data from 'path_to_train'
    X, y = load_dataset(path_to_train)

    # Holdout
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=3./10, random_state=2020)

    # Use SVM from scikit-learn
    clf = svm.SVC(kernel='rbf', C=100000, gamma=0.01, probability=True)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    # Compute Accuracy
    acc = accuracy_score(y_test, y_pred)
    print('\nAccuracy:', acc*100)

    # Save model by using dump (joblib)
    dump(clf, model_path)
    print("\nModel was trained and saved at " + model_path + "\n")

    return model_name
