# import tensorflow as tf
# from tensorflow.keras import datasets, layers, models
# from face_wrapper import FaceWrapper as FW
import cv2
import os
import logger
# validation_split: Float between 0 and 1. 
# Fraction of the training data to be used as validation data. 
# The model will set apart this fraction of the training data, 
# will not train on it, and will evaluate the loss and any model metrics on this data at the end of each epoch. 
# The validation data is selected from the last samples in the x and y data provided, before shuffling.

def build_model():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dense(3, activation='softmax')) # no softmax activation applied, we will use logit
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def loadData(dataDir):
    import numpy as np
    X, Y = [], []
    _, dirs, _ = next(os.walk(dataDir))
    i = 0
    for d in dirs:
        _, _, files = next(os.walk('/'.join([dataDir, d])))
        for f in files:
            X += [cv2.imread('/'.join([dataDir, d, f]))/255.0]
            Y += [i]
        i += 1
    return np.array(X), np.array(Y)

def main():
    import numpy as np
    logger.debug("import data...")
    X, Y = loadData('./faces')
    # X = 

    # X = np.array([1, 2, 3, 4, 5])
    # Y = np.array([0, 0, 0, 1, 1])
    # print(Y.shape)
    # return
    from sklearn.model_selection import KFold as KF
    skf = KF(n_splits=8, shuffle=True, random_state=100000000)

    # Y = np.array([i for i in range(8)])
    for trainIdx, testIdx in skf.split(Y):
        X_train, X_test, Y_train, Y_test = X[trainIdx], X[testIdx], Y[trainIdx], Y[testIdx]
        print(trainIdx)
        # print(Y[testIdx])
        # print(Y_train.shape)
main()
# import tensorflow as tf
# from tensorflow.keras import datasets, layers, models
# from face_wrapper import FaceWrapper as FW 

# def buildCNN(self):

#     """Build a CNN model, the build process is attached to the FaceWrapper workflow
#     Returns:
#         Keras model
#     """
#     model = models.Sequential()
#     model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)))
#     model.add(layers.MaxPooling2D((2, 2)))
#     model.add(layers.Conv2D(64, (3, 3), activation='relu'))
#     model.add(layers.MaxPooling2D((2, 2)))
#     model.add(layers.Flatten())
#     model.add(layers.Dense(16, activation='relu'))
#     model.add(layers.Dense(3, activation='softmax')) # no softmax activation applied, we will use logit
#     model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#     return model

# def main():
#     face = FW()
#     face._build_model = buildCNN
#     face.classes = ['phuc', 'me', 'ngoc']

#     # this will run the workflow as describe in flow.png
#     face.runWorkFlow(epochs=3).save('sample')
# main()