# import modules
import tensorflow
from tensorflow import keras 
import logger 
import os
import time
import numpy as np 

class FaceWrapper:
    def __init__(self):
        # configs goes here
        self.faceWidth = 128
        self.faceHeight = 128
        self.batchSize = 32
        self.datasetDir = './faces'
        self.model = None

    def importDataset(self):
        # preprocess training data
        train_datagen = keras.preprocessing.image.ImageDataGenerator(rescale=1./255, validation_split=1./3)

        # split dataset
        train_set = train_datagen.flow_from_directory(
            self.datasetDir,
            seed=10,
            target_size=(self.faceHeight, self.faceWidth),
            class_mode='categorical', 
            batch_size=self.batchSize, 
            subset="training",
        )

        test_set = train_datagen.flow_from_directory(
            self.datasetDir,
            seed=10,
            target_size=(self.faceHeight, self.faceWidth),
            class_mode='categorical', 
            batch_size=self.batchSize, 
            subset="validation"
        )

        return train_set, test_set

    def buildModel(self):
        model = keras.Sequential([
            keras.layers.Flatten(input_shape=(self.faceHeight, self.faceWidth, 3)),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(2, activation='softmax'),
        ])
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    
    def runWorkFlow(self, epochs=20): # exec the whole workflow
        train, test = self.importDataset()
        self.model = self.buildModel()
        logger.debug(train, test)
        self.train(train, test, epochs=epochs)
        
    def train(self, train_set, test_set, epochs=20):
        self.model.fit(
            train_set, # data to train, a format of (X, y)
            steps_per_epoch=len(train_set), # The number of batch iterations before a training epoch is considered finished. Ignore if whole data is load, in our case, we need this in order to iterate over batches; hence make sure that generator can generate at least: steps_per_epoch * epochs batches
            epochs=epochs, # number of epochs, i.e times that iteration of updating weights goes
            validation_data=test_set, # validation
            validation_steps=len(test_set)
        )

    def predict(self, X):
        return self.model.predict(np.array(X))
        
    def save(self, id):
        self.model.save(id)
        return self 

    def load(self, id):
        self.model = keras.models.load_model(id)
        return self 

if __name__=='__main__':
    logger.debug('Tensorflow version:', tensorflow.__version__)