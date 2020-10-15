# import modules
import tensorflow
from tensorflow import keras 
import logger 
import os
import time
import numpy as np 
import json

class FaceWrapper:
    """FaceWrapper model is responsible for interventing preprocessing and training steps for experiments
    Drawback: Temporarily no k-fold supported
    """

    def __init__(self):
        # configs goes here
        self.faceWidth = 128
        self.faceHeight = 128
        self.batchSize = 32
        self.datasetDir = './faces'
        self.saveDir = './models'
        self.classes = []

        # model name to produce log file
        self.modelName = 'default'

        # model
        self.model = None
        self.history = None

        # we might want to override these function to intervent steps for researching
        # steps of using these callback please refer to './train.py' for details
        self._import_dataset = None 
        self._preprocess = None 
        self._build_model = None 
        self._train = None
        self._predict = None 

    def importDataset(self):

        if self._import_dataset:
            return self._import_dataset(self)
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

    def preprocess(self, X):
        if self._preprocess:
            return self._preprocess(self, X)
        return X 

    def buildModel(self):
        if self._build_model:
            return self._build_model(self)

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
        train = self.preprocess(train)
        test = self.preprocess(test)
        self.model = self.buildModel()
        # logger.debug(train, test)
        self.train(train, test, epochs=epochs)
        return self
        
    def train(self, train_set, test_set, epochs=20):

        if self._train:
            self.history = self._train(self, train_set, test_set)
        else: 
            self.history = self.model.fit(
                train_set, # data to train, a format of (X, y)
                steps_per_epoch=len(train_set), # The number of batch iterations before a training epoch is considered finished. Ignore if whole data is load, in our case, we need this in order to iterate over batches; hence make sure that generator can generate at least: steps_per_epoch * epochs batches
                epochs=epochs, # number of epochs, i.e times that iteration of updating weights goes
                validation_data=test_set, # validation
                validation_steps=len(test_set)
            )
        return self

    def predict(self, X):
        if self._predict:
            logger.debug("Call custom prediction")
            return self._predict(self, X)

        predictions = self.model.predict(np.array(X))
        return predictions
        
    def save(self, id):
        directory = os.path.join(self.saveDir, id)
        if not os.path.isdir(directory):
            os.mkdir(directory)
        
        # save model to file
        model_dir = os.path.join(directory, id+'.h5')
        self.model.save(model_dir)

        # self meta data alongside model
        meta_dir = os.path.join(directory, id+'.meta')
        with open(meta_dir, 'w') as f:
            print(json.dumps(self.classes), file=f)
            print(self.history.history, file=f)
        return self

    def load(self, id):
        directory = os.path.join(self.saveDir, id)
        model_dir = os.path.join(directory, id + '.h5')
        meta_dir  = os.path.join(directory, id + '.meta')

        self.model = keras.models.load_model(model_dir)

        with open(meta_dir, 'r') as f:
            classes = f.readline()
            self.classes = json.loads(classes)
            logger.debug("Loaded classes:", self.classes)
        return self 

if __name__=='__main__':
    logger.debug('Tensorflow version:', tensorflow.__version__)
