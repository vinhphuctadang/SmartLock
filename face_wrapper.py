# import modules
import tensorflow
from tensorflow import keras
import logger
import os
import time
import numpy as np
import json
import matplotlib.pyplot as plt


class FaceWrapper:
    """FaceWrapper model is responsible for interventing preprocessing and training steps for experiments
    Drawback: Temporarily no k-fold supported
    """

    def __init__(self):
        # configs goes here
        self.faceWidth = 128
        self.faceHeight = 128
        self.faceChannel = 3
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
        train_datagen = keras.preprocessing.image.ImageDataGenerator(
            rescale=1./255, validation_split=1./3)
        test_datagen = keras.preprocessing.image.ImageDataGenerator(
            rescale=1./255, validation_split=1./3)

        # split dataset
        train_set = train_datagen.flow_from_directory(
            self.datasetDir,
            seed=10,
            target_size=(self.faceHeight, self.faceWidth),
            class_mode='categorical',
            color_mode='rgb' if self.faceChannel == 3 else 'grayscale',
            batch_size=self.batchSize,
            subset="training",
        )
        valid_set = train_datagen.flow_from_directory(
            self.datasetDir,
            seed=10,
            target_size=(self.faceHeight, self.faceWidth),
            class_mode='categorical',
            color_mode='rgb' if self.faceChannel == 3 else 'grayscale',
            batch_size=self.batchSize,
            subset="validation"
        )

        test_set = test_datagen.flow_from_directory(
            self.datasetDir,
            seed=10,
            target_size=(self.faceHeight, self.faceWidth),
            class_mode='categorical',
            color_mode='rgb' if self.faceChannel == 3 else 'grayscale',
            batch_size=1,
            subset="validation"
        )

        return train_set, valid_set, test_set

    def preprocess(self, X):
        if self._preprocess:
            return self._preprocess(self, X)
        return X

    def buildModel(self):
        if self._build_model:
            return self._build_model(self)

        model = keras.Sequential([
            keras.layers.Flatten(input_shape=(
                self.faceHeight, self.faceWidth, self.faceChannel)),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(2, activation='softmax'),
        ])
        model.compile(optimizer='adam',
                      loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def runWorkFlow(self, epochs=20):  # exec the whole workflow
        train, valid, test = self.importDataset()
        train = self.preprocess(train)
        valid = self.preprocess(valid)
        self.model = self.buildModel()
        # logger.debug(train, valid)
        self.train(train, valid, test, epochs=epochs)
        return self

    def plot_fig(self, epoch_arr, name, filepath, train, val):
        plt.plot(epoch_arr, train, 'g-o', label='Training ' + name)
        plt.plot(epoch_arr, val, 'r-o', label='Validation ' + name)
        plt.title('Testing Accuracy & ' + name)
        plt.legend()
        plt.xlabel("Epochs")
        plt.ylabel(name)
        plt.savefig(filepath)
        plt.clf()

    def train(self, train_set, valid_set, test_set, epochs=20):

        if self._train:
            self.history = self._train(self, train_set, valid_set)
        else:
            time_start = time.time()
            STEP_SIZE_TRAIN = max(1, train_set.samples //
                                  train_set.batch_size)
            STEP_SIZE_VALID = max(1, valid_set.samples //
                                  valid_set.batch_size)
            self.history = self.model.fit(
                train_set,  # data to train, a format of (X, y)
                # The number of batch iterations before a training epoch is considered finished. Ignore if whole data is load, in our case, we need this in order to iterate over batches; hence make sure that generator can generate at least: steps_per_epoch * epochs batches
                steps_per_epoch=STEP_SIZE_TRAIN,
                epochs=epochs,  # number of epochs, i.e times that iteration of updating weights goes
                validation_data=valid_set,  # validation
                validation_steps=STEP_SIZE_VALID
            )
            STEP_SIZE_TEST = max(1, test_set.samples //
                                 test_set.batch_size)
            (eval_loss, eval_accuracy) = self.model.evaluate(
                test_set, steps=STEP_SIZE_TEST, verbose=1)
            print('Loss = ', eval_loss)
            print('Accuracy = ', eval_accuracy)
            print("Total run-time: %f seconds" % (time.time() - time_start))
            # os.system(
            #     "say Your experiment has finished. Please collect your result")
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

        # plot figure after training model
        epochs = len(self.history.history['accuracy'])
        epoch_arr = [i for i in range(epochs)]
        train_acc = self.history.history['accuracy']
        train_loss = self.history.history['loss']

        val_acc = self.history.history['val_accuracy']
        val_loss = self.history.history['val_loss']

        self.plot_fig(epoch_arr, 'Accuracy',
                      os.path.join(directory, id+'_acc.png'), train_acc, val_acc)
        self.plot_fig(epoch_arr, 'Loss', os.path.join(directory, id+'_loss.png'),
                      train_loss, val_loss)
        return self

    def load(self, id):
        directory = os.path.join(self.saveDir, id)
        model_dir = os.path.join(directory, id + '.h5')
        meta_dir = os.path.join(directory, id + '.meta')

        self.model = keras.models.load_model(model_dir)

        with open(meta_dir, 'r') as f:
            classes = f.readline()
            self.classes = json.loads(classes)
            logger.debug("Loaded classes:", self.classes)
        return self


if __name__ == '__main__':
    logger.debug('Tensorflow version:', tensorflow.__version__)
