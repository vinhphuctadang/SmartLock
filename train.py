import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from face_wrapper import FaceWrapper as FW 

def buildCNN(self):
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dense(2, activation='softmax')) # no softmax activation applied, we will use logit
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def main():
    face = FW()
    face._build_model = buildCNN
    face.classes = ['phuc', 'quy']

    face.runWorkFlow(epochs=3).save('sample')
main()